"""Run the golden set on both architectures and emit metrics.

Default: deterministic judge + evaluators, writes JSON + Markdown side-by-side
to report/. With --langsmith (and LANGSMITH_API_KEY) it also uploads the dataset
and runs LangSmith Experiments with the same evaluators for native trace metrics.

Usage:
    python evals/run_experiments.py                       # metrics -> report/
    python evals/run_experiments.py --langsmith --llm-judge
"""
from __future__ import annotations

import json
import statistics
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "evals"))

from finance_coach import endpoint  # noqa: E402
from finance_coach.config import SETTINGS  # noqa: E402
import evaluators as E  # noqa: E402
from golden_set import GOLDEN, by_category  # noqa: E402

OUT_DIR = ROOT / "report"
OUT_DIR.mkdir(exist_ok=True)


def _pctl(values: list[float], p: float) -> float:
    if not values:
        return 0.0
    s = sorted(values)
    k = (len(s) - 1) * p
    f = int(k)
    c = min(f + 1, len(s) - 1)
    return round(s[f] + (s[c] - s[f]) * (k - f), 1)


def run_architecture(arch: str, use_llm_judge: bool = False) -> dict:
    rows = []
    latencies, costs, tokens = [], [], []
    succ, toolacc, ground = [], [], []
    agent_cost: dict[str, float] = defaultdict(float)
    overheads = []

    for task in GOLDEN:
        res = endpoint.answer(task["query"], arch, history=task.get("history"))
        tool_results = [s.data for s in res.trace if s.kind == "tool" and s.data]
        s = E.judge_success(task, res.answer, res.tools_used, use_llm=use_llm_judge)
        ta = E.tool_selection_accuracy(task.get("expected_tools", []), res.tools_used)
        g = E.groundedness(res.answer, tool_results)
        succ.append(s); toolacc.append(ta); ground.append(g)
        latencies.append(res.latency_ms); costs.append(res.cost); tokens.append(res.total_tokens)
        overheads.append(res.inter_agent_overhead_pct)
        for st in res.trace:
            if st.kind in {"agent", "llm"}:
                agent_cost[st.name] += st.cost
        rows.append({
            "id": task["id"], "category": task["category"], "success": s,
            "tool_accuracy": ta, "groundedness": g,
            "latency_ms": round(res.latency_ms, 1), "cost": round(res.cost, 6),
            "tokens": res.total_tokens, "tools": res.tools_used,
            "agents": res.agents_used, "answer": res.answer,
        })

    n = len(GOLDEN)
    return {
        "architecture": arch,
        "n_tasks": n,
        "success_rate": round(sum(succ) / n, 3),
        "tool_selection_accuracy": round(sum(toolacc) / n, 3),
        "groundedness": round(sum(ground) / n, 3),
        "latency_p50_ms": _pctl(latencies, 0.5),
        "latency_p95_ms": _pctl(latencies, 0.95),
        "cost_per_task": round(sum(costs) / n, 6),
        "tokens_per_task": round(sum(tokens) / n, 1),
        "inter_agent_overhead_pct": round(sum(overheads) / n, 2),
        "cost_breakdown_by_agent": {k: round(v, 6) for k, v in sorted(
            agent_cost.items(), key=lambda x: -x[1])},
        "rows": rows,
    }


def render_markdown(crew: dict, base: dict) -> str:
    def line(label, key, fmt="{}"):
        return f"| {label} | {fmt.format(base[key])} | {fmt.format(crew[key])} |"

    md = ["# Golden Set Results — Crew vs Baseline", "",
          f"LLM: **OpenRouter** · Tasks: {crew['n_tasks']} "
          f"({', '.join(f'{k}:{v}' for k, v in by_category().items())})", "",
          "| Metric | Baseline | Crew |", "|---|---|---|",
          line("success_rate", "success_rate"),
          line("tool_selection_accuracy", "tool_selection_accuracy"),
          line("groundedness", "groundedness"),
          line("latency_p50 (ms)", "latency_p50_ms"),
          line("latency_p95 (ms)", "latency_p95_ms"),
          line("cost_per_task ($)", "cost_per_task"),
          line("tokens_per_task", "tokens_per_task"),
          line("inter_agent_overhead_pct", "inter_agent_overhead_pct"),
          ""]
    md.append("## Cost breakdown by agent (crew)\n")
    md.append("| Agent | Cost ($) |\n|---|---|")
    for k, v in crew["cost_breakdown_by_agent"].items():
        md.append(f"| {k} | {v} |")
    md.append("\n## Per-task success (1 = pass)\n")
    md.append("| Task | Category | Baseline | Crew |\n|---|---|---|---|")
    by_id = {r["id"]: r for r in base["rows"]}
    for r in crew["rows"]:
        b = by_id[r["id"]]
        md.append(f"| {r['id']} | {r['category']} | {b['success']:.0f} | {r['success']:.0f} |")
    return "\n".join(md)


def maybe_langsmith():
    """Upload dataset + run LangSmith experiments when configured."""
    if not SETTINGS.langsmith_api_key:
        print("[langsmith] no LANGSMITH_API_KEY — skipping cloud experiments.")
        return
    try:
        from langsmith import Client
    except Exception as e:
        print(f"[langsmith] client unavailable: {e}")
        return
    client = Client()
    ds_name = "personal-finance-coach-golden"
    try:
        ds = client.create_dataset(ds_name, description="Finance coach golden set")
    except Exception:
        ds = client.read_dataset(dataset_name=ds_name)
    existing = list(client.list_examples(dataset_id=ds.id))
    if not existing:
        for t in GOLDEN:
            client.create_example(
                inputs={"query": t["query"], "history": t.get("history")},
                outputs={"must_include": t.get("must_include", []),
                         "expected_tools": t.get("expected_tools", [])},
                dataset_id=ds.id, metadata={"id": t["id"], "category": t["category"]},
            )
        print(f"[langsmith] uploaded {len(GOLDEN)} examples to '{ds_name}'.")

    def make_target(arch):
        def target(inputs: dict) -> dict:
            res = endpoint.answer(inputs["query"], arch, history=inputs.get("history"))
            return {"answer": res.answer, "tools_used": res.tools_used,
                    "tool_results": [s.data for s in res.trace if s.kind == "tool"]}
        return target

    def ev_success(run, example):
        task = {"query": example.inputs["query"],
                "must_include": example.outputs.get("must_include", []),
                "expected_tools": example.outputs.get("expected_tools", [])}
        score = E.judge_success(task, run.outputs["answer"], run.outputs["tools_used"])
        return {"key": "success", "score": score}

    def ev_tools(run, example):
        score = E.tool_selection_accuracy(example.outputs.get("expected_tools", []),
                                          run.outputs["tools_used"])
        return {"key": "tool_selection_accuracy", "score": score}

    def ev_ground(run, example):
        score = E.groundedness(run.outputs["answer"], run.outputs.get("tool_results", []))
        return {"key": "groundedness", "score": score}

    for arch in ["baseline", "crew"]:
        client.evaluate(make_target(arch), data=ds_name,
                        evaluators=[ev_success, ev_tools, ev_ground],
                        experiment_prefix=f"finance-{arch}")
        print(f"[langsmith] experiment finished for {arch}.")


def main():
    use_ls = "--langsmith" in sys.argv
    llm_judge = "--llm-judge" in sys.argv
    crew = run_architecture("crew", use_llm_judge=llm_judge)
    base = run_architecture("baseline", use_llm_judge=llm_judge)
    (OUT_DIR / "results.json").write_text(
        json.dumps({"crew": crew, "baseline": base}, ensure_ascii=False, indent=2))
    md = render_markdown(crew, base)
    (OUT_DIR / "results.md").write_text(md)
    print(md)
    if use_ls:
        maybe_langsmith()


if __name__ == "__main__":
    main()
