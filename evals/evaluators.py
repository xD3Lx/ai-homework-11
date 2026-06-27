"""Custom evaluators for the golden set.

Three quality metrics required by the brief:
  * success_rate            — did the answer pass the (LLM/deterministic) judge
  * tool_selection_accuracy — were the expected tools actually used
  * groundedness            — are the numbers in the answer backed by tool data

Each evaluator returns a float in [0,1] so it works as a LangSmith evaluator and
in the local runner. `judge_success` uses deterministic substring/number checks
by default; pass use_llm=True to defer to an LLM-as-judge.
"""
from __future__ import annotations

import re
from typing import Any

NUM_RE = re.compile(r"-?\d[\d,]*\.?\d*")


def _nums(text: str) -> set[float]:
    out: set[float] = set()
    for m in NUM_RE.findall(text or ""):
        try:
            out.add(round(float(m.replace(",", "")), 2))
        except ValueError:
            pass
    return out


def tool_selection_accuracy(expected: list[str], used: list[str]) -> float:
    """Fraction of expected tools that were actually used (1.0 if none expected)."""
    exp = set(expected)
    if not exp:
        # for no-tool tasks, accurate iff no tools were used
        return 1.0 if not used else 0.0
    return round(len(exp & set(used)) / len(exp), 3)


def groundedness(answer: str, tool_results: list[dict]) -> float:
    """Share of numbers in the answer that appear in the tool outputs.

    Ignores trivially small integers (0-31: dates/counts) to focus on monetary
    claims. Returns 1.0 when the answer makes no substantive numeric claim.
    """
    grounded_pool: set[float] = set()
    for r in tool_results:
        grounded_pool |= _nums(_flatten(r))
    # also accept simple derived values (half, x12, percentages) within tolerance
    answer_nums = {n for n in _nums(answer) if abs(n) > 31}
    if not answer_nums:
        return 1.0
    ok = 0
    for n in answer_nums:
        if _is_grounded(n, grounded_pool):
            ok += 1
    return round(ok / len(answer_nums), 3)


def _is_grounded(n: float, pool: set[float], tol: float = 0.02) -> bool:
    for p in pool:
        if p == 0:
            continue
        if abs(n - p) <= max(0.5, abs(p) * tol):
            return True
        # accept common derivations: half, double, *12 (annualisation), *0.4
        for factor in (0.5, 2.0, 12.0, 0.4, 0.6):
            if abs(n - p * factor) <= max(0.5, abs(p * factor) * tol):
                return True
    return False


def _flatten(obj: Any) -> str:
    return str(obj)


def judge_success(task: dict, answer: str, used_tools: list[str],
                  use_llm: bool = False) -> float:
    """Deterministic judge: required substrings present, forbidden absent,
    and key tools used. Set use_llm=True (online) to defer to an LLM judge."""
    if use_llm:
        return _llm_judge(task, answer)
    ans = (answer or "").lower()
    for sub in task.get("must_include", []):
        if sub.lower() not in ans:
            return 0.0
    for sub in task.get("must_not_include", []):
        if sub.lower() in ans:
            return 0.0
    # tool expectation: at least one expected tool used (or none expected)
    exp = set(task.get("expected_tools", []))
    if exp and not (exp & set(used_tools)):
        return 0.0
    return 1.0


def _llm_judge(task: dict, answer: str) -> float:
    """LLM-as-judge (used in online mode)."""
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
    from finance_coach import llm

    prompt = (
        f"Запит: {task['query']}\n"
        f"Очікувані факти/підрядки: {task.get('must_include')}\n"
        f"Відповідь агента: {answer}\n\n"
        "Чи коректно відповідь обробляє запит і містить очікувані факти? "
        'Поверни лише JSON {"pass": true|false}.'
    )
    resp = llm.chat([{"role": "user", "content": prompt}])
    import json
    try:
        return 1.0 if json.loads(resp.content).get("pass") else 0.0
    except Exception:
        return 0.0
