"""Multi-agent crew (LangGraph) — Personal Finance Coach.

Five specialised agents:
  * Router       — classifies intent, routes the request
  * Guardian     — fraud escalation & out-of-scope rejection
  * DataAnalyst  — runs the finance tools (tool_use loop)
  * Advisor      — turns numbers into concrete, actionable advice
  * Synthesizer  — composes the final user-facing answer (friendly 'ти' tone)

Topology:
    START → router → {fraud→guardian, out_of_scope→guardian, else→data_analyst}
    data_analyst → advisor → synthesizer → END
    guardian → END

Uses LangGraph when installed. If LangGraph is unavailable the SAME node
functions run through a minimal sequential executor, so behaviour is identical
and the offline test suite still exercises the full crew.
"""
from __future__ import annotations

import json
import time
from typing import Any, TypedDict

from . import llm, tools
from .config import SETTINGS
from .obs import traceable
from .types import RunResult, TraceStep

try:
    from langgraph.graph import END, START, StateGraph  # type: ignore

    HAVE_LANGGRAPH = True
except Exception:
    HAVE_LANGGRAPH = False


class CrewState(TypedDict, total=False):
    query: str
    history: list[dict]
    route: str
    ctx: dict
    tool_results: list[dict]
    draft: str
    answer: str
    result: RunResult


# ---- node implementations ---------------------------------------------------

def _hist_users(state: CrewState) -> list[str]:
    return [h["content"] for h in state.get("history", []) if h.get("role") == "user"]


def node_router(state: CrewState) -> CrewState:
    res = state["result"]
    s0 = time.perf_counter()
    resp = llm.complete("router", state["query"], history=_hist_users(state),
                        model=SETTINGS.model_cheap)
    ctx = json.loads(resp.content)
    res.agents_used.append("router")
    res.inter_agent_tokens += resp.usage["prompt_tokens"]
    _acc(res, resp, "agent", "router", f"intent={ctx['intent']}",
         (time.perf_counter() - s0) * 1000)
    route = ctx["intent"] if ctx["intent"] in {"fraud", "out_of_scope"} else "analyze"
    return {"ctx": ctx, "route": route}


def node_guardian(state: CrewState) -> CrewState:
    res = state["result"]
    ctx = state["ctx"]
    s0 = time.perf_counter()
    resp = llm.complete("guardian", state["query"], ctx=ctx, model=SETTINGS.model_cheap)
    res.agents_used.append("guardian")
    res.inter_agent_tokens += resp.usage["prompt_tokens"]
    results: list[dict] = []
    if ctx["intent"] == "fraud":
        r = tools.call_tool("detect_suspicious", {"account": "credit_card"})
        results.append(r)
        res.tools_used.append("detect_suspicious")
        res.trace.append(TraceStep(kind="tool", name="detect_suspicious",
                                   detail="account=credit_card", data=r))
    _acc(res, resp, "agent", "guardian", resp.content, (time.perf_counter() - s0) * 1000)
    # compose final via synthesizer-style completion
    final = llm.complete("synthesizer", state["query"], ctx=ctx, results=results,
                         model=SETTINGS.model_smart)
    _acc(res, final, "agent", "synthesizer", "escalation/oos reply", 0.0)
    res.agents_used.append("synthesizer")
    return {"answer": final.content, "tool_results": results}


def node_data_analyst(state: CrewState) -> CrewState:
    res = state["result"]
    messages = [
        {"role": "system", "content": "Ти — Data Analyst. Викликай інструменти, щоб "
                                      "дістати точні числа для запиту. Не вигадуй суми."},
    ]
    for h in state.get("history", []):
        messages.append(h)
    messages.append({"role": "user", "content": state["query"]})

    results: list[dict] = []
    for _ in range(4):
        s0 = time.perf_counter()
        resp = llm.chat(messages, tools=tools.TOOL_SCHEMAS,
                        model=SETTINGS.model_cheap, role="data_analyst")
        _acc(res, resp, "agent", "data_analyst",
             "tool calls" if resp.tool_calls else "done", (time.perf_counter() - s0) * 1000)
        if "data_analyst" not in res.agents_used:
            res.agents_used.append("data_analyst")
        if not resp.tool_calls:
            break
        messages.append({
            "role": "assistant", "content": resp.content,
            "tool_calls": [
                {"id": tc["id"], "type": "function",
                 "function": {"name": tc["name"], "arguments": json.dumps(tc["arguments"])}}
                for tc in resp.tool_calls
            ],
        })
        for tc in resp.tool_calls:
            ts0 = time.perf_counter()
            r = tools.call_tool(tc["name"], tc["arguments"])
            results.append(r)
            res.tools_used.append(tc["name"])
            res.trace.append(TraceStep(kind="tool", name=tc["name"],
                                       detail=json.dumps(tc["arguments"], ensure_ascii=False),
                                       latency_ms=(time.perf_counter() - ts0) * 1000, data=r))
            messages.append({"role": "tool", "tool_call_id": tc["id"], "name": tc["name"],
                             "content": json.dumps(r, ensure_ascii=False)})
    return {"tool_results": results}


def node_advisor(state: CrewState) -> CrewState:
    res = state["result"]
    ctx = state["ctx"]
    if ctx["intent"] not in {"advice", "multistep"}:
        return {"draft": ""}  # stat queries skip advisory shaping
    s0 = time.perf_counter()
    resp = llm.complete("advisor", state["query"], history=_hist_users(state),
                        results=state.get("tool_results", []), ctx=ctx,
                        model=SETTINGS.model_smart)
    res.agents_used.append("advisor")
    res.inter_agent_tokens += resp.usage["prompt_tokens"]
    _acc(res, resp, "agent", "advisor", "advice draft", (time.perf_counter() - s0) * 1000)
    return {"draft": resp.content}


def node_synthesizer(state: CrewState) -> CrewState:
    res = state["result"]
    ctx = state["ctx"]
    s0 = time.perf_counter()
    resp = llm.complete("synthesizer", state["query"], history=_hist_users(state),
                        results=state.get("tool_results", []),
                        draft=state.get("draft") or None, ctx=ctx,
                        model=SETTINGS.model_smart)
    res.agents_used.append("synthesizer")
    res.inter_agent_tokens += resp.usage["prompt_tokens"]
    _acc(res, resp, "agent", "synthesizer", "final answer",
         (time.perf_counter() - s0) * 1000)
    return {"answer": resp.content}


def _acc(res: RunResult, resp, kind, name, detail, latency_ms) -> None:
    res.prompt_tokens += resp.usage["prompt_tokens"]
    res.completion_tokens += resp.usage["completion_tokens"]
    res.cost += resp.cost()
    res.trace.append(TraceStep(
        kind=kind, name=name, detail=detail, latency_ms=latency_ms,
        tokens=resp.usage["prompt_tokens"] + resp.usage["completion_tokens"], cost=resp.cost(),
    ))


# ---- graph assembly ---------------------------------------------------------

def _route_decision(state: CrewState) -> str:
    return state["route"]


def build_graph():
    g = StateGraph(CrewState)
    g.add_node("router", node_router)
    g.add_node("guardian", node_guardian)
    g.add_node("data_analyst", node_data_analyst)
    g.add_node("advisor", node_advisor)
    g.add_node("synthesizer", node_synthesizer)
    g.add_edge(START, "router")
    g.add_conditional_edges("router", _route_decision,
                            {"fraud": "guardian", "out_of_scope": "guardian",
                             "analyze": "data_analyst"})
    g.add_edge("guardian", END)
    g.add_edge("data_analyst", "advisor")
    g.add_edge("advisor", "synthesizer")
    g.add_edge("synthesizer", END)
    return g.compile()


_GRAPH = build_graph() if HAVE_LANGGRAPH else None


def _run_fallback(state: CrewState) -> CrewState:
    """Identical node sequence without LangGraph (for envs lacking the dep)."""
    state.update(node_router(state))
    if state["route"] in {"fraud", "out_of_scope"}:
        state.update(node_guardian(state))
        return state
    state.update(node_data_analyst(state))
    state.update(node_advisor(state))
    state.update(node_synthesizer(state))
    return state


@traceable(name="crew", tags=["architecture:crew"])
def run(query: str, history: list[dict] | None = None) -> RunResult:
    t0 = time.perf_counter()
    res = RunResult(answer="", architecture="crew")
    state: CrewState = {"query": query, "history": history or [], "result": res,
                        "tool_results": []}
    if HAVE_LANGGRAPH:
        out = _GRAPH.invoke(state)
    else:
        out = _run_fallback(state)
    res.answer = out.get("answer", "")
    res.latency_ms = (time.perf_counter() - t0) * 1000
    return res
