"""Single-agent baseline: one LLM + a plain tool_use loop, no framework.

Same tool set and endpoint contract as the crew, so the two are directly
comparable in LangSmith experiments.
"""
from __future__ import annotations

import json
import time

from .obs import traceable

from . import llm, tools
from .format import normalize_money
from .config import SETTINGS
from .types import RunResult, TraceStep

SYSTEM_PROMPT = (
    "Ти — Personal Finance Coach у банківському застосунку. Відповідай дружньо, "
    "на 'ти', без менторства. Усі числа бери ВИКЛЮЧНО з результатів інструментів — "
    "ніколи не вигадуй суми. Якщо запит про підозрілу транзакцію/fraud — не вирішуй "
    "сам, направ користувача до служби підтримки. Якщо запит поза скоупом (інвестиції, "
    "покупка активів тощо) — ввічливо відмов і запропонуй доступні функції. "
    f"Сьогоднішня дата (today) у системі: {SETTINGS.today.isoformat()} — рахуй усі "
    "відносні періоди саме від неї, а не від реальної поточної дати. Для відносних "
    "діапазонів використовуй параметр `period` (last_week, this_month тощо), а не "
    "власні start/end. "
    "ВАЛЮТА в датасеті — долари США: завжди показуй суми у '$' з КРАПКОЮ як "
    "десятковим роздільником (напр. 2.90), ніколи не пиши 'грн' чи інші валюти."
)

MAX_STEPS = 5


@traceable(name="baseline_agent", tags=["architecture:baseline"])
def run(query: str, history: list[dict] | None = None) -> RunResult:
    """history: list of {'role','content'} prior turns (multi-turn context)."""
    t0 = time.perf_counter()
    res = RunResult(answer="", architecture="baseline", agents_used=["baseline"])

    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    for h in history or []:
        messages.append(h)
    messages.append({"role": "user", "content": query})

    for _ in range(MAX_STEPS):
        s0 = time.perf_counter()
        resp = llm.chat(messages, tools=tools.TOOL_SCHEMAS,
                        model=SETTINGS.model_smart, role="baseline")
        dt = (time.perf_counter() - s0) * 1000
        res.prompt_tokens += resp.usage["prompt_tokens"]
        res.completion_tokens += resp.usage["completion_tokens"]
        res.cost += resp.cost()
        res.trace.append(TraceStep(
            kind="llm", name="baseline", detail="tool decision" if resp.tool_calls else "final answer",
            latency_ms=dt, tokens=resp.usage["prompt_tokens"] + resp.usage["completion_tokens"],
            cost=resp.cost(),
        ))

        if not resp.tool_calls:
            res.answer = resp.content or ""
            break

        # record assistant tool-call turn
        messages.append({
            "role": "assistant",
            "content": resp.content,
            "tool_calls": [
                {"id": tc["id"], "type": "function",
                 "function": {"name": tc["name"], "arguments": json.dumps(tc["arguments"])}}
                for tc in resp.tool_calls
            ],
        })
        for tc in resp.tool_calls:
            ts0 = time.perf_counter()
            result = tools.call_tool(tc["name"], tc["arguments"])
            tdt = (time.perf_counter() - ts0) * 1000
            res.tools_used.append(tc["name"])
            res.trace.append(TraceStep(
                kind="tool", name=tc["name"],
                detail=json.dumps(tc["arguments"], ensure_ascii=False),
                latency_ms=tdt, data=result,
            ))
            messages.append({
                "role": "tool", "tool_call_id": tc["id"], "name": tc["name"],
                "content": json.dumps(result, ensure_ascii=False),
            })
    else:
        if not res.answer:
            res.answer = "Не вдалося завершити аналіз за відведену кількість кроків."

    res.answer = normalize_money(res.answer)
    res.latency_ms = (time.perf_counter() - t0) * 1000
    return res
