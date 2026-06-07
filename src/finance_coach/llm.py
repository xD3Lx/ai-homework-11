"""Unified LLM interface over OpenRouter with a deterministic offline emulator.

Two entry points:
  * chat()     — message-level tool-calling loop (baseline agent, data analyst)
  * complete() — single role-tagged completion (router, guardian, advisor, synth)

In ONLINE mode both call OpenRouter (OpenAI-compatible). In OFFLINE mode both are
served by offline_brain so the system is fully runnable without an API key.
"""
from __future__ import annotations

import json
import uuid
from dataclasses import dataclass, field
from typing import Any

from . import offline_brain as brain
from .config import SETTINGS, price_for


@dataclass
class LLMResponse:
    content: str | None = None
    tool_calls: list[dict] = field(default_factory=list)  # {id,name,arguments}
    usage: dict = field(default_factory=lambda: {"prompt_tokens": 0, "completion_tokens": 0})
    model: str = ""

    def cost(self) -> float:
        p = price_for(self.model)
        return round(
            self.usage["prompt_tokens"] / 1e6 * p["in"]
            + self.usage["completion_tokens"] / 1e6 * p["out"],
            6,
        )


def _est_tokens(text: str) -> int:
    return max(1, len(text or "") // 4)


def _openrouter_client():
    from openai import OpenAI

    return OpenAI(
        api_key=SETTINGS.openrouter_api_key,
        base_url=SETTINGS.openrouter_base_url,
    )


# ---- chat (tool loop) -------------------------------------------------------

def chat(
    messages: list[dict],
    tools: list[dict] | None = None,
    model: str | None = None,
    role: str = "baseline",
) -> LLMResponse:
    model = model or SETTINGS.model_smart
    if SETTINGS.offline:
        return _mock_chat(messages, tools, model, role)

    client = _openrouter_client()
    kwargs: dict[str, Any] = {"model": model, "messages": messages, "temperature": 0}
    if tools:
        kwargs["tools"] = tools
        kwargs["tool_choice"] = "auto"
    resp = client.chat.completions.create(**kwargs)
    msg = resp.choices[0].message
    tcs = []
    for tc in (msg.tool_calls or []):
        try:
            args = json.loads(tc.function.arguments or "{}")
        except json.JSONDecodeError:
            args = {}
        tcs.append({"id": tc.id, "name": tc.function.name, "arguments": args})
    usage = {
        "prompt_tokens": resp.usage.prompt_tokens,
        "completion_tokens": resp.usage.completion_tokens,
    }
    return LLMResponse(content=msg.content, tool_calls=tcs, usage=usage, model=model)


def _mock_chat(messages, tools, model, role) -> LLMResponse:
    """Emulate OpenAI tool-calling using offline_brain."""
    users = [m["content"] for m in messages if m.get("role") == "user"]
    query = users[-1] if users else ""
    history = users[:-1]
    has_tool_results = any(m.get("role") == "tool" for m in messages)

    ctx = brain.classify(query, history)

    # First pass: decide on tool calls
    if not has_tool_results:
        planned = brain.plan_tools(query, ctx)
        if planned:
            tcs = [
                {"id": f"call_{uuid.uuid4().hex[:8]}", "name": p["name"],
                 "arguments": p["arguments"]}
                for p in planned
            ]
            prompt_tok = _est_tokens(" ".join(m.get("content") or "" for m in messages))
            return LLMResponse(
                content=None, tool_calls=tcs,
                usage={"prompt_tokens": prompt_tok, "completion_tokens": 30 * len(tcs)},
                model=model,
            )

    # Second pass (or no tools needed): compose the final answer
    results = []
    for m in messages:
        if m.get("role") == "tool":
            try:
                results.append(json.loads(m["content"]))
            except (json.JSONDecodeError, TypeError):
                pass
    answer = brain.compose_answer(query, ctx, results)
    prompt_tok = _est_tokens(" ".join(m.get("content") or "" for m in messages))
    return LLMResponse(
        content=answer, tool_calls=[],
        usage={"prompt_tokens": prompt_tok, "completion_tokens": _est_tokens(answer)},
        model=model,
    )


# ---- complete (role-tagged single turn) -------------------------------------

ROLE_SYSTEM = {
    "router": "Ти — маршрутизатор запитів фінансового помічника. Класифікуй запит "
              "користувача та визнач, які кроки потрібні.",
    "guardian": "Ти — модуль безпеки. Виявляй fraud та запити поза скоупом.",
    "advisor": "Ти — фінансовий радник. Давай конкретні поради на основі чисел, "
               "дружній тон на 'ти', без загальних рекомендацій.",
    "synthesizer": "Ти збираєш фінальну відповідь користувачу: дружній тон на 'ти', "
                   "числа лише з наданих даних.",
}


def complete(
    role: str,
    query: str,
    *,
    history: list[str] | None = None,
    results: list[dict] | None = None,
    draft: str | None = None,
    ctx: dict | None = None,
    model: str | None = None,
) -> LLMResponse:
    model = model or SETTINGS.model_smart
    if SETTINGS.offline:
        return _mock_complete(role, query, history, results, draft, ctx, model)

    # ONLINE: build a prompt per role and call the LLM
    sys = ROLE_SYSTEM.get(role, "Ти — корисний асистент.")
    parts = [f"Запит користувача: {query}"]
    if history:
        parts.append("Попередні репліки користувача: " + " | ".join(history))
    if results is not None:
        parts.append("Дані з інструментів (JSON):\n" + json.dumps(results, ensure_ascii=False))
    if draft:
        parts.append("Чернетка відповіді для полірування:\n" + draft)
    if role == "router":
        parts.append('Поверни JSON: {"intent": "...", "category": "...|null", '
                     '"merchant": "...|null", "period": "...|null"}.')
    if role == "guardian":
        parts.append('Поверни JSON: {"action": "escalate|reject|allow", "reason": "..."}.')
    user = "\n\n".join(parts)
    resp = chat([{"role": "system", "content": sys}, {"role": "user", "content": user}],
                model=model, role=role)
    return resp


def _mock_complete(role, query, history, results, draft, ctx, model) -> LLMResponse:
    ctx = ctx or brain.classify(query, history)
    if role == "router":
        content = json.dumps(ctx, ensure_ascii=False)
    elif role == "guardian":
        if ctx["intent"] == "fraud":
            action = {"action": "escalate", "reason": "suspected fraud"}
        elif ctx["intent"] == "out_of_scope":
            action = {"action": "reject", "reason": "out of scope"}
        else:
            action = {"action": "allow", "reason": "in scope"}
        content = json.dumps(action, ensure_ascii=False)
    elif role in {"advisor", "synthesizer"}:
        content = draft if draft else brain.compose_answer(query, ctx, results or [])
    else:
        content = brain.compose_answer(query, ctx, results or [])
    return LLMResponse(
        content=content, tool_calls=[],
        usage={"prompt_tokens": _est_tokens(query), "completion_tokens": _est_tokens(content)},
        model=model,
    )
