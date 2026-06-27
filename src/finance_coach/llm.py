"""Unified LLM interface over OpenRouter (OpenAI-compatible).

Two entry points:
  * chat()     — message-level tool-calling loop (baseline agent, data analyst)
  * complete() — single role-tagged completion (router, guardian, advisor, synth)
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field

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


def _openrouter_client():
    if not SETTINGS.openrouter_api_key:
        raise RuntimeError(
            "OPENROUTER_API_KEY is not set. Add it to your .env to run the agents."
        )
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
    client = _openrouter_client()
    kwargs: dict = {"model": model, "messages": messages, "temperature": 0}
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
    sys = ROLE_SYSTEM.get(role, "Ти — корисний асистент.")
    parts = [f"Запит користувача: {query}"]
    if history:
        parts.append("Попередні репліки користувача: " + " | ".join(history))
    if results is not None:
        parts.append("Дані з інструментів (JSON):\n" + json.dumps(results, ensure_ascii=False))
    if draft:
        parts.append("Чернетка відповіді для полірування:\n" + draft)
    if role == "router":
        parts.append(
            'Поверни ЛИШЕ JSON без жодного тексту до чи після: '
            '{"intent": "stat|advice|multistep|fraud|out_of_scope", '
            '"category": "...|null", "merchant": "...|null", "period": "...|null"}.'
        )
    if role == "guardian":
        parts.append('Поверни ЛИШЕ JSON: {"action": "escalate|reject|allow", "reason": "..."}.')
    user = "\n\n".join(parts)
    return chat(
        [{"role": "system", "content": sys}, {"role": "user", "content": user}],
        model=model, role=role,
    )
