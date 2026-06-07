"""Shared endpoint: dispatch a query to either architecture on one contract.

Also provides a tiny in-memory multi-turn session so follow-ups like
'а за місяць?' resolve against the previous turn.
"""
from __future__ import annotations

from . import baseline, crew
from .config import SETTINGS
from .types import RunResult

ARCHITECTURES = {"crew": crew.run, "baseline": baseline.run}


def answer(query: str, architecture: str = "crew",
           history: list[dict] | None = None) -> RunResult:
    if architecture not in ARCHITECTURES:
        raise ValueError(f"unknown architecture: {architecture}")
    return ARCHITECTURES[architecture](query, history=history)


class Session:
    """Maintains conversational history for multi-turn context."""

    def __init__(self, architecture: str = "crew", max_turns: int = 6):
        self.architecture = architecture
        self.max_turns = max_turns
        self.history: list[dict] = []

    def ask(self, query: str) -> RunResult:
        res = answer(query, self.architecture, history=self.history)
        self.history.append({"role": "user", "content": query})
        self.history.append({"role": "assistant", "content": res.answer})
        # keep history bounded
        self.history = self.history[-2 * self.max_turns:]
        return res

    def reset(self) -> None:
        self.history = []


def mode() -> str:
    return "offline (mock LLM)" if SETTINGS.offline else "online (OpenRouter)"
