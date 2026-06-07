"""Shared result/trace types for both architectures."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class TraceStep:
    kind: str            # "agent" | "tool" | "llm"
    name: str            # agent name or tool name
    detail: str = ""     # short human-readable detail
    latency_ms: float = 0.0
    tokens: int = 0
    cost: float = 0.0
    data: Any = None     # raw payload (tool result / args)


@dataclass
class RunResult:
    answer: str
    architecture: str
    trace: list[TraceStep] = field(default_factory=list)
    latency_ms: float = 0.0
    cost: float = 0.0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    tools_used: list[str] = field(default_factory=list)
    agents_used: list[str] = field(default_factory=list)
    # multi-agent specific
    inter_agent_tokens: int = 0

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens

    @property
    def inter_agent_overhead_pct(self) -> float:
        return round(100 * self.inter_agent_tokens / self.total_tokens, 2) if self.total_tokens else 0.0

    def to_dict(self) -> dict:
        return {
            "answer": self.answer,
            "architecture": self.architecture,
            "latency_ms": round(self.latency_ms, 1),
            "cost": round(self.cost, 6),
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "total_tokens": self.total_tokens,
            "tools_used": self.tools_used,
            "agents_used": self.agents_used,
            "inter_agent_overhead_pct": self.inter_agent_overhead_pct,
            "trace": [
                {
                    "kind": s.kind, "name": s.name, "detail": s.detail,
                    "latency_ms": round(s.latency_ms, 1), "tokens": s.tokens,
                    "cost": round(s.cost, 6),
                }
                for s in self.trace
            ],
        }
