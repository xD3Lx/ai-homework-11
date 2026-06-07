"""Central configuration. Reads .env; degrades gracefully to offline mock mode."""
from __future__ import annotations

import os
from dataclasses import dataclass
from datetime import date
from pathlib import Path

try:
    from dotenv import load_dotenv

    load_dotenv()
except Exception:  # dotenv optional
    pass

import tempfile

ROOT = Path(__file__).resolve().parents[2]
DATA_CSV = ROOT / "starter" / "data" / "transactions.csv"
# SQLite lives in a writable temp dir (mounted folders may reject sqlite locks).
# Override with FINANCE_COACH_DB if you want it elsewhere.
DB_PATH = Path(
    os.getenv("FINANCE_COACH_DB", str(Path(tempfile.gettempdir()) / "finance_coach.db"))
)


def _to_bool(v: str | None, default: bool = False) -> bool:
    if v is None:
        return default
    return v.strip().lower() in {"1", "true", "yes", "on"}


@dataclass(frozen=True)
class Settings:
    openrouter_api_key: str = os.getenv("OPENROUTER_API_KEY", "").strip()
    openrouter_base_url: str = os.getenv(
        "OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1"
    )
    model_smart: str = os.getenv("MODEL_SMART", "anthropic/claude-3.5-sonnet")
    model_cheap: str = os.getenv("MODEL_CHEAP", "anthropic/claude-3.5-haiku")
    langsmith_api_key: str = os.getenv("LANGSMITH_API_KEY", "").strip()
    langsmith_project: str = os.getenv("LANGSMITH_PROJECT", "personal-finance-coach")
    now: str = os.getenv("FINANCE_COACH_NOW", "2025-11-30")
    _force_offline: bool = _to_bool(os.getenv("FINANCE_COACH_OFFLINE"), False)

    @property
    def offline(self) -> bool:
        """Offline (deterministic mock) when forced or no OpenRouter key."""
        return self._force_offline or not self.openrouter_api_key

    @property
    def today(self) -> date:
        return date.fromisoformat(self.now)


SETTINGS = Settings()

# Approx OpenRouter prices ($ per 1M tokens) for cost estimation in offline traces.
MODEL_PRICES = {
    "anthropic/claude-3.5-sonnet": {"in": 3.0, "out": 15.0},
    "anthropic/claude-3.5-haiku": {"in": 0.8, "out": 4.0},
    "openai/gpt-4o-mini": {"in": 0.15, "out": 0.6},
}


def price_for(model: str) -> dict[str, float]:
    return MODEL_PRICES.get(model, {"in": 1.0, "out": 5.0})
