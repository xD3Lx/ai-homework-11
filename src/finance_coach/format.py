"""Deterministic money formatting for final answers.

The dataset is single-currency. LLMs occasionally render amounts with a comma
decimal separator or guess a wrong currency (e.g. 'грн') from Ukrainian merchant
names. This module enforces the dataset's real currency and a dot decimal on the
final answer, independent of what the model produced. Self-contained: it only
reads the dataset's `currency` column.
"""
from __future__ import annotations

import re
from functools import lru_cache

from . import data as D

CURRENCY_SYMBOLS = {
    "USD": "$", "EUR": "€", "GBP": "£", "UAH": "₴", "PLN": "zł",
    "JPY": "¥", "CHF": "CHF", "CAD": "C$", "AUD": "A$",
}

# Currency tokens the model might wrongly emit; all are rewritten to the
# dataset's actual currency symbol. Longest-first so 'грн.' beats 'грн'.
_FOREIGN_TOKENS = sorted(
    ["грн.", "грн", "гривень", "гривнях", "гривні", "гривня", "₴",
     "UAH", "uah", "EUR", "eur", "€", "USD", "usd"],
    key=len, reverse=True,
)

_DECIMAL_COMMA = re.compile(r"(\d),(\d{1,2})(?!\d)")


@lru_cache(maxsize=1)
def dataset_currency() -> str:
    """The single currency code present in the dataset (defaults to USD)."""
    df = D.load_df()
    codes = [str(c) for c in df["currency"].dropna().unique()]
    return codes[0] if codes else "USD"


def dataset_symbol() -> str:
    code = dataset_currency()
    return CURRENCY_SYMBOLS.get(code, code)


def normalize_money(text: str | None, symbol: str | None = None) -> str:
    """Force dot decimals and the dataset currency symbol in `text`."""
    if not text:
        return text or ""
    symbol = symbol or dataset_symbol()
    # 1) comma decimal -> dot (3-digit thousands groups like 1,167 stay intact)
    out = _DECIMAL_COMMA.sub(r"\1.\2", text)
    # 2) rewrite any wrong currency token to the dataset symbol
    for tok in _FOREIGN_TOKENS:
        if tok == symbol:
            continue
        out = re.sub(rf"(?<!\w){re.escape(tok)}(?!\w)", symbol, out)
    # 3) tidy accidental duplicates like "$ $" / "$$"
    out = re.sub(r"\$\s+\$", "$", out)
    out = re.sub(r"\$\$+", "$", out)
    return out
