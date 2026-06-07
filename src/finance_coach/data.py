"""Data layer: loads transactions.csv into SQLite and exposes a cached DataFrame.

SQLite satisfies the 'Storage' requirement; pandas is used for analytics on top.
"""
from __future__ import annotations

import sqlite3
from datetime import date, timedelta
from functools import lru_cache

import pandas as pd

from .config import DATA_CSV, DB_PATH, SETTINGS


def build_db() -> None:
    """(Re)build the SQLite database from the CSV. Idempotent."""
    df = pd.read_csv(DATA_CSV)
    con = sqlite3.connect(DB_PATH)
    df.to_sql("transactions", con, if_exists="replace", index=False)
    con.execute("CREATE INDEX IF NOT EXISTS idx_cat ON transactions(category)")
    con.execute("CREATE INDEX IF NOT EXISTS idx_merchant ON transactions(merchant)")
    con.commit()
    con.close()


@lru_cache(maxsize=1)
def load_df() -> pd.DataFrame:
    """Cached transactions DataFrame, read from SQLite (built on first use)."""
    if not DB_PATH.exists():
        build_db()
    con = sqlite3.connect(DB_PATH)
    df = pd.read_sql("SELECT * FROM transactions", con)
    con.close()
    df["date"] = pd.to_datetime(df["date"])
    df["amount"] = df["amount"].astype(float)
    df["recurring"] = df["recurring"].astype(str).str.lower().isin({"true", "1"})
    return df


# ---- relative-date resolution (anchored to SETTINGS.today) -------------------

def today() -> date:
    return SETTINGS.today


def resolve_period(period: str | None) -> tuple[date | None, date | None]:
    """Map a named period to an inclusive (start, end) date range.

    Supported: last_week, this_week, last_month, this_month, last_30_days,
    last_3_months, this_year, last_year, ytd, all. None -> (None, None).
    """
    if not period or period == "all":
        return None, None
    t = today()
    p = period.lower().strip()
    if p == "last_week":
        end = t - timedelta(days=t.weekday() + 1)  # last Sunday
        start = end - timedelta(days=6)
        return start, end
    if p == "this_week":
        start = t - timedelta(days=t.weekday())
        return start, t
    if p == "last_30_days":
        return t - timedelta(days=29), t
    if p == "this_month":
        return date(t.year, t.month, 1), t
    if p == "last_month":
        first_this = date(t.year, t.month, 1)
        last_prev = first_this - timedelta(days=1)
        return date(last_prev.year, last_prev.month, 1), last_prev
    if p == "last_3_months":
        return t - timedelta(days=89), t
    if p in {"this_year", "ytd"}:
        return date(t.year, 1, 1), t
    if p == "last_year":
        return date(t.year - 1, 1, 1), date(t.year - 1, 12, 31)
    return None, None


def filter_df(
    df: pd.DataFrame,
    start: date | None = None,
    end: date | None = None,
    category: str | None = None,
    merchant: str | None = None,
    account: str | None = None,
) -> pd.DataFrame:
    out = df
    if start is not None:
        out = out[out["date"].dt.date >= start]
    if end is not None:
        out = out[out["date"].dt.date <= end]
    if category:
        out = out[out["category"].str.lower() == category.lower()]
    if merchant:
        out = out[out["merchant"].str.contains(merchant, case=False, na=False)]
    if account:
        out = out[out["account"].str.lower() == account.lower()]
    return out
