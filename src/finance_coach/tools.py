"""Finance tools — deterministic functions over the transaction data.

Every tool returns a JSON-serialisable dict. All numeric claims an agent makes
must come from these tools (groundedness), so each result echoes the exact
figures used.
"""
from __future__ import annotations

from datetime import date, datetime
from typing import Any, Callable

import pandas as pd

from . import data as D


def _round(x: float) -> float:
    return round(float(x), 2)


def _sample(df: pd.DataFrame, n: int = 5) -> list[dict]:
    cols = ["date", "merchant", "amount", "category", "account"]
    rows = df.sort_values("date", ascending=False).head(n)[cols]
    out = []
    for _, r in rows.iterrows():
        out.append(
            {
                "date": r["date"].strftime("%Y-%m-%d %H:%M"),
                "merchant": r["merchant"],
                "amount": _round(r["amount"]),
                "category": r["category"],
                "account": r["account"],
            }
        )
    return out


# --- tools -------------------------------------------------------------------

def get_spending(
    category: str | None = None,
    merchant: str | None = None,
    period: str | None = None,
    start: str | None = None,
    end: str | None = None,
    account: str | None = None,
) -> dict[str, Any]:
    """Total spend (expenses only) for an optional category/merchant/period."""
    df = D.load_df()
    s, e = D.resolve_period(period)
    if start:
        s = date.fromisoformat(start)
    if end:
        e = date.fromisoformat(end)
    f = D.filter_df(df, s, e, category, merchant, account)
    f = f[f["amount"] < 0]  # expenses only
    total = _round(-f["amount"].sum())
    return {
        "tool": "get_spending",
        "filters": {
            "category": category,
            "merchant": merchant,
            "period": period,
            "start": str(s) if s else None,
            "end": str(e) if e else None,
            "account": account,
        },
        "total_spent": total,
        "transaction_count": int(len(f)),
        "avg_transaction": _round(-f["amount"].mean()) if len(f) else 0.0,
        "sample": _sample(f),
    }


def top_categories(period: str | None = None, n: int = 5) -> dict[str, Any]:
    """Top expense categories for a period (excludes salary/credit_payment)."""
    df = D.load_df()
    s, e = D.resolve_period(period)
    f = D.filter_df(df, s, e)
    f = f[(f["amount"] < 0) & (~f["category"].isin(["salary", "credit_payment"]))]
    g = (-f.groupby("category")["amount"].sum()).sort_values(ascending=False)
    items = [{"category": c, "total": _round(v)} for c, v in g.head(n).items()]
    return {
        "tool": "top_categories",
        "period": period,
        "start": str(s) if s else None,
        "end": str(e) if e else None,
        "top": items,
    }


def last_payment(merchant: str) -> dict[str, Any]:
    """Date and amount of the most recent payment to a merchant."""
    df = D.load_df()
    f = D.filter_df(df, merchant=merchant)
    if f.empty:
        return {"tool": "last_payment", "merchant": merchant, "found": False}
    row = f.sort_values("date").iloc[-1]
    return {
        "tool": "last_payment",
        "merchant": row["merchant"],
        "found": True,
        "date": row["date"].strftime("%Y-%m-%d"),
        "amount": _round(row["amount"]),
        "days_ago": (D.today() - row["date"].date()).days,
    }


def list_subscriptions() -> dict[str, Any]:
    """All recurring subscriptions with monthly cost and a forgotten-sub flag.

    Forgotten = recurring monthly merchant whose last charge is >50 days ago.
    """
    df = D.load_df()
    sub = df[(df["category"] == "subscriptions")]
    out = []
    total_active = 0.0
    for merchant, g in sub.groupby("merchant"):
        last = g["date"].max().date()
        days_ago = (D.today() - last).days
        monthly = _round(-g.sort_values("date")["amount"].iloc[-1])
        forgotten = days_ago > 50
        out.append(
            {
                "merchant": merchant,
                "monthly_cost": monthly,
                "charges": int(len(g)),
                "last_charge": str(last),
                "days_since_last": days_ago,
                "likely_forgotten": forgotten,
            }
        )
        if not forgotten:
            total_active += monthly
    out.sort(key=lambda x: x["monthly_cost"], reverse=True)
    return {
        "tool": "list_subscriptions",
        "subscriptions": out,
        "active_monthly_total": _round(total_active),
        "forgotten": [s for s in out if s["likely_forgotten"]],
    }


def time_of_day_breakdown(
    category: str = "delivery", period: str | None = None, night_hour: int = 21
) -> dict[str, Any]:
    """Share of a category's orders placed at/after night_hour (impulse pattern)."""
    df = D.load_df()
    s, e = D.resolve_period(period)
    f = D.filter_df(df, s, e, category)
    f = f[f["amount"] < 0]
    if f.empty:
        return {"tool": "time_of_day_breakdown", "category": category, "count": 0}
    late = f[f["date"].dt.hour >= night_hour]
    return {
        "tool": "time_of_day_breakdown",
        "category": category,
        "night_hour": night_hour,
        "count": int(len(f)),
        "late_count": int(len(late)),
        "late_pct": _round(100 * len(late) / len(f)),
        "late_total": _round(-late["amount"].sum()),
        "total": _round(-f["amount"].sum()),
    }


def compare_periods(
    period_a: str, period_b: str, category: str | None = None
) -> dict[str, Any]:
    """Compare total spend between two named periods (a vs b)."""
    a = get_spending(category=category, period=period_a)
    b = get_spending(category=category, period=period_b)
    diff = _round(a["total_spent"] - b["total_spent"])
    pct = _round(100 * diff / b["total_spent"]) if b["total_spent"] else None
    return {
        "tool": "compare_periods",
        "category": category,
        "period_a": {"name": period_a, "total": a["total_spent"]},
        "period_b": {"name": period_b, "total": b["total_spent"]},
        "difference": diff,
        "pct_change": pct,
    }


def monthly_summary(month: str | None = None) -> dict[str, Any]:
    """Income, expenses, net for a month (YYYY-MM) and a naive month-end projection."""
    df = D.load_df()
    t = D.today()
    if month is None:
        month = f"{t.year}-{t.month:02d}"
    y, m = (int(x) for x in month.split("-"))
    f = df[(df["date"].dt.year == y) & (df["date"].dt.month == m)]
    income = _round(f[f["amount"] > 0]["amount"].sum())
    expenses = _round(-f[f["amount"] < 0]["amount"].sum())
    net = _round(income - expenses)
    # naive projection: scale expenses to full month if month is in progress
    last_day = f["date"].max().day if not f.empty else 1
    days_in_month = pd.Period(f"{y}-{m:02d}").days_in_month
    proj_expenses = _round(expenses / last_day * days_in_month) if last_day else expenses
    proj_net = _round(income - proj_expenses)
    return {
        "tool": "monthly_summary",
        "month": month,
        "income": income,
        "expenses": expenses,
        "net": net,
        "projected_expenses": proj_expenses,
        "projected_net": proj_net,
        "days_elapsed": int(last_day),
        "days_in_month": int(days_in_month),
    }


def detect_suspicious(account: str | None = "credit_card") -> dict[str, Any]:
    """Flag potentially fraudulent transactions (large foreign/credit charges)."""
    df = D.load_df()
    f = D.filter_df(df, account=account)
    f = f[f["amount"] < 0]
    # heuristic: amount well above the typical credit-card transaction
    threshold = 150.0
    flagged = f[f["amount"].abs() >= threshold]
    items = _sample(flagged, n=10)
    return {
        "tool": "detect_suspicious",
        "account": account,
        "threshold": threshold,
        "flagged_count": int(len(flagged)),
        "flagged": items,
    }


def project_savings(
    category: str, reduction_pct: float, period: str | None = None
) -> dict[str, Any]:
    """Project monthly & annual savings from reducing a category by reduction_pct.

    Average monthly spend = total category spend over the whole dataset divided by
    the number of distinct calendar months in the dataset
    (i.e. the average of the per-month costs). The `period` argument is accepted
    for API compatibility but ignored — the projection is always dataset-wide.
    """
    df = D.load_df()
    f = df[(df["category"].str.lower() == category.lower()) & (df["amount"] < 0)]
    months = int(df["date"].dt.to_period("M").nunique()) or 1
    total = -f["amount"].sum()
    monthly = total / months
    saved_monthly = _round(monthly * reduction_pct / 100.0)
    return {
        "tool": "project_savings",
        "category": category,
        "reduction_pct": reduction_pct,
        "basis": "full_dataset",
        "months_in_dataset": months,
        "total_spend": _round(total),
        "avg_monthly_spend": _round(monthly),
        "monthly_savings": saved_monthly,
        "annual_savings": _round(saved_monthly * 12),
    }


def weekend_vs_weekday() -> dict[str, Any]:
    """Average expense transaction size on weekends vs weekdays."""
    df = D.load_df()
    f = df[df["amount"] < 0].copy()
    f["weekend"] = f["date"].dt.weekday >= 5
    wknd = _round(-f[f["weekend"]]["amount"].mean())
    week = _round(-f[~f["weekend"]]["amount"].mean())
    return {
        "tool": "weekend_vs_weekday",
        "weekday_avg": week,
        "weekend_avg": wknd,
        "weekend_premium_pct": _round(100 * (wknd - week) / week) if week else None,
    }


# --- registry ----------------------------------------------------------------

TOOLS: dict[str, Callable[..., dict]] = {
    "get_spending": get_spending,
    "top_categories": top_categories,
    "last_payment": last_payment,
    "list_subscriptions": list_subscriptions,
    "time_of_day_breakdown": time_of_day_breakdown,
    "compare_periods": compare_periods,
    "monthly_summary": monthly_summary,
    "detect_suspicious": detect_suspicious,
    "project_savings": project_savings,
    "weekend_vs_weekday": weekend_vs_weekday,
}

# OpenAI/OpenRouter-style tool schemas
TOOL_SCHEMAS: list[dict] = [
    {
        "type": "function",
        "function": {
            "name": "get_spending",
            "description": (
                "Total expenses for an optional category, merchant and/or period. "
                "ALWAYS select the relative period via the `period` enum — it is "
                "resolved server-side against the system's current date, so never "
                "compute calendar dates yourself."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "category": {
                        "type": "string",
                        "enum": [
                            "coffee", "groceries", "restaurants", "delivery",
                            "transport", "entertainment", "shopping", "health",
                            "subscriptions", "utilities", "travel", "credit_payment",
                        ],
                    },
                    "merchant": {"type": "string"},
                    "period": {
                        "type": "string",
                        "description": "Relative window, resolved against the system 'today'.",
                        "enum": [
                            "last_week", "this_week", "this_month", "last_month",
                            "last_30_days", "last_3_months", "this_year",
                            "last_year", "all",
                        ],
                    },
                    "account": {"type": "string", "enum": ["main_debit", "credit_card"]},
                },
                "required": ["period"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "top_categories",
            "description": "Top expense categories for a period.",
            "parameters": {
                "type": "object",
                "properties": {
                    "period": {"type": "string"},
                    "n": {"type": "integer", "default": 5},
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "last_payment",
            "description": "Most recent payment to a given merchant.",
            "parameters": {
                "type": "object",
                "properties": {"merchant": {"type": "string"}},
                "required": ["merchant"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "list_subscriptions",
            "description": "All recurring subscriptions; flags likely-forgotten ones.",
            "parameters": {"type": "object", "properties": {}},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "time_of_day_breakdown",
            "description": "Share of a category's orders placed late at night (impulse).",
            "parameters": {
                "type": "object",
                "properties": {
                    "category": {"type": "string", "default": "delivery"},
                    "period": {"type": "string"},
                    "night_hour": {"type": "integer", "default": 21},
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "compare_periods",
            "description": "Compare total spend between two named periods.",
            "parameters": {
                "type": "object",
                "properties": {
                    "period_a": {"type": "string"},
                    "period_b": {"type": "string"},
                    "category": {"type": "string"},
                },
                "required": ["period_a", "period_b"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "monthly_summary",
            "description": "Income, expenses, net and projection for a month (YYYY-MM).",
            "parameters": {
                "type": "object",
                "properties": {"month": {"type": "string"}},
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "detect_suspicious",
            "description": "Flag potentially fraudulent large/foreign transactions.",
            "parameters": {
                "type": "object",
                "properties": {"account": {"type": "string"}},
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "project_savings",
            "description": "Project monthly & annual savings from cutting a category.",
            "parameters": {
                "type": "object",
                "properties": {
                    "category": {"type": "string"},
                    "reduction_pct": {"type": "number"},
                    "period": {"type": "string"},
                },
                "required": ["category", "reduction_pct"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "weekend_vs_weekday",
            "description": "Average transaction size on weekends vs weekdays.",
            "parameters": {"type": "object", "properties": {}},
        },
    },
]


def call_tool(name: str, arguments: dict[str, Any]) -> dict[str, Any]:
    """Dispatch a tool call by name with kwargs; never raises on bad args."""
    fn = TOOLS.get(name)
    if fn is None:
        return {"error": f"unknown tool: {name}"}
    try:
        return fn(**(arguments or {}))
    except Exception as exc:  # surface, don't crash the agent loop
        return {"error": f"{name} failed: {exc}", "arguments": arguments}
