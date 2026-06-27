"""Unit tests for the deterministic finance tools and evaluators.

These do not call any LLM, so they run without an API key.
Run: pytest -q
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "evals"))

from finance_coach import tools  # noqa: E402
import evaluators as E  # noqa: E402


def test_get_spending_coffee_last_week():
    r = tools.get_spending(category="coffee", period="last_week")
    assert r["total_spent"] == 21.6
    assert r["transaction_count"] == 4


def test_top_categories_this_month():
    r = tools.top_categories(period="this_month")
    assert r["top"][0]["category"] == "groceries"


def test_last_payment_netflix():
    r = tools.last_payment("Netflix")
    assert r["found"] and r["date"] == "2025-11-10" and r["amount"] == -12.0


def test_forgotten_subscription_detected():
    r = tools.list_subscriptions()
    assert "Sportlife" in {f["merchant"] for f in r["forgotten"]}


def test_suspicious_flags_two():
    r = tools.detect_suspicious(account="credit_card")
    assert r["flagged_count"] == 2
    assert {"Booking.com", "AliExpress"} <= {f["merchant"] for f in r["flagged"]}


def test_project_savings_delivery():
    r = tools.project_savings(category="delivery", reduction_pct=50)
    assert r["annual_savings"] > 0


def test_monthly_summary_november():
    r = tools.monthly_summary("2025-11")
    assert r["income"] == 2400.0
    assert round(r["net"], 2) == 60.76


def test_delivery_late_night_pattern():
    r = tools.time_of_day_breakdown(category="delivery", period="last_3_months")
    assert r["late_pct"] >= 40  # impulse pattern present


def test_unknown_tool_safe():
    assert "error" in tools.call_tool("nope", {})


def test_groundedness_evaluator():
    results = [{"total_spent": 84.1}]
    assert E.groundedness("Ти витратив $84.10", results) == 1.0
    assert E.groundedness("Ти витратив $5000", results) < 1.0


def test_tool_selection_accuracy():
    assert E.tool_selection_accuracy(["get_spending"], ["get_spending"]) == 1.0
    assert E.tool_selection_accuracy([], ["get_spending"]) == 0.0
