"""Unit + smoke tests. Run: FINANCE_COACH_OFFLINE=1 pytest -q"""
import os
import sys
from pathlib import Path

os.environ.setdefault("FINANCE_COACH_OFFLINE", "1")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "evals"))

from finance_coach import tools, endpoint  # noqa: E402
import evaluators as E  # noqa: E402
from golden_set import GOLDEN  # noqa: E402


def test_get_spending_coffee_last_week():
    r = tools.get_spending(category="coffee", period="last_week")
    assert r["total_spent"] == 21.6
    assert r["transaction_count"] == 4


def test_last_payment_netflix():
    r = tools.last_payment("Netflix")
    assert r["found"] and r["date"] == "2025-11-10" and r["amount"] == -12.0


def test_forgotten_subscription_detected():
    r = tools.list_subscriptions()
    forgotten = {f["merchant"] for f in r["forgotten"]}
    assert "Sportlife" in forgotten


def test_suspicious_flags_two():
    r = tools.detect_suspicious(account="credit_card")
    assert r["flagged_count"] == 2
    merchants = {f["merchant"] for f in r["flagged"]}
    assert {"Booking.com", "AliExpress"} <= merchants


def test_project_savings_delivery():
    r = tools.project_savings(category="delivery", reduction_pct=50)
    assert r["annual_savings"] > 0


def test_unknown_tool_safe():
    assert "error" in tools.call_tool("nope", {})


def test_both_architectures_answer():
    for arch in ["baseline", "crew"]:
        res = endpoint.answer("скільки на каву минулого тижня?", arch)
        assert "21.6" in res.answer or "21" in res.answer
        assert res.tools_used  # at least one tool


def test_fraud_escalation_does_not_self_resolve():
    res = endpoint.answer("я не робив транзакцію на карті, це fraud", "crew")
    assert "підтримк" in res.answer.lower()


def test_out_of_scope_rejected():
    res = endpoint.answer("купи акції Apple", "crew")
    assert "поза" in res.answer.lower()
    assert not res.tools_used


def test_golden_set_success_both():
    for arch in ["baseline", "crew"]:
        passed = 0
        for t in GOLDEN:
            res = endpoint.answer(t["query"], arch, history=t.get("history"))
            passed += E.judge_success(t, res.answer, res.tools_used)
        assert passed / len(GOLDEN) >= 0.9, f"{arch} success too low: {passed}/{len(GOLDEN)}"


def test_multiturn_context_inherited():
    s = endpoint.Session("crew")
    s.ask("скільки на каву минулого тижня?")
    r = s.ask("а за місяць?")
    assert "84" in r.answer  # inherited 'coffee' + month period
