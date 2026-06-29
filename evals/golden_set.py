"""Golden set — 18 tasks across the three query categories plus edge cases.

Each task carries the expected intent, the tools that SHOULD be involved, and
reference facts (substrings/numbers the answer must contain) used by the
deterministic success judge. References are grounded in the provided dataset
(anchored to today = 2025-11-30).
"""
from __future__ import annotations

GOLDEN: list[dict] = [
    # --- category 1: stats & facts ---
    {
        "id": "stat_coffee_lastweek",
        "query": "скільки я витратив на каву минулого тижня?",
        "category": "stat",
        "intent": "stat",
        "expected_tools": ["get_spending"],
        "must_include": ["21.6", "4"],
    },
    {
        "id": "stat_top_categories",
        "query": "топ-5 категорій витрат за листопад?",
        "category": "stat",
        "intent": "stat",
        "expected_tools": ["top_categories"],
        "must_include": ["продукт", "1167"],
    },
    {
        "id": "stat_netflix_last",
        "query": "коли була дата останнього платежу за Netflix?",
        "category": "stat",
        "intent": "stat",
        "expected_tools": ["last_payment"],
        "must_include": ["2025-11-10", "12"],
    },
    {
        "id": "stat_groceries_month",
        "query": "скільки пішло на продукти цього місяця?",
        "category": "stat",
        "intent": "stat",
        "expected_tools": ["get_spending"],
        "must_include": ["1167"],
    },
    {
        "id": "stat_delivery_year",
        "query": "скільки всього на доставку цього року?",
        "category": "stat",
        "intent": "stat",
        "expected_tools": ["get_spending"],
        "must_include": ["$1312"],
    },
    # --- category 2: data-grounded savings advice ---
    {
        "id": "advice_save_200",
        "query": "де можу зекономити $200 цього місяця?",
        "category": "advice",
        "intent": "advice",
        "expected_tools": ["list_subscriptions", "time_of_day_breakdown", "get_spending"],
        "must_include": ["Доставка", "Sportlife", "$"],
    },
    {
        "id": "advice_subscriptions",
        "query": "на які підписки витрачається найбільше та чи всі вони необхідні?",
        "category": "advice",
        "intent": "advice",
        "expected_tools": ["list_subscriptions"],
        "must_include": ["Sportlife", "15"],
    },
    {
        "id": "advice_credit_card",
        "query": "як швидше виплатити кредитну картку?",
        "category": "advice",
        "intent": "advice",
        "expected_tools": ["get_summary", "monthly_summary", "top-categories"],
        "must_include": ["60.76"],
    },
    {
        "id": "advice_forgotten_sub",
        "query": "чи є у мене забуті підписки, якими я не користуюся?",
        "category": "advice",
        "intent": "advice",
        "expected_tools": ["list_subscriptions"],
        "must_include": ["Sportlife"],
    },
    # --- category 3: multi-step analysis & synthesis ---
    {
        "id": "multi_delivery_halve",
        "query": "якщо зменшити витрати на доставку вдвічі — яка економія за рік?",
        "category": "multistep",
        "intent": "multistep",
        "expected_tools": ["project_savings"],
        "must_include": ["764", "рік"],
    },
    {
        "id": "multi_month_positive",
        "query": "чи буде поточний місяць закрито у плюс?",
        "category": "multistep",
        "intent": "multistep",
        "expected_tools": ["monthly_summary"],
        "must_include": ["плюс", "60"],
    },
    {
        "id": "multi_year_compare",
        "query": "порівняй мої витрати поточного року з минулорічними",
        "category": "multistep",
        "intent": "multistep",
        "expected_tools": ["compare_periods"],
        "must_include": ["Різниця"],
    },
    {
        "id": "multi_coffee_halve",
        "query": "скільки зекономлю за рік, якщо вдвічі менше витрачати на каву?",
        "category": "multistep",
        "intent": "multistep",
        "expected_tools": ["project_savings"],
        "must_include": ["рік"],
    },
    # --- edge: fraud escalation ---
    {
        "id": "fraud_booking",
        "query": "на моїй карті $703 в Booking.com 15 січня, я не робив цю транзакцію",
        "category": "fraud",
        "intent": "fraud",
        "expected_tools": ["detect_suspicious"],
        "must_include": ["підозріл"],
        "must_not_include": ["заблокував картку"],
    },
    {
        "id": "fraud_unknown",
        "query": "бачу незнайому транзакцію на карті, це точно не я",
        "category": "fraud",
        "intent": "fraud",
        "expected_tools": ["detect_suspicious"],
        "must_include": ["підозріл", "AliExpress", "Booking.com"],
    },
    #--- edge: out of scope ---
    {
        "id": "oos_stocks",
        "query": "купи мені акції Apple на $500",
        "category": "out_of_scope",
        "intent": "out_of_scope",
        "expected_tools": [],
        "must_include": [],
    },
    {
        "id": "oos_crypto",
        "query": "інвестуй мої гроші в біткоїн",
        "category": "out_of_scope",
        "intent": "out_of_scope",
        "expected_tools": [],
        "must_include": [],
    },
    # --- edge: multi-turn follow-up ---
    {
        "id": "multiturn_coffee_month",
        "query": "а за місяць?",
        "category": "stat",
        "intent": "stat",
        "expected_tools": ["get_spending"],
        "history": [
            {"role": "user", "content": "скільки на каву минулого тижня?"},
            {"role": "assistant", "content": "$21.60 на каву."},
        ],
        "must_include": ["84"],
    },
]


def by_category() -> dict[str, int]:
    out: dict[str, int] = {}
    for t in GOLDEN:
        out[t["category"]] = out.get(t["category"], 0) + 1
    return out
