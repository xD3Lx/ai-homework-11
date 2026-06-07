"""Deterministic offline 'brain'.

Powers OFFLINE/mock mode so the whole pipeline (tools, agents, golden set, UI)
runs and produces correct, grounded, friendly answers without any API key.
It is a small rule-based NLU + planner + answer composer — NOT a language model.
In ONLINE mode this module is bypassed and a real LLM does the reasoning.
"""
from __future__ import annotations

import re
from typing import Any

# ---- lexicons ---------------------------------------------------------------

CATEGORY_KEYWORDS = {
    "coffee": ["кав", "coffee", "aroma", "lviv croissants", "blue bottle"],
    "delivery": ["доставк", "delivery", "glovo", "bolt food", "uber eats", "їжу додому"],
    "restaurants": ["ресторан", "restaurant", "puzata", "kanapa", "кафе"],
    "groceries": ["продукт", "groceries", "atb", "silpo", "novus", "auchan", "супермаркет"],
    "subscriptions": ["підписк", "subscription", "netflix", "spotify", "apple one", "sportlife"],
    "transport": ["транспорт", "таксі", "taxi", "uklon", "поїздк"],
    "entertainment": ["розваг", "entertainment", "кіно", "cinema", "steam"],
    "shopping": ["шопінг", "shopping", "одяг", "rozetka", "comfy", "zara", "h&m"],
    "health": ["здоров", "health", "аптек", "pharmacy", "клінік", "лік"],
    "utilities": ["комунал", "utilities", "світло", "інтернет", "мобільн"],
    "travel": ["подорож", "travel", "booking"],
}

MERCHANTS = [
    "Netflix", "Spotify", "Apple One", "iCloud Storage", "Sportlife",
    "Glovo", "Bolt Food", "Uber Eats", "ATB", "Silpo", "Novus",
    "Booking.com", "AliExpress", "Rozetka", "Comfy",
]

PERIOD_PATTERNS = [
    (r"минул\w*\s+тиж", "last_week"),
    (r"last\s+week", "last_week"),
    (r"цьо\w*\s+тиж|this\s+week", "this_week"),
    (r"минул\w*\s+місяц|last\s+month", "last_month"),
    (r"цьо\w*\s+місяц|this\s+month|за\s+місяць", "this_month"),
    (r"минул\w*\s+рок|торік|last\s+year", "last_year"),
    (r"цьо\w*\s+рок|this\s+year|поточн\w*\s+рок", "this_year"),
    (r"останн\w*\s+3\s+місяц|3\s+month|трьом", "last_3_months"),
    (r"30\s+дн", "last_30_days"),
]

FRAUD_KW = ["не робив", "не робила", "не я", "fraud", "шахрай", "підозріл",
            "не здійснював", "не здійснювала", "вкрал", "украл", "незнайом"]
OOS_KW = ["купи акці", "продай акці", "biticoin", "біткоїн", "крипт", "акції",
          "інвестуй", "погод", "weather", "анекдот", "вірш", "рецепт"]
ADVICE_KW = ["зекономи", "економи", "economy", "save", "заощади", "виплатити",
             "погасити", "скоротити", "де можу", "які підписки", "необхідн",
             "забут", "не користу", "не потрібн"]
MULTISTEP_KW = ["порівня", "compare", "у плюс", "в плюс", "закри", "якщо зменш",
                "за рік", "проєкц", "прогноз", "вдвічі"]


def _find_category(text: str) -> str | None:
    t = text.lower()
    for cat, kws in CATEGORY_KEYWORDS.items():
        if any(k in t for k in kws):
            return cat
    return None


def _find_merchant(text: str) -> str | None:
    t = text.lower()
    for m in MERCHANTS:
        if m.lower() in t:
            return m
    return None


def _find_period(text: str) -> str | None:
    t = text.lower()
    for pat, name in PERIOD_PATTERNS:
        if re.search(pat, t):
            return name
    return None


# ---- intent classification --------------------------------------------------

def classify(query: str, history: list[str] | None = None) -> dict[str, Any]:
    """Return {intent, category, merchant, period} using query + prior user turns."""
    t = query.lower()
    history = history or []

    intent = "stat"
    if any(k in t for k in FRAUD_KW):
        intent = "fraud"
    elif any(k in t for k in OOS_KW):
        intent = "out_of_scope"
    elif any(k in t for k in MULTISTEP_KW):
        intent = "multistep"
    elif any(k in t for k in ADVICE_KW):
        intent = "advice"

    category = _find_category(query)
    merchant = _find_merchant(query)
    period = _find_period(query)

    # multi-turn: inherit category/merchant from the previous user turn when the
    # follow-up doesn't name one (e.g. "$45" -> "а за місяць?").
    if category is None and merchant is None and history:
        for prev in reversed(history):
            category = category or _find_category(prev)
            merchant = merchant or _find_merchant(prev)
            if category or merchant:
                break

    return {
        "intent": intent,
        "category": category,
        "merchant": merchant,
        "period": period,
    }


# ---- tool planning ----------------------------------------------------------

def plan_tools(query: str, ctx: dict[str, Any]) -> list[dict[str, Any]]:
    """Decide which tools to call (name + arguments) for a classified query."""
    intent = ctx["intent"]
    cat, merch, period = ctx["category"], ctx["merchant"], ctx["period"]
    calls: list[dict[str, Any]] = []
    t = query.lower()

    if intent == "fraud":
        return [{"name": "detect_suspicious", "arguments": {"account": "credit_card"}}]

    if intent == "out_of_scope":
        return []

    if intent == "stat":
        if "топ" in t or "top" in t:
            calls.append({"name": "top_categories",
                          "arguments": {"period": period or "this_month"}})
        elif merch and ("останн" in t or "дата" in t or "last" in t):
            calls.append({"name": "last_payment", "arguments": {"merchant": merch}})
        else:
            args: dict[str, Any] = {"period": period or "this_month"}
            if cat:
                args["category"] = cat
            if merch:
                args["merchant"] = merch
            calls.append({"name": "get_spending", "arguments": args})
        return calls

    if intent == "advice":
        if "підписк" in t or "subscription" in t:
            calls.append({"name": "list_subscriptions", "arguments": {}})
        elif "картк" in t or "кредит" in t or "card" in t:
            calls.append({"name": "monthly_summary", "arguments": {}})
            calls.append({"name": "get_spending",
                          "arguments": {"category": "credit_payment", "period": "this_year"}})
        else:
            # general "where can I save" -> survey the big discretionary buckets
            calls.append({"name": "list_subscriptions", "arguments": {}})
            calls.append({"name": "time_of_day_breakdown",
                          "arguments": {"category": "delivery", "period": "last_3_months"}})
            calls.append({"name": "get_spending",
                          "arguments": {"category": "delivery", "period": "last_3_months"}})
            calls.append({"name": "get_spending",
                          "arguments": {"category": "coffee", "period": "last_3_months"}})
        return calls

    if intent == "multistep":
        if "вдвічі" in t or "зменш" in t or "economy" in t or "за рік" in t:
            calls.append({"name": "project_savings",
                          "arguments": {"category": cat or "delivery", "reduction_pct": 50}})
        elif "плюс" in t or "закри" in t:
            calls.append({"name": "monthly_summary", "arguments": {}})
        else:  # year over year comparison
            calls.append({"name": "compare_periods",
                          "arguments": {"period_a": "this_year", "period_b": "last_year",
                                        **({"category": cat} if cat else {})}})
        return calls

    return calls


# ---- helpers ----------------------------------------------------------------

CAT_UA = {
    "coffee": "каву", "delivery": "доставку їжі", "restaurants": "ресторани",
    "groceries": "продукти", "subscriptions": "підписки", "transport": "транспорт",
    "entertainment": "розваги", "shopping": "шопінг", "health": "здоров'я",
    "utilities": "комуналку", "travel": "подорожі",
}


def _money(x: float) -> str:
    return f"${x:,.0f}" if abs(x) >= 100 or float(x).is_integer() else f"${x:,.2f}"


def _by_tool(results: list[dict], name: str) -> dict | None:
    for r in results:
        if r.get("tool") == name:
            return r
    return None


# ---- answer composition -----------------------------------------------------

def compose_answer(query: str, ctx: dict, results: list[dict]) -> str:
    """Build the final, grounded, friendly (ти) answer from tool results."""
    intent = ctx["intent"]

    if intent == "out_of_scope":
        return (
            "Це поза моїми можливостями — я допомагаю лише з твоїми витратами та "
            "заощадженнями (статистика, поради щодо економії, аналіз підписок). "
            "Інвестиціями чи покупкою активів я не займаюся. Хочеш, подивимось, "
            "де цього місяця можна зекономити?"
        )

    if intent == "fraud":
        susp = _by_tool(results, "detect_suspicious") or {}
        lst = susp.get("flagged", [])
        lines = [
            "Схоже на можливий fraud. Блокування картки та оформлення chargeback "
            "виходять за межі моїх можливостей — це робить служба підтримки. "
            "Рекомендовані дії:",
            "",
            "1. Заблокувати картку: Картки → ця карта → Заблокувати",
            "2. Звернутися до служби підтримки через чат застосунку — у них окрема "
            "процедура для disputed transactions",
        ]
        if lst:
            flagged = ", ".join(
                f"{x['merchant']} {_money(abs(x['amount']))} ({x['date'][:10]})" for x in lst
            )
            lines += ["", f"До речі, по цій картці я бачу ще {len(lst)} велик(і) "
                          f"транзакці(ї), варто перевірити їх теж: {flagged}."]
        return "\n".join(lines)

    if intent == "stat":
        top = _by_tool(results, "top_categories")
        if top:
            items = "\n".join(
                f"{i}. {CAT_UA.get(x['category'], x['category'])} — {_money(x['total'])}"
                for i, x in enumerate(top["top"], 1)
            )
            return f"Топ категорій витрат за період:\n\n{items}"
        lp = _by_tool(results, "last_payment")
        if lp:
            if not lp.get("found"):
                return f"Не знайшла платежів за «{lp['merchant']}» у твоїй історії."
            return (f"Останній платіж за {lp['merchant']} — {lp['date']} на "
                    f"{_money(abs(lp['amount']))} ({lp['days_ago']} дн. тому).")
        gs = _by_tool(results, "get_spending")
        if gs:
            cat = gs["filters"].get("category")
            label = CAT_UA.get(cat, cat) if cat else "витрат"
            merch = gs["filters"].get("merchant")
            subj = merch or (f"на {label}" if cat else "загалом")
            if gs["transaction_count"] == 0:
                return f"За цей період витрат {('на ' + label) if cat else ''} не було."
            return (f"{_money(gs['total_spent'])} {subj}. Це {gs['transaction_count']} "
                    f"транзакці(й), у середньому {_money(gs['avg_transaction'])} кожна.")
        return "Не вдалося порахувати — уточни категорію або період, будь ласка."

    if intent == "advice":
        return _compose_advice(query, ctx, results)

    if intent == "multistep":
        return _compose_multistep(query, ctx, results)

    return "Уточни, будь ласка, що саме показати — категорію та період."


def _compose_advice(query: str, ctx: dict, results: list[dict]) -> str:
    subs = _by_tool(results, "list_subscriptions")
    deliv_time = _by_tool(results, "time_of_day_breakdown")
    deliv_spend = None
    coffee_spend = None
    for r in results:
        if r.get("tool") == "get_spending":
            c = r["filters"].get("category")
            if c == "delivery":
                deliv_spend = r
            elif c == "coffee":
                coffee_spend = r

    # subscription-focused advice
    if subs and not deliv_spend:
        forgotten = subs.get("forgotten", [])
        lines = [f"Активні підписки коштують {_money(subs['active_monthly_total'])}/міс. "]
        if forgotten:
            f = forgotten[0]
            lines.append(
                f"\nОкремо звертаю увагу: {f['merchant']} {_money(f['monthly_cost'])}/міс — "
                f"остання транзакція {f['last_charge']} ({f['days_since_last']} дн. тому), "
                f"ймовірно забута підписка. Варто перевірити і скасувати: −"
                f"{_money(f['monthly_cost'])}/міс."
            )
        top = ", ".join(
            f"{s['merchant']} {_money(s['monthly_cost'])}"
            for s in subs["subscriptions"][:4]
        )
        lines.append(f"\nНайбільші активні: {top}.")
        return "".join(lines)

    # credit-card payoff advice
    ms = _by_tool(results, "monthly_summary")
    if ms and "картк" in query.lower():
        return (
            f"Цього місяця чистий результат {_money(ms['net'])} (дохід {_money(ms['income'])}, "
            f"витрати {_money(ms['expenses'])}). Щоб швидше закрити картку: спрямовуй "
            f"будь-який місячний профіцит у повне погашення замість мінімального платежу "
            f"$50 — мінімальний платіж майже весь йде у відсотки. Конкретний крок: "
            f"налаштуй автосписання повного балансу в день зарплати (25-го)."
        )

    # general "where to save $X" — combine delivery + subscriptions + coffee
    target = _extract_target_amount(query)
    blocks: list[str] = []
    total = 0.0
    if deliv_spend and deliv_time:
        months = 3
        monthly = deliv_spend["total_spent"] / months
        save = monthly / 2
        total += save
        blocks.append(
            f"**1. Доставка їжі — {_money(monthly)}/міс.** {deliv_time['late_pct']:.0f}% "
            f"замовлень після 21:00 — імпульсні. Зменшення вдвічі = ~{_money(save)}."
        )
    if subs:
        f = (subs.get("forgotten") or [None])[0]
        if f:
            total += f["monthly_cost"]
            blocks.append(
                f"**2. Підписки — {_money(subs['active_monthly_total'])}/міс.** "
                f"{f['merchant']} {_money(f['monthly_cost'])}/міс — остання транзакція "
                f"{f['days_since_last']} дн. тому, ймовірно забута. Рекомендую перевірити. "
                f"−{_money(f['monthly_cost'])}/міс."
            )
    if coffee_spend:
        monthly = coffee_spend["total_spent"] / 3
        save = monthly * 0.4
        total += save
        blocks.append(
            f"**3. Кава поза домом — {_money(monthly)}/міс.** Не пропоную відмовлятися "
            f"повністю, але готувати вдома 2 дні на тиждень = ~{_money(save)}."
        )

    head = "На основі даних за останні 3 місяці виділяю реалістичні варіанти:\n\n"
    body = "\n\n".join(blocks)
    tail = f"\n\nЗагалом ~{_money(total)}."
    if target:
        if total >= target:
            tail += f" Це покриває цільові {_money(target)}. З чого почнемо?"
        else:
            tail += (f" До цільових {_money(target)} не вистачає — потрібне глибше "
                     f"скорочення в одній з категорій. З чого почнемо?")
    else:
        tail += " З чого почнемо?"
    return head + body + tail


def _compose_multistep(query: str, ctx: dict, results: list[dict]) -> str:
    ps = _by_tool(results, "project_savings")
    if ps:
        return (
            f"Зараз на {CAT_UA.get(ps['category'], ps['category'])} йде ~"
            f"{_money(ps['avg_monthly_spend'])}/міс. Якщо скоротити на "
            f"{ps['reduction_pct']:.0f}%, економія складе {_money(ps['monthly_savings'])}/міс, "
            f"тобто {_money(ps['annual_savings'])} за рік."
        )
    ms = _by_tool(results, "monthly_summary")
    if ms:
        verdict = "у плюс" if ms["projected_net"] >= 0 else "у мінус"
        return (
            f"За поточними темпами місяць закриється {verdict}: прогноз чистого результату "
            f"{_money(ms['projected_net'])} (дохід {_money(ms['income'])}, прогноз витрат "
            f"{_money(ms['projected_expenses'])}; пройшло {ms['days_elapsed']} з "
            f"{ms['days_in_month']} днів)."
        )
    cp = _by_tool(results, "compare_periods")
    if cp:
        a, b = cp["period_a"], cp["period_b"]
        direction = "більше" if cp["difference"] > 0 else "менше"
        pct = f"{abs(cp['pct_change']):.0f}%" if cp["pct_change"] is not None else "—"
        note = ""
        if b["name"] == "last_year":
            note = (" Зверни увагу: дані за минулий рік охоплюють лише грудень, "
                    "тож пряме порівняння обмежене.")
        return (
            f"Витрати {a['name']} — {_money(a['total'])}, {b['name']} — {_money(b['total'])}. "
            f"Різниця {_money(abs(cp['difference']))} ({direction}, {pct}).{note}"
        )
    return "Потрібно більше даних для цього аналізу — уточни період або категорію."


def _extract_target_amount(text: str) -> float | None:
    m = re.search(r"\$?\s*(\d{2,4})", text)
    if m:
        return float(m.group(1))
    return None
