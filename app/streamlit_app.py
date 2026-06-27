"""Streamlit demo for the Personal Finance Coach.

Run:  streamlit run app/streamlit_app.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "evals"))

from finance_coach import endpoint  # noqa: E402
from finance_coach.config import SETTINGS  # noqa: E402

st.set_page_config(page_title="Personal Finance Coach", page_icon="💸", layout="wide")

# ---- sidebar ---------------------------------------------------------------
with st.sidebar:
    st.header("⚙️ Налаштування")
    arch = st.radio("Архітектура", ["crew", "baseline"], index=0,
                    help="crew = multi-agent LangGraph · baseline = single-agent")
    st.caption("LLM: **OpenRouter**")
    smart = st.text_input("MODEL_SMART", SETTINGS.model_smart)
    cheap = st.text_input("MODEL_CHEAP", SETTINGS.model_cheap)
    st.caption(f"Сьогодні (для відносних дат): {SETTINGS.today}")
    if st.button("🗑️ Очистити діалог"):
        st.session_state.pop("session", None)
        st.session_state.pop("turns", None)
        st.rerun()
    if not SETTINGS.openrouter_api_key:
        st.warning("OPENROUTER_API_KEY не задано — додай ключ у .env, щоб працювали агенти.")

# session objects
if st.session_state.get("arch") != arch:
    st.session_state["arch"] = arch
    st.session_state["session"] = endpoint.Session(arch)
    st.session_state["turns"] = []
sess: endpoint.Session = st.session_state["session"]

tab_chat, tab_eval = st.tabs(["💬 Chat", "📊 Eval"])

# ---- chat tab --------------------------------------------------------------
with tab_chat:
    st.subheader("Помічник з особистих фінансів")
    for turn in st.session_state.get("turns", []):
        with st.chat_message("user"):
            st.markdown(turn["q"])
        with st.chat_message("assistant"):
            st.markdown(turn["a"])
            res = turn["res"]
            with st.expander(
                f"🔎 Trace · {res.architecture} · {res.latency_ms:.0f} ms · "
                f"${res.cost:.5f} · {res.total_tokens} tok"
            ):
                cols = st.columns(4)
                cols[0].metric("Агенти", " → ".join(dict.fromkeys(res.agents_used)) or "—")
                cols[1].metric("Інструменти", ", ".join(res.tools_used) or "—")
                cols[2].metric("Tokens", res.total_tokens)
                cols[3].metric("Overhead", f"{res.inter_agent_overhead_pct}%")
                st.markdown("**Послідовність кроків:**")
                st.dataframe(
                    [
                        {"kind": s.kind, "name": s.name, "detail": s.detail[:60],
                         "ms": round(s.latency_ms, 1), "tok": s.tokens,
                         "cost$": round(s.cost, 6)}
                        for s in res.trace
                    ],
                    use_container_width=True, hide_index=True,
                )

    query = st.chat_input("Спитай про витрати, підписки, економію…")
    if query:
        with st.spinner("Аналізую…"):
            res = sess.ask(query)
        st.session_state.setdefault("turns", []).append(
            {"q": query, "a": res.answer, "res": res})
        st.rerun()

    with st.expander("💡 Приклади запитів"):
        st.markdown(
            "- скільки витратив на каву минулого тижня?\n"
            "- топ-5 категорій витрат за листопад?\n"
            "- де можу зекономити $200 цього місяця?\n"
            "- на які підписки витрачається найбільше та чи всі потрібні?\n"
            "- якщо зменшити витрати на доставку вдвічі — яка економія за рік?\n"
            "- на моїй карті $703 в Booking.com, я не робив цю транзакцію\n"
            "- купи мені акції Apple"
        )

# ---- eval tab --------------------------------------------------------------
with tab_eval:
    st.subheader("Golden set — crew vs baseline")
    st.caption("Прогін 18 задач через обидві архітектури з custom-евалюаторами.")
    if st.button("▶️ Запустити golden set"):
        import evaluators as E
        from golden_set import GOLDEN

        prog = st.progress(0.0)
        summary = {}
        per_task = []
        for ai, a in enumerate(["baseline", "crew"]):
            succ = ta = gr = 0.0
            lat = cost = 0.0
            for i, t in enumerate(GOLDEN):
                r = endpoint.answer(t["query"], a, history=t.get("history"))
                tr = [s.data for s in r.trace if s.kind == "tool" and s.data]
                s = E.judge_success(t, r.answer, r.tools_used)
                acc = E.tool_selection_accuracy(t.get("expected_tools", []), r.tools_used)
                g = E.groundedness(r.answer, tr)
                succ += s; ta += acc; gr += g; lat += r.latency_ms; cost += r.cost
                if a == "crew":
                    per_task.append({"task": t["id"], "category": t["category"],
                                     "success": int(s), "tool_acc": acc, "ground": g})
                prog.progress((ai * len(GOLDEN) + i + 1) / (2 * len(GOLDEN)))
            n = len(GOLDEN)
            summary[a] = {
                "success_rate": round(succ / n, 3),
                "tool_selection_accuracy": round(ta / n, 3),
                "groundedness": round(gr / n, 3),
                "avg_latency_ms": round(lat / n, 1),
                "cost_per_task": round(cost / n, 6),
            }
        st.markdown("### Підсумок")
        st.dataframe(
            [{"metric": k,
              "baseline": summary["baseline"][k], "crew": summary["crew"][k]}
             for k in summary["baseline"]],
            use_container_width=True, hide_index=True,
        )
        st.markdown("### Деталізація (crew)")
        st.dataframe(per_task, use_container_width=True, hide_index=True)
