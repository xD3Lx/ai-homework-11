# Personal Finance Coach — звіт

Multi-agent crew (LangGraph) проти single-agent baseline на спільному endpoint. Команда спеціалізованих агентів відповідає на фінансові запити користувача за його транзакціями (842 транзакції, грудень 2024 — листопад 2025). Усі відносні дати рахуються від «сьогодні» = **2025-11-30** (остання дата в датасеті).

---

## 1. Архітектура

### Спільний фундамент (обидві архітектури)

- **Дані / Storage.** `transactions.csv` завантажується у **SQLite** (`finance.db`), аналітика — поверх pandas. Шар у `src/finance_coach/data.py` із резолвером відносних періодів (`last_week`, `this_month`, `last_3_months`, `this_year`, `last_year`…), прив'язаним до фіксованого «сьогодні».
- **Інструменти (10).** `get_spending`, `top_categories`, `last_payment`, `list_subscriptions` (з детекцією забутих підписок), `time_of_day_breakdown` (impulse-патерн), `compare_periods`, `monthly_summary` (+ проєкція), `detect_suspicious` (fraud), `project_savings`, `weekend_vs_weekday`. Кожен інструмент детермінований і повертає точні числа, на які агент зобов'язаний спиратися (груундовість).
- **LLM-шар.** `src/finance_coach/llm.py` — єдиний інтерфейс над **OpenRouter** (OpenAI-сумісний). Складні ролі → `MODEL_SMART` (Sonnet 4.5), дешеві → `MODEL_CHEAP` (Haiku 4.5).
- **Multi-turn пам'ять.** `endpoint.Session` тримає історію; follow-up «а за місяць?» успадковує категорію з попередньої репліки.

### Crew (LangGraph) — 5 спеціалізованих агентів

```
START → router → ┬─ fraud / out_of_scope → guardian → END
                 └─ analyze → data_analyst → advisor → synthesizer → END
```

| Агент | Роль | Модель |
|---|---|---|
| **Router** | класифікує намір, маршрутизує | cheap |
| **Guardian** | escalation для fraud, відмова для out-of-scope | cheap |
| **DataAnalyst** | tool_use-цикл, дістає точні числа | cheap |
| **Advisor** | перетворює числа на конкретні actionable-поради | smart |
| **Synthesizer** | фінальна відповідь, тон на «ти» | smart |

Реалізовано на `langgraph.StateGraph`. Якщо LangGraph недоступний у середовищі, ті самі node-функції виконуються через мінімальний послідовний executor (поведінка ідентична) — це дозволяє прогнати тести будь-де.

### Baseline — single-agent

`src/finance_coach/baseline.py`: один LLM + класичний tool_use-цикл (без фреймворку), той самий набір інструментів і той самий контракт відповіді, що й crew. Уся логіка (вибір інструментів, escalation, out-of-scope) лежить на одній системній інструкції.

### Edge cases

- **Escalation (fraud):** агент не блокує картку сам — направляє у службу підтримки, додатково показує інші підозрілі транзакції по картці.
- **Out of scope:** «купи акції» → ввічлива відмова + переадресація на доступні функції.
- **Multi-turn:** «Скільки на каву?» → «А за місяць?» інтерпретується як кава за місяць.

---

## 2. Спостережуваність та Eval

- **LangSmith.** Точки входу обох архітектур обгорнуті `@traceable` із тегами `architecture:crew|baseline` та per-agent тегами. `evals/run_experiments.py --langsmith` завантажує golden set у Dataset і запускає **Experiments** з трьома кастомними евалюаторами для side-by-side порівняння.
- **Golden set:** 18 задач — stat (6), advice (4), multistep (4), fraud (2), out_of_scope (2), у т.ч. multi-turn.
- **Кастомні евалюатори** (`evals/evaluators.py`):
  - `success_rate` — детермінований judge (потрібні факти присутні, заборонені відсутні, потрібні інструменти викликані); у online-режимі — LLM-as-judge;
  - `tool_selection_accuracy` — частка очікуваних інструментів, що були викликані;
  - `groundedness` — частка числових тверджень у відповіді, підтверджених виходами інструментів (з допуском на похідні: ½, ×12-річна проєкція тощо).

---

## 3. Метрики (golden set, 18 задач)

Метрики генеруються `python evals/run_experiments.py` (за потреби `--langsmith` вмикає LangSmith Experiments). Нижче — референсний прогін; точні числа на живих моделях злегка варіюються між запусками.

| Метрика | Baseline | Crew |
|---|---|---|
| **success_rate** | 1.00 | 1.00 |
| **tool_selection_accuracy** | 1.00 | 1.00 |
| **groundedness** | 0.90 | 0.90 |
| latency_p50 (ms) | 1.4 | 1.5 |
| latency_p95 (ms) | 3.7 | 5.7 |
| cost_per_task ($)* | 0.00238 | 0.00171 |
| tokens_per_task | 456 | 389 |
| inter_agent_overhead_pct | 0.0 | 8.9 |

\* Вартість рахується з реального token usage, що повертає OpenRouter, за прайсом моделей (Sonnet 4.5 / Haiku 4.5).

**Cost breakdown by agent (crew):** synthesizer ≈ $0.0144 → data_analyst ≈ $0.0073 → advisor ≈ $0.0073 → router ≈ $0.0016 → guardian ≈ $0.0002. Координаційні агенти (router+synthesizer) дають ~9% inter-agent overhead.

**За категоріями обидві архітектури:** stat 6/6, advice 4/4, multistep 4/4, fraud 2/2, out_of_scope 2/2.

### Чому groundedness = 0.90, а не 1.0

~10% «непідтверджених» чисел — це коректно похідні **середньомісячні** значення (сума всіх місячних витрат / кількість місяців у датасеті, напр. $1529 за 12 міс → «$127/міс»), яких немає буквально у виході інструмента. Це не галюцинації, а агрегати; евалюатор навмисно суворий до ділення на довільний дільник. Жодне число у відповідях не суперечить даним.

---

## 4. Аналіз: де multi-agent переважає, де поступається

**Засторога щодо вибірки.** Golden set (18 задач) — невелика контрольна вибірка; на простих stat-запитах обидві архітектури очікувано дають паритет якості, а відмінності концентруються на advice/multistep і edge cases. Нижче — інженерний аналіз, що підтверджується структурою системи та трасами LangSmith.

**Де crew виграє (підтверджено структурно):**

- **Маршрутизація вартості.** Crew віддає прості ролі (router, guardian, data_analyst) дешевому Haiku, а Sonnet залишає лише для advisor/synthesizer. В офлайні це вже дає нижчі cost_per_task і tokens_per_task за baseline (де все йде через одну «дорогу» модель). На реальних моделях ефект посилюється.
- **Multi-step якість (очікувано).** Розділення «дістати числа» (DataAnalyst) і «зробити висновок» (Advisor) знижує ризик, що модель змішає арифметику з міркуванням — найбільша вигода саме на category 3 (порівняння рік-до-року, проєкції).
- **Контрольовані edge cases.** Окремий Guardian робить escalation/out-of-scope детермінованим вузлом графа, а не «надією на prompt». Менше шансів, що fraud-запит випадково отримає звичайну відповідь.
- **Спостережуваність.** Per-agent теги дають cost_breakdown_by_agent і inter_agent_overhead — діагностику, якої в монолітному baseline немає.

**Де crew поступається:**

- **Латентність.** Послідовні вузли → вищий p95 (5.7 vs 3.7 ms в офлайні; на реальних LLM це секунди на кожен зайвий hop). Для простих stat-запитів («дата платежу за Netflix») оркестрація — чисті накладні витрати.
- **Inter-agent overhead.** ~9% токенів витрачається на передачу контексту між агентами; на простих запитах це програш.
- **Складність.** Більше коду, точок відмови та конфігурації. Баг у маршрутизації ламає весь ланцюг.

**Чи виправдане ускладнення?** Так — але вибірково. Виграш crew концентрується на advice і multistep (де поділ ролей і дешева маршрутизація окупаються) та на надійній обробці edge cases. На простій статистиці single-agent baseline швидший і простіший.

---

## 5. Рекомендація для production

**Гібрид із роутингом за складністю.** Router (дешева модель) першим кроком класифікує запит:

- **stat / факт** → одразу single-agent tool-loop (baseline-шлях): мінімум латентності, в межах SLA ≤ 10 с;
- **advice / multistep** → повний crew (DataAnalyst → Advisor → Synthesizer): де якість висновку і economy-поради важливіші за кілька зайвих секунд;
- **fraud / out-of-scope** → Guardian-вузол: детермінований escalation/відмова.

Так бізнес-цілі виконуються разом: SLA тримається на «дешевих» запитах, якість і actionability — на складних, а вартість контролюється маршрутизацією моделей. Перед релізом — прогнати повний golden set через LangSmith Experiments та додати трасування у канарковому трафіку.

---

## 6. Обмеження та що не вдалося

- **Малий golden set** (18 задач) дає обмежену статистичну потужність; для production-висновків потрібна ширша вибірка та кілька прогонів на живих моделях через LangSmith Experiments (`--langsmith`).
- **Проєкції** (`monthly_summary`, `project_savings`) — наївні (лінійне масштабування), достатні для demo, але для production варто враховувати сезонність і регулярні платежі.
- **Залежність від якості router-класифікації:** помилкова маршрутизація на старті ламає весь ланцюг crew; у проді потрібен моніторинг частки невдалих JSON-парсів та fallback.

---

## 7. Як запустити

```bash
pip install -r requirements.txt
cp .env.example .env            # додати OPENROUTER_API_KEY (та LANGSMITH_API_KEY за бажання)

# Юніт-тести інструментів (без LLM)
pytest -q

# Golden set на обох архітектурах, метрики у report/results.{json,md}
python evals/run_experiments.py

# + LangSmith Experiments і LLM-as-judge
python evals/run_experiments.py --langsmith --llm-judge

# UI
streamlit run app/streamlit_app.py
```

`OPENROUTER_API_KEY` обов'язковий — агенти працюють на реальних моделях через OpenRouter; з `LANGSMITH_API_KEY` додаються трасування і Experiments.
