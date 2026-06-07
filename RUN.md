# Personal Finance Coach — як запустити

Multi-agent crew (LangGraph) vs single-agent baseline для фінансових запитів за транзакціями користувача.

## Встановлення

```bash
pip install -r requirements.txt
cp .env.example .env     # опційно: OPENROUTER_API_KEY, LANGSMITH_API_KEY
```

Без ключів усе працює в **offline-режимі** (детермінований mock-LLM) — зручно для тестів і demo.
Додайте `OPENROUTER_API_KEY`, щоб увімкнути реальні моделі; `LANGSMITH_API_KEY` — для трасування й Experiments.

## Команди

| Дія | Команда |
|---|---|
| Тести | `FINANCE_COACH_OFFLINE=1 pytest -q` |
| Golden set + метрики | `FINANCE_COACH_OFFLINE=1 python evals/run_experiments.py` |
| Те саме з LangSmith | `python evals/run_experiments.py --langsmith --llm-judge` |
| UI | `streamlit run app/streamlit_app.py` |

Метрики пишуться у `report/results.json` та `report/results.md`. Повний звіт — `report/REPORT.md`.

## Структура

```
src/finance_coach/
  config.py        # налаштування, offline-детект, ціни моделей
  data.py          # CSV → SQLite + резолвер відносних дат
  tools.py         # 10 фінансових інструментів + схеми для tool-calling
  llm.py           # OpenRouter + offline-емулятор (chat / complete)
  offline_brain.py # детермінований NLU + планувальник + композер відповідей
  baseline.py      # single-agent tool_use loop
  crew.py          # LangGraph: router, guardian, data_analyst, advisor, synthesizer
  endpoint.py      # спільний endpoint + multi-turn Session
  types.py         # RunResult / TraceStep
  obs.py           # LangSmith @traceable shim
evals/
  golden_set.py    # 18 задач
  evaluators.py    # success_rate, tool_selection_accuracy, groundedness
  run_experiments.py
app/streamlit_app.py
tests/test_tools.py
report/REPORT.md
```

## Конфігурація (.env)

- `OPENROUTER_API_KEY`, `MODEL_SMART`, `MODEL_CHEAP` — LLM.
- `LANGSMITH_API_KEY`, `LANGSMITH_PROJECT` — спостережуваність.
- `FINANCE_COACH_OFFLINE=1` — примусовий offline-режим.
- `FINANCE_COACH_NOW=2025-11-30` — «сьогодні» для відносних дат (кінець датасету).
