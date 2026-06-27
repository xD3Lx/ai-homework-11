# Personal Finance Coach — як запустити

Multi-agent crew (LangGraph) vs single-agent baseline для фінансових запитів за транзакціями користувача.

## Встановлення

```bash
pip install -r requirements.txt
cp .env.example .env     # додайте OPENROUTER_API_KEY (та LANGSMITH_API_KEY за бажання)
```

`OPENROUTER_API_KEY` обов'язковий — агенти працюють на реальних моделях через OpenRouter.
`LANGSMITH_API_KEY` вмикає трасування й Experiments.

## Команди

| Дія | Команда |
|---|---|
| Тести (інструменти, без LLM) | `pytest -q` |
| Golden set + метрики | `python evals/run_experiments.py` |
| Те саме з LangSmith | `python evals/run_experiments.py --langsmith --llm-judge` |
| UI | `streamlit run app/streamlit_app.py` |

Метрики пишуться у `report/results.json` та `report/results.md`. Повний звіт — `report/REPORT.md`.

## Деплой (Docker / Fly.io)

Локально в Docker:

```bash
docker build -t finance-coach .
docker run -p 8080:8080 -e OPENROUTER_API_KEY=sk-or-... finance-coach
# відкрити http://localhost:8080
```

На Fly.io (`Dockerfile` + `fly.toml` уже в репо):

```bash
fly launch --no-deploy            # або: fly apps create personal-finance-coach
fly secrets set OPENROUTER_API_KEY=sk-or-...   # обов'язково
fly secrets set LANGSMITH_API_KEY=lsv2-...     # опційно
fly deploy
```

Streamlit слухає порт 8080 (Fly health-check — `/_stcore/health`). Ключі задаються через `fly secrets`, не в `fly.toml`.

## Структура

```
src/finance_coach/
  config.py        # налаштування, offline-детект, ціни моделей
  data.py          # CSV → SQLite + резолвер відносних дат
  tools.py         # 10 фінансових інструментів + схеми для tool-calling
  llm.py           # OpenRouter клієнт (chat / complete)
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
- `FINANCE_COACH_NOW=2025-11-30` — «сьогодні» для відносних дат (кінець датасету).
