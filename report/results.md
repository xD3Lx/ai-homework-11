# Golden Set Results — Crew vs Baseline

Mode: **offline (mock LLM)** · Tasks: 18 (stat:6, advice:4, multistep:4, fraud:2, out_of_scope:2)

| Metric | Baseline | Crew |
|---|---|---|
| success_rate | 1.0 | 1.0 |
| tool_selection_accuracy | 1.0 | 1.0 |
| groundedness | 0.9 | 0.9 |
| latency_p50 (ms) | 1.3 | 1.5 |
| latency_p95 (ms) | 3.3 | 5.4 |
| cost_per_task ($) | 0.002383 | 0.00171 |
| tokens_per_task | 456.2 | 388.5 |
| inter_agent_overhead_pct | 0.0 | 8.86 |

## Cost breakdown by agent (crew)

| Agent | Cost ($) |
|---|---|
| synthesizer | 0.014382 |
| data_analyst | 0.007327 |
| advisor | 0.007296 |
| router | 0.001552 |
| guardian | 0.000219 |

## Per-task success (1 = pass)

| Task | Category | Baseline | Crew |
|---|---|---|---|
| stat_coffee_lastweek | stat | 1 | 1 |
| stat_top_categories | stat | 1 | 1 |
| stat_netflix_last | stat | 1 | 1 |
| stat_groceries_month | stat | 1 | 1 |
| stat_delivery_year | stat | 1 | 1 |
| advice_save_200 | advice | 1 | 1 |
| advice_subscriptions | advice | 1 | 1 |
| advice_credit_card | advice | 1 | 1 |
| advice_forgotten_sub | advice | 1 | 1 |
| multi_delivery_halve | multistep | 1 | 1 |
| multi_month_positive | multistep | 1 | 1 |
| multi_year_compare | multistep | 1 | 1 |
| multi_coffee_halve | multistep | 1 | 1 |
| fraud_booking | fraud | 1 | 1 |
| fraud_unknown | fraud | 1 | 1 |
| oos_stocks | out_of_scope | 1 | 1 |
| oos_crypto | out_of_scope | 1 | 1 |
| multiturn_coffee_month | stat | 1 | 1 |