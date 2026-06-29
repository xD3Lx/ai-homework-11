# Golden Set Results — Crew vs Baseline

LLM: **OpenRouter** · Tasks: 18 (stat:6, advice:4, multistep:4, fraud:2, out_of_scope:2)

| Metric | Baseline | Crew |
|---|---|---|
| success_rate | 0.944 | 1.0 |
| tool_selection_accuracy | 0.907 | 0.944 |
| groundedness | 0.794 | 0.781 |
| latency_p50 (ms) | 7758.8 | 12553.9 |
| latency_p95 (ms) | 13069.2 | 33735.9 |
| cost_per_task ($) | 0.015396 | 0.014341 |
| tokens_per_task | 3854.6 | 4623.2 |
| inter_agent_overhead_pct | 0.0 | 18.16 |

## Cost breakdown by agent (crew)

| Agent | Cost ($) |
|---|---|
| synthesizer | 0.109692 |
| data_analyst | 0.078471 |
| advisor | 0.062922 |
| router | 0.005864 |
| guardian | 0.001185 |

## Per-task success (1 = pass)

| Task | Category | Baseline | Crew |
|---|---|---|---|
| stat_coffee_lastweek | stat | 1 | 1 |
| stat_top_categories | stat | 1 | 1 |
| stat_netflix_last | stat | 1 | 1 |
| stat_groceries_month | stat | 1 | 1 |
| stat_delivery_year | stat | 1 | 1 |
| advice_save_200 | advice | 0 | 1 |
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