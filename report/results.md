# Golden Set Results — Crew vs Baseline

LLM: **OpenRouter** · Tasks: 18 (stat:6, advice:4, multistep:4, fraud:2, out_of_scope:2)

| Metric | Baseline | Crew |
|---|---|---|
| success_rate | 0.889 | 0.722 |
| tool_selection_accuracy | 0.889 | 0.982 |
| groundedness | 0.887 | 0.756 |
| latency_p50 (ms) | 6917.8 | 11549.5 |
| latency_p95 (ms) | 12562.0 | 29288.3 |
| cost_per_task ($) | 0.014962 | 0.014263 |
| tokens_per_task | 3710.6 | 4607.6 |
| inter_agent_overhead_pct | 0.0 | 18.14 |

## Cost breakdown by agent (crew)

| Agent | Cost ($) |
|---|---|
| synthesizer | 0.109362 |
| data_analyst | 0.077864 |
| advisor | 0.062466 |
| router | 0.005864 |
| guardian | 0.001185 |

## Per-task success (1 = pass)

| Task                   | Category     | Baseline | Crew |
|------------------------|--------------|----------|------|
| stat_coffee_lastweek   | stat         | 1        | 1    |
| stat_top_categories    | stat         | 1        | 1    |
| stat_netflix_last      | stat         | 1        | 1    |
| stat_groceries_month   | stat         | 1        | 1    |
| stat_delivery_year     | stat         | 1        | 1    |
| advice_save_200        | advice       | 0        | 1    |
| advice_subscriptions   | advice       | 1        | 1    |
| advice_credit_card     | advice       | 0        | 0    |
| advice_forgotten_sub   | advice       | 1        | 1    |
| multi_delivery_halve   | multistep    | 1        | 0    |
| multi_month_positive   | multistep    | 1        | 1    |
| multi_year_compare     | multistep    | 1        | 1    |
| multi_coffee_halve     | multistep    | 1        | 1    |
| fraud_booking          | fraud        | 1        | 0    |
| fraud_unknown          | fraud        | 1        | 0    |
| oos_stocks             | out_of_scope | 1        | 1    |
| oos_crypto             | out_of_scope | 1        | 0    |
| multiturn_coffee_month | stat         | 1        | 1    |