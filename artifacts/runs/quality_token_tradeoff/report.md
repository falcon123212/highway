# Quality Token Tradeoff Smoke

This smoke compares a full-context reflective baseline against Highway selected context.
Both paths use the same deterministic reflective answerer; only the context size differs.

| Metric | Value |
|---|---:|
| Queries | 20 |
| Baseline EM | 100.00% |
| Highway EM | 100.00% |
| Quality delta | 0.00 pp |
| Avg baseline prompt tokens | 17170.00 |
| Avg Highway prompt tokens | 66.50 |
| Avg prompt tokens avoided | 17103.50 |
| Avg prompt tokens avoided pct | 99.61% |
| Avg baseline output tokens | 14.00 |
| Avg Highway output tokens | 14.00 |
| Avg KV bytes avoided estimated | 1681342464.00 |
| Avg cost avoided estimated USD | 0.01710350 |
| Avg Highway context latency | 5.16 ms |

Metrics JSON: `artifacts/runs/quality_token_tradeoff/metrics.json`
