# Token Economics Smoke

This smoke validates the no-LLM context runtime token economics path.

| Metric | Value |
|---|---:|
| Queries | 20 |
| Average baseline input tokens | 15130.00 |
| Average actual input tokens | 21.50 |
| Average avoided input tokens | 15108.50 |
| Average avoided input tokens pct | 99.86% |
| Average KV bytes estimated | 2113536.00 |
| Average KV bytes avoided estimated | 1485225984.00 |
| Average cost estimated USD | 0.00002150 |
| Average cost avoided estimated USD | 0.01510850 |
| Average rows scanned | 100.50 |
| Average blocks materialized | 25.50 |
| p95 context latency | 9.21 ms |

Metrics JSON: `artifacts/runs/token_economics_smoke/metrics.json`
