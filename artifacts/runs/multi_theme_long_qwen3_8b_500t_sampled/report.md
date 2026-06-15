# Multi-Theme Long LLM Benchmark

Verdict: VALIDATING
Model: `qwen3:8b`

| Turns | Answer OK | Source attr | Hallucination | Coherence | Long-range | Avoided input | Prompt distinct | Context p95 |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 500 | 100.00% | 100.00% | 0.00% | 100.00% | 100.00% | 96.95% | 100.00% | 2.12 ms |

Average baseline blocks: `252.00`.
Average Highway blocks: `2.00`.
Poison fail rate: `0.00%`.

Metrics JSON: `artifacts/runs/multi_theme_long_qwen3_8b_500t_sampled/metrics.json`
Records JSONL: `artifacts/runs/multi_theme_long_qwen3_8b_500t_sampled/records.jsonl`
