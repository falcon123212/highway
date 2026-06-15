# Local LLM Quality Benchmark

Verdict: VALIDATING
Model: `qwen3:8b`

| Size | Baseline EM | Highway EM | Quality delta | Source attr | Coherence | Avoided tokens | Baseline TTFT p95 | Highway TTFT p95 | Context p95 | Metrics complete |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 80 | 100.00% | 100.00% | 0.00 pp | 100.00% | 100.00% | 93.02% | 1356.44 ms | 107.62 ms | 4.70 ms | 100.00% |

## Interpretation

Qwen 0.5B style runs are integration smokes. Quality claims require a stronger local model, starting with a 1.5B class model.
Token savings are accepted only when factual quality, source attribution, and multi-turn coherence do not regress.

Metrics JSON: `artifacts/runs/local_llm_quality_qwen3_8b_smoke/metrics.json`
Records JSONL: `artifacts/runs/local_llm_quality_qwen3_8b_smoke/records.jsonl`
