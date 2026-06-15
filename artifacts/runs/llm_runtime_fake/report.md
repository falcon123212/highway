# LLM Runtime Fake Benchmark

Verdict: VALIDATING

| Size | Baseline EM | Highway EM | Quality delta | Avoided tokens | Baseline TTFT p95 | Highway TTFT p95 | Baseline total p95 | Highway total p95 | Context p95 | Rows scanned | Blocks materialized | Metrics complete |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1,000 | 100.00% | 100.00% | 0.00 pp | 99.68% | 429.02 ms | 1.43 ms | 485.02 ms | 57.30 ms | 2.49 ms | 1.20 | 1.20 | 100.00% |
| 10,000 | 100.00% | 100.00% | 0.00 pp | 99.97% | 4254.02 ms | 1.43 ms | 4310.02 ms | 57.30 ms | 5.11 ms | 1.20 | 1.20 | 100.00% |
| 100,000 | 100.00% | 100.00% | 0.00 pp | 100.00% | 42504.03 ms | 1.43 ms | 42560.03 ms | 57.30 ms | 16.85 ms | 1.20 | 1.20 | 100.00% |

## Why this matters

Token economy is accepted only when answer quality stays correct.
The fake client isolates Highway runtime behavior from model randomness.
A real local LLM should only be connected after this benchmark remains validating.

Metrics JSON: `artifacts/runs/llm_runtime_fake/metrics.json`
Records JSONL: `artifacts/runs/llm_runtime_fake/records.jsonl`
