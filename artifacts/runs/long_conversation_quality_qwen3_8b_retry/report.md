# Long Conversation Quality Benchmark

Verdict: VALIDATING
Model: `qwen3:8b`

| Turns | Answer OK | Source attr | Hallucination | Coherence | Avoided input | Output over budget | Retry rate | Context p95 |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 4 | 100.00% | 100.00% | 0.00% | 100.00% | 94.82% | 0.00% | 0.00% | 9.65 ms |

Average output tokens saved by retry: `0.00`.

This benchmark separates context quality, answer quality, input-token economy, and output-token budget control.

Metrics JSON: `artifacts/runs/long_conversation_quality_qwen3_8b_retry/metrics.json`
Records JSONL: `artifacts/runs/long_conversation_quality_qwen3_8b_retry/records.jsonl`
