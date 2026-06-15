# Long Conversation Quality Benchmark

Verdict: NON_VALIDATING
Model: `qwen3:8b`

| Turns | Answer OK | Source attr | Hallucination | Coherence | Avoided input | Output over budget | Context p95 |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 4 | 100.00% | 100.00% | 0.00% | 100.00% | 95.46% | 50.00% | 11.46 ms |

This benchmark separates context quality, answer quality, input-token economy, and output-token budget control.

Metrics JSON: `artifacts/runs/long_conversation_quality_qwen3_8b/metrics.json`
Records JSONL: `artifacts/runs/long_conversation_quality_qwen3_8b/records.jsonl`
