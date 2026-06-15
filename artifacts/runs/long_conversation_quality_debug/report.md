# Long Conversation Quality Benchmark

Verdict: NON_VALIDATING
Model: `contract_aware_fake`

| Turns | Answer OK | Source attr | Hallucination | Coherence | Avoided input | Output over budget | Context p95 |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 8 | 100.00% | 100.00% | 0.00% | 100.00% | 72.60% | 0.00% | 5.01 ms |

This benchmark separates context quality, answer quality, input-token economy, and output-token budget control.

Metrics JSON: `artifacts/runs/long_conversation_quality_debug/metrics.json`
Records JSONL: `artifacts/runs/long_conversation_quality_debug/records.jsonl`
