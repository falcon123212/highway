# Long Conversation Quality Benchmark

Verdict: VALIDATING
Model: `contract_aware_fake`

| Turns | Answer OK | Source attr | Hallucination | Coherence | Avoided input | Output over budget | Retry rate | Context p95 |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 12 | 100.00% | 100.00% | 0.00% | 100.00% | 98.09% | 0.00% | 0.00% | 6.64 ms |

Average output tokens saved by retry: `0.00`.

This benchmark separates context quality, answer quality, input-token economy, and output-token budget control.

Metrics JSON: `artifacts/runs/long_conversation_quality_fake/metrics.json`
Records JSONL: `artifacts/runs/long_conversation_quality_fake/records.jsonl`
