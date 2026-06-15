# Long Conversation Quality Benchmark

Verdict: NON_VALIDATING
Model: `contract_aware_fake`

| Turns | Answer OK | Source attr | Hallucination | Coherence | Avoided input | Output over budget | Prompt distinct | Context p95 |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 4 | 100.00% | 100.00% | 25.00% | 100.00% | 98.24% | 0.00% | 100.00% | 9.98 ms |

Average output tokens saved by retry: `0.00`.
Retry rate: `0.00%`.
Poison fail rate: `100.00%`.
Average baseline blocks: `168.00`.
Average Highway blocks: `1.50`.
Average baseline prompt tokens: `5050.00`.
Average Highway prompt tokens: `88.75`.

This benchmark separates context quality, answer quality, input-token economy, and output-token budget control.

Metrics JSON: `artifacts/runs/long_conversation_quality_poison/metrics.json`
Records JSONL: `artifacts/runs/long_conversation_quality_poison/records.jsonl`
