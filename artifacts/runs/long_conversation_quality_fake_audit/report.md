# Long Conversation Quality Benchmark

Verdict: VALIDATING
Model: `contract_aware_fake`

| Turns | Answer OK | Source attr | Hallucination | Coherence | Avoided input | Output over budget | Prompt distinct | Context p95 |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 12 | 100.00% | 100.00% | 0.00% | 100.00% | 98.09% | 0.00% | 100.00% | 6.85 ms |

Average output tokens saved by retry: `0.00`.
Retry rate: `0.00%`.
Poison fail rate: `0.00%`.
Average baseline blocks: `168.00`.
Average Highway blocks: `2.33`.
Average baseline prompt tokens: `5050.42`.
Average Highway prompt tokens: `96.33`.

This benchmark separates context quality, answer quality, input-token economy, and output-token budget control.

Metrics JSON: `artifacts/runs/long_conversation_quality_fake_audit/metrics.json`
Records JSONL: `artifacts/runs/long_conversation_quality_fake_audit/records.jsonl`
