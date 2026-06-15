# Long Conversation Quality Benchmark

Verdict: VALIDATING
Model: `qwen3:8b`

| Turns | Answer OK | Source attr | Hallucination | Coherence | Avoided input | Output over budget | Prompt distinct | Context p95 |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 25 | 100.00% | 100.00% | 0.00% | 100.00% | 94.97% | 0.00% | 100.00% | 3.81 ms |

Average output tokens saved by retry: `0.00`.
Retry rate: `0.00%`.
Poison fail rate: `0.00%`.
Average baseline blocks: `168.00`.
Average Highway blocks: `2.28`.
Average baseline prompt tokens: `5050.44`.
Average Highway prompt tokens: `96.04`.

This benchmark separates context quality, answer quality, input-token economy, and output-token budget control.

Metrics JSON: `artifacts/runs/long_conversation_quality_qwen3_8b_25t/metrics.json`
Records JSONL: `artifacts/runs/long_conversation_quality_qwen3_8b_25t/records.jsonl`
