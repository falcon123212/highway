# SWE-bench Verified ContextPack Benchmark - VALIDATING

Dataset: `SWE-bench/SWE-bench_Verified` split `test`
Records: `1`

| Mode | Recall@1 | Recall@3 | Recall@5 | Precision@5 | Tokens avoided | p95 compile |
|---|---:|---:|---:|---:|---:|---:|
| `highway_contextpack` | 0.00% | 100.00% | 100.00% | 20.00% | 99.93% | 1925.47 ms |

## Audit

- Prompt distinct rate: `100.00%`
- Avg baseline blocks: `3406.00`
- Avg Highway blocks: `5.00`
- Avg tokens avoided: `99.93%`
- Poison fail rate: `0.00%`

## Files

- Metrics JSON: `artifacts/runs/swebench_verified_fileloc_1/metrics.json`
- Records JSONL: `artifacts/runs/swebench_verified_fileloc_1/records.jsonl`
- Prompts: `artifacts/runs/swebench_verified_fileloc_1/prompts`
