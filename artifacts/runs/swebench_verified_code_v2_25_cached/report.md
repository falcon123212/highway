# SWE-bench Verified ContextPack Benchmark - NON_VALIDATING

Dataset: `SWE-bench/SWE-bench_Verified` split `test`
Records: `25`

| Mode | Recall@1 | Recall@3 | Recall@5 | Precision@5 | Tokens avoided | p95 compile |
|---|---:|---:|---:|---:|---:|---:|
| `highway_contextpack` | 12.00% | 20.00% | 24.00% | 4.80% | 99.89% | 2285.53 ms |
| `highway_code_contextpack_v2` | 30.00% | 34.00% | 42.00% | 8.80% | 99.89% | 5144.76 ms |

## Symbol Localization

| Mode | Symbol@1 | Symbol@3 | Symbol@5 | Hunk area | Relevant lines | Tokens/relevant line |
|---|---:|---:|---:|---:|---:|---:|
| `highway_contextpack` | 12.00% | 20.00% | 24.00% | 24.00% | 24.00% | 284.27 |
| `highway_code_contextpack_v2` | 30.29% | 34.29% | 42.29% | 41.68% | 41.68% | 259.71 |

## Audit

- Prompt distinct rate: `100.00%`
- Avg baseline blocks: `2175.88`
- Avg Highway blocks: `5.00`
- Avg tokens avoided: `99.89%`
- Repo index cache hit rate: `100.00%`
- Repo index build p95: `0.00 ms`
- Repo index load p95: `856.18 ms`
- Poison fail rate: `0.00%`

## Files

- Metrics JSON: `artifacts/runs/swebench_verified_code_v2_25_cached/metrics.json`
- Records JSONL: `artifacts/runs/swebench_verified_code_v2_25_cached/records.jsonl`
- Prompts: `artifacts/runs/swebench_verified_code_v2_25_cached/prompts`
