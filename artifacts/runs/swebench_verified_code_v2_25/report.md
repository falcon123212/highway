# SWE-bench Verified ContextPack Benchmark - NON_VALIDATING

Dataset: `SWE-bench/SWE-bench_Verified` split `test`
Records: `25`

| Mode | Recall@1 | Recall@3 | Recall@5 | Precision@5 | Tokens avoided | p95 compile |
|---|---:|---:|---:|---:|---:|---:|
| `highway_contextpack` | 12.00% | 20.00% | 24.00% | 4.80% | 99.89% | 2008.84 ms |
| `highway_code_contextpack_v2` | 30.00% | 34.00% | 42.00% | 8.80% | 99.89% | 10204.31 ms |

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
- Repo index cache hit rate: `0.00%`
- Repo index build p95: `19488.05 ms`
- Repo index load p95: `19584.37 ms`
- Poison fail rate: `0.00%`

## Files

- Metrics JSON: `artifacts/runs/swebench_verified_code_v2_25/metrics.json`
- Records JSONL: `artifacts/runs/swebench_verified_code_v2_25/records.jsonl`
- Prompts: `artifacts/runs/swebench_verified_code_v2_25/prompts`
