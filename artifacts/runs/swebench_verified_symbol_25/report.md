# SWE-bench Verified ContextPack Benchmark - NON_VALIDATING

Dataset: `SWE-bench/SWE-bench_Verified` split `test`
Records: `25`

| Mode | Recall@1 | Recall@3 | Recall@5 | Precision@5 | Tokens avoided | p95 compile |
|---|---:|---:|---:|---:|---:|---:|
| `bm25_topk` | 12.00% | 16.00% | 24.00% | 4.80% | 99.89% | 2044.73 ms |
| `hybrid` | 12.00% | 20.00% | 24.00% | 4.80% | 99.89% | 2129.06 ms |
| `highway_contextpack` | 12.00% | 20.00% | 24.00% | 4.80% | 99.89% | 2077.70 ms |

## Symbol Localization

| Mode | Symbol@1 | Symbol@3 | Symbol@5 | Hunk area | Relevant lines | Tokens/relevant line |
|---|---:|---:|---:|---:|---:|---:|
| `bm25_topk` | 12.00% | 16.00% | 24.00% | 24.00% | 24.00% | 285.48 |
| `hybrid` | 12.00% | 20.00% | 24.00% | 24.00% | 24.00% | 284.27 |
| `highway_contextpack` | 12.00% | 20.00% | 24.00% | 24.00% | 24.00% | 284.27 |

## Audit

- Prompt distinct rate: `100.00%`
- Avg baseline blocks: `2175.88`
- Avg Highway blocks: `5.00`
- Avg tokens avoided: `99.89%`
- Poison fail rate: `0.00%`

## Files

- Metrics JSON: `artifacts/runs/swebench_verified_symbol_25/metrics.json`
- Records JSONL: `artifacts/runs/swebench_verified_symbol_25/records.jsonl`
- Prompts: `artifacts/runs/swebench_verified_symbol_25/prompts`
