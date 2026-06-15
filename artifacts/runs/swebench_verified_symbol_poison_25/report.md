# SWE-bench Verified ContextPack Benchmark - NON_VALIDATING

Dataset: `SWE-bench/SWE-bench_Verified` split `test`
Records: `25`

| Mode | Recall@1 | Recall@3 | Recall@5 | Precision@5 | Tokens avoided | p95 compile |
|---|---:|---:|---:|---:|---:|---:|
| `highway_contextpack` | 0.00% | 0.00% | 0.00% | 0.00% | 99.90% | 2000.88 ms |

## Symbol Localization

| Mode | Symbol@1 | Symbol@3 | Symbol@5 | Hunk area | Relevant lines | Tokens/relevant line |
|---|---:|---:|---:|---:|---:|---:|
| `highway_contextpack` | 0.00% | 0.00% | 0.00% | 0.00% | 0.00% | 265.45 |

## Audit

- Prompt distinct rate: `100.00%`
- Avg baseline blocks: `2175.88`
- Avg Highway blocks: `4.76`
- Avg tokens avoided: `99.90%`
- Poison fail rate: `24.00%`

## Files

- Metrics JSON: `artifacts/runs/swebench_verified_symbol_poison_25/metrics.json`
- Records JSONL: `artifacts/runs/swebench_verified_symbol_poison_25/records.jsonl`
- Prompts: `artifacts/runs/swebench_verified_symbol_poison_25/prompts`
