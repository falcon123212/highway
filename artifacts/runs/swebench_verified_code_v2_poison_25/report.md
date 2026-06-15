# SWE-bench Verified ContextPack Benchmark - NON_VALIDATING

Dataset: `SWE-bench/SWE-bench_Verified` split `test`
Records: `25`

| Mode | Recall@1 | Recall@3 | Recall@5 | Precision@5 | Tokens avoided | p95 compile |
|---|---:|---:|---:|---:|---:|---:|
| `highway_code_contextpack_v2` | 0.00% | 0.00% | 0.00% | 0.00% | 99.90% | 5044.44 ms |

## Symbol Localization

| Mode | Symbol@1 | Symbol@3 | Symbol@5 | Hunk area | Relevant lines | Tokens/relevant line |
|---|---:|---:|---:|---:|---:|---:|
| `highway_code_contextpack_v2` | 0.00% | 0.00% | 0.00% | 0.00% | 0.00% | 237.34 |

## Audit

- Prompt distinct rate: `100.00%`
- Avg baseline blocks: `2175.88`
- Avg Highway blocks: `4.56`
- Avg tokens avoided: `99.90%`
- Repo index cache hit rate: `100.00%`
- Repo index build p95: `0.00 ms`
- Repo index load p95: `255.65 ms`
- Poison fail rate: `44.00%`

## Files

- Metrics JSON: `artifacts/runs/swebench_verified_code_v2_poison_25/metrics.json`
- Records JSONL: `artifacts/runs/swebench_verified_code_v2_poison_25/records.jsonl`
- Prompts: `artifacts/runs/swebench_verified_code_v2_poison_25/prompts`
