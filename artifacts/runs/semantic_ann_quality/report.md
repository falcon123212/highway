# Semantic ANN Quality Benchmark

Verdict: NON_VALIDATING

| Size | Strategy | ANN cap | Lexical cap | EM | Recall@k | p95 latency | Rows scanned | Reranker p95 | Blocks materialized | Metrics complete |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1,000 | ooc_full_scan | 200 | 0 | 100.00% | 100.00% | 10.02 ms | 1000.00 | 0.00 ms | 50.00 | 100.00% |
| 1,000 | ooc_ann_hnsw | 200 | 0 | 100.00% | 22.50% | 7.18 ms | 200.00 | 0.00 ms | 50.00 | 100.00% |
| 1,000 | ooc_ann_pruned_hybrid | 200 | 0 | 100.00% | 24.00% | 7.06 ms | 200.00 | 0.00 ms | 50.00 | 100.00% |
| 1,000 | ooc_semantic_rescue_hybrid | 200 | 0 | 100.00% | 99.70% | 13.14 ms | 1000.00 | 0.00 ms | 50.00 | 100.00% |
| 1,000 | ooc_semantic_lexical_rescue | 200 | 1000 | 100.00% | 99.30% | 9.28 ms | 200.90 | 0.00 ms | 50.00 | 100.00% |
| 1,000 | ooc_semantic_lexical_rescue | 200 | 5000 | 100.00% | 99.30% | 10.01 ms | 200.90 | 0.00 ms | 50.00 | 100.00% |
| 1,000 | ooc_semantic_rerank_rescue | 200 | 1000 | 100.00% | 43.00% | 30.38 ms | 200.90 | 20.84 ms | 50.00 | 100.00% |
| 1,000 | ooc_semantic_rerank_rescue | 200 | 5000 | 100.00% | 43.00% | 29.78 ms | 200.90 | 20.51 ms | 50.00 | 100.00% |
| 1,000 | ooc_semantic_field_rescue | 200 | 1000 | 100.00% | 99.30% | 10.00 ms | 215.70 | 0.00 ms | 50.00 | 100.00% |
| 1,000 | ooc_semantic_field_rescue | 200 | 5000 | 100.00% | 99.30% | 10.53 ms | 215.70 | 0.00 ms | 50.00 | 100.00% |
| 1,000 | ooc_full_scan | 500 | 0 | 100.00% | 100.00% | 10.79 ms | 1000.00 | 0.00 ms | 50.00 | 100.00% |
| 1,000 | ooc_ann_hnsw | 500 | 0 | 100.00% | 51.00% | 8.51 ms | 500.00 | 0.00 ms | 50.00 | 100.00% |
| 1,000 | ooc_ann_pruned_hybrid | 500 | 0 | 100.00% | 51.70% | 8.43 ms | 500.00 | 0.00 ms | 50.00 | 100.00% |
| 1,000 | ooc_semantic_rescue_hybrid | 500 | 0 | 100.00% | 100.00% | 15.00 ms | 1000.00 | 0.00 ms | 50.00 | 100.00% |
| 1,000 | ooc_semantic_lexical_rescue | 500 | 1000 | 100.00% | 100.00% | 11.79 ms | 500.40 | 0.00 ms | 50.00 | 100.00% |
| 1,000 | ooc_semantic_lexical_rescue | 500 | 5000 | 100.00% | 100.00% | 10.38 ms | 500.40 | 0.00 ms | 50.00 | 100.00% |
| 1,000 | ooc_semantic_rerank_rescue | 500 | 1000 | 100.00% | 47.70% | 61.99 ms | 500.40 | 51.17 ms | 50.00 | 100.00% |
| 1,000 | ooc_semantic_rerank_rescue | 500 | 5000 | 100.00% | 47.70% | 60.72 ms | 500.40 | 49.70 ms | 50.00 | 100.00% |
| 1,000 | ooc_semantic_field_rescue | 500 | 1000 | 100.00% | 100.00% | 11.93 ms | 508.65 | 0.00 ms | 50.00 | 100.00% |
| 1,000 | ooc_semantic_field_rescue | 500 | 5000 | 100.00% | 100.00% | 11.97 ms | 508.65 | 0.00 ms | 50.00 | 100.00% |
| 1,000 | ooc_full_scan | 1000 | 0 | 100.00% | 100.00% | 11.05 ms | 1000.00 | 0.00 ms | 50.00 | 100.00% |
| 1,000 | ooc_ann_hnsw | 1000 | 0 | 100.00% | 72.40% | 8.97 ms | 906.95 | 0.00 ms | 50.00 | 100.00% |
| 1,000 | ooc_ann_pruned_hybrid | 1000 | 0 | 100.00% | 74.00% | 9.53 ms | 906.95 | 0.00 ms | 50.00 | 100.00% |
| 1,000 | ooc_semantic_rescue_hybrid | 1000 | 0 | 100.00% | 91.70% | 14.99 ms | 1000.00 | 0.00 ms | 50.00 | 100.00% |
| 1,000 | ooc_semantic_lexical_rescue | 1000 | 1000 | 100.00% | 98.50% | 13.59 ms | 907.05 | 0.00 ms | 50.00 | 100.00% |
| 1,000 | ooc_semantic_lexical_rescue | 1000 | 5000 | 100.00% | 98.50% | 13.53 ms | 907.05 | 0.00 ms | 50.00 | 100.00% |
| 1,000 | ooc_semantic_rerank_rescue | 1000 | 1000 | 100.00% | 48.00% | 107.89 ms | 907.05 | 94.06 ms | 50.00 | 100.00% |
| 1,000 | ooc_semantic_rerank_rescue | 1000 | 5000 | 100.00% | 48.00% | 110.86 ms | 907.05 | 95.46 ms | 50.00 | 100.00% |
| 1,000 | ooc_semantic_field_rescue | 1000 | 1000 | 100.00% | 93.80% | 13.98 ms | 908.40 | 0.00 ms | 50.00 | 100.00% |
| 1,000 | ooc_semantic_field_rescue | 1000 | 5000 | 100.00% | 93.80% | 12.87 ms | 908.40 | 0.00 ms | 50.00 | 100.00% |
| 10,000 | ooc_full_scan | 200 | 0 | 100.00% | 100.00% | 51.46 ms | 10000.00 | 0.00 ms | 50.00 | 100.00% |
| 10,000 | ooc_ann_hnsw | 200 | 0 | 100.00% | 17.30% | 8.52 ms | 200.00 | 0.00 ms | 50.00 | 100.00% |
| 10,000 | ooc_ann_pruned_hybrid | 200 | 0 | 100.00% | 18.70% | 9.24 ms | 200.00 | 0.00 ms | 50.00 | 100.00% |
| 10,000 | ooc_semantic_rescue_hybrid | 200 | 0 | 100.00% | 83.00% | 89.20 ms | 10000.00 | 0.00 ms | 50.00 | 100.00% |
| 10,000 | ooc_semantic_lexical_rescue | 200 | 1000 | 100.00% | 95.20% | 46.15 ms | 201.00 | 0.00 ms | 50.00 | 100.00% |
| 10,000 | ooc_semantic_lexical_rescue | 200 | 5000 | 100.00% | 95.20% | 40.22 ms | 201.00 | 0.00 ms | 50.00 | 100.00% |
| 10,000 | ooc_semantic_rerank_rescue | 200 | 1000 | 100.00% | 79.30% | 63.99 ms | 201.00 | 23.04 ms | 50.00 | 100.00% |
| 10,000 | ooc_semantic_rerank_rescue | 200 | 5000 | 100.00% | 79.30% | 70.16 ms | 201.00 | 26.71 ms | 50.00 | 100.00% |
| 10,000 | ooc_semantic_field_rescue | 200 | 1000 | 100.00% | 95.20% | 40.44 ms | 218.50 | 0.00 ms | 50.00 | 100.00% |
| 10,000 | ooc_semantic_field_rescue | 200 | 5000 | 100.00% | 95.20% | 39.57 ms | 218.50 | 0.00 ms | 50.00 | 100.00% |
| 10,000 | ooc_full_scan | 500 | 0 | 100.00% | 100.00% | 53.90 ms | 10000.00 | 0.00 ms | 50.00 | 100.00% |
| 10,000 | ooc_ann_hnsw | 500 | 0 | 100.00% | 14.50% | 9.11 ms | 500.00 | 0.00 ms | 50.00 | 100.00% |
| 10,000 | ooc_ann_pruned_hybrid | 500 | 0 | 100.00% | 15.90% | 9.15 ms | 500.00 | 0.00 ms | 50.00 | 100.00% |
| 10,000 | ooc_semantic_rescue_hybrid | 500 | 0 | 100.00% | 85.20% | 90.70 ms | 10000.00 | 0.00 ms | 50.00 | 100.00% |
| 10,000 | ooc_semantic_lexical_rescue | 500 | 1000 | 100.00% | 74.60% | 41.01 ms | 501.00 | 0.00 ms | 50.00 | 100.00% |
| 10,000 | ooc_semantic_lexical_rescue | 500 | 5000 | 100.00% | 74.60% | 43.10 ms | 501.00 | 0.00 ms | 50.00 | 100.00% |
| 10,000 | ooc_semantic_rerank_rescue | 500 | 1000 | 100.00% | 53.10% | 98.38 ms | 501.00 | 58.90 ms | 50.00 | 100.00% |
| 10,000 | ooc_semantic_rerank_rescue | 500 | 5000 | 100.00% | 53.10% | 97.65 ms | 501.00 | 55.77 ms | 50.00 | 100.00% |
| 10,000 | ooc_semantic_field_rescue | 500 | 1000 | 100.00% | 74.60% | 40.91 ms | 518.00 | 0.00 ms | 50.00 | 100.00% |
| 10,000 | ooc_semantic_field_rescue | 500 | 5000 | 100.00% | 74.60% | 40.27 ms | 518.00 | 0.00 ms | 50.00 | 100.00% |
| 10,000 | ooc_full_scan | 1000 | 0 | 100.00% | 100.00% | 57.14 ms | 10000.00 | 0.00 ms | 50.00 | 100.00% |
| 10,000 | ooc_ann_hnsw | 1000 | 0 | 100.00% | 11.10% | 11.99 ms | 1000.00 | 0.00 ms | 50.00 | 100.00% |
| 10,000 | ooc_ann_pruned_hybrid | 1000 | 0 | 100.00% | 12.50% | 13.03 ms | 1000.00 | 0.00 ms | 50.00 | 100.00% |
| 10,000 | ooc_semantic_rescue_hybrid | 1000 | 0 | 100.00% | 99.20% | 91.10 ms | 10000.00 | 0.00 ms | 50.00 | 100.00% |
| 10,000 | ooc_semantic_lexical_rescue | 1000 | 1000 | 100.00% | 73.70% | 46.07 ms | 1000.00 | 0.00 ms | 50.00 | 100.00% |
| 10,000 | ooc_semantic_lexical_rescue | 1000 | 5000 | 100.00% | 73.70% | 46.37 ms | 1001.00 | 0.00 ms | 50.00 | 100.00% |
| 10,000 | ooc_semantic_rerank_rescue | 1000 | 1000 | 100.00% | 51.80% | 161.18 ms | 1000.00 | 117.41 ms | 50.00 | 100.00% |
| 10,000 | ooc_semantic_rerank_rescue | 1000 | 5000 | 100.00% | 51.80% | 159.36 ms | 1001.00 | 116.55 ms | 50.00 | 100.00% |
| 10,000 | ooc_semantic_field_rescue | 1000 | 1000 | 100.00% | 73.70% | 46.12 ms | 1000.00 | 0.00 ms | 50.00 | 100.00% |
| 10,000 | ooc_semantic_field_rescue | 1000 | 5000 | 100.00% | 73.70% | 47.90 ms | 1017.25 | 0.00 ms | 50.00 | 100.00% |
| 100,000 | ooc_full_scan | 200 | 0 | 100.00% | 100.00% | 452.92 ms | 100000.00 | 0.00 ms | 50.00 | 100.00% |
| 100,000 | ooc_ann_hnsw | 200 | 0 | 100.00% | 12.70% | 12.52 ms | 200.00 | 0.00 ms | 50.00 | 100.00% |
| 100,000 | ooc_ann_pruned_hybrid | 200 | 0 | 100.00% | 14.00% | 13.72 ms | 200.00 | 0.00 ms | 50.00 | 100.00% |
| 100,000 | ooc_semantic_rescue_hybrid | 200 | 0 | 100.00% | 65.20% | 664.84 ms | 200.00 | 0.00 ms | 50.00 | 100.00% |
| 100,000 | ooc_semantic_lexical_rescue | 200 | 1000 | 100.00% | 75.80% | 322.10 ms | 201.05 | 0.00 ms | 50.00 | 100.00% |
| 100,000 | ooc_semantic_lexical_rescue | 200 | 5000 | 100.00% | 75.80% | 340.94 ms | 201.05 | 0.00 ms | 50.00 | 100.00% |
| 100,000 | ooc_semantic_rerank_rescue | 200 | 1000 | 100.00% | 65.20% | 361.06 ms | 201.05 | 26.42 ms | 50.00 | 100.00% |
| 100,000 | ooc_semantic_rerank_rescue | 200 | 5000 | 100.00% | 65.20% | 350.82 ms | 201.05 | 25.84 ms | 50.00 | 100.00% |
| 100,000 | ooc_semantic_field_rescue | 200 | 1000 | 100.00% | 75.80% | 336.04 ms | 219.00 | 0.00 ms | 50.00 | 100.00% |
| 100,000 | ooc_semantic_field_rescue | 200 | 5000 | 100.00% | 75.80% | 329.75 ms | 219.00 | 0.00 ms | 50.00 | 100.00% |
| 100,000 | ooc_full_scan | 500 | 0 | 100.00% | 100.00% | 459.39 ms | 100000.00 | 0.00 ms | 50.00 | 100.00% |
| 100,000 | ooc_ann_hnsw | 500 | 0 | 100.00% | 12.50% | 12.20 ms | 500.00 | 0.00 ms | 50.00 | 100.00% |
| 100,000 | ooc_ann_pruned_hybrid | 500 | 0 | 100.00% | 13.90% | 12.96 ms | 500.00 | 0.00 ms | 50.00 | 100.00% |
| 100,000 | ooc_semantic_rescue_hybrid | 500 | 0 | 100.00% | 63.90% | 642.93 ms | 500.00 | 0.00 ms | 50.00 | 100.00% |
| 100,000 | ooc_semantic_lexical_rescue | 500 | 1000 | 100.00% | 74.00% | 342.92 ms | 501.00 | 0.00 ms | 50.00 | 100.00% |
| 100,000 | ooc_semantic_lexical_rescue | 500 | 5000 | 100.00% | 74.00% | 328.16 ms | 501.00 | 0.00 ms | 50.00 | 100.00% |
| 100,000 | ooc_semantic_rerank_rescue | 500 | 1000 | 100.00% | 63.90% | 395.06 ms | 501.00 | 60.87 ms | 50.00 | 100.00% |
| 100,000 | ooc_semantic_rerank_rescue | 500 | 5000 | 100.00% | 63.90% | 393.01 ms | 501.00 | 66.23 ms | 50.00 | 100.00% |
| 100,000 | ooc_semantic_field_rescue | 500 | 1000 | 100.00% | 74.00% | 345.78 ms | 518.90 | 0.00 ms | 50.00 | 100.00% |
| 100,000 | ooc_semantic_field_rescue | 500 | 5000 | 100.00% | 74.00% | 338.09 ms | 518.90 | 0.00 ms | 50.00 | 100.00% |
| 100,000 | ooc_full_scan | 1000 | 0 | 100.00% | 100.00% | 452.78 ms | 100000.00 | 0.00 ms | 50.00 | 100.00% |
| 100,000 | ooc_ann_hnsw | 1000 | 0 | 100.00% | 10.70% | 15.85 ms | 1000.00 | 0.00 ms | 50.00 | 100.00% |
| 100,000 | ooc_ann_pruned_hybrid | 1000 | 0 | 100.00% | 12.10% | 15.01 ms | 1000.00 | 0.00 ms | 50.00 | 100.00% |
| 100,000 | ooc_semantic_rescue_hybrid | 1000 | 0 | 100.00% | 56.30% | 654.60 ms | 1000.00 | 0.00 ms | 50.00 | 100.00% |
| 100,000 | ooc_semantic_lexical_rescue | 1000 | 1000 | 100.00% | 64.00% | 320.97 ms | 1000.00 | 0.00 ms | 50.00 | 100.00% |
| 100,000 | ooc_semantic_lexical_rescue | 1000 | 5000 | 100.00% | 64.00% | 328.72 ms | 1001.00 | 0.00 ms | 50.00 | 100.00% |
| 100,000 | ooc_semantic_rerank_rescue | 1000 | 1000 | 100.00% | 56.60% | 478.33 ms | 1000.00 | 129.01 ms | 50.00 | 100.00% |
| 100,000 | ooc_semantic_rerank_rescue | 1000 | 5000 | 100.00% | 56.60% | 455.88 ms | 1001.00 | 120.43 ms | 50.00 | 100.00% |
| 100,000 | ooc_semantic_field_rescue | 1000 | 1000 | 100.00% | 64.00% | 341.67 ms | 1000.00 | 0.00 ms | 50.00 | 100.00% |
| 100,000 | ooc_semantic_field_rescue | 1000 | 5000 | 100.00% | 64.00% | 337.30 ms | 1018.90 | 0.00 ms | 50.00 | 100.00% |

## Best Compromises

| Size | Best recall | Best latency | Best recall <100ms | Best recall <200ms | Best tradeoff |
|---:|---|---|---|---|---|
| 1,000 | ooc_full_scan cap 200 lex 0 (100.00%) | ooc_ann_pruned_hybrid cap 200 lex 0 (7.06 ms) | ooc_full_scan cap 200 lex 0 (100.00%, 10.02 ms) | ooc_full_scan cap 200 lex 0 (100.00%, 10.02 ms) | ooc_semantic_lexical_rescue cap 200 lex 1000 (99.30%, 9.28 ms) |
| 10,000 | ooc_full_scan cap 200 lex 0 (100.00%) | ooc_ann_hnsw cap 200 lex 0 (8.52 ms) | ooc_full_scan cap 200 lex 0 (100.00%, 51.46 ms) | ooc_full_scan cap 200 lex 0 (100.00%, 51.46 ms) | ooc_semantic_field_rescue cap 200 lex 5000 (95.20%, 39.57 ms) |
| 100,000 | ooc_full_scan cap 200 lex 0 (100.00%) | ooc_ann_hnsw cap 500 lex 0 (12.20 ms) | ooc_ann_pruned_hybrid cap 200 lex 0 (14.00%, 13.72 ms) | ooc_ann_pruned_hybrid cap 200 lex 0 (14.00%, 13.72 ms) | ooc_ann_pruned_hybrid cap 500 lex 0 (13.90%, 12.96 ms) |

Validation requires the best semantic path to reach the recall gate under 200 ms p95 on the executed tiers.

If the verdict is NON_VALIDATING, broad real-LLM semantic demos remain blocked. The next options are a stronger embedder, a cross-encoder/reranker stage, or a more specialized lexical candidate index before mmap rerank.

This benchmark is allowed to be NON_VALIDATING. Its job is to expose semantic ANN quality risk before a real LLM is connected.
Metrics JSON: `artifacts/runs/semantic_ann_quality/metrics.json`
Records JSONL: `artifacts/runs/semantic_ann_quality/records.jsonl`
