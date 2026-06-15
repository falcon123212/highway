# Semantic ANN Quality Benchmark

Verdict: NON_VALIDATING

| Size | Strategy | ANN cap | Lexical cap | Rerank in/out | EM | Recall@k | p95 latency | Rows scanned | Reranker p95 | Reranker avail | Blocks materialized | Metrics complete |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1,000 | ooc_full_scan | 200 | 0 | 0/0 | 100.00% | 100.00% | 10.25 ms | 1000.00 | 0.00 ms | 0.00% | 50.00 | 100.00% |
| 1,000 | ooc_ann_hnsw | 200 | 0 | 0/0 | 100.00% | 22.40% | 7.72 ms | 200.00 | 0.00 ms | 0.00% | 50.00 | 100.00% |
| 1,000 | ooc_semantic_field_rescue | 200 | 5000 | 0/0 | 100.00% | 99.00% | 12.04 ms | 215.65 | 0.00 ms | 0.00% | 50.00 | 100.00% |
| 1,000 | ooc_semantic_cross_encoder_rescue | 200 | 5000 | 200/200 | 100.00% | 42.00% | 1701.15 ms | 200.00 | 1668.61 ms | 100.00% | 50.00 | 100.00% |
| 10,000 | ooc_full_scan | 200 | 0 | 0/0 | 100.00% | 100.00% | 43.68 ms | 10000.00 | 0.00 ms | 0.00% | 50.00 | 100.00% |
| 10,000 | ooc_ann_hnsw | 200 | 0 | 0/0 | 100.00% | 17.70% | 8.39 ms | 200.00 | 0.00 ms | 0.00% | 50.00 | 100.00% |
| 10,000 | ooc_semantic_field_rescue | 200 | 5000 | 0/0 | 100.00% | 95.20% | 32.65 ms | 218.50 | 0.00 ms | 0.00% | 50.00 | 100.00% |
| 10,000 | ooc_semantic_cross_encoder_rescue | 200 | 5000 | 200/200 | 100.00% | 78.60% | 1628.74 ms | 200.00 | 1568.60 ms | 100.00% | 50.00 | 100.00% |
| 100,000 | ooc_full_scan | 200 | 0 | 0/0 | 100.00% | 100.00% | 391.75 ms | 100000.00 | 0.00 ms | 0.00% | 50.00 | 100.00% |
| 100,000 | ooc_ann_hnsw | 200 | 0 | 0/0 | 100.00% | 11.60% | 10.91 ms | 200.00 | 0.00 ms | 0.00% | 50.00 | 100.00% |
| 100,000 | ooc_semantic_field_rescue | 200 | 5000 | 0/0 | 100.00% | 74.70% | 287.27 ms | 219.00 | 0.00 ms | 0.00% | 50.00 | 100.00% |
| 100,000 | ooc_semantic_cross_encoder_rescue | 200 | 5000 | 200/200 | 100.00% | 65.00% | 2121.26 ms | 200.00 | 1688.71 ms | 100.00% | 50.00 | 100.00% |

## Best Compromises

| Size | Best recall | Best latency | Best recall <100ms | Best recall <200ms | Best tradeoff |
|---:|---|---|---|---|---|
| 1,000 | ooc_full_scan cap 200 lex 0 (100.00%) | ooc_ann_hnsw cap 200 lex 0 (7.72 ms) | ooc_full_scan cap 200 lex 0 (100.00%, 10.25 ms) | ooc_full_scan cap 200 lex 0 (100.00%, 10.25 ms) | ooc_full_scan cap 200 lex 0 (100.00%, 10.25 ms) |
| 10,000 | ooc_full_scan cap 200 lex 0 (100.00%) | ooc_ann_hnsw cap 200 lex 0 (8.39 ms) | ooc_full_scan cap 200 lex 0 (100.00%, 43.68 ms) | ooc_full_scan cap 200 lex 0 (100.00%, 43.68 ms) | ooc_semantic_field_rescue cap 200 lex 5000 (95.20%, 32.65 ms) |
| 100,000 | ooc_full_scan cap 200 lex 0 (100.00%) | ooc_ann_hnsw cap 200 lex 0 (10.91 ms) | ooc_ann_hnsw cap 200 lex 0 (11.60%, 10.91 ms) | ooc_ann_hnsw cap 200 lex 0 (11.60%, 10.91 ms) | ooc_ann_hnsw cap 200 lex 0 (11.60%, 10.91 ms) |

Validation requires the best semantic path to reach the recall gate under 200 ms p95 on the executed tiers.

If the verdict is NON_VALIDATING, broad real-LLM semantic demos remain blocked. The next options are a stronger embedder, a cross-encoder/reranker stage, or a more specialized lexical candidate index before mmap rerank.

This benchmark is allowed to be NON_VALIDATING. Its job is to expose semantic ANN quality risk before a real LLM is connected.
Metrics JSON: `artifacts/runs/semantic_cross_encoder_full/metrics.json`
Records JSONL: `artifacts/runs/semantic_cross_encoder_full/records.jsonl`
