# Semantic ANN Quality Benchmark

Verdict: VALIDATING

| Size | Strategy | ANN cap | Lexical cap | Rerank in/out | EM | Recall@k | p95 latency | Rows scanned | Reranker p95 | Reranker avail | Blocks materialized | Metrics complete |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1,000 | ooc_full_scan | 200 | 0 | 0/0 | 100.00% | 100.00% | 30.14 ms | 1000.00 | 0.00 ms | 0.00% | 50.00 | 100.00% |
| 1,000 | ooc_ann_hnsw | 200 | 0 | 0/0 | 65.00% | 42.30% | 47.86 ms | 200.00 | 0.00 ms | 0.00% | 50.00 | 100.00% |
| 1,000 | ooc_semantic_field_rescue | 200 | 1000 | 0/0 | 100.00% | 47.40% | 48.42 ms | 208.15 | 0.00 ms | 0.00% | 50.00 | 100.00% |
| 1,000 | ooc_semantic_field_rescue | 200 | 5000 | 0/0 | 100.00% | 47.40% | 46.20 ms | 208.15 | 0.00 ms | 0.00% | 50.00 | 100.00% |
| 1,000 | ooc_full_scan | 500 | 0 | 0/0 | 100.00% | 100.00% | 31.73 ms | 1000.00 | 0.00 ms | 0.00% | 50.00 | 100.00% |
| 1,000 | ooc_ann_hnsw | 500 | 0 | 0/0 | 65.00% | 57.50% | 46.78 ms | 360.35 | 0.00 ms | 0.00% | 50.00 | 100.00% |
| 1,000 | ooc_semantic_field_rescue | 500 | 1000 | 0/0 | 100.00% | 57.10% | 47.42 ms | 367.25 | 0.00 ms | 0.00% | 50.00 | 100.00% |
| 1,000 | ooc_semantic_field_rescue | 500 | 5000 | 0/0 | 100.00% | 57.10% | 46.50 ms | 367.25 | 0.00 ms | 0.00% | 50.00 | 100.00% |
| 1,000 | ooc_full_scan | 1000 | 0 | 0/0 | 100.00% | 100.00% | 32.58 ms | 1000.00 | 0.00 ms | 0.00% | 50.00 | 100.00% |
| 1,000 | ooc_ann_hnsw | 1000 | 0 | 0/0 | 65.00% | 76.30% | 45.17 ms | 360.35 | 0.00 ms | 0.00% | 50.00 | 100.00% |
| 1,000 | ooc_semantic_field_rescue | 1000 | 1000 | 0/0 | 100.00% | 92.70% | 46.02 ms | 367.25 | 0.00 ms | 0.00% | 50.00 | 100.00% |
| 1,000 | ooc_semantic_field_rescue | 1000 | 5000 | 0/0 | 100.00% | 92.70% | 46.47 ms | 367.25 | 0.00 ms | 0.00% | 50.00 | 100.00% |
| 10,000 | ooc_full_scan | 200 | 0 | 0/0 | 100.00% | 100.00% | 78.65 ms | 10000.00 | 0.00 ms | 0.00% | 50.00 | 100.00% |
| 10,000 | ooc_ann_hnsw | 200 | 0 | 0/0 | 35.00% | 59.30% | 51.72 ms | 200.00 | 0.00 ms | 0.00% | 50.00 | 100.00% |
| 10,000 | ooc_semantic_field_rescue | 200 | 1000 | 0/0 | 100.00% | 80.40% | 53.11 ms | 215.40 | 0.00 ms | 0.00% | 50.00 | 100.00% |
| 10,000 | ooc_semantic_field_rescue | 200 | 5000 | 0/0 | 100.00% | 80.40% | 50.82 ms | 215.40 | 0.00 ms | 0.00% | 50.00 | 100.00% |
| 10,000 | ooc_full_scan | 500 | 0 | 0/0 | 100.00% | 100.00% | 81.66 ms | 10000.00 | 0.00 ms | 0.00% | 50.00 | 100.00% |
| 10,000 | ooc_ann_hnsw | 500 | 0 | 0/0 | 35.00% | 33.40% | 56.17 ms | 500.00 | 0.00 ms | 0.00% | 50.00 | 100.00% |
| 10,000 | ooc_semantic_field_rescue | 500 | 1000 | 0/0 | 100.00% | 53.60% | 60.31 ms | 515.40 | 0.00 ms | 0.00% | 50.00 | 100.00% |
| 10,000 | ooc_semantic_field_rescue | 500 | 5000 | 0/0 | 100.00% | 53.60% | 51.04 ms | 515.40 | 0.00 ms | 0.00% | 50.00 | 100.00% |
| 10,000 | ooc_full_scan | 1000 | 0 | 0/0 | 100.00% | 100.00% | 79.02 ms | 10000.00 | 0.00 ms | 0.00% | 50.00 | 100.00% |
| 10,000 | ooc_ann_hnsw | 1000 | 0 | 0/0 | 35.00% | 15.90% | 48.33 ms | 685.90 | 0.00 ms | 0.00% | 50.00 | 100.00% |
| 10,000 | ooc_semantic_field_rescue | 1000 | 1000 | 0/0 | 100.00% | 34.80% | 49.34 ms | 700.80 | 0.00 ms | 0.00% | 50.00 | 100.00% |
| 10,000 | ooc_semantic_field_rescue | 1000 | 5000 | 0/0 | 100.00% | 34.80% | 49.79 ms | 700.80 | 0.00 ms | 0.00% | 50.00 | 100.00% |
| 100,000 | ooc_full_scan | 200 | 0 | 0/0 | 100.00% | 100.00% | 475.65 ms | 100000.00 | 0.00 ms | 0.00% | 50.00 | 100.00% |
| 100,000 | ooc_ann_hnsw | 200 | 0 | 0/0 | 30.00% | 62.20% | 55.40 ms | 200.00 | 0.00 ms | 0.00% | 50.00 | 100.00% |
| 100,000 | ooc_semantic_field_rescue | 200 | 1000 | 0/0 | 100.00% | 82.30% | 51.14 ms | 214.85 | 0.00 ms | 0.00% | 50.00 | 100.00% |
| 100,000 | ooc_semantic_field_rescue | 200 | 5000 | 0/0 | 100.00% | 82.30% | 52.28 ms | 214.85 | 0.00 ms | 0.00% | 50.00 | 100.00% |
| 100,000 | ooc_full_scan | 500 | 0 | 0/0 | 100.00% | 100.00% | 494.18 ms | 100000.00 | 0.00 ms | 0.00% | 50.00 | 100.00% |
| 100,000 | ooc_ann_hnsw | 500 | 0 | 0/0 | 30.00% | 61.10% | 77.20 ms | 500.00 | 0.00 ms | 0.00% | 50.00 | 100.00% |
| 100,000 | ooc_semantic_field_rescue | 500 | 1000 | 0/0 | 100.00% | 81.00% | 81.85 ms | 514.85 | 0.00 ms | 0.00% | 50.00 | 100.00% |
| 100,000 | ooc_semantic_field_rescue | 500 | 5000 | 0/0 | 100.00% | 81.00% | 67.88 ms | 514.85 | 0.00 ms | 0.00% | 50.00 | 100.00% |
| 100,000 | ooc_full_scan | 1000 | 0 | 0/0 | 100.00% | 100.00% | 540.43 ms | 100000.00 | 0.00 ms | 0.00% | 50.00 | 100.00% |
| 100,000 | ooc_ann_hnsw | 1000 | 0 | 0/0 | 30.00% | 58.10% | 55.28 ms | 902.10 | 0.00 ms | 0.00% | 50.00 | 100.00% |
| 100,000 | ooc_semantic_field_rescue | 1000 | 1000 | 0/0 | 90.00% | 69.20% | 59.70 ms | 911.20 | 0.00 ms | 0.00% | 50.00 | 100.00% |
| 100,000 | ooc_semantic_field_rescue | 1000 | 5000 | 0/0 | 100.00% | 76.90% | 64.91 ms | 916.70 | 0.00 ms | 0.00% | 50.00 | 100.00% |

## Best Compromises

| Size | Best recall | Best latency | Best recall <100ms | Best recall <200ms | Best tradeoff |
|---:|---|---|---|---|---|
| 1,000 | ooc_full_scan cap 200 lex 0 (100.00%) | ooc_full_scan cap 200 lex 0 (30.14 ms) | ooc_full_scan cap 200 lex 0 (100.00%, 30.14 ms) | ooc_full_scan cap 200 lex 0 (100.00%, 30.14 ms) | ooc_full_scan cap 200 lex 0 (100.00%, 30.14 ms) |
| 10,000 | ooc_full_scan cap 200 lex 0 (100.00%) | ooc_ann_hnsw cap 1000 lex 0 (48.33 ms) | ooc_full_scan cap 200 lex 0 (100.00%, 78.65 ms) | ooc_full_scan cap 200 lex 0 (100.00%, 78.65 ms) | ooc_semantic_field_rescue cap 200 lex 5000 (80.40%, 50.82 ms) |
| 100,000 | ooc_full_scan cap 200 lex 0 (100.00%) | ooc_semantic_field_rescue cap 200 lex 1000 (51.14 ms) | ooc_semantic_field_rescue cap 200 lex 1000 (82.30%, 51.14 ms) | ooc_semantic_field_rescue cap 200 lex 1000 (82.30%, 51.14 ms) | ooc_semantic_field_rescue cap 200 lex 1000 (82.30%, 51.14 ms) |

## Embedding Backend

| Size | Backend | Model | Dim | Local only | Batch | Encode latency | Fallback |
|---:|---|---|---:|---:|---:|---:|---|
| 1,000 | sentence_transformer | BAAI/bge-small-en-v1.5 | 384 | True | 64 | 5583.58 ms |  |
| 10,000 | sentence_transformer | BAAI/bge-small-en-v1.5 | 384 | True | 64 | 55833.61 ms |  |
| 100,000 | sentence_transformer | BAAI/bge-small-en-v1.5 | 384 | True | 64 | 606847.35 ms |  |

Validation requires the best semantic path to reach the recall gate under 200 ms p95 on the executed tiers.

If the verdict is NON_VALIDATING, broad real-LLM semantic demos remain blocked. The next options are a stronger embedder, a cross-encoder/reranker stage, or a more specialized lexical candidate index before mmap rerank.

This benchmark is allowed to be NON_VALIDATING. Its job is to expose semantic ANN quality risk before a real LLM is connected.
Metrics JSON: `artifacts/runs/semantic_real_embedder_full/metrics.json`
Records JSONL: `artifacts/runs/semantic_real_embedder_full/records.jsonl`
