# Semantic ANN Quality Benchmark

Verdict: NON_VALIDATING

| Size | Strategy | ANN cap | Lexical cap | EM | Recall@k | p95 latency | Rows scanned | Reranker p95 | Blocks materialized | Metrics complete |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 100,000 | ooc_full_scan | 200 | 0 | 100.00% | 100.00% | 431.72 ms | 100000.00 | 0.00 ms | 50.00 | 100.00% |
| 100,000 | ooc_ann_hnsw | 200 | 0 | 100.00% | 12.60% | 13.45 ms | 200.00 | 0.00 ms | 50.00 | 100.00% |
| 100,000 | ooc_semantic_lexical_rescue | 200 | 5000 | 100.00% | 75.70% | 290.27 ms | 201.05 | 0.00 ms | 50.00 | 100.00% |
| 100,000 | ooc_semantic_field_rescue | 200 | 5000 | 100.00% | 75.70% | 299.08 ms | 219.00 | 0.00 ms | 50.00 | 100.00% |

## Best Compromises

| Size | Best recall | Best latency | Best recall <100ms | Best recall <200ms | Best tradeoff |
|---:|---|---|---|---|---|
| 100,000 | ooc_full_scan cap 200 lex 0 (100.00%) | ooc_ann_hnsw cap 200 lex 0 (13.45 ms) | ooc_ann_hnsw cap 200 lex 0 (12.60%, 13.45 ms) | ooc_ann_hnsw cap 200 lex 0 (12.60%, 13.45 ms) | ooc_ann_hnsw cap 200 lex 0 (12.60%, 13.45 ms) |

Validation requires the best semantic path to reach the recall gate under 200 ms p95 on the executed tiers.

If the verdict is NON_VALIDATING, broad real-LLM semantic demos remain blocked. The next options are a stronger embedder, a cross-encoder/reranker stage, or a more specialized lexical candidate index before mmap rerank.

This benchmark is allowed to be NON_VALIDATING. Its job is to expose semantic ANN quality risk before a real LLM is connected.
Metrics JSON: `artifacts/runs/semantic_ann_quality_field_probe/metrics.json`
Records JSONL: `artifacts/runs/semantic_ann_quality_field_probe/records.jsonl`
