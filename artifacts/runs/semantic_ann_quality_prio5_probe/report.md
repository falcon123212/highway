# Semantic ANN Quality Benchmark

Verdict: NON_VALIDATING

| Size | Strategy | ANN cap | Lexical cap | EM | Recall@k | p95 latency | Rows scanned | Blocks materialized | Metrics complete |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 100,000 | ooc_full_scan | 200 | 0 | 100.00% | 100.00% | 424.28 ms | 100000.00 | 50.00 | 100.00% |
| 100,000 | ooc_ann_hnsw | 200 | 0 | 95.00% | 12.80% | 13.41 ms | 200.00 | 50.00 | 100.00% |
| 100,000 | ooc_semantic_lexical_rescue | 200 | 1000 | 100.00% | 76.00% | 317.59 ms | 201.10 | 50.00 | 100.00% |
| 100,000 | ooc_semantic_rerank_rescue | 200 | 1000 | 100.00% | 65.20% | 330.86 ms | 201.10 | 50.00 | 100.00% |

## Best Compromises

| Size | Best recall | Best latency | Best recall <100ms | Best recall <200ms | Best tradeoff |
|---:|---|---|---|---|---|
| 100,000 | ooc_full_scan cap 200 lex 0 (100.00%) | ooc_ann_hnsw cap 200 lex 0 (13.41 ms) | ooc_ann_hnsw cap 200 lex 0 (12.80%, 13.41 ms) | ooc_ann_hnsw cap 200 lex 0 (12.80%, 13.41 ms) | ooc_ann_hnsw cap 200 lex 0 (12.80%, 13.41 ms) |

Validation requires the best semantic path to reach the recall gate under 200 ms p95 on the executed tiers.

If the verdict is NON_VALIDATING, broad real-LLM semantic demos remain blocked. The next options are a stronger embedder, a cross-encoder/reranker stage, or a more specialized lexical candidate index before mmap rerank.

This benchmark is allowed to be NON_VALIDATING. Its job is to expose semantic ANN quality risk before a real LLM is connected.
Metrics JSON: `artifacts/runs/semantic_ann_quality_prio5_probe/metrics.json`
Records JSONL: `artifacts/runs/semantic_ann_quality_prio5_probe/records.jsonl`
