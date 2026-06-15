# Semantic ANN Quality Benchmark

Verdict: NON_VALIDATING

| Size | Strategy | ANN cap | Lexical cap | EM | Recall@k | p95 latency | Rows scanned | Reranker p95 | Blocks materialized | Metrics complete |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 100,000 | ooc_full_scan | 200 | 0 | 100.00% | 100.00% | 435.38 ms | 100000.00 | 0.00 ms | 50.00 | 100.00% |
| 100,000 | ooc_ann_flat | 200 | 0 | 100.00% | 12.80% | 12.80 ms | 200.00 | 0.00 ms | 50.00 | 100.00% |
| 100,000 | ooc_ann_hnsw | 200 | 0 | 100.00% | 12.80% | 13.13 ms | 200.00 | 0.00 ms | 50.00 | 100.00% |

## Best Compromises

| Size | Best recall | Best latency | Best recall <100ms | Best recall <200ms | Best tradeoff |
|---:|---|---|---|---|---|
| 100,000 | ooc_full_scan cap 200 lex 0 (100.00%) | ooc_ann_flat cap 200 lex 0 (12.80 ms) | ooc_ann_flat cap 200 lex 0 (12.80%, 12.80 ms) | ooc_ann_flat cap 200 lex 0 (12.80%, 12.80 ms) | ooc_ann_flat cap 200 lex 0 (12.80%, 12.80 ms) |

Validation requires the best semantic path to reach the recall gate under 200 ms p95 on the executed tiers.

If the verdict is NON_VALIDATING, broad real-LLM semantic demos remain blocked. The next options are a stronger embedder, a cross-encoder/reranker stage, or a more specialized lexical candidate index before mmap rerank.

This benchmark is allowed to be NON_VALIDATING. Its job is to expose semantic ANN quality risk before a real LLM is connected.
Metrics JSON: `artifacts/runs/semantic_ann_quality_flat_probe/metrics.json`
Records JSONL: `artifacts/runs/semantic_ann_quality_flat_probe/records.jsonl`
