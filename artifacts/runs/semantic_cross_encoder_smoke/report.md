# Semantic ANN Quality Benchmark

Verdict: QUALITY_ONLY

| Size | Strategy | ANN cap | Lexical cap | Rerank in/out | EM | Recall@k | p95 latency | Rows scanned | Reranker p95 | Reranker avail | Blocks materialized | Metrics complete |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 80 | ooc_full_scan | 20 | 0 | 0/0 | 100.00% | 100.00% | 5.59 ms | 80.00 | 0.00 ms | 0.00% | 50.00 | 100.00% |
| 80 | ooc_semantic_cross_encoder_rescue | 20 | 40 | 40/20 | 100.00% | 100.00% | 3477.91 ms | 50.00 | 3457.72 ms | 100.00% | 50.00 | 100.00% |

## Best Compromises

| Size | Best recall | Best latency | Best recall <100ms | Best recall <200ms | Best tradeoff |
|---:|---|---|---|---|---|
| 80 | ooc_full_scan cap 20 lex 0 (100.00%) | ooc_full_scan cap 20 lex 0 (5.59 ms) | ooc_full_scan cap 20 lex 0 (100.00%, 5.59 ms) | ooc_full_scan cap 20 lex 0 (100.00%, 5.59 ms) | ooc_full_scan cap 20 lex 0 (100.00%, 5.59 ms) |

Validation requires the best semantic path to reach the recall gate under 200 ms p95 on the executed tiers.

If the verdict is NON_VALIDATING, broad real-LLM semantic demos remain blocked. The next options are a stronger embedder, a cross-encoder/reranker stage, or a more specialized lexical candidate index before mmap rerank.

This benchmark is allowed to be NON_VALIDATING. Its job is to expose semantic ANN quality risk before a real LLM is connected.
Metrics JSON: `artifacts/runs/semantic_cross_encoder_smoke/metrics.json`
Records JSONL: `artifacts/runs/semantic_cross_encoder_smoke/records.jsonl`
