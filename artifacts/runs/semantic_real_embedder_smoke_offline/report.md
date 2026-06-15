# Semantic ANN Quality Benchmark

Verdict: NON_VALIDATING

| Size | Strategy | ANN cap | Lexical cap | Rerank in/out | EM | Recall@k | p95 latency | Rows scanned | Reranker p95 | Reranker avail | Blocks materialized | Metrics complete |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1,000 | ooc_full_scan | 200 | 0 | 0/0 | 100.00% | 100.00% | 26.90 ms | 1000.00 | 0.00 ms | 0.00% | 50.00 | 100.00% |
| 1,000 | ooc_ann_hnsw | 200 | 0 | 0/0 | 50.00% | 38.50% | 51.84 ms | 200.00 | 0.00 ms | 0.00% | 50.00 | 100.00% |
| 1,000 | ooc_semantic_field_rescue | 200 | 5000 | 0/0 | 100.00% | 78.50% | 55.18 ms | 204.00 | 0.00 ms | 0.00% | 50.00 | 100.00% |

## Best Compromises

| Size | Best recall | Best latency | Best recall <100ms | Best recall <200ms | Best tradeoff |
|---:|---|---|---|---|---|
| 1,000 | ooc_full_scan cap 200 lex 0 (100.00%) | ooc_full_scan cap 200 lex 0 (26.90 ms) | ooc_full_scan cap 200 lex 0 (100.00%, 26.90 ms) | ooc_full_scan cap 200 lex 0 (100.00%, 26.90 ms) | ooc_full_scan cap 200 lex 0 (100.00%, 26.90 ms) |

## Embedding Backend

| Size | Backend | Model | Dim | Local only | Batch | Encode latency | Fallback |
|---:|---|---|---:|---:|---:|---:|---|
| 1,000 | sentence_transformer | BAAI/bge-small-en-v1.5 | 384 | True | 64 | 5322.95 ms |  |

Validation requires the best semantic path to reach the recall gate under 200 ms p95 on the executed tiers.

If the verdict is NON_VALIDATING, broad real-LLM semantic demos remain blocked. The next options are a stronger embedder, a cross-encoder/reranker stage, or a more specialized lexical candidate index before mmap rerank.

This benchmark is allowed to be NON_VALIDATING. Its job is to expose semantic ANN quality risk before a real LLM is connected.
Metrics JSON: `artifacts/runs/semantic_real_embedder_smoke_offline/metrics.json`
Records JSONL: `artifacts/runs/semantic_real_embedder_smoke_offline/records.jsonl`
