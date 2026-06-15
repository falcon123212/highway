# OOC Scale-Up Benchmark

| Size | Strategy | Cap | Count | EM | No-leak | ANN used | Recall | Mean ms | p95 ms | Rows scanned | Blocks mat. | Bytes read | Index MB |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1000 | legacy_memory_scan |  | 20 | 100.00% | 100.00% |  |  | 3.86 | 5.48 | 1000.0 | 1000.0 | 0.0 | 6.030 |
| 1000 | ooc_full_scan | 200 | 20 | 100.00% | 100.00% |  |  | 10.95 | 12.41 | 1000.0 | 50.0 | 14633.6 | 6.030 |
| 1000 | ooc_marker_entity_pruned | 200 | 20 | 100.00% | 100.00% |  |  | 7.09 | 11.03 | 130.3 | 32.9 | 9684.6 | 6.030 |
| 1000 | ooc_ann_flat | 200 | 20 | 100.00% | 100.00% | 100.00% | 31.40% | 8.19 | 9.06 | 200.0 | 50.0 | 13435.9 | 6.030 |
| 1000 | ooc_ann_hnsw | 200 | 20 | 100.00% | 100.00% | 100.00% | 31.40% | 8.38 | 9.29 | 200.0 | 50.0 | 13436.0 | 6.030 |
| 1000 | ooc_ann_pruned_hybrid | 200 | 20 | 100.00% | 100.00% | 0.00% |  | 6.90 | 10.14 | 130.3 | 32.9 | 9684.6 | 6.030 |
| 1000 | ooc_candidate_cap_sweep | 20 | 20 | 100.00% | 100.00% |  |  | 10.18 | 11.46 | 1000.0 | 50.0 | 14754.2 | 6.030 |
| 1000 | ooc_candidate_cap_sweep | 50 | 20 | 100.00% | 100.00% |  |  | 10.20 | 11.39 | 1000.0 | 50.0 | 14753.6 | 6.030 |
| 1000 | ooc_candidate_cap_sweep | 200 | 20 | 100.00% | 100.00% |  |  | 10.39 | 11.58 | 1000.0 | 50.0 | 14633.6 | 6.030 |
| 10000 | legacy_memory_scan |  | 20 | 100.00% | 100.00% |  |  | 9.47 | 11.55 | 10000.0 | 10000.0 | 0.0 | 60.606 |
| 10000 | ooc_full_scan | 200 | 20 | 100.00% | 100.00% |  |  | 37.64 | 51.00 | 10000.0 | 50.0 | 14707.8 | 60.606 |
| 10000 | ooc_marker_entity_pruned | 200 | 20 | 100.00% | 100.00% |  |  | 17.25 | 36.25 | 130.3 | 32.9 | 9714.4 | 60.606 |
| 10000 | ooc_ann_flat | 200 | 20 | 100.00% | 100.00% | 100.00% | 49.40% | 9.83 | 10.58 | 200.0 | 50.0 | 13416.2 | 60.606 |
| 10000 | ooc_ann_hnsw | 200 | 20 | 100.00% | 100.00% | 100.00% | 40.80% | 9.42 | 10.51 | 200.0 | 50.0 | 13416.7 | 60.606 |
| 10000 | ooc_ann_pruned_hybrid | 200 | 20 | 100.00% | 100.00% | 0.00% |  | 16.74 | 33.73 | 130.3 | 32.9 | 9714.4 | 60.606 |
| 10000 | ooc_candidate_cap_sweep | 20 | 20 | 100.00% | 100.00% |  |  | 37.19 | 48.51 | 10000.0 | 50.0 | 14784.0 | 60.606 |
| 10000 | ooc_candidate_cap_sweep | 50 | 20 | 100.00% | 100.00% |  |  | 35.95 | 48.70 | 10000.0 | 50.0 | 14701.6 | 60.606 |
| 10000 | ooc_candidate_cap_sweep | 200 | 20 | 100.00% | 100.00% |  |  | 36.39 | 47.22 | 10000.0 | 50.0 | 14707.8 | 60.606 |
| 100000 | ooc_full_scan | 200 | 20 | 100.00% | 100.00% |  |  | 291.48 | 415.88 | 100000.0 | 50.0 | 14541.5 | 609.471 |
| 100000 | ooc_marker_entity_pruned | 200 | 20 | 100.00% | 100.00% |  |  | 114.72 | 281.22 | 130.3 | 32.9 | 9729.1 | 609.471 |
| 100000 | ooc_ann_flat | 200 | 20 | 100.00% | 100.00% | 100.00% | 64.00% | 24.06 | 25.53 | 200.0 | 50.0 | 13466.0 | 609.471 |
| 100000 | ooc_ann_hnsw | 200 | 20 | 100.00% | 100.00% | 100.00% | 19.90% | 12.52 | 13.28 | 200.0 | 50.0 | 13464.6 | 609.471 |
| 100000 | ooc_ann_pruned_hybrid | 200 | 20 | 100.00% | 100.00% | 0.00% |  | 114.39 | 274.78 | 130.3 | 32.9 | 9729.1 | 609.471 |
| 100000 | ooc_candidate_cap_sweep | 20 | 20 | 100.00% | 100.00% |  |  | 279.38 | 401.38 | 100000.0 | 50.0 | 14814.1 | 609.471 |
| 100000 | ooc_candidate_cap_sweep | 50 | 20 | 100.00% | 100.00% |  |  | 271.12 | 398.30 | 100000.0 | 50.0 | 14541.5 | 609.471 |
| 100000 | ooc_candidate_cap_sweep | 200 | 20 | 100.00% | 100.00% |  |  | 275.39 | 385.48 | 100000.0 | 50.0 | 14541.5 | 609.471 |

## Verdict

Full mmap scan stays below the 500 ms p95 threshold through 100000 blocks on this run (415.88 ms p95).

1000 blocks: pruning scanned 86.96% fewer embedding rows and materialized 34.30% fewer blocks.
10000 blocks: pruning scanned 98.70% fewer embedding rows and materialized 34.30% fewer blocks.
100000 blocks: pruning scanned 99.87% fewer embedding rows and materialized 34.30% fewer blocks.

ANN acceleration was active in this run:
1000 blocks: best ANN strategy ooc_ann_flat reached 9.06 ms p95 vs 12.41 ms full-scan p95 (1.4x faster), with 80.00% fewer embedding rows reranked and 31.40% recall@k.
10000 blocks: best ANN strategy ooc_ann_hnsw reached 10.51 ms p95 vs 51.00 ms full-scan p95 (4.9x faster), with 98.00% fewer embedding rows reranked and 40.80% recall@k.
100000 blocks: best ANN strategy ooc_ann_hnsw reached 13.28 ms p95 vs 415.88 ms full-scan p95 (31.3x faster), with 99.80% fewer embedding rows reranked and 19.90% recall@k.
