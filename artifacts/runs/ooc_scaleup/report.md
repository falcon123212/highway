# OOC Scale-Up Benchmark

| Size | Strategy | Cap | Count | EM | No-leak | Mean ms | p95 ms | Rows scanned | Blocks mat. | Bytes read | Index MB |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1000 | legacy_memory_scan |  | 20 | 100.00% | 100.00% | 2.31 | 3.09 | 1000.0 | 1000.0 | 0.0 | 2.840 |
| 1000 | ooc_full_scan | 200 | 20 | 100.00% | 100.00% | 9.97 | 11.56 | 1000.0 | 50.0 | 14748.1 | 2.840 |
| 1000 | ooc_marker_entity_pruned | 200 | 20 | 100.00% | 100.00% | 2.28 | 2.84 | 1.0 | 1.0 | 338.4 | 2.840 |
| 1000 | ooc_candidate_cap_sweep | 20 | 20 | 100.00% | 100.00% | 9.61 | 10.54 | 1000.0 | 50.0 | 14747.0 | 2.840 |
| 1000 | ooc_candidate_cap_sweep | 50 | 20 | 100.00% | 100.00% | 10.88 | 13.15 | 1000.0 | 50.0 | 14746.0 | 2.840 |
| 1000 | ooc_candidate_cap_sweep | 200 | 20 | 100.00% | 100.00% | 9.38 | 10.18 | 1000.0 | 50.0 | 14748.1 | 2.840 |
| 10000 | legacy_memory_scan |  | 20 | 100.00% | 100.00% | 7.65 | 9.01 | 10000.0 | 10000.0 | 0.0 | 28.716 |
| 10000 | ooc_full_scan | 200 | 20 | 100.00% | 100.00% | 33.66 | 37.62 | 10000.0 | 50.0 | 14775.9 | 28.716 |
| 10000 | ooc_marker_entity_pruned | 200 | 20 | 100.00% | 100.00% | 2.40 | 2.76 | 1.0 | 1.0 | 338.4 | 28.716 |
| 10000 | ooc_candidate_cap_sweep | 20 | 20 | 100.00% | 100.00% | 32.98 | 36.72 | 10000.0 | 50.0 | 14777.0 | 28.716 |
| 10000 | ooc_candidate_cap_sweep | 50 | 20 | 100.00% | 100.00% | 35.14 | 40.81 | 10000.0 | 50.0 | 14775.6 | 28.716 |
| 10000 | ooc_candidate_cap_sweep | 200 | 20 | 100.00% | 100.00% | 33.17 | 38.08 | 10000.0 | 50.0 | 14775.9 | 28.716 |
| 50000 | ooc_full_scan | 200 | 20 | 100.00% | 100.00% | 126.12 | 149.01 | 50000.0 | 50.0 | 14604.7 | 144.585 |
| 50000 | ooc_marker_entity_pruned | 200 | 20 | 100.00% | 100.00% | 2.38 | 2.92 | 1.0 | 1.0 | 338.4 | 144.585 |
| 50000 | ooc_candidate_cap_sweep | 20 | 20 | 100.00% | 100.00% | 122.76 | 146.43 | 50000.0 | 50.0 | 14804.5 | 144.585 |
| 50000 | ooc_candidate_cap_sweep | 50 | 20 | 100.00% | 100.00% | 122.90 | 152.32 | 50000.0 | 50.0 | 14604.6 | 144.585 |
| 50000 | ooc_candidate_cap_sweep | 200 | 20 | 100.00% | 100.00% | 123.04 | 146.14 | 50000.0 | 50.0 | 14604.7 | 144.585 |
| 100000 | ooc_full_scan | 200 | 20 | 100.00% | 100.00% | 241.37 | 291.99 | 100000.0 | 50.0 | 14608.4 | 290.542 |
| 100000 | ooc_marker_entity_pruned | 200 | 20 | 100.00% | 100.00% | 2.30 | 2.61 | 1.0 | 1.0 | 338.4 | 290.542 |
| 100000 | ooc_candidate_cap_sweep | 20 | 20 | 100.00% | 100.00% | 234.88 | 278.60 | 100000.0 | 50.0 | 14806.5 | 290.542 |
| 100000 | ooc_candidate_cap_sweep | 50 | 20 | 100.00% | 100.00% | 234.87 | 280.65 | 100000.0 | 50.0 | 14608.2 | 290.542 |
| 100000 | ooc_candidate_cap_sweep | 200 | 20 | 100.00% | 100.00% | 243.28 | 283.10 | 100000.0 | 50.0 | 14608.4 | 290.542 |

## Verdict

Full mmap scan stays below the 500 ms p95 threshold through 100000 blocks on this run (291.99 ms p95).

1000 blocks: pruning scanned 99.90% fewer embedding rows and materialized 98.00% fewer blocks.
10000 blocks: pruning scanned 99.99% fewer embedding rows and materialized 98.00% fewer blocks.
50000 blocks: pruning scanned 100.00% fewer embedding rows and materialized 98.00% fewer blocks.
100000 blocks: pruning scanned 100.00% fewer embedding rows and materialized 98.00% fewer blocks.
