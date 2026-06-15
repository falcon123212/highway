# OOC Scale-Up Benchmark

| Size | Strategy | Cap | Count | EM | No-leak | ANN used | Recall | Mean ms | p95 ms | Rows scanned | Blocks mat. | Bytes read | Index MB |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1000 | legacy_memory_scan |  | 10 | 100.00% | 100.00% |  |  | 3.74 | 6.15 | 1000.0 | 1000.0 | 0.0 | 2.840 |
| 1000 | ooc_full_scan | 1000 | 10 | 100.00% | 100.00% |  |  | 105.37 | 109.69 | 1000.0 | 1000.0 | 266609.0 | 2.840 |
| 1000 | ooc_marker_entity_pruned | 1000 | 10 | 100.00% | 100.00% |  |  | 2.85 | 3.57 | 1.0 | 1.0 | 337.9 | 2.840 |
| 1000 | ooc_candidate_cap_sweep | 20 | 10 | 100.00% | 100.00% |  |  | 100.53 | 102.33 | 1000.0 | 1000.0 | 266609.0 | 2.840 |
| 1000 | ooc_candidate_cap_sweep | 50 | 10 | 100.00% | 100.00% |  |  | 99.40 | 103.19 | 1000.0 | 1000.0 | 266609.0 | 2.840 |
| 1000 | ooc_candidate_cap_sweep | 1000 | 10 | 100.00% | 100.00% |  |  | 103.38 | 106.47 | 1000.0 | 1000.0 | 266609.0 | 2.840 |

## Verdict

Full mmap scan stays below the 500 ms p95 threshold through 1000 blocks on this run (109.69 ms p95).

1000 blocks: pruning scanned 99.90% fewer embedding rows and materialized 99.90% fewer blocks.
