# OOC Scale-Up Benchmark

| Size | Strategy | Cap | Count | EM | No-leak | ANN used | Recall | Mean ms | p95 ms | Rows scanned | Blocks mat. | Bytes read | Index MB |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1000 | legacy_memory_scan |  | 10 | 100.00% | 100.00% |  |  | 2.81 | 5.31 | 1000.0 | 1000.0 | 0.0 | 2.840 |
| 1000 | ooc_full_scan | 200 | 10 | 100.00% | 100.00% |  |  | 10.52 | 11.60 | 1000.0 | 50.0 | 14017.5 | 2.840 |
| 1000 | ooc_marker_entity_pruned | 200 | 10 | 100.00% | 100.00% |  |  | 2.90 | 4.33 | 1.0 | 1.0 | 337.9 | 2.840 |
| 1000 | ooc_candidate_cap_sweep | 20 | 10 | 100.00% | 100.00% |  |  | 10.16 | 11.07 | 1000.0 | 50.0 | 14016.2 | 2.840 |
| 1000 | ooc_candidate_cap_sweep | 50 | 10 | 100.00% | 100.00% |  |  | 10.12 | 11.18 | 1000.0 | 50.0 | 14015.2 | 2.840 |
| 1000 | ooc_candidate_cap_sweep | 200 | 10 | 100.00% | 100.00% |  |  | 10.22 | 11.20 | 1000.0 | 50.0 | 14017.5 | 2.840 |

## Verdict

Full mmap scan stays below the 500 ms p95 threshold through 1000 blocks on this run (11.60 ms p95).

1000 blocks: pruning scanned 99.90% fewer embedding rows and materialized 98.00% fewer blocks.
