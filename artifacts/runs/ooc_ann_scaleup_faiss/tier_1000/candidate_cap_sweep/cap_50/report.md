# OOC Scale-Up Benchmark

| Size | Strategy | Cap | Count | EM | No-leak | ANN used | Recall | Mean ms | p95 ms | Rows scanned | Blocks mat. | Bytes read | Index MB |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
|  | ooc_full_scan | 50 | 20 | 100.00% | 100.00% |  |  | 10.20 | 11.39 | 1000.0 | 50.0 | 14753.6 | 6.030 |

## Verdict

Full mmap scan stays below the 500 ms p95 threshold through the largest measured tier blocks on this run (11.39 ms p95).
