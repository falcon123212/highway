# OOC Scale-Up Benchmark - Candidate Cap Sweep

| Size | Strategy | Cap | Count | EM | No-leak | ANN used | Recall | Mean ms | p95 ms | Rows scanned | Blocks mat. | Bytes read | Index MB |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
|  | ooc_full_scan | 20 | 20 | 100.00% | 100.00% |  |  | 9.94 | 10.91 | 1000.0 | 50.0 | 14754.2 | 2.841 |
|  | ooc_full_scan | 50 | 20 | 100.00% | 100.00% |  |  | 10.07 | 11.01 | 1000.0 | 50.0 | 14753.6 | 2.841 |
|  | ooc_full_scan | 200 | 20 | 100.00% | 100.00% |  |  | 10.24 | 11.14 | 1000.0 | 50.0 | 14633.6 | 2.841 |

## Verdict

Full mmap scan stays below the 500 ms p95 threshold through the largest measured tier blocks on this run (11.14 ms p95).
