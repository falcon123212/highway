# OOC Scale-Up Benchmark - Candidate Cap Sweep

| Size | Strategy | Cap | Count | EM | No-leak | ANN used | Recall | Mean ms | p95 ms | Rows scanned | Blocks mat. | Bytes read | Index MB |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
|  | ooc_full_scan | 20 | 20 | 100.00% | 100.00% |  |  | 36.96 | 47.24 | 10000.0 | 50.0 | 14784.0 | 60.606 |
|  | ooc_full_scan | 50 | 20 | 100.00% | 100.00% |  |  | 35.37 | 46.33 | 10000.0 | 50.0 | 14701.6 | 60.606 |
|  | ooc_full_scan | 200 | 20 | 100.00% | 100.00% |  |  | 36.93 | 49.44 | 10000.0 | 50.0 | 14707.8 | 60.606 |

## Verdict

Full mmap scan stays below the 500 ms p95 threshold through the largest measured tier blocks on this run (49.44 ms p95).
