# OOC Scale-Up Benchmark - Candidate Cap Sweep

| Size | Strategy | Cap | Count | EM | No-leak | ANN used | Recall | Mean ms | p95 ms | Rows scanned | Blocks mat. | Bytes read | Index MB |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
|  | ooc_full_scan | 20 | 10 | 100.00% | 100.00% |  |  | 100.53 | 102.33 | 1000.0 | 1000.0 | 266609.0 | 2.840 |
|  | ooc_full_scan | 50 | 10 | 100.00% | 100.00% |  |  | 99.40 | 103.19 | 1000.0 | 1000.0 | 266609.0 | 2.840 |
|  | ooc_full_scan | 1000 | 10 | 100.00% | 100.00% |  |  | 103.38 | 106.47 | 1000.0 | 1000.0 | 266609.0 | 2.840 |

## Verdict

Full mmap scan stays below the 500 ms p95 threshold through the largest measured tier blocks on this run (106.47 ms p95).
