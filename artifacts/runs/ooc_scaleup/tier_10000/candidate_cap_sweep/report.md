# OOC Scale-Up Benchmark - Candidate Cap Sweep

| Size | Strategy | Cap | Count | EM | No-leak | Mean ms | p95 ms | Rows scanned | Blocks mat. | Bytes read | Index MB |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
|  | ooc_full_scan | 20 | 20 | 100.00% | 100.00% | 32.98 | 36.72 | 10000.0 | 50.0 | 14777.0 | 28.716 |
|  | ooc_full_scan | 50 | 20 | 100.00% | 100.00% | 35.14 | 40.81 | 10000.0 | 50.0 | 14775.6 | 28.716 |
|  | ooc_full_scan | 200 | 20 | 100.00% | 100.00% | 33.17 | 38.08 | 10000.0 | 50.0 | 14775.9 | 28.716 |

## Verdict

Full mmap scan stays below the 500 ms p95 threshold through the largest measured tier blocks on this run (38.08 ms p95).
