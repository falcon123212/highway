# OOC Scale-Up Benchmark - Candidate Cap Sweep

| Size | Strategy | Cap | Count | EM | No-leak | Mean ms | p95 ms | Rows scanned | Blocks mat. | Bytes read | Index MB |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
|  | ooc_full_scan | 20 | 20 | 100.00% | 100.00% | 9.61 | 10.54 | 1000.0 | 50.0 | 14747.0 | 2.840 |
|  | ooc_full_scan | 50 | 20 | 100.00% | 100.00% | 10.88 | 13.15 | 1000.0 | 50.0 | 14746.0 | 2.840 |
|  | ooc_full_scan | 200 | 20 | 100.00% | 100.00% | 9.38 | 10.18 | 1000.0 | 50.0 | 14748.1 | 2.840 |

## Verdict

Full mmap scan stays below the 500 ms p95 threshold through the largest measured tier blocks on this run (10.18 ms p95).
