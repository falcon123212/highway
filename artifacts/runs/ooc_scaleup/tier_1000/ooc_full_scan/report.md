# OOC Scale-Up Benchmark

| Size | Strategy | Cap | Count | EM | No-leak | Mean ms | p95 ms | Rows scanned | Blocks mat. | Bytes read | Index MB |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
|  | ooc_full_scan | 200 | 20 | 100.00% | 100.00% | 9.97 | 11.56 | 1000.0 | 50.0 | 14748.1 | 2.840 |

## Verdict

Full mmap scan stays below the 500 ms p95 threshold through the largest measured tier blocks on this run (11.56 ms p95).
