# OOC Scale-Up Benchmark

| Size | Strategy | Cap | Count | EM | No-leak | Mean ms | p95 ms | Rows scanned | Blocks mat. | Bytes read | Index MB |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
|  | ooc_full_scan | 50 | 20 | 100.00% | 100.00% | 35.14 | 40.81 | 10000.0 | 50.0 | 14775.6 | 28.716 |

## Verdict

Full mmap scan stays below the 500 ms p95 threshold through the largest measured tier blocks on this run (40.81 ms p95).
