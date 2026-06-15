# OOC Scale-Up Benchmark

| Size | Strategy | Cap | Count | EM | No-leak | Mean ms | p95 ms | Rows scanned | Blocks mat. | Bytes read | Index MB |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
|  | ooc_full_scan | 50 | 20 | 100.00% | 100.00% | 234.87 | 280.65 | 100000.0 | 50.0 | 14608.2 | 290.542 |

## Verdict

Full mmap scan stays below the 500 ms p95 threshold through the largest measured tier blocks on this run (280.65 ms p95).
