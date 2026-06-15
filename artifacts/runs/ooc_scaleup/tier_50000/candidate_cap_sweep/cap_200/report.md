# OOC Scale-Up Benchmark

| Size | Strategy | Cap | Count | EM | No-leak | Mean ms | p95 ms | Rows scanned | Blocks mat. | Bytes read | Index MB |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
|  | ooc_full_scan | 200 | 20 | 100.00% | 100.00% | 123.04 | 146.14 | 50000.0 | 50.0 | 14604.7 | 144.585 |

## Verdict

Full mmap scan stays below the 500 ms p95 threshold through the largest measured tier blocks on this run (146.14 ms p95).
