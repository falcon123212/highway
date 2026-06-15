# OOC Scale-Up Benchmark

| Size | Strategy | Cap | Count | EM | No-leak | ANN used | Recall | Mean ms | p95 ms | Rows scanned | Blocks mat. | Bytes read | Index MB |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
|  | ooc_full_scan | 200 | 20 | 100.00% | 100.00% |  |  | 273.93 | 395.17 | 100000.0 | 50.0 | 14541.5 | 290.543 |

## Verdict

Full mmap scan stays below the 500 ms p95 threshold through the largest measured tier blocks on this run (395.17 ms p95).
