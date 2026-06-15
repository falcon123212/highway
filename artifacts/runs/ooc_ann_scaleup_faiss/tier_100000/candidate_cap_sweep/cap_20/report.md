# OOC Scale-Up Benchmark

| Size | Strategy | Cap | Count | EM | No-leak | ANN used | Recall | Mean ms | p95 ms | Rows scanned | Blocks mat. | Bytes read | Index MB |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
|  | ooc_full_scan | 20 | 20 | 100.00% | 100.00% |  |  | 279.38 | 401.38 | 100000.0 | 50.0 | 14814.1 | 609.471 |

## Verdict

Full mmap scan stays below the 500 ms p95 threshold through the largest measured tier blocks on this run (401.38 ms p95).
