# OOC Scale-Up Benchmark - Candidate Cap Sweep

| Size | Strategy | Cap | Count | EM | No-leak | ANN used | Recall | Mean ms | p95 ms | Rows scanned | Blocks mat. | Bytes read | Index MB |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
|  | ooc_full_scan | 20 | 20 | 100.00% | 100.00% |  |  | 275.27 | 392.00 | 100000.0 | 50.0 | 14814.1 | 290.543 |
|  | ooc_full_scan | 50 | 20 | 100.00% | 100.00% |  |  | 281.61 | 400.08 | 100000.0 | 50.0 | 14541.5 | 290.543 |
|  | ooc_full_scan | 200 | 20 | 100.00% | 100.00% |  |  | 279.72 | 415.30 | 100000.0 | 50.0 | 14541.5 | 290.543 |

## Verdict

Full mmap scan stays below the 500 ms p95 threshold through the largest measured tier blocks on this run (415.30 ms p95).
