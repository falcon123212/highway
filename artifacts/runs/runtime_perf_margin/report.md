# Runtime Perf Margin Benchmark

Verdict: VALIDATING

| Size | Workload | Context p95 | Runtime p95 | Rows scanned | Blocks materialized | Hotset hits | Tokens avoided | KV avoided | Metrics complete |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 1,000 | structured_exact | 3.26 ms | 0.09 ms | 1.20 | 1.20 | 0.00 | 15075.50 | 1481981952.00 | 100.00% |
| 10,000 | structured_exact | 3.59 ms | 0.10 ms | 1.20 | 1.20 | 0.00 | 150075.50 | 14753021952.00 | 100.00% |
| 100,000 | structured_exact | 21.06 ms | 0.09 ms | 1.20 | 1.20 | 0.00 | 1500075.50 | 147463421952.00 | 100.00% |

This benchmark measures structured marker/entity retrieval and fake runtime response without double retrieval.
Metrics JSON: `artifacts/runs/runtime_perf_margin/metrics.json`
Records JSONL: `artifacts/runs/runtime_perf_margin/records.jsonl`
