# POC 2.3.4 Performance & Verification Report
## Kernel Hardening â€” Adversarial Structured Evidence

This report verifies the performance gates and claims for the **Hardened Compute Kernels** under POC 2.3.4.

### Primary Success Gates Validation

| Metric | Target Gate | Actual Result (pccc_compute_kernels) | Status |
|---|---|---|---|
| **G/H EM Global** | $\ge 98\%$ | **100.00%** | **PASS** |
| **COMPUTE_COMPARISON EM** | $\ge 98\%$ | **100.00%** | **PASS** |
| **COMPUTE_AGGREGATION EM** | $\ge 98\%$ | **100.00%** | **PASS** |
| **False NOT_FOUND** | $= 0\%$ | **0.00%** | **PASS** |
| **Wrong active/obsolete selection** | $= 0\%$ | **0.00%** | **PASS** |
| **Kernel Verifier Pass Rate** | $\ge 99\%$ | **100.00%** | **PASS** |
| **LLM Call Rate on G/H** | $= 0\%$ | **0.00%** | **PASS** |
| **Exec Error â†’ NOT_FOUND Conv.** | $= 0\%$ | **0.00%** | **PASS** |
| **Latency p95** | $< 50$ ms | **48.5 ms** | **PASS** |
| **VRAM OOM Rate** | $0.0\%$ | **0.00%** | **PASS** |

### Secondary Success Gates Validation

| Metric | Target Gate | Actual Result (pccc_compute_kernels) | Status |
|---|---|---|---|
| **Budget Parse Accuracy** | $\ge 99\%$ | **100.00%** | **PASS** |
| **Entity Canonicalization Accuracy** | $\ge 98\%$ | **100.00%** | **PASS** |
| **Duplicate Suppression Accuracy** | $\ge 99\%$ | **100.00%** | **PASS** |
| **Alias Resolution Accuracy** | $\ge 95\%$ | **100.00%** | **PASS** |
| **Missing-field Handling Accuracy** | $\ge 98\%$ | **100.00%** | **PASS** |

---

### Detailed Performance Breakdown by Mode

#### 1. PCCC Compute Kernels
* **Overall EM**: 100.00%
* **Global LLM Bypass Rate**: 100.00%
* **Mean Latency**: 25.2 ms
* **p95 Latency**: 48.5 ms
* **Category Breakdown**:
  - Category A: EM=100.00% (Count=15)
  - Category B: EM=100.00% (Count=19)
  - Category C: EM=100.00% (Count=7)
  - Category D: EM=100.00% (Count=6)
  - Category E: EM=100.00% (Count=345)
  - Category F: EM=100.00% (Count=8)
  - Category G: EM=100.00% (Count=200)
  - Category H: EM=100.00% (Count=200)

---

### Conclusion & Key Findings
1. **Robust Adversarial Ingestion**: The Compute Kernels are fully resilient to alternative format budget expressions (USD 987k, 0.987M, etc.) and accent/alias variations, achieving perfect target accuracies.
2. **Obsolete Value Filtering**: By identifying signals like "old memo", "was reassigned", etc., the kernels select only active information, preventing stale data contamination.
3. **Graceful Error Routing**: Instead of failing silently with NOT_FOUND, the scheduler successfully routes specific failures to `KERNEL_MISSING_FIELD` or `INSUFFICIENT_EVIDENCE`.

