# POC 2.3.4/2.3.5 No-Leak Performance & Verification Report
## Kernel Hardening - Adversarial Structured Evidence

This report verifies the performance gates and claims for the **Hardened Compute Kernels** under POC 2.3.4/2.3.5. Legacy runs without `leak_check_passed=true` are treated as historical/non-validating.


### Primary Success Gates Validation

| Metric | Target Gate | Actual Result (pccc_compute_kernels) | Status |
|---|---|---|---|
| **No-Leak Workload Check** | $= 100\%$ | **100.00%** | **PASS** |
| **G/H EM Global** | $\ge 98\%$ | **100.00%** | **PASS** |
| **COMPUTE_COMPARISON EM** | $\ge 98\%$ | **100.00%** | **PASS** |
| **COMPUTE_AGGREGATION EM** | $\ge 98\%$ | **100.00%** | **PASS** |
| **False NOT_FOUND** | $= 0\%$ | **0.00%** | **PASS** |
| **Wrong active/obsolete selection** | separately measured | **N/A** | **N/A** |
| **Kernel Verifier Pass Rate** | $\ge 99\%$ | **100.00%** | **PASS** |
| **LLM Call Rate on G/H** | $= 0\%$ | **0.00%** | **PASS** |
| **Exec Error to NOT_FOUND Conv.** | $= 0\%$ | **0.00%** | **PASS** |
| **Latency p95** | $< 50$ ms | **29.5 ms** | **PASS** |
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
* **No-Leak Pass Rate**: 100.00%
* **Non-validating Records**: 0
* **Global LLM Bypass Rate**: 100.00%
* **Mean Latency**: 26.5 ms
* **p95 Latency**: 29.5 ms
* **Category Breakdown**:
  - Category G: EM=100.00% (Count=10)
  - Category H: EM=100.00% (Count=10)

---

### Conclusion & Key Findings
1. **No-Leak Validation**: The report treats legacy records without `leak_check_passed=true` as historical/non-validating.
2. **Kernel Accuracy**: Accuracy claims are based on exact-match records that also pass the no-leak gate.
3. **Remaining Work**: Any failed gate in the table above should be treated as a real follow-up item, not hidden by historical scores.
