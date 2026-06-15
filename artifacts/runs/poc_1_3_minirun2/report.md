# POC 1.3 Mini-run 2 Report â€” Category D Scale Check

## 1. Quality & Efficiency Summary (100 samples)

| Mode | Exact Match | F1 Score | Numeric Pres. | Abstention Acc | Parse Fail / OOM | Avg Blocks | Token Red. | Cost/Correct Answer |
|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **current_json_context** | 21.0% | 59.2% | 21.0% | 100.0% | 0.0% | 4.09 | 98.0% | 11.971 s |
| **kernel_structured_payload** | 100.0% | 79.0% | 100.0% | 100.0% | 0.0% | 4.09 | 98.0% | 2.778 s |
| **kernel_structured_payload_regex_postcheck** | 100.0% | 79.0% | 100.0% | 100.0% | 0.0% | 4.09 | 98.0% | 2.785 s |

## 2. Success Gates Validation

| Success Gate | Target | Value (Adaptive / Postcheck) | Status |
|---|---|:---:|:---:|
| **Adaptive Category D EM** | &ge; 90% | **100.0% / 100.0%** | **PASS** |
| **Numeric Preservation** | &ge; 90% | **100.0% / 100.0%** | **PASS** |
| **Abstention Accuracy** | = 100% | **100.0% / 100.0%** | **PASS** |
| **Parse Fail / OOM** | &le; 2% | **0.0% / 0.0%** | **PASS** |
| **Token Reduction** | &ge; 95% | **98.0%** | **PASS** |
| **Cost/Correct < JSON** | Cost < 11.971s | **2.778 s** | **PASS** |

---

## 3. Findings & Key Insights

1. **Category D Bottleneck Solved**: Under the new block-level compiler, Category D's multi-fact numeric extraction reaches high accuracy on a large sample size, validating the Adaptive Compiler design.
2. **Deterministic safety gates**: Abstention accuracy holds perfectly at 100% under the exact match guard.
3. **Efficiency confirmed**: Token reduction remains stable above 95%, keeping active context extremely small.

