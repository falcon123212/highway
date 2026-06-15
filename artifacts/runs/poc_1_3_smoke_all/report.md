# POC 1.3 Smoke Test â€” Adaptive vs JSON Baseline (All Categories)

## 1. Quality & Format Metrics

| Mode | Exact Match | Suffix Err (Cat E) | Abstention Acc | Parse Fail / OOM | Avg Blocks | Token Red. |
|---|:---:|:---:|:---:|:---:|:---:|:---:|
| **predictor_cached_guarded_json** | 84.0% | 0.0% | 100.0% | 0.0% | 8.26 | 95.9% |
| **predictor_cached_guarded_adaptive_kernel** | 100.0% | 0.0% | 100.0% | 0.0% | 8.26 | 95.9% |

## 2. Exact Match Breakdown by Category

| Mode | Category A | Category B | Category C | Category D | Category E |
|---|:---:|:---:|:---:|:---:|:---:|
| **predictor_cached_guarded_json** | 100.0% | 100.0% | 100.0% | 20.0% | 100.0% |
| **predictor_cached_guarded_adaptive_kernel** | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% |

## 3. Success Gates Validation

| Success Gate | Target | Value | Status |
|---|---|:---:|:---:|
| **No crash / Parse fail** | &le; 2% | **0.0%** | **PASS** |
| **Suffix Error (Cat E)** | = 0% | **0.0%** | **PASS** |
| **Abstention Accuracy** | = 100% | **100.0%** | **PASS** |
| **Category D (Adaptive > JSON)** | Adaptive (100.0%) > JSON (20.0%) | **+80.0 pts** | **PASS** |
| **A/B/C/E Compatibility** | Adaptive (100.0%) &ge; JSON (100.0%) - 2% | **+0.0 pts** | **PASS** |

---

## 4. Key Takeaways

1. **All-Category Compatibility**: The Adaptive compiler does not break the standard single-fact categories (A, B, C, E) and maintains perfect parity or exceeds baseline scores.
2. **Abstention & Suffix Safety preserved**: Suffix distraction remains at 0% and abstentions remain at 100% accuracy, validating the integrated guards.
3. **Category D Lift confirmed**: Structuring multi-fact payloads directly resolves the Qwen 0.5B multi-fact extraction failures.

