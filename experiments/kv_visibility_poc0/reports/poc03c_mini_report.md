# POC 0.3c-mini — Visibility Fixes Verification Report

Status: **PASS**

Model: **Qwen/Qwen2.5-3B-Instruct**
Samples: **120** (60 Category E, 30 Category C, 30 Category D)
Context size: **50 blocks (~6.5k tokens)**

## Comparison Table:

| Metric | Hybrid | Old Predictor (Full) | New Predictor (No-Pos + Prompt Fix) |
|---|---|---|---|
| **Exact Match (Overall)** | 94.2% | 64.2% | 91.7% |
| **Numeric Preservation** | 96.7% | 65.0% | 97.5% |
| **Gold Block Recall** | 100.0% | 100.0% | 100.0% |
| **Average Kept Blocks** | 4.51 | 14.44 | 4.51 |
| **Token Reduction** | 91.0% | 71.1% | 91.0% |
| **Selector Latency** | 13.97 ms | 25.08 ms | 24.90 ms |
| **LLM Prefill TTFT** | 154.6 ms | 415.2 ms | 154.9 ms |

## Category-Specific breakdown:

### Category E (Unseen projects & Suffix distractors):
*   **Old Predictor EM**: 78.3%
*   **New Predictor EM**: 95.0%
*   **Delta EM**: +16.7 pts
*   **Old Predictor Suffix Error Rate**: 21.7%
*   **New Predictor Suffix Error Rate**: 5.0%
*   **Hybrid Suffix Error Rate**: 5.0%

### Category C (Contradiction accuracy):
*   **New Predictor Contradiction Accuracy**: 100.0% (Expected: &ge; 95%)

### Category D (Multi-fact recall):
*   **New Predictor Multi-fact Recall**: 100.0% (Expected: &ge; 95%)

## Success Gates Status:

| Gate | Target | Value | Status |
|---|---|---|---|
| **Gold Block Recall** | 100% | 100.0% | **PASS** |
| **Category E EM** | &ge; old predictor + 10 pts | +16.7 pts | **PASS** |
| **Suffix Error Rate** | &le; 10% | 5.0% | **PASS** |
| **Numeric Preservation** | &ge; 85% | 97.5% | **PASS** |
| **Avg Kept Blocks** | &le; 6 | 4.51 | **PASS** |
| **Token Reduction** | &ge; 85% | 91.0% | **PASS** |
| **TTFT vs old predictor** | &le; old predictor | New 154.9 ms vs Old 415.2 ms | **PASS** |
| **Contradiction Accuracy** | &ge; 95% | 100.0% | **PASS** |
| **Multi-fact Recall** | &ge; 95% | 100.0% | **PASS** |

## Verdict:
The position-ablation and strict-prompt fixes successfully resolve Category E distractor errors while maintaining perfect gold block recall, resulting in significantly higher token savings and faster TTFT.

