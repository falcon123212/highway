# POC 0.1 — Pre-Prefill Visibility Predictor Report

Status: **FAIL**

## Configuration
- **Validation Test Samples**: 100
- **Average Blocks per Prompt**: 50.0
- **Block Size**: 128 tokens
- **No Full-Context LLM Pass**: **TRUE** (No prefill attention pass required!)

## Quality & Accuracy Comparison

| Metric | Predictor (POC 0.1) | Oracle (POC 0) | Dense (MiniLM) | Hybrid | Random |
|---|---|---|---|---|---|
| **Exact Match** | 23.0% | 23.0% | 20.0% | 23.0% | 17.0% |
| **Numeric Preservation** | 39.0% | - | 41.0% | 38.0% | - |

## Evidence and Culling Performance

- **Gold Block Recall**: 100.0% (Target: &ge; 99.0%) &rarr; **PASS**
- **Average Kept Blocks**: 15.2 / 50 (Target: &le; 20.0) &rarr; **PASS**
- **Token Reduction**: 77.6% (Target: &ge; 60.0%) &rarr; **PASS**
- **Exact Match vs Dense**: Predictor is +3.0 pts relative to Dense (Target: &ge; +5.0 pts) &rarr; **FAIL**

## Latency Metrics (TTFT)
- **Predictor TTFT**: 1898.9 ms
- **Hybrid TTFT**: 1900.1 ms

## Verdict
The predictor failed to satisfy all validation gates. Review classifier accuracy and recall.

