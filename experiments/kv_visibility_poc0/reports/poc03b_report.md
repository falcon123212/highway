# POC 0.3b — Clean Cost Dominance Confirmation

Status: **FAIL**

Model: **Qwen/Qwen2.5-3B-Instruct**
Samples: **100**
Context sizes: **6.5k / 26k / 52k tokens** (50 / 200 / 400 blocks)

## Quality:
*   **Predictor EM (6.5k)**: 71.0%
*   **Hybrid EM (6.5k)**: 79.0%
*   **Delta**: -8.0 pts (Gate: within &plusmn;1 pt)
*   **Numeric Preservation**: 72.0% (Gate: &ge; 90%)
*   **Gold Block Recall**: 100.0% (Gate: &ge; 99%)
*   **Evidence Block Recall**: 74.0%
*   **Contradiction Accuracy**: 100.0%
*   **Multi-fact Recall**: 100.0%

## Cost & Latencies (6.5k scale):
*   **Selector Latency**: 24.64 ms (Gate: &le; 100 ms)
*   **Hybrid Retrieval Latency**: 13.61 ms (BM25 + Dense + Combining)
*   **TTFT Full**: 1205.1 ms
*   **TTFT Hybrid**: 136.2 ms
*   **TTFT Predictor**: 135.7 ms
*   **TTFT Reduction vs Full**: 88.7% (Gate: &ge; 50%)
*   **TTFT Difference vs Hybrid**: -0.3% (Gate: &plusmn;5%)
*   **Token Reduction**: 72.4% (Gate: &ge; 70%)

## Peak VRAM & KV Cache:
*   **Peak VRAM Total Full**: 8022.5 MB
*   **Peak VRAM Total Hybrid**: 6645.3 MB
*   **Peak VRAM Total Predictor**: 6641.1 MB
*   **Peak VRAM Total Reduction vs Full**: 17.2% (Gate: &ge; 15%)
*   **Estimated KV Cache (52k)**: 103.1 MB (vs Full: 452.1 MB)
*   **Estimated KV Reduction (52k)**: 77.2% (Gate: &ge; 60%)

## Throughput & Batch Capacity:
*   **Max Batch Size Full**: 1.0
*   **Max Batch Size Hybrid**: 3.0
*   **Max Batch Size Predictor**: 3.0
*   **Max Batch Size vs Full Scaling**: 3.0x (Gate: &ge; 2.0x)
*   **OOM Rate (52k)**: Predictor 0.0% vs Full 0.0%

## Success Gates Status:

| Gate | Target | Value | Status |
|---|---|---|---|
| **Gold Block Recall** | &ge; 99% | 100.0% | **PASS** |
| **Exact Match vs Hybrid** | &ge; Hybrid - 1 pt | -8.0 pts | **FAIL** |
| **Numeric Preservation** | &ge; 90% | 72.0% | **FAIL** |
| **Token Reduction** | &ge; 70% | 72.4% | **PASS** |
| **Selector Latency** | &le; 100 ms | 53.78 ms | **PASS** |
| **TTFT vs Full** | &ge; 50% | 88.7% | **PASS** |
| **TTFT vs Hybrid** | &plusmn;5% (or &le; 20 ms) | -0.3% | **PASS** |
| **Peak VRAM Total Reduction vs Full** | &ge; 15% | 17.2% | **PASS** |
| **Estimated KV Reduction** | &ge; 60% | 70.8% | **PASS** |
| **Max Batch Size vs Full** | &ge; &times;2 | 3.0x | **PASS** |
| **OOM Rate** | Predictor &le; Full | Predictor 0.0% vs Full 0.0% | **PASS** |
| **52k Context** | Run Stable | 0.0% OOM Rate | **PASS** |

## Verdict:
The Predictor failed to satisfy all cost-dominance validation gates. Review model latency or memory utilization.

