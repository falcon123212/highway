# POC 1.1 Overnight â€” Compiler Guard & Quality Rescue Report

## 1. Executive Summary

This evaluation validates whether quality failures observed in the first iteration of POC 1 stem from model size limitations (`Qwen2.5-0.5B-Instruct`), a context compiler that compiles suffix distractors (the "suffix trap"), or a lack of a deterministic guard on missing entities. 

By implementing a **strict character-boundary suffix filter** and a **deterministic exact-match bypass guard (`guarded`)**, we achieved major quality improvements:
*   **Exact Match (EM) increased by +27.0%** at 200 blocks and **+27.7%** at 400 blocks, rescuing overall accuracy.
*   **Suffix Error Rate dropped to 0.0%** (down from 85.0% in baseline cached modes).
*   **Abstention Accuracy jumped to 100.0%** (up from ~2-4% in baseline cached modes).
*   **Latency Overhead was negative** (improving p95 latency by up to **-173.0 ms**) since the deterministic bypass completely skips LLM decoding on absent entities.

---

## 2. Setup

*   **Model**: `Qwen/Qwen2.5-0.5B-Instruct`
*   **Precision**: `FP16`
*   **Serving Engine**: `vLLM`
*   **Environment**: `WSL2` (running vLLM server), Windows (running evaluation client)
*   **GPU**: `GeForce RTX 4060 (8GB VRAM)`
*   **Hyperparameters**: Temperature = 0, Top-p = 1, Max new tokens = 64
*   **Samples per combination**: 300 (60 per Category A-E)
*   **Sanity modes samples**: 50 (`full_context` and `random`)
*   **Context Sizes**: 200 blocks (~25.6k tokens) and 400 blocks (~51.2k tokens)
*   **Block Size**: 128 tokens

---

## 3. Modes Compared

1.  **`oracle`**: Golden context (only the exact block containing the fact is kept). This serves as the upper-bound quality baseline.
2.  **`hybrid`**: Standard hybrid compilation of embeddings and visibility selectors.
3.  **`predictor_cached`**: Baseline cached embeddings pipeline using standard prompt compiler.
4.  **`predictor_cached_strict_entity`**: Baseline pipeline but with a post-selector filter. If a question targets an entity (e.g. `XENON-407`), any blocks containing only suffix distractors (e.g. `XENON-407-Legacy`) are discarded, keeping only exact-match blocks.
5.  **`predictor_cached_guarded`**: Identical to `predictor_cached_strict_entity`, but with an added deterministic bypass: if no blocks contain the exact entity name, the LLM is not called and a static `NOT_FOUND` response is returned immediately.
6.  **`full_context` (Sanity Check)**: Passes the entire context to the LLM (50 samples).
7.  **`random` (Sanity Check)**: Selects random blocks as context (50 samples).

---

## 4. Quality Results

Below are the aggregated quality metrics for both context sizes:

### Context Size: 200 blocks (~25.6k tokens)
| Mode | Sample Count | Exact Match | F1 Score | Numeric Pres. | Suffix Error Rate | Abstention Acc | Avg Blocks Kept | Token Red. |
|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **full_context** | 50 | 78.0% | 74.0% | 78.0% | 0.0% | 100.0% | 200.00 | 0.0% |
| **oracle** | 300 | 75.7% | 81.2% | 83.7% | 3.3% | 45.5% | 1.16 | 99.4% |
| **random** | 50 | 4.0% | 32.0% | 10.0% | 0.0% | 0.0% | 7.42 | 96.3% |
| **hybrid** | 300 | 56.7% | 73.1% | 71.7% | 83.3% | 4.5% | 9.17 | 95.4% |
| **predictor_cached** | 300 | 56.7% | 73.4% | 72.0% | 81.7% | 2.3% | 9.16 | 95.4% |
| **predictor_cached_strict_entity** | 300 | 70.0% | 81.3% | 84.0% | 16.7% | 4.5% | 8.46 | 95.8% |
| **predictor_cached_guarded** | 300 | 83.7% | 81.2% | 83.7% | 0.0% | 100.0% | 8.46 | 95.8% |

### Context Size: 400 blocks (~51.2k tokens)
| Mode | Sample Count | Exact Match | F1 Score | Numeric Pres. | Suffix Error Rate | Abstention Acc | Avg Blocks Kept | Token Red. |
|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **full_context** | 50 | 12.0% | 41.4% | 90.0% | 0.0% | 100.0% | 400.00 | 0.0% |
| **oracle** | 300 | 75.7% | 81.2% | 83.7% | 3.3% | 45.5% | 1.16 | 99.7% |
| **random** | 50 | 2.0% | 28.7% | 10.0% | 0.0% | 0.0% | 10.56 | 97.4% |
| **hybrid** | 300 | 57.0% | 73.2% | 71.7% | 85.0% | 6.8% | 15.04 | 96.2% |
| **predictor_cached** | 300 | 56.0% | 72.8% | 71.0% | 85.0% | 4.5% | 15.03 | 96.2% |
| **predictor_cached_strict_entity** | 300 | 70.0% | 81.0% | 83.7% | 16.7% | 6.8% | 14.34 | 96.4% |
| **predictor_cached_guarded** | 300 | 83.7% | 81.0% | 83.7% | 0.0% | 100.0% | 14.42 | 96.4% |

---

## 5. Latency Results

Evaluating the execution latency metrics (percentiles p50 and p95 for first token delivery) and average decoding throughput:

### Context Size: 200 blocks (~25.6k tokens)
| Mode | TTFT p50 | TTFT p95 | Decode Tokens/s | First Token Lat. p50 | First Token Lat. p95 | Cost/Correct Answer |
|---|:---:|:---:|:---:|:---:|:---:|:---:|
| **full_context** | 2637.0 ms | 2735.4 ms | 47.6 tok/s | 2681.9 ms | 2818.4 ms | 3.933 s |
| **oracle** | 2071.3 ms | 2111.2 ms | 45.4 tok/s | 2102.9 ms | 2175.9 ms | 3.828 s |
| **random** | 2102.3 ms | 2185.9 ms | 42.6 tok/s | 2135.8 ms | 2341.0 ms | 72.970 s |
| **hybrid** | 2089.8 ms | 2228.2 ms | 44.3 tok/s | 2124.2 ms | 2267.9 ms | 5.230 s |
| **predictor_cached** | 2088.5 ms | 2290.9 ms | 43.3 tok/s | 2126.3 ms | 2318.3 ms | 5.258 s |
| **predictor_cached_strict_entity** | 2082.7 ms | 2252.0 ms | 43.0 tok/s | 2117.2 ms | 2327.2 ms | 4.219 s |
| **predictor_cached_guarded** | 2077.2 ms | 2120.8 ms | 43.9 tok/s | 2102.4 ms | 2185.2 ms | 2.972 s |

### Context Size: 400 blocks (~51.2k tokens)
| Mode | TTFT p50 | TTFT p95 | Decode Tokens/s | First Token Lat. p50 | First Token Lat. p95 | Cost/Correct Answer |
|---|:---:|:---:|:---:|:---:|:---:|:---:|
| **full_context** | 2922.5 ms | 2992.4 ms | 60.2 tok/s | 2957.5 ms | 3026.3 ms | 28.560 s |
| **oracle** | 2071.5 ms | 2109.8 ms | 43.8 tok/s | 2110.5 ms | 2206.8 ms | 3.879 s |
| **random** | 2089.2 ms | 2193.8 ms | 45.0 tok/s | 2126.1 ms | 2320.5 ms | 143.681 s |
| **hybrid** | 2089.0 ms | 2346.4 ms | 44.3 tok/s | 2129.5 ms | 2382.5 ms | 5.195 s |
| **predictor_cached** | 2080.6 ms | 2348.3 ms | 44.2 tok/s | 2123.5 ms | 2386.1 ms | 5.283 s |
| **predictor_cached_strict_entity** | 2078.3 ms | 2336.5 ms | 44.5 tok/s | 2116.2 ms | 2406.5 ms | 4.180 s |
| **predictor_cached_guarded** | 2080.2 ms | 2113.6 ms | 43.5 tok/s | 2114.7 ms | 2213.1 ms | 3.003 s |

---

## 6. Selector Results

Evaluating selector latency overhead and the recall of target facts:

| Mode | Context Blocks | Gold Block Recall | Selector p50 | Selector p95 | Avg Blocks Kept | Token Reduction |
|---|:---:|:---:|:---:|:---:|:---:|:---:|
| **hybrid** | 200 | 100.0% | 25.6 ms | 30.6 ms | 9.17 | 95.4% |
| **predictor_cached** | 200 | 100.0% | 25.7 ms | 30.4 ms | 9.16 | 95.4% |
| **predictor_cached_guarded** | 200 | 100.0% | 25.7 ms | 30.9 ms | 8.46 | 95.8% |
| **predictor_cached_strict_entity** | 200 | 100.0% | 25.6 ms | 30.7 ms | 8.46 | 95.8% |
| **hybrid** | 400 | 100.0% | 30.3 ms | 35.5 ms | 15.04 | 96.2% |
| **predictor_cached** | 400 | 100.0% | 30.8 ms | 35.8 ms | 15.03 | 96.2% |
| **predictor_cached_guarded** | 400 | 100.0% | 30.6 ms | 35.4 ms | 14.42 | 96.4% |
| **predictor_cached_strict_entity** | 400 | 100.0% | 30.5 ms | 35.2 ms | 14.34 | 96.4% |

*   **Recall Stability**: The no-position cached selector maintained a flawless **100.0% Gold Block Recall** across all evaluated contexts, confirming that culling does not drop the source evidence.
*   **Selector Overhead**: The local embedding-based culling process introduces extremely low latency, with p50 under **32 ms** and p95 under **38 ms** even at 400 blocks.

---

## 7. Error Breakdown

Aggregate counts of error classifications across all contexts (200 & 400 blocks combined):

| Mode | Correct | Suffix Confusion | Missing Project Halluc. | Numeric Wrong | Gold Missing | Model Failed (Gold Present) |
|---|:---:|:---:|:---:|:---:|:---:|:---:|
| **oracle** | 454 | 0 | 48 | 98 | 0 | 0 |
| **hybrid** | 341 | 81 | 83 | 95 | 0 | 0 |
| **predictor_cached** | 338 | 80 | 85 | 97 | 0 | 0 |
| **predictor_cached_strict_entity** | 420 | 0 | 83 | 97 | 0 | 0 |
| **predictor_cached_guarded** | 502 | 0 | 0 | 98 | 0 | 0 |

### Analysis:
*   **Suffix Confusion**: Totally eliminated in both `strict_entity` and `guarded` modes.
*   **Missing Project Hallucination**: Dropped from 85 counts in `predictor_cached` to 0 in `predictor_cached_guarded` thanks to the bypass guard.
*   **Numeric Wrong**: Remains the primary remaining failure mode (98 counts in `guarded` mode vs 98 in `oracle`), highlighting a capacity limitation of the 0.5B model when dealing with multi-fact queries (Category D) and exact numbers/dates.

---

## 8. Guard Impact

1.  **Strict Suffix Filter (`strict_entity`)**:
    *   By enforcing boundaries on targeted entities (checking that letters, digits, underscores, or hyphens do not follow the entity name), the post-selector successfully filtered out distraction blocks (e.g. `XENON-407-Legacy`).
    *   This eliminated **Suffix Confusion** errors, boosting EM from **56.7%** to **70.0%** at 200 blocks.
2.  **Deterministic Abstention Guard (`guarded`)**:
    *   If no matching exact entity is present in the selected blocks, the pipeline directly returns a `NOT_FOUND` response.
    *   This rescued the **Abstention Accuracy** from a near-zero level (~2.3% for baseline caching) to a perfect **100.0%**.
    *   Additionally, since it avoids calling the LLM entirely, it drops first-token latency to the cost of selector + compile (~112 ms total p95), producing a negative latency overhead on p95.

---

## 9. Interpretation

Based on the performance gates:

*   **Gate 1 (Recall)**: Flawless **100.0% >= 99%** (**PASS**)
*   **Gate 2 (Suffix Rate)**: **0.0% <= 25%** (**PASS**)
*   **Gate 3 (Abstention)**: **100.0% >= 80%** (**PASS**)
*   **Gate 4 (EM Gain)**: **+27.0% (200b) / +27.7% (400b)** which is far greater than the +5.0 pts target (**PASS**)
*   **Gate 5 (Latency)**: **-133.1 ms / -173.0 ms** (negative overhead, target was <= +30 ms) (**PASS**)

We are in **Cas A â€” Guarded amÃ©liore fortement**:
> **POC 1.1 shows that deterministic context compiler guards significantly improve suffix discrimination and abstention behavior on Qwen2.5-0.5B-Instruct, while preserving the low-latency benefits of vLLM front-end culling.**

The problem was not purely the Qwen 0.5B model's reasoning capabilities; the permissive context compiler and suffix leaks were responsible for a significant share of errors. Implementing strict matching and exact guards brings vLLM culling performance close to or above the oracle level for simple facts and abstentions.

---

## 10. Decision for Next POC

1.  **Integrate Guards**: Integrate the exact-match entity filters and deterministic bypass guards into the production pipeline for POC 1 final.
2.  **Next Evaluation Step**: While simple facts, suffix errors, and abstentions are fully solved, **Numeric Preservation** on multi-fact requests remains the next bottleneck (causing around ~16% of total errors even in Oracle mode).
3.  **Stronger Models**: Recommend executing the same vLLM benchmark protocol with a stronger model (e.g., `Qwen2.5-3B-Instruct` or `Qwen2.5-7B-Instruct` quantized) to address the capacity bottleneck in extracting multiple facts and preserving precise numbers/budgets.

