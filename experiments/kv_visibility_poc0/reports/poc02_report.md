# POC 0.2 — Hardening Brutal — Report

Status: **FAIL**

## 1. Test 1 & 2: Generalization & Split Evaluation

| Split Mode | Exact Match | Gold Block Recall | Average Kept Blocks | Generalization Gate | Status |
|---|---|---|---|---|---|
| **Split 1 (Standard Mixed)** | 75.0% | 100.0% | 13.70 | Reference | - |
| **Split 2 (ABCD &rarr; Test E: Unseen projects)** | 89.0% | 100.0% | 16.69 | Unseen Project Recall &ge; 99% | **PASS** |
| **Split 3 (ABE &rarr; Test CD: Cross-category)** | 50.0% | 100.0% | 13.42 | Category Generalization | - |

---

## 2. Test 3: Ablation Analysis (Ablating Position Features)

| Feature Configuration | Gold Block Recall | Exact Match | Average Kept Blocks | Ablation Gate | Status |
|---|---|---|---|---|---|
| **Full Features** | 100.0% | 75.0% | 13.70 | - | - |
| **No Position Features** | 100.0% | 73.0% | 4.18 | Recall &ge; 95% | **PASS** |
| **Semantic / Entity Only** | 100.0% | 73.0% | 4.18 | - | - |

---

## 3. Test 4: Long Context Scaling Analysis

| Context Size | Gold Block Recall | Average Kept Blocks | Selector Latency | LLM Prefill Time |
|---|---|---|---|---|
| **50 Blocks (~6.5k tokens)** | 100.0% | 14.70 | 21.86 ms | 1870.2 ms |
| **100 Blocks (~13.0k tokens)** | 100.0% | 22.90 | 22.02 ms | 1954.5 ms |
| **200 Blocks (~26.0k tokens)** | 100.0% | 39.80 | 24.57 ms | 2000.3 ms |
| **400 Blocks (~52.0k tokens)** | 100.0% | 72.50 | 30.06 ms | 2192.9 ms |

---

## 4. Test 5: Quality & Accuracy Comparison (Qwen-3B + JSON Output)

| Metric | Predictor (POC 0.2) | Dense (MiniLM) | Hybrid | Random |
|---|---|---|---|---|
| **Exact Match** | 75.0% | 73.0% | 72.0% | 18.0% |
| **Numeric Preservation** | 76.0% | 79.0% | 78.0% | - |
| **TTFT Proxy** | 2884.2 ms | - | 2944.9 ms | - |

---

## 5. Success Gates Status

| Gate | Target | Value | Status |
|---|---|---|---|
| **Gold Block Recall** | &ge; 99% | 100.0% | **PASS** |
| **Token Reduction** | &ge; 70% | 72.6% | **PASS** |
| **Selector Latency CPU** | &le; 100 ms | 21.86 ms | **PASS** |
| **Mixed Test Exact Match** | &ge; Hybrid + 5.0 pts | 75.0% (target: &ge; 77.0%) | **FAIL** |
| **No-position Gold Recall** | &ge; 95% | 100.0% | **PASS** |
| **Contradiction Accuracy** | &ge; 90% | 100.0% | **PASS** |
| **Multi-fact Recall** | &ge; 90% | 100.0% | **PASS** |
| **Generalization to Unseen Projects** | PASS | 100.0% Recall | **PASS** |
| **End-to-end TTFT Reduction** | &ge; 40% | 31.3% | **FAIL** |

## Verdict
The predictor failed to satisfy all validation gates. Review classifier accuracy, recall, and generalization metrics.

