# POC 1.3 Mini-run 3 Report â€” Budget Sweep

## 1. Quality & Efficiency Summary (100 mixed samples, 400 blocks)

| Budget | Overall EM | Gold Recall | Suffix Error | Abstention Acc | Parse Fail | Avg Blocks Kept | Token Reduction |
|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **max_kept_4** | 97.0% | 97.0% | 0.0% | 100.0% | 0.0% | 3.46 | 99.1% |
| **max_kept_6** | 100.0% | 100.0% | 0.0% | 100.0% | 0.0% | 5.32 | 98.7% |
| **max_kept_8** | 100.0% | 100.0% | 0.0% | 100.0% | 0.0% | 7.32 | 98.2% |
| **max_kept_12** | 100.0% | 100.0% | 0.0% | 100.0% | 0.0% | 11.32 | 97.2% |
| **max_kept_16** | 100.0% | 100.0% | 0.0% | 100.0% | 0.0% | 15.32 | 96.2% |

## 2. Category-Specific Exact Match Breakdown

| Budget | Category A | Category B | Category C | Category D | Category E |
|---|:---:|:---:|:---:|:---:|:---:|
| **max_kept_4** | 100.0% | 100.0% | 100.0% | 100.0% | 85.0% |
| **max_kept_6** | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% |
| **max_kept_8** | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% |
| **max_kept_12** | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% |
| **max_kept_16** | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% |

## 3. Success Gates Validation

### Budget: max_kept_4

| Success Gate | Target | Actual | Status |
|---|---|:---:|:---:|
| **Overall EM** | &ge; 95% | **97.0%** | **PASS** |
| **Category D EM** | &ge; 90% | **100.0%** | **PASS** |
| **Gold Recall** | &ge; 99% | **97.0%** | **FAIL** |
| **Suffix Error (Cat E)** | = 0% | **0.0%** | **PASS** |
| **Abstention Accuracy** | = 100% | **100.0%** | **PASS** |
| **Parse Fail / OOM** | = 0% | **0.0%** | **PASS** |
| **Token Reduction** | &ge; 96% | **99.1%** | **PASS** |

### Budget: max_kept_6

| Success Gate | Target | Actual | Status |
|---|---|:---:|:---:|
| **Overall EM** | &ge; 95% | **100.0%** | **PASS** |
| **Category D EM** | &ge; 90% | **100.0%** | **PASS** |
| **Gold Recall** | &ge; 99% | **100.0%** | **PASS** |
| **Suffix Error (Cat E)** | = 0% | **0.0%** | **PASS** |
| **Abstention Accuracy** | = 100% | **100.0%** | **PASS** |
| **Parse Fail / OOM** | = 0% | **0.0%** | **PASS** |
| **Token Reduction** | &ge; 96% | **98.7%** | **PASS** |

### Budget: max_kept_8

| Success Gate | Target | Actual | Status |
|---|---|:---:|:---:|
| **Overall EM** | &ge; 95% | **100.0%** | **PASS** |
| **Category D EM** | &ge; 90% | **100.0%** | **PASS** |
| **Gold Recall** | &ge; 99% | **100.0%** | **PASS** |
| **Suffix Error (Cat E)** | = 0% | **0.0%** | **PASS** |
| **Abstention Accuracy** | = 100% | **100.0%** | **PASS** |
| **Parse Fail / OOM** | = 0% | **0.0%** | **PASS** |
| **Token Reduction** | &ge; 96% | **98.2%** | **PASS** |

### Budget: max_kept_12

| Success Gate | Target | Actual | Status |
|---|---|:---:|:---:|
| **Overall EM** | &ge; 95% | **100.0%** | **PASS** |
| **Category D EM** | &ge; 90% | **100.0%** | **PASS** |
| **Gold Recall** | &ge; 99% | **100.0%** | **PASS** |
| **Suffix Error (Cat E)** | = 0% | **0.0%** | **PASS** |
| **Abstention Accuracy** | = 100% | **100.0%** | **PASS** |
| **Parse Fail / OOM** | = 0% | **0.0%** | **PASS** |
| **Token Reduction** | &ge; 96% | **97.2%** | **PASS** |

### Budget: max_kept_16

| Success Gate | Target | Actual | Status |
|---|---|:---:|:---:|
| **Overall EM** | &ge; 95% | **100.0%** | **PASS** |
| **Category D EM** | &ge; 90% | **100.0%** | **PASS** |
| **Gold Recall** | &ge; 99% | **100.0%** | **PASS** |
| **Suffix Error (Cat E)** | = 0% | **0.0%** | **PASS** |
| **Abstention Accuracy** | = 100% | **100.0%** | **PASS** |
| **Parse Fail / OOM** | = 0% | **0.0%** | **PASS** |
| **Token Reduction** | &ge; 96% | **96.2%** | **PASS** |


## 4. Decision Matrix & Sweet Spot Recommendation

ðŸ“Œ **DECISION: Sweet spot (max_kept = 6)**. Le budget de 4 blocs dÃ©grade la qualitÃ© (notamment la CatÃ©gorie D), mais max_kept_6 permet de valider toutes les success gates avec un excellent taux de culling.

