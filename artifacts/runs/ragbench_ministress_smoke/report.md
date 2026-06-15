# RAGBench Mini Stress Test v0 Report

**Status**: VALIDATING
**Model**: `ragbench_grounded_fake`
**Configs**: covidqa, cuad, finqa, hotpotqa, techqa
**Count**: 25 cases

## Performance Table Comparison

| Metric | Full local | BM25 local | Highway local | **Pruned local** | BM25 global | Dense global | Hybrid global | Highway global | **Pruned global** | **Pruned global BM25 S1** |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| Input tokens (avg) | 1744.4 | 226.8 | 1259.2 | **312.8** | 225.4 | 168.6 | 208.5 | 2386.0 | **421.9** | **370.7** |
| Input tokens ratio | 100.0% | 13.0% | 72.2% | **17.9%** | 12.9% | 9.7% | 12.0% | 136.8% | **24.2%** | **21.3%** |
| Utilized recall | N/A | 24.33% | 69.76% | **43.13%** | 19.00% | 0.00% | 16.33% | 25.00% | **3.60%** | **28.27%** |
| Relevant recall | N/A | 25.46% | 69.89% | **45.82%** | 21.08% | 4.00% | 17.95% | 25.19% | **5.90%** | **27.48%** |
| Answer correctness | 100.00% | 88.00% | 88.00% | **80.00%** | 76.00% | 8.00% | 72.00% | 36.00% | **28.00%** | **72.00%** |
| Attribution accuracy | 100.00% | 88.00% | 88.00% | **80.00%** | 76.00% | 8.00% | 72.00% | 36.00% | **28.00%** | **72.00%** |
| Grounded success rate | 100.00% | 88.00% | 88.00% | **80.00%** | 76.00% | 8.00% | 72.00% | 36.00% | **28.00%** | **72.00%** |
| Hallucination rate | 0.00% | 0.00% | 0.00% | **0.00%** | 0.00% | 0.00% | 0.00% | 0.00% | **0.00%** | **0.00%** |
| Tokens / correct grounded | 1744.4 | 228.7 | 1333.8 | **321.4** | 207.9 | 156.0 | 192.6 | 3044.4 | **470.7** | **357.3** |
| Tokens / attempted success | 1744.4 | 257.7 | 1430.9 | **390.9** | 296.6 | 2107.0 | 289.6 | 6627.8 | **1506.7** | **514.9** |
| Tokens / correct only | 1744.4 | 257.7 | 1430.9 | **390.9** | 296.6 | 2107.0 | 289.6 | 6627.8 | **1506.7** | **514.9** |

## Global-Specific Retrieval Metrics

| Metric | BM25 global | Dense global | Hybrid global | Highway global | **Pruned global** | **Pruned global BM25 S1** |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| case_hit_rate | 92.00% | 8.00% | 92.00% | **56.00%** | **56.00%** | **92.00%** |
| doc_hit_rate | 76.00% | 8.00% | 72.00% | **36.00%** | **28.00%** | **72.00%** |
| support_sentence_recall | 19.00% | 0.00% | 16.33% | **25.00%** | **3.60%** | **28.27%** |
| distractor_selection_rate | 68.00% | 98.40% | 74.40% | **90.22%** | **88.03%** | **72.99%** |

## Token Ratio Metrics

| Metric | BM25 local | Highway local | **Pruned local** | BM25 global | Dense global | Hybrid global | Highway global | **Pruned global** | **Pruned global BM25 S1** |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| ratio_of_averages | 13.00% | 72.18% | **17.93%** | 12.92% | 9.66% | 11.95% | 136.78% | **24.18%** | **21.25%** |
| mean_of_case_ratios | 32.35% | 71.43% | **41.23%** | 30.70% | 25.92% | 29.47% | 457.78% | **63.34%** | **53.31%** |

## Poisoning & Security Gates

| Metric | Highway local | **Pruned local** | Highway global | **Pruned global** | **Pruned global BM25 S1** |
| :--- | :---: | :---: | :---: | :---: | :---: |
| Poison false validation rate | 0.00% | **0.00%** | 0.00% | **0.00%** | **0.00%** |
| Poison on initially valid | 0.00% | **0.00%** | 0.00% | **0.00%** | **0.00%** |
| Poison initially valid N | 22 | **20** | 9 | **7** | **18** |
| Poison false validation count | 0 | **0** | 0 | **0** | **0** |

## Validation Gates (POC 16.1.1)

- **duplicate_source_id**: `0` (Expected: 0)
- **support_key_mapping_accuracy**: `100.0%` (Expected: 100.0%)

## POC 16.5 / 16.6 — Diagnostic Gates

**Run size**: `smoke` (25 cases)

| Gate | Value | Target | Status |
| :--- | :---: | :---: | :---: |
| grounded_success_ge_88 | 80.00 | 88.00 | ❌ FAIL |
| avg_tokens_le_500 | 312.76 | 500.00 | ✅ PASS |
| utilized_recall_ge_bm25 | 43.13 | 24.33 | ✅ PASS |
| tokens_per_attempted_success_le_600 | 390.95 | 600.00 | ✅ PASS |
| poison_initially_valid_zero | 0.00 | 0.00 | ✅ PASS |
| global_grounded_success_ge_85 | 28.00 | 85.00 | ❌ FAIL |
| global_avg_tokens_le_500 | 421.88 | 500.00 | ✅ PASS |
| global_case_hit_rate_ge_92 | 56.00 | 92.00 | ❌ FAIL |
| global_distractor_rate_le_50 | 88.03 | 50.00 | ❌ FAIL |
| global_bm25_stage1_grounded_success_ge_70 | 72.00 | 70.00 | ✅ PASS |

## Document Aggregation Strategy Sweep (Stage 1 Retrieval)

| Stage 1 | Aggregation Strategy | Case Hit Rate | Doc Hit Rate | Support Sentence Recall | Distractor Selection Rate |
| :--- | :--- | :---: | :---: | :---: | :---: |
| HYBRID | `sum_score` | 56.00% | 28.00% | 3.60% | 88.03% |
| HYBRID | `max_score` | 92.00% | 80.00% | 26.27% | 70.34% |
| HYBRID | `top3_avg_score` | 92.00% | 80.00% | 26.93% | 70.97% |
| HYBRID | `bm25_doc_score + max_sentence_score` | 92.00% | 76.00% | 28.27% | 70.87% |
| HYBRID | `bm25_doc_score + top3_sentence_score` | 92.00% | 80.00% | 32.27% | 71.41% |
| BM25 | `sum_score` | 88.00% | 56.00% | 15.27% | 79.10% |
| BM25 | `max_score` | 92.00% | 80.00% | 28.27% | 69.56% |
| BM25 | `top3_avg_score` | 92.00% | 80.00% | 33.60% | 70.77% |
| BM25 | `bm25_doc_score + max_sentence_score` | 92.00% | 72.00% | 28.27% | 72.99% |
| BM25 | `bm25_doc_score + top3_sentence_score` | 92.00% | 76.00% | 29.60% | 70.37% |

## Files Written

*   **Metrics JSON**: `artifacts/runs/ragbench_ministress_smoke/metrics.json`
*   **Records JSONL**: `artifacts/runs/ragbench_ministress_smoke/records.jsonl`
