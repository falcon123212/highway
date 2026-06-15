# RAGBench Mini Stress Test v0 Report

**Status**: VALIDATING
**Model**: `ragbench_grounded_fake`
**Configs**: covidqa, cuad, finqa, hotpotqa, techqa
**Count**: 5 cases

## Performance Table Comparison

| Metric | Full local | BM25 local | Highway local | **Pruned local** | BM25 global | Dense global | Hybrid global | Highway global | **Pruned global** | **Pruned global BM25 S1** |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| Input tokens (avg) | 1360.4 | 197.6 | 1052.2 | **317.8** | 211.2 | 130.4 | 195.0 | 769.8 | **428.0** | **413.6** |
| Input tokens ratio | 100.0% | 14.5% | 77.3% | **23.4%** | 15.5% | 9.6% | 14.3% | 56.6% | **31.5%** | **30.4%** |
| Utilized recall | N/A | 26.67% | 66.67% | **60.00%** | 13.33% | 0.00% | 6.67% | 46.67% | **0.00%** | **20.00%** |
| Relevant recall | N/A | 23.43% | 68.00% | **65.14%** | 14.86% | 0.00% | 10.86% | 48.00% | **0.00%** | **20.00%** |
| Answer correctness | 100.00% | 80.00% | 100.00% | **100.00%** | 80.00% | 20.00% | 80.00% | 60.00% | **20.00%** | **80.00%** |
| Attribution accuracy | 100.00% | 80.00% | 100.00% | **100.00%** | 80.00% | 20.00% | 80.00% | 60.00% | **20.00%** | **80.00%** |
| Grounded success rate | 100.00% | 80.00% | 100.00% | **100.00%** | 80.00% | 20.00% | 80.00% | 60.00% | **20.00%** | **80.00%** |
| Hallucination rate | 0.00% | 0.00% | 0.00% | **0.00%** | 0.00% | 0.00% | 0.00% | 0.00% | **0.00%** | **0.00%** |
| Tokens / correct grounded | 1360.4 | 213.5 | 1052.2 | **317.8** | 209.2 | 111.0 | 185.0 | 909.0 | **347.0** | **439.5** |
| Tokens / attempted success | 1360.4 | 247.0 | 1052.2 | **317.8** | 264.0 | 652.0 | 243.8 | 1283.0 | **2140.0** | **517.0** |
| Tokens / correct only | 1360.4 | 247.0 | 1052.2 | **317.8** | 264.0 | 652.0 | 243.8 | 1283.0 | **2140.0** | **517.0** |

## Global-Specific Retrieval Metrics

| Metric | BM25 global | Dense global | Hybrid global | Highway global | **Pruned global** | **Pruned global BM25 S1** |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| case_hit_rate | 100.00% | 40.00% | 100.00% | **60.00%** | **80.00%** | **100.00%** |
| doc_hit_rate | 80.00% | 20.00% | 80.00% | **60.00%** | **20.00%** | **80.00%** |
| support_sentence_recall | 13.33% | 0.00% | 6.67% | **46.67%** | **0.00%** | **20.00%** |
| distractor_selection_rate | 72.00% | 96.00% | 72.00% | **82.43%** | **89.09%** | **75.09%** |

## Token Ratio Metrics

| Metric | BM25 local | Highway local | **Pruned local** | BM25 global | Dense global | Hybrid global | Highway global | **Pruned global** | **Pruned global BM25 S1** |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| ratio_of_averages | 14.53% | 77.34% | **23.36%** | 15.52% | 9.59% | 14.33% | 56.59% | **31.46%** | **30.40%** |
| mean_of_case_ratios | 34.34% | 73.39% | **50.24%** | 36.08% | 20.74% | 32.78% | 121.41% | **78.06%** | **73.93%** |

## Poisoning & Security Gates

| Metric | Highway local | **Pruned local** | Highway global | **Pruned global** | **Pruned global BM25 S1** |
| :--- | :---: | :---: | :---: | :---: | :---: |
| Poison false validation rate | 0.00% | **0.00%** | 0.00% | **0.00%** | **0.00%** |
| Poison on initially valid | 0.00% | **0.00%** | 0.00% | **0.00%** | **0.00%** |
| Poison initially valid N | 5 | **5** | 3 | **1** | **4** |
| Poison false validation count | 0 | **0** | 0 | **0** | **0** |

## Validation Gates (POC 16.1.1)

- **duplicate_source_id**: `0` (Expected: 0)
- **support_key_mapping_accuracy**: `100.0%` (Expected: 100.0%)

## POC 16.5 / 16.6 — Diagnostic Gates

**Run size**: `smoke` (5 cases)

| Gate | Value | Target | Status |
| :--- | :---: | :---: | :---: |
| grounded_success_ge_88 | 100.00 | 88.00 | ✅ PASS |
| avg_tokens_le_500 | 317.80 | 500.00 | ✅ PASS |
| utilized_recall_ge_bm25 | 60.00 | 26.67 | ✅ PASS |
| tokens_per_attempted_success_le_600 | 317.80 | 600.00 | ✅ PASS |
| poison_initially_valid_zero | 0.00 | 0.00 | ✅ PASS |
| global_grounded_success_ge_85 | 20.00 | 85.00 | ❌ FAIL |
| global_avg_tokens_le_500 | 428.00 | 500.00 | ✅ PASS |
| global_case_hit_rate_ge_92 | 80.00 | 92.00 | ❌ FAIL |
| global_distractor_rate_le_50 | 89.09 | 50.00 | ❌ FAIL |
| global_bm25_stage1_grounded_success_ge_70 | 80.00 | 70.00 | ✅ PASS |

## Document Aggregation Strategy Sweep (Stage 1 Retrieval)

| Stage 1 | Aggregation Strategy | Case Hit Rate | Doc Hit Rate | Support Sentence Recall | Distractor Selection Rate |
| :--- | :--- | :---: | :---: | :---: | :---: |
| HYBRID | `sum_score` | 80.00% | 20.00% | 0.00% | 89.09% |
| HYBRID | `max_score` | 100.00% | 60.00% | 20.00% | 75.27% |
| HYBRID | `top3_avg_score` | 100.00% | 80.00% | 6.67% | 79.27% |
| HYBRID | `bm25_doc_score + max_sentence_score` | 100.00% | 80.00% | 20.00% | 75.09% |
| HYBRID | `bm25_doc_score + top3_sentence_score` | 100.00% | 80.00% | 20.00% | 75.09% |
| BM25 | `sum_score` | 100.00% | 40.00% | 0.00% | 82.00% |
| BM25 | `max_score` | 100.00% | 80.00% | 20.00% | 67.67% |
| BM25 | `top3_avg_score` | 100.00% | 80.00% | 20.00% | 72.00% |
| BM25 | `bm25_doc_score + max_sentence_score` | 100.00% | 80.00% | 20.00% | 75.09% |
| BM25 | `bm25_doc_score + top3_sentence_score` | 100.00% | 80.00% | 20.00% | 70.00% |

## Files Written

*   **Metrics JSON**: `artifacts/runs/ragbench_ministress_debug/metrics.json`
*   **Records JSONL**: `artifacts/runs/ragbench_ministress_debug/records.jsonl`
