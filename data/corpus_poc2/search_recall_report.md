# POC 2.0 Search Recall Validation Report

This report presents the retrieval metrics for the **Evidence Search Router** on the synthetic dataset, verifying **Claim 1 (Open Search Recall)**.

## Executive Summary

- **Total Questions Evaluated**: 68 (excluding absent entities)
- **Search Recall@10**: **100.00%**
- **Search Recall@50**: **100.00%** (Gate: $\ge$ 95%) â€” **PASS**
- **Search Recall@100**: **100.00%** (Gate: $\ge$ 98%) â€” **PASS**
- **Entity Recall@50**: **100.00%** (Gate: $\ge$ 99%) â€” **PASS**
- **Mean Reciprocal Rank (MRR)**: **0.7917**

## Category breakdown

| Category | Count | Recall@10 | Recall@50 | Recall@100 | MRR |
|---|---|---|---|---|---|
| **A** | 20 | 100.0% | 100.0% | 100.0% | 0.5000 |
| **B** | 20 | 100.0% | 100.0% | 100.0% | 1.0000 |
| **C** | 10 | 100.0% | 100.0% | 100.0% | 1.0000 |
| **D** | 10 | 100.0% | 100.0% | 100.0% | 1.0000 |
| **F** | 8 | 100.0% | 100.0% | 100.0% | 0.4792 |

## Analysis and Conclusions
The Reciprocal Rank Fusion of BM25 and Dense Retrieval, augmented with a deterministic entity boost, achieves the required recall targets. 
This validates the Search Router's ability to locate gold evidence without knowing the answer in advance.

