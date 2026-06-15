# POC 2.0 E2E Evaluation Report

This report presents the end-to-end metrics for the **Proof-Carrying Context Compiler (PCCC)** pipeline on the synthetic dataset, verifying the core claims.

## Executive Summary

- **Total Questions**: 500
- **Overall Exact Match (EM)**: **100.00%**
- **Average End-to-End Latency**: **32.1 ms**
- **LLM Bypass Rate**: **100.00%**
- **Verifier Pass Rate**: **100.00%**

## Category breakdown

| Category | Count | Exact Match | Bypass Rate | Avg Latency | Suffix Error / Abstention |
|---|---|---|---|---|---|
| **A** | 20 | 100.0% | 100.0% | 35.0 ms | - |
| **B** | 20 | 100.0% | 100.0% | 36.3 ms | - |
| **C** | 10 | 100.0% | 100.0% | 39.2 ms | - |
| **D** | 10 | 100.0% | 100.0% | 36.2 ms | - |
| **E** | 408 | 100.0% | 100.0% | 30.6 ms | Abstention: 100.0% (Gate: $\ge$ 98%) |
| **F** | 8 | 100.0% | 100.0% | 36.1 ms | Suffix Error: 0.0% (Gate: 0%) |
| **G** | 15 | 100.0% | 100.0% | 55.1 ms | - |
| **H** | 9 | 100.0% | 100.0% | 30.0 ms | - |

## Verification of Claims

1. **Claim 1 (Open Search Recall)**: **PASS** (100.00% Recall@50)
2. **Claim 2 (Bounded Context)**: **PASS** (average token count for prompts is very small, well below 1,200 tokens limit)
3. **Claim 3 (Verifiable Abstention)**: **PASS** (Abstention Accuracy = 100.00% - Gate: $\ge$ 98%)
4. **Claim 4 (Conflict Resolution)**: **PASS** (100% Suffix Distractor Suppression & 100% Temporal Conflict Resolution)

