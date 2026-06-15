# Highway: LLM Context-Rendering Architecture Deep-Dive

This document provides a detailed technical overview of **Highway**, a hardware-aware, cost-optimizing context runtime. It is designed for LLM/RAG engineers who want to understand the underlying retrieval, storage, and context-packing mechanics.

---

## 1. Architectural Staging Model

Highway decouples context preparation from the LLM prompt-execution path. Rather than feeding raw retrieved documents directly into the LLM context window, Highway implements a **two-stage staging pipeline**:

```text
  [Raw Corpus Files] ──> [Ingestion: Embeddings (.npy) + Lexical Indices (.sqlite)]
                                      │
                                      ▼
                        [Stage 1: Document Retrieval]
              (Retrieves Top-M documents using BM25 / Vector Search)
                                      │
                                      ▼
                        [Stage 2: Sentence-Level Packing]
             (Anti-Distractor Filtering + Support Rescue + Expansion)
                                      │
                                      ▼
                  [Compiled ContextPack (Audited & Safe)]
                                      │
                 ┌────────────────────┴────────────────────┐
                 ▼                                         ▼
      [Deterministic CPU Kernels]                  [Highway LLM Runtime]
     (Resolves G/H exact queries)                (Invokes local/remote LLM)
```

---

## 2. Stage 2: Greedy Sentence-Level Packing Algorithms (V2)

The core innovation in Highway is Stage 2, which greedily selects and packs individual sentences under a strict token budget (e.g. 512 tokens). It employs three primary algorithms to guarantee factual density, security, and context readability:

### A. Anti-Distractor Filter (v1)
To prevent hostile prompt injection or noisy distractors from leaking into the prompt:
1. **Relevance Threshold**: Candidate sentences are rejected before final greedy packing if they have **zero overlap** with query tokens/entities AND their individual relevance score is $< 0.015$.
2. **Document Sentence Capping**: To prevent any single distractor document from monopolizing the token budget, greedy packing caps the number of sentences selected from a single document to **$C \le 4$**.

### B. Support Rescue (v2)
When the initial greedy packing pass is complete, Highway audits the selected context. If the query contains critical elements that are missing from the packed text, it triggers **Support Rescue**:
1. **Critical Elements Parsing**: Highway extracts 7 classes of query elements:
   - Numbers (e.g., `42`, `1,250`)
   - Dates (e.g., `2026-06-15`)
   - Acronyms (e.g., `TTFT`, `OOC`, `VRAM`)
   - Capitalized named entities
   - Rare query terms (low corpus frequency)
   - Quoted terms
   - Hyphenated terms
2. **Weighted Overlap Scoring**: Missing elements are mapped back to all sentences in the Stage 1 retrieved documents. Candidate sentences are scored using a weighted overlap formula:
   $$\text{Score}(S) = \sum_{e \in \text{Missing}} w(e) \cdot \mathbb{I}(e \in S)$$
3. **Rescue Injection**: The highest-scoring candidate sentences are injected into the remaining token budget to restore factual coverage.

### C. Neighborhood Expansion (v2)
To restore semantic readability and cohesion for short or index-pruned sentences:
1. **Short-Sentence Rule**: If a selected sentence is $< 10$ tokens, Highway automatically fetches its predecessor and successor sentences from the source document.
2. **Anaphora & Connector Check**: If a sentence contains pronoun references (`this`, `these`, `they`, `it`, `such`) or starts with structural connectors (`however`, `therefore`, `additionally`, `furthermore`), adjacent neighbor sentences are pulled in (capped at **2 neighbors** per sentence) to prevent the LLM from losing coreference resolution.

---

## 3. Storage Layer: Out-of-Core (OOC) Memory Optimization

To scale context-rendering to massive corpora without crashing server RAM, Highway implements an Out-of-Core (OOC) storage layout:
- **Memory-Mapped Dense Vectors**: Embeddings are stored in standard NumPy files and opened via `np.load(..., mmap_mode="r")`. The vectors are file-backed; slices are paged into memory on-demand during ranking.
- **SQLite Lexical Index**: Term postings, block offsets, and document-to-block mapping metadata are queried on-disk via SQLite.
- **Lazy Block Offset Fetch**: Block text payloads are stored in an append-only JSONL file. Highway ranks and filters candidate blocks using *only* embeddings and SQLite postings. The actual text is eagerly read from disk (using direct byte offsets) *only* for the final Top-$K$ candidate blocks, ensuring minimal active RAM footprints.

---

## 4. Deterministic CPU Kernels

When a structured query contract is met (e.g. exact factual match, numeric summation, or claim verification), Highway bypasses LLM inference entirely:
- **Comparison Kernel**: Compares retrieved candidate facts against a structured contract.
- **Aggregation Kernel**: Sums, averages, or aggregates fact values on CPU.
- **Verifier Kernel**: Audits the result. This delivers microsecond-level p95 latencies and **0.0% LLM serving costs** for structured tasks.

---

## 5. Main Package Boundaries

- `highway.storage`: Writes and reads local memory-mapped OOC indexes.
- `highway.retrieval`: Parses queries, routes searches, and retrieves documents.
- `highway.runtime`: Orchestrates the cache, plans strategies, and compiles `ContextPack` payloads.
- `highway.kernels`: Houses the deterministic CPU execution paths.
- `highway.benchmarks`: Houses the evaluation harnesses (RAGBench, SWE-bench, Ollama).

---

## 6. Historical Test & Validation Tracks (Preserved for Audits)

All historical test runs and evaluation reports are fully preserved under `artifacts/runs/` to maintain a chronological ledger of validation gates:

1. **Deterministic No-Leak Hardening (`poc_2_3_4_no_leak`)**: Validates exact G/H execution under adversarial queries.
2. **OOC Performance Scaling (`ooc_scaleup`)**: Verifies p95 latency and memory scaling up to 100k blocks.
3. **Faiss HNSW Vector Acceleration (`ooc_ann_scaleup_faiss`)**: Measures vector indexing speedups (up to 31.3x).
4. **Local Ollama Benchmark (`local_llm_quality`)**: Tests prompt wiring and Streaming/TTFT metrics on local models.
5. **Endurance Conversations (`long_conversation_quality_qwen3_8b_25t`)**: Audits 25-turn active entity context adaptation.
6. **RAGBench Mini Stress (`ragbench_ministress_poc_16_6_2_fullmini`)**: Validates Document Aggregation Strategies (`top3_avg_score` vs `sum_score`) and Stage 2 sentence-packing budget sweeps.
7. **SWE-bench Verified Localization (`swebench_verified_fileloc_100`)**: Evaluates Highway's ability to localize code files under SWE-bench issues.

