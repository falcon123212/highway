# Highway

[![GitHub Repository](https://img.shields.io/badge/GitHub-Repository-blue?logo=github)](https://github.com/falcon123212/highway)

**Highway** is a context optimization runtime that reduces the cost and latency of running large language models (LLMs) on long documents. 

By filtering, pruning, and packing long documents into compact prompts before they are sent to the LLM, Highway makes AI applications faster, cheaper, and more reliable.

---

## 🔗 Useful Links

- **Repository**: [https://github.com/falcon123212/highway](https://github.com/falcon123212/highway)
- **Detailed Test Results & Benchmarks**: [docs/validated_results.md](docs/validated_results.md)
- **Technical Architecture Guide**: [docs/architecture/overview.md](docs/architecture/overview.md)
- **Memory & Storage Optimization Guide**: [docs/architecture/hardware_context_runtime.md](docs/architecture/hardware_context_runtime.md)
- **System Roadmap & Timeline**: [docs/roadmap_to_llm_runtime.md](docs/roadmap_to_llm_runtime.md)

---

## 💡 What Problem Does Highway Solve?

When you ask an LLM questions about very long files (like codebases, financial sheets, or legal documents):
1. **High VRAM Costs**: Sending raw documents consumes massive GPU memory (KV-cache pressure), which limits how many users can query the model at the same time.
2. **Slow Responses**: The model takes several seconds just to read the long prompt (Prefill Latency / TTFT).
3. **Hallucinations**: LLMs often get distracted by irrelevant text inside long documents ("Lost in the Middle").

Highway solves this by **decoupling retrieval from generation**: it identifies the exact sentences needed to answer the query, verifies their safety, and packs them into a tiny, high-density context (typically 512 tokens), leaving out all the distractor text.

---

## ⚡ Key Results at a Glance (POC 16.6.2)

On a benchmark sweep of 100 cases using real-world RAG datasets (RAGBench):
- **Cost Savings**: Prompts are compressed from 1,744.4 tokens to **460.7 tokens** (a **73.6% reduction**).
- **GPU Memory Avoided**: Active memory usage for processing prompts drops by **~78%**, allowing 4x higher concurrency.
- **Latency Speedup**: Response prefill time (TTFT) drops by **74%** (from 21.8 ms to 5.7 ms).
- **Improved Accuracy**: Grounded Success Rate is boosted from **44% to 63%** (a **+19.00% absolute increase**) compared to standard document retrieval by focusing only on the highest-relevance sentences.
- **Poison Proof**: The context assembler has a **0.00% false validation rate** on poisoned/malicious prompts.

---

## 🛠️ How It Works (Simplified)

```text
  [Raw Corpus / Codebase] ──> [Ingestion & Indexing (mmap embeddings + SQLite)]
                                    │
                                    ▼
       [Query] ──> [Stage 1: Retrieve Relevant Documents (BM25/Hybrid)]
                                    │
                                    ▼
           [Stage 2: Pack & Prune Sentences (Support Rescue & Filtering)]
                                    │
                                    ▼
                     [Compiled Compact ContextPack (512 tokens)]
                                    │
                                    ▼
                           [LLM or CPU Kernel] ──> [Correct Answer]
```

1. **Ingest**: Documents are broken into searchable blocks. Embeddings are mapped directly to disk (`mmap`), consuming minimal active RAM.
2. **Stage 1 (Retrieve)**: The query is parsed to identify candidate documents.
3. **Stage 2 (Pack & Prune)**: An intelligent assembler extracts only the critical sentences, rescues missing factual terms (dates, numbers, entities), expands neighbors to maintain readability, and drops distractor text.
4. **Answer**: The compact context is sent to the LLM, or resolved instantly via local CPU kernels if the answer is computable.

---

## 📂 Project Layout

- [src/highway/](file:///c:/Users/nicol/Documents/Highway/src/highway/): Core source code (runtime, retrieval, database storage, and evaluation benchmarks).
- [tests/](file:///c:/Users/nicol/Documents/Highway/tests/): Automated unit tests verifying program correctness.
- [data/](file:///c:/Users/nicol/Documents/Highway/data/): Corpus documents, QA data, and index offsets.
- [docs/](file:///c:/Users/nicol/Documents/Highway/docs/): Architectural guides, roadmaps, and validated results.
- [artifacts/runs/](file:///c:/Users/nicol/Documents/Highway/artifacts/runs/): Historical test reports, metric sweeps, and execution logs of all benchmark runs (preserves `report.md`, `metrics.json`, `records.jsonl`, `*.csv` configurations; heavy `.npy` embeddings/`.sqlite` indexes are ignored).


---

## 🚀 Running Verification Tests

To verify that the local installation is working correctly, run the python compiler and the regression tests:

```powershell
# 1. Run all unit tests
pytest tests -q

# 2. Compile all source files to verify syntax
python -m compileall src tests
```

