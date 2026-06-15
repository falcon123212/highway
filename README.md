# Highway

[![GitHub Repository](https://img.shields.io/badge/GitHub-Repository-blue?logo=github)](https://github.com/falcon123212/highway)

**Highway** is a context optimization runtime that reduces the cost and latency of running large language models (LLMs) on long documents. 

By filtering, pruning, and packing long documents into compact prompts before they are sent to the LLM, Highway makes AI applications faster, cheaper, and more reliable.

Highway is inspired by the systems design of offline rendering pipelines: instead of processing an entire scene naively, it builds a minimal working set, manages data residency, and spends computation only where it improves the final output. In Highway, the “scene” is a large text corpus, the “working set” is a compact evidence bundle, and the expensive computation is LLM prefill and generation.

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

## 🎨 Inspiration from Offline Rendering

Highway is partly inspired by systems used in offline rendering and production graphics pipelines.

This does not mean that Highway directly applies rendering algorithms to language models.
The analogy is more specific: offline renderers are designed to process extremely large scenes that cannot be loaded or evaluated naively at once. They rely on acceleration structures, working-set management, out-of-core storage, caching, scheduling, and quality/cost trade-offs.

Highway explores whether similar systems principles can be useful for long-context LLM execution.

In offline rendering, a renderer does not evaluate every triangle, texture, light path, or asset with equal priority for every pixel. It uses spatial structures, visibility, sampling, caching, and level-of-detail strategies to focus computation on the parts of the scene that matter.

In Highway, the equivalent problem is context execution:

> The model should not receive every document, block, or sentence with equal priority.
> The runtime should identify the minimal evidence set required for the current query and compile it into a compact, auditable context.

The relevant inspiration is therefore not visual rendering itself, but the underlying resource-management philosophy.

### Relevant Parallels

| Offline Rendering Concept                      | Highway Equivalent                                             |
| ---------------------------------------------- | -------------------------------------------------------------- |
| Scene too large to fit fully in active memory  | Corpus too large to send fully to the LLM                      |
| Out-of-core geometry and texture streaming     | Out-of-core text, embeddings, and source offsets               |
| Acceleration structures such as BVH            | Retrieval indices, postings, embeddings, entity/marker filters |
| Working set for a camera ray or tile           | Working set for a user query                                   |
| Visibility and relevance filtering             | Evidence selection and sentence packing                        |
| Level of detail                                | Context compression and token-budgeted packing                 |
| Cache residency                                | Hot/warm/cold context and block residency                      |
| Prefetching likely-needed data                 | Lazy fetch and candidate block loading                         |
| Sampling trade-offs                            | Recall/latency/token-budget trade-offs                         |
| Denoising / reconstruction from sparse samples | Answer generation from compact evidence                        |

The central idea is similar:

> Do not process the entire world.
> Build the smallest useful working set, then spend computation where it matters.

### What Transfers Well

The following offline-rendering ideas appear relevant to Highway:

1. **Out-of-core execution**
   Large scenes are often streamed from disk or memory-mapped storage. Highway applies a similar idea to large corpora through memory-mapped embeddings, SQLite postings, and lazy text fetches.

2. **Acceleration before expensive computation**
   Renderers use acceleration structures to avoid brute-force intersection. Highway uses retrieval, filtering, and packing before invoking the LLM, because LLM prefill is the expensive step.

3. **Working-set construction**
   Rendering systems build a small active set of geometry, textures, and lights for a specific view or path. Highway builds a compact evidence set for a specific query.

4. **Residency management**
   Production renderers decide what data should stay hot, what can be evicted, and what should be fetched later. Highway’s `ResidencyManager` explores the same type of policy for text blocks, embeddings, and session context.

5. **Cost-aware quality control**
   Rendering constantly balances quality, time, memory, and sampling budget. Highway similarly balances groundedness, recall, latency, token count, and KV-cache pressure.

6. **Deterministic computation where possible**
   Renderers rely on deterministic kernels for geometry, shading, filtering, and sampling. Highway similarly routes structured operations such as comparison, aggregation, filtering, and simple fact verification to CPU-side kernels when an LLM call is unnecessary.

### What Does Not Transfer Directly

The analogy has limits.

Language is not geometry.
A relevant sentence is not equivalent to a visible triangle.
Semantic relevance is noisier, more ambiguous, and more context-dependent than spatial visibility.

Several rendering ideas do not transfer cleanly:

* spatial locality does not always map to semantic locality;
* a document’s position in a corpus does not imply usefulness;
* semantic recall cannot be solved by geometry-style visibility alone;
* missing one small sentence can break factual correctness;
* unlike rendering, there is no objective pixel-level ground truth for many language tasks;
* LLM generation introduces uncertainty after context selection.

For this reason, Highway should not be described as “ray tracing for LLMs” or as a direct adaptation of offline rendering.

A more accurate description is:

> Highway borrows the systems mindset of offline rendering — out-of-core execution, acceleration structures, working-set construction, residency management, and cost-aware computation — and applies it to long-context LLM pipelines.

### Why This Matters

Long-context LLM execution is becoming a systems problem.

The bottleneck is not only model intelligence.
It is also how much context is loaded, how it is selected, how it is verified, how much memory it consumes, and whether expensive model calls are used only when necessary.

Offline rendering is a useful reference because it has spent decades solving a similar class of problem:

> How do you produce a high-quality result from a dataset too large to process naively?

Highway applies that question to LLM context:

> How do you produce a grounded answer from a corpus too large to fit efficiently into the prompt?

This is the design space Highway is exploring.

---

## ⚡ Why Compute Still Matters

Highway does not remove the need for compute.

Large language models remain expensive to run, especially at scale. The cost of inference is driven by several factors:

* model size,
* prompt length,
* prefill latency,
* KV-cache memory,
* output length,
* batching strategy,
* hardware architecture,
* serving runtime,
* and concurrency requirements.

Highway focuses on one specific part of this problem:

> Reducing the amount of unnecessary context that reaches the model.

This matters because long-context inference is not free. Every extra token in the prompt must be processed during prefill and contributes to KV-cache memory pressure. In high-concurrency environments, this can reduce throughput, increase latency, and require more GPU memory per active request.

Highway should therefore be understood as a compute-allocation layer, not a compute replacement.

The objective is to spend expensive model compute only on the evidence that is likely to matter.

---

### Where Compute Is Used in a Highway Pipeline

A Highway-style system still requires compute at multiple stages:

| Stage            | Compute Type            | Purpose                                                      |
| ---------------- | ----------------------- | ------------------------------------------------------------ |
| Ingestion        | CPU / optional GPU      | Parse documents, split blocks, create offsets, build indices |
| Embedding        | GPU or CPU              | Convert text blocks into vector representations              |
| Retrieval        | CPU / memory / disk I/O | Find candidate blocks or documents                           |
| Reranking        | CPU or GPU              | Improve candidate ordering using stronger models             |
| Sentence packing | CPU                     | Select, filter, rescue, and compress evidence                |
| Verification     | CPU / optional LLM      | Check answer contracts, source coverage, and grounding       |
| Generation       | GPU / accelerator       | Run the LLM on the final compact context                     |

Highway does not make these stages disappear.

Instead, it tries to move work away from the most expensive part of the stack — long LLM prefill and large KV-cache residency — into cheaper and more controllable stages such as retrieval, filtering, deterministic kernels, and compact context assembly.

---

### Prefill vs Decode

Highway primarily targets the input side of inference.

For many long-document workloads, the model spends a significant amount of time processing the prompt before producing the first output token. This is the prefill phase.

If a query sends thousands or tens of thousands of unnecessary tokens to the model, the system pays for them even if the answer only depends on a few sentences.

Highway attempts to reduce this cost by compiling a smaller `ContextPack`.

This can help reduce:

* prefill latency,
* prompt-side memory usage,
* KV-cache growth,
* active request memory,
* and serving pressure under concurrency.

However, Highway does not directly reduce the cost of generating many output tokens. If a workload is decode-bound because the model must generate a very long answer, context compression alone will not solve the full compute problem.

---

### KV-Cache Pressure

For transformer inference, long prompts increase KV-cache memory usage.

This matters because KV-cache memory can become a limiting factor for serving multiple users or long sessions. Even if the model weights fit on the GPU, the active context for many concurrent requests can consume substantial memory.

Highway’s approach is to avoid placing irrelevant evidence into the model context in the first place.

The intended effect is:

```text
Less irrelevant prompt context
        ↓
Lower prefill work
        ↓
Lower KV-cache pressure
        ↓
Higher possible concurrency
        ↓
Lower serving cost per useful answer
```

This is especially relevant for enterprise search, codebase QA, legal review, support logs, and long-document analysis, where the raw corpus is much larger than the actual evidence required for a given answer.

---

### Why Not Just Use a Longer-Context Model?

Long-context models are useful, but they do not eliminate the need for context management.

A larger context window allows more text to be provided to the model. It does not guarantee that:

* the model will focus on the right evidence,
* irrelevant text will not distract the answer,
* source grounding will be correct,
* latency will remain acceptable,
* serving concurrency will remain economical,
* or the system will be auditable.

Highway is based on the assumption that long context and context optimization are complementary.

A stronger long-context model can still benefit from receiving a smaller, cleaner, source-grounded context.

The point is not:

> “Do not use long-context models.”

The point is:

> “Do not waste long-context capacity on avoidable distractors.”

---

### Compute Trade-Offs

Highway introduces its own compute overhead.

Retrieval, packing, reranking, verification, and indexing all cost something. The system is only useful when this overhead is lower than the cost of sending excessive context to the LLM, or when the additional verification improves reliability enough to justify the cost.

This creates a trade-off:

```text
Extra CPU / retrieval / reranking work
        vs
Avoided LLM prefill / KV-cache / hallucination risk
```

Highway is most likely to be useful when:

* the corpus is large;
* the useful evidence is sparse;
* source grounding matters;
* many concurrent users query the system;
* the model context would otherwise contain many distractors;
* prefill latency or GPU memory is a bottleneck;
* and deterministic checks can replace some LLM calls.

Highway is less likely to help when:

* the input is already short;
* the full document is genuinely required;
* output generation dominates total latency;
* the retriever fails to find relevant candidates;
* the use case does not require citations or grounding;
* or the overhead of retrieval and packing exceeds the savings from compression.

---

### Practical Implication

Highway is not a claim that compute is no longer needed.

It is a claim that long-context LLM systems need better compute discipline.

Instead of treating the LLM as the first component that sees everything, Highway treats the LLM as the final expensive stage in a larger runtime pipeline.

The design goal is:

> Use cheap deterministic compute to reduce expensive probabilistic compute.

Or more concretely:

> Use indexing, retrieval, filtering, packing, and verification to reduce the amount of GPU inference spent on irrelevant context.

This is the core systems argument behind Highway.

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

