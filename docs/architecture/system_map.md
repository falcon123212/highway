# Highway System Map

```text
workload generation
        |
        v
data/corpus_poc2 documents + data/workloads
        |
        v
ingestion: documents -> legacy index or out-of-core index
        |
        v
storage: in-memory index or mmap embeddings + SQLite postings + lazy block offsets
        |
        v
retrieval: query parser -> search router(auto) -> evidence resolver -> IR builder
        |
        v
runtime foundations: ContextAdapter -> HighwayContextEngine -> ContextPack
        |
        v
runtime execution: scheduler -> cache manager -> compiler/verifier -> optional LLMRuntime
        |
        v
kernels: comparison, aggregation, fact lookup, claim verification
        |
        v
benchmarks: result JSONL -> no-leak gated report -> validated metrics
```

## Main Package Boundaries

- `highway.workloads`: creates adversarial and no-leak workloads.
- `highway.ingestion`: indexes corpus documents.
- `highway.storage`: writes and reads optional local out-of-core indexes.
- `highway.retrieval`: parses questions and selects evidence.
- `highway.runtime`: orchestrates cache, retrieval, kernels, and LLM fallback.
- `highway.kernels`: deterministic CPU paths for computable structured queries.
- `highway.benchmarks`: evaluates runs and writes reports.

## LLM Runtime Direction

The next public runtime layer is `HighwayContextEngine`. It should produce an audited `ContextPack` before any LLM call. The planned flow is:

```text
user turn + session state
        |
        v
ContextAdapter -> strategy decision
        |
        v
HighwayContextEngine -> retrieval, evidence resolution, residency metrics
        |
        v
ContextPack -> citations, blocks, metrics, warnings
        |
        v
HighwayLLMRuntime -> optional prompt/client layer
```

The LLM consumes the `ContextPack`; it does not select raw corpus data directly.

## Public Commands

The root wrappers are the stable command interface:

```text
build_poc234_kernel_hardening_workload.py
run_poc234_kernel_hardening.py
eval_pccc_benchmark.py
```
