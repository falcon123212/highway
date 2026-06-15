# Hardware-Aware Context Runtime

Highway now has an optional local out-of-core retrieval path. The goal is to move toward a context-rendering runtime: the system should not assume that every block, embedding, and candidate can live in RAM.

## Runtime Shape

```text
documents
   |
   v
out-of-core ingestion
   |
   +-- blocks.jsonl          append-only block payloads
   +-- block_offsets.json    direct byte offsets for lazy fetch
   +-- embeddings.npy        mmap-backed dense vectors
   +-- postings.sqlite       lexical/entity postings
   +-- vector_index*.json    optional ANN metadata
   +-- *.faiss               optional Faiss candidate indexes
   +-- manifest.json         layout contract
   |
   v
SearchRouter(auto)
   |
   v
OutOfCoreIndex
   |
   +-- SQLite postings -> exact pruned candidates
   +-- VectorCandidateIndex -> ANN candidates
   +-- mmap rerank -> exact candidate scoring
   +-- byte-offset fetch -> lazy block materialization
   |
   v
ResidencyManager -> ContextPack metrics
   |
   v
HighwayContextEngine -> optional HighwayLLMRuntime
```

## Hardware Shortcuts

- Dense vectors are opened with `np.load(..., mmap_mode="r")`, so the array is file-backed and can be sliced without eagerly loading the entire `.npy`.
- Blocks are fetched by byte offset only after ranking, so top-k retrieval does not materialize the full corpus.
- SQLite postings hold lexical/entity terms on disk and are used to seed candidates before block text is read.
- Optional `VectorCandidateIndex` backends can seed dense candidates with `numpy_flat`, `faiss_flat`, `faiss_hnsw`, or `faiss_ivf_flat`.
- ANN candidates are never trusted directly: Highway reranks them against the mmap embeddings before fetching blocks.
- `HardwareBudget` caps candidate count, index scan window, context tokens, and resident bytes.
- `ResidencyManager` records bytes read, blocks materialized, hotset hits, evictions, and observed resident memory.
- Planned `HighwayContextEngine` will make retrieval usable without an LLM by returning an audited `ContextPack` containing selected blocks, source attribution, strategy decisions, and hardware metrics.
- Planned `HighwayLLMRuntime` will consume only `ContextPack` objects. The LLM will not directly drive corpus selection.

## Token Throughput And Savings

The next runtime layer must measure the cost avoided by not sending full context to the model. Highway should report token economics alongside storage metrics:

- `baseline_input_tokens`: naive/full-context prompt estimate.
- `actual_input_tokens`: tokens in the compiled context pack.
- `avoided_input_tokens`: baseline minus actual.
- `output_tokens`: generated or fake-client output tokens.
- `ttft_ms`: time to first token.
- `decode_ms`: generation time after first token.
- `input_tokens_per_second`: prefill throughput estimate.
- `output_tokens_per_second`: decode throughput.
- `kv_bytes_estimated`: estimated KV memory for the actual prompt.
- `kv_bytes_avoided_estimated`: estimated KV memory avoided by context reduction.
- `cost_avoided_estimated_usd`: configurable price-model savings.

KV estimate V1:

```text
kv_bytes = input_tokens * layers * hidden_size * 2 * bytes_per_element
```

The default `bytes_per_element` is `2` for FP16/BF16. If model shape is unknown, KV estimates should be reported as unavailable with a warning instead of guessed silently.

## Compatibility

The legacy index remains valid:

```text
data/corpus_poc2/index
```

The out-of-core index is additive:

```text
data/corpus_poc2/index_ooc
```

`SearchRouter(storage_mode="auto")` uses the out-of-core path only when it finds a `highway_out_of_core_v1` manifest. Otherwise it falls back to the legacy in-memory path.

Faiss is optional. If Faiss or the ANN file is missing, ANN strategies report `ann_available=false` and fall back to exact mmap/pruned retrieval without breaking `SearchRouter`.

## Showcase Commands

```powershell
$env:PYTHONPATH = "src"
python -m highway.ingestion.ingest --corpus-dir data/corpus_poc2 --output-dir data/corpus_poc2/index_ooc --layout out_of_core
python -m highway.ingestion.ingest --corpus-dir data/corpus_poc2 --output-dir data/corpus_poc2/index_ooc --layout out_of_core --vector-backend faiss_hnsw
python build_poc234_kernel_hardening_workload.py --corpus data/corpus_poc2/index_ooc --output artifacts/runs/poc_2_3_4_ooc/workload_20.jsonl --n-comparison 10 --n-aggregation 10 --seed 42
python run_poc234_kernel_hardening.py --run-name poc_2_3_4_ooc_smoke --corpus data/corpus_poc2/index_ooc --workload artifacts/runs/poc_2_3_4_ooc/workload_20.jsonl --output artifacts/runs/poc_2_3_4_ooc/results_smoke.jsonl --summary artifacts/runs/poc_2_3_4_ooc/summary_smoke.md
python run_ooc_scaleup_benchmark.py --sizes 1000,10000,100000 --queries 20 --seed 42 --strategy all --ann-backends faiss_flat,faiss_hnsw --mixed-query-set marker,entity,semantic --output-dir artifacts/runs/ooc_ann_scaleup
```

## Current Boundary

This is local out-of-core retrieval, not distributed petabyte execution yet. It establishes the runtime contract needed for later object storage, shard scheduling, prefetch policy, token/KV economics, and real KV/VRAM residency control.
