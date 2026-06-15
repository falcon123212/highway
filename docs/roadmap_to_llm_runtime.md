# Highway Roadmap To LLM Runtime

Highway is now a serious research prototype, but it should not be presented as a production LLM runtime yet. The validated core is the storage/retrieval/performance foundation: no-leak workloads, local out-of-core indexing, optional ANN/Faiss acceleration, lazy block fetch, and reproducible reports.

The next objective is to build the runtime foundations before letting a real LLM drive anything. The LLM should consume an audited `ContextPack`; it should never select raw corpus data directly.

## Current State: 65% Done

- No-leak workload and reporting are validating: `100%` no-leak and `100%` G/H exact match on the reference smoke.
- Out-of-core storage exists: mmap embeddings, SQLite postings, byte-offset lazy fetch, and manifest-based compatibility.
- ANN/Faiss is optional and working: `faiss_hnsw` is available locally while default runtime behavior still works without Faiss.
- Hardware metrics are measured: latency, rows scanned, bytes read, blocks materialized, resident memory, ANN usage, and fallback reasons.
- `HighwayContextEngine` now produces a serializable `ContextPack` without any LLM call.
- `ContextAdapter` now provides a first-pass session-aware strategy plan for marker, entity, semantic, and follow-up turns.
- `TokenEconomics` now reports baseline input tokens, actual context tokens, avoided tokens, estimated KV bytes, cost estimates, and throughput fields without requiring a real LLM.
- A quality/token tradeoff smoke now compares full-context reflective answers against Highway context answers, with answer quality and token savings measured in the same record.
- `HighwayLLMRuntime` now has a fake-client path: `ContextPack -> prompt -> DeterministicReflectiveClient -> token economics`.
- A local Ollama benchmark path now exists for real-model smoke runs and quality/coherence measurement. It remains optional and reports `SKIPPED` when Ollama or the requested model is absent.
- A real-time answer guard now exists for long conversations: `AnswerContractCompiler -> AnswerVerifier` controls answer type, allowed sources, hallucination checks, and output-token budgets. Prio 11 now passes the compiled budget to Ollama and can retry one compact answer without a second retrieval.
- The structured exact runtime path now has a dedicated perf-margin run under `artifacts/runs/runtime_perf_margin/`.
- The semantic ANN path now has dedicated quality runs under `artifacts/runs/semantic_ann_quality/` and `artifacts/runs/semantic_real_embedder_full/`; Prio 4 added lexical rescue, Prio 5 added a deterministic local reranker, Prio 6 added field-posting rescue, Prio 7 validated the optional CrossEncoder load/cache path, and Prio 8 added a real local BGE embedder path. The strict `100k` semantic path is now `VALIDATING`.
- Current ANN/Faiss run: `artifacts/runs/ooc_ann_scaleup_faiss/report.md`.
- Current quality/token run: `artifacts/runs/quality_token_tradeoff/report.md`.
- Current fake LLM runtime run: `artifacts/runs/llm_runtime_fake/report.md`.
- Current local LLM target: `artifacts/runs/local_llm_quality/` once an Ollama model is available.
- Current long-conversation proof: `artifacts/runs/long_conversation_quality_qwen3_8b_25t/report.md`.
- Current multi-theme long-horizon proof: `artifacts/runs/multi_theme_long_qwen3_8b_100t/report.md`, plus sampled 500-turn endurance under `artifacts/runs/multi_theme_long_qwen3_8b_500t_sampled/report.md`.
- Current external-benchmark harnesses: RAGBench exists for Headroom-like RAG checks, the RAGBench Mini Stress Test (`artifacts/runs/ragbench_ministress_poc_16_6_2_fullmini/report.md`) validates global document selection/aggregation tuning under POC 16.6.2 sweeps, and SWE-bench Verified is now the principal dev-agent benchmark path. The first SWE-bench gate is file localization from `problem_statement -> ContextPack`, measured against gold patch files at the repository `base_commit`.

Key validated signal from the ANN run:

| Tier | Full mmap p95 | Best ANN p95 | Speedup | Rows reranked | No-leak | G/H EM |
|---:|---:|---:|---:|---:|---:|---:|
| 1,000 blocks | 12.41 ms | 9.06 ms | 1.4x | 200 | 100.00% | 100.00% |
| 10,000 blocks | 51.00 ms | 10.51 ms | 4.9x | 200 | 100.00% | 100.00% |
| 100,000 blocks | 415.88 ms | 13.28 ms | 31.3x | 200 | 100.00% | 100.00% |

Current caveat: HNSW alone is still too low recall, and CrossEncoder on CPU is too slow. The real BGE embedder plus field rescue now crosses the `100k` recall gate and stays under the `200 ms` interactive target.

Current quality/token signal:

| Run | Baseline EM | Highway EM | Quality delta | Avg baseline prompt tokens | Avg Highway prompt tokens | Avoided prompt tokens |
|---|---:|---:|---:|---:|---:|---:|
| `quality_token_tradeoff` | 100.00% | 100.00% | 0.00 pp | 17,170.00 | 66.50 | 99.61% |

This is still a deterministic fake-answerer smoke, not a real LLM evaluation. Its value is that it introduces the right gate: token economy is not considered valid unless answer quality stays flat.

Current fake LLM runtime signal:

| Tier | Baseline EM | Highway EM | Quality delta | Avoided tokens | Baseline TTFT p95 | Highway TTFT p95 | Context p95 |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1,000 blocks | 100.00% | 100.00% | 0.00 pp | 99.68% | 429.02 ms | 1.43 ms | 1.97 ms |
| 10,000 blocks | 100.00% | 100.00% | 0.00 pp | 99.97% | 4,254.02 ms | 1.43 ms | 4.43 ms |
| 100,000 blocks | 100.00% | 100.00% | 0.00 pp | 100.00% | 42,504.03 ms | 1.43 ms | 14.08 ms |

This run is validating for the fake-client path. It proves the runtime contract and metric accounting. The real local LLM path now exists through Ollama, but quality claims require a measured model run.

Local LLM benchmark status:

| Model class | Purpose | Status |
|---|---|---|
| Qwen 0.5B | Runtime smoke: prompt, streaming, TTFT, tokens/s, parsing | Implemented target |
| Qwen 1.5B+ | First useful quality/coherence check | Next validating run |
| 3B+ quantized | Stronger local quality if 8 GB VRAM allows it | Later optimization |

The local LLM benchmark accepts token savings only if factual quality, source attribution, and multi-turn coherence do not regress versus baseline. The first real `qwen3:8b` micro-smoke is now `VALIDATING`: it proved the runtime path, answered the requested project identity correctly, preserved Highway source attribution, and avoided `93.02%` of input tokens. The evaluator separately records `full_exact_match=false` because the model omitted the budget, but `answer_satisfies_question=true` because the question asked which project, not the budget amount.

Long-conversation answer-guard status:

| Run | Turns | Answer OK | Source attr | Hallucination | Coherence | Input avoided | Output over budget | Prompt distinct | Status |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| `long_conversation_quality_fake_audit` | 12 | 100.00% | 100.00% | 0.00% | 100.00% | 98.09% | 0.00% | 100.00% | VALIDATING |
| `long_conversation_quality_poison` | 4 | 100.00% | 100.00% | 25.00% | 100.00% | 98.24% | 0.00% | 100.00% | NON_VALIDATING |
| `long_conversation_quality_qwen3_8b_12t` | 12 | 100.00% | 100.00% | 0.00% | 100.00% | 94.94% | 0.00% | 100.00% | VALIDATING |
| `long_conversation_quality_qwen3_8b_25t` | 25 | 100.00% | 100.00% | 0.00% | 100.00% | 94.97% | 0.00% | 100.00% | VALIDATING |
| `long_conversation_quality_qwen3_8b` | 4 | 100.00% | 100.00% | 0.00% | 100.00% | 95.46% | 50.00% | n/a | HISTORICAL NON_VALIDATING |

This is the intended split after Prio 12: Highway proves real-time context compilation, answer auditing, output-budget control, and auditable prompt separation. That audit discipline has now been moved onto SWE-bench Verified for the dev-agent path; the current blocker is code candidate generation quality, not prompt auditing.

Multi-theme long-horizon status:

| Run | Turns | LLM cadence | Answer OK | Source attr | Hallucination | Coherence | Long-range recall | Input avoided | Context p95 | Status |
|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---|
| `multi_theme_long_fake_100t` | 100 | fake every turn | 100.00% | 100.00% | 0.00% | 100.00% | 100.00% | 97.45% | 2.12 ms | VALIDATING |
| `multi_theme_long_poison_100t` | 100 | fake every turn | 100.00% | 100.00% | 17.00% | 100.00% | 100.00% | 97.71% | 1.97 ms | NON_VALIDATING |
| `multi_theme_long_qwen3_8b_100t` | 100 | qwen every turn | 100.00% | 100.00% | 0.00% | 100.00% | 100.00% | 94.96% | 2.42 ms | VALIDATING |
| `multi_theme_long_qwen3_8b_500t_sampled` | 500 | qwen every 5 turns | 100.00% | 100.00% | 0.00% | 100.00% | 100.00% | 96.95% | 2.12 ms | VALIDATING |
| `multi_theme_long_fake_1000t` | 1000 | fake every turn | 100.00% | 100.00% | 0.00% | 100.00% | 100.00% | 97.45% | 2.27 ms | VALIDATING |

Prio 13 proves that the audited prompt path scales beyond a toy conversation: multiple themes, topic switches, long-range recall, poison checks, and baseline-vs-Highway prompt hashes all stay measurable. It does not yet prove broad open-ended assistant quality. The next proof should move the same harness onto a real dev/code workload with natural tickets, docs, logs, commits, and ambiguous user turns.

RAGBench Headroom-like status:

| Run | Dataset path | Status | Answer OK | Source attr | Tokens avoided | Prompt distinct | Poison |
|---|---|---|---:|---:|---:|---:|---:|
| `highway_ragbench_offline_fake_smoke` | RAGBench-shaped offline rows | VALIDATING | 100.00% | 100.00% | 65.93% | 100.00% | n/a |
| `highway_ragbench_offline_poison_smoke` | RAGBench-shaped offline rows | NON_VALIDATING | 0.00% | 0.00% | 78.93% | 100.00% | 100.00% fail |
| `highway_ragbench_skip_smoke` | `galileo-ai/ragbench` without `datasets` extra | SKIPPED | n/a | n/a | n/a | n/a | n/a |

Prio 14 moves the Headroom comparison from synthetic-only prompts toward an external RAG protocol: `full context -> compiled ContextPack`, plus source hashes, source removal, and cost per correct grounded answer. The harness is implemented; the next operational step is installing `.[benchmark]`, caching RAGBench, and running the fake and qwen3 smoke over real configs.

RAGBench Mini Stress Test (POC 16.6.2) status:

| Run / Sweep Case | Strategy / Mode | Status | Grounded Success | Avg Input Tokens | Recall (utilized) | Poison Rate |
|---|---|---|---:|---:|---:|---:|
| `ragbench_ministress_smoke` | `highway_pruned_local` | VALIDATING | 80.00% | 312.8 | 43.13% | 0.00% |
| `ragbench_ministress_smoke` | `highway_pruned_global_bm25_stage1` | VALIDATING | 72.00% | 370.7 | 28.27% | 0.00% |
| `ragbench_ministress_poc_16_6_2_fullmini` | `highway_pruned_global_bm25_top3avg` | VALIDATING | 63.00% | 460.7 | 33.60% | 0.00% |
| `ragbench_ministress_poc_16_6_2_fullmini` | `highway_pruned_global_bm25_max` | VALIDATING | 61.00% | 471.8 | 28.27% | 0.00% |
| `ragbench_ministress_smoke` | `highway_pruned_global` | VALIDATING | 28.00% | 421.9 | 3.60% | 0.00% |

This run validates the Global Two-Stage Sentence-Packer under POC 16.6 and global aggregation tuning under POC 16.6.1/16.6.2. Switching from the naive `sum_score` document aggregation to `top3_avg_score` resolves the document routing bias, boosting Grounded Success Rate to 63.00% on a full 100-case sweep (a +19.00% absolute increase) while preserving 0.00% false validation rate on poisoned tests.

SWE-bench Verified dev-agent status:

| Stage | Goal | Status |
|---|---|---|
| SWE-0 file localization | Recover gold patch files with fewer tokens | Implemented, code-aware v2 improves but remains NON_VALIDATING |
| SWE-1 symbol localization | Recover functions/classes or hunk areas | Implemented, code-aware v2 improves but remains NON_VALIDATING |
| SWE-2 patch planning | Qwen outputs target files/symbols/edit intent/tests | Planned |
| SWE-3 patch generation | Qwen emits unified diff on a small subset | Planned |
| SWE-4 poison | Remove gold files and require NON_VALIDATING | Implemented for missing gold file |

Prio 15 changes the external benchmark priority: RAGBench is useful for RAG-style QA, but SWE-bench Verified is the stronger proof for Highway as a contextual inference runtime for development. The first real SWE run is intentionally not hidden: `swebench_verified_symbol_25` is `NON_VALIDATING` with `24.00%` file recall@5 and `24.00%` symbol recall@5 despite `99.89%` token reduction. The code-aware v2 path improves the measured `25`-issue subset to `42.00%` file recall@5 and `42.29%` symbol recall@5 while preserving `99.89%` token reduction and `100.00%` prompt distinctness. Top-10 raises hunk-area recall to `53.68%`, but file recall stays capped at `42.00%`.

Current SWE conclusion: the benchmark is auditable and the token economy is real, but the dev-agent retrieval quality is not yet sufficient. Patch planning should wait until a stronger candidate generator adds traceback/error extraction, import graph neighbors, test-to-source mapping, and repo-specific symbol aliases.

Updated structured exact margin:

| Tier | Workload | Context p95 | Runtime p95 | Rows scanned | Metrics complete |
|---:|---|---:|---:|---:|---:|
| 1,000 blocks | structured_exact | 2.30 ms | 0.09 ms | 1.20 | 100.00% |
| 10,000 blocks | structured_exact | 3.56 ms | 0.07 ms | 1.20 | 100.00% |
| 100,000 blocks | structured_exact | 14.96 ms | 0.10 ms | 1.20 | 100.00% |

Semantic ANN status:

| Tier | Best synthetic recall@k | Real BGE field recall@k | Real BGE field p95 | Fast ANN recall@k | Fast ANN p95 | Status |
|---:|---:|---:|---:|---:|---:|---|
| 1,000 blocks | 100.00% | 92.70% | 46.02 ms | 76.30% | 45.17 ms | VALIDATING |
| 10,000 blocks | 95.20% | 80.40% | 53.11 ms | 59.30% | 51.72 ms | VALIDATING |
| 100,000 blocks | 75.80% | 82.30% | 51.14 ms | 62.20% | 55.40 ms | VALIDATING |

The `100k` semantic retrieval blocker is now cleared for the synthetic scale-up benchmark. Prio 8 proves that better embeddings matter: `BAAI/bge-small-en-v1.5` plus field rescue reaches `82.30%` recall at `51.14 ms` p95. The next proof should move from synthetic retrieval to a real dev-agent workload and then a controlled local LLM client.

CrossEncoder status:

| Run | Strategy | Reranker available | 100k EM | 100k Recall@k | 100k p95 | Status |
|---|---|---:|---:|---:|---:|---|
| `semantic_cross_encoder_full` | ooc_semantic_cross_encoder_rescue | 100.00% | 100.00% | 65.00% | 2121.26 ms | NON_VALIDATING |

This run proves model caching and inference, not semantic success. It stays non-validating because recall remains below `80%` and p95 latency is far above the `200 ms` interactive target.

## Remaining Work

| Area | Weight | Goal |
|---|---:|---|
| Adaptive runtime + LLM API | 20% | Turn retrieval into a context engine |
| Memory/KV/cache policy | 10% | Control hot/warm/cold context and KV load |
| Real benchmarks + polish | 5-10% | Prove value on dev/code workloads |

## Non-Negotiable Order

1. Runtime contracts.
2. Context engine without LLM.
3. Cache/KV policy.
4. Token economics measurement.
5. Real LLM integration.
6. Real dev-agent benchmark.

## Target Architecture

```text
user turn + session state
        |
        v
ContextAdapter
        |
        v
HighwayContextEngine
        |
        +-- StrategyPlanner
        +-- SearchRouter(auto)
        +-- OutOfCoreIndex / VectorCandidateIndex
        +-- EvidenceResolver
        +-- ResidencyManager
        |
        v
ContextPack
        |
        v
HighwayLLMRuntime
        |
        v
LLM client
```

The first implementation target was `HighwayContextEngine -> ContextPack`. `HighwayLLMRuntime` now exists for the fake-client path; real model clients come after the fake runtime remains validating.

## Phase 1: Runtime Foundations Before LLM

Status: implemented for the no-real-LLM path and extended with an optional Ollama benchmark path. Continue treating 0.5B runs as integration smoke until a 1.5B+ model clears quality/coherence gates.

Target public types:

```python
@dataclass(frozen=True)
class ContextRequest:
    user_turn: str
    session_id: str = "default"
    token_budget: int = 4096
    latency_budget_ms: float = 100.0
    strategy: str = "auto"

@dataclass(frozen=True)
class ContextBlock:
    block_id: str
    source_file: str
    text: str
    score: float
    reason: str

@dataclass(frozen=True)
class ContextPack:
    request: ContextRequest
    blocks: list[ContextBlock]
    query_ir: dict
    metrics: dict
    warnings: list[str]
```

Target engine:

```python
class HighwayContextEngine:
    def retrieve(self, request: ContextRequest) -> ContextPack:
        ...
```

Required metrics in every `ContextPack`:

- `strategy_used`
- `bytes_read`
- `embedding_rows_scanned`
- `blocks_materialized`
- `ann_used`
- `ann_backend`
- `latency_ms`
- `context_input_tokens_estimated`

Acceptance:

- `ContextPack` is produced without a real LLM. Current status: implemented.
- Existing wrappers keep their behavior. Current status: covered by full test suite.
- `pytest tests -q` remains green. Current status: green in the latest local verification.

## Phase 2: Real-Time Context Adapter

Status: first-pass implementation exists. Next hardening step is richer query classification and measured multi-turn benchmarks.

Target types:

```python
@dataclass
class SessionState:
    session_id: str
    active_entities: list[str]
    active_sources: list[str]
    pinned_block_ids: list[str]
    last_strategy: str | None
    turn_count: int

class ContextAdapter:
    def plan(self, request: ContextRequest, state: SessionState) -> dict:
        ...
```

Policy V1:

- Marker/entity query: exact pruning first.
- Semantic query without strong entity: HNSW candidates, then mmap rerank.
- Follow-up query: prioritize active entities, active sources, and pinned blocks.
- Insufficient result: fallback to full/pruned according to budget.
- Never use `query_id` as a retrieval signal.

Acceptance:

- Multi-turn retrieval reuses previous entities/sources. Current status: covered by unit tests.
- The chosen strategy and reasons are visible in `ContextPack.metrics`. Current status: implemented.
- Next required proof: benchmark a real multi-turn workload under `artifacts/runs/context_adapter_multiturn/`.

## Phase 3: Token Throughput And Savings Model

Status: initial implementation exists. Highway can now report token economics before any real LLM integration.

Target type:

```python
@dataclass(frozen=True)
class TokenEconomics:
    baseline_input_tokens: int
    actual_input_tokens: int
    avoided_input_tokens: int
    output_tokens: int
    ttft_ms: float
    decode_ms: float
    total_llm_ms: float
    input_tokens_per_second: float
    output_tokens_per_second: float
    effective_tokens_per_second: float
    kv_bytes_estimated: int | None
    kv_bytes_avoided_estimated: int | None
    cost_estimated_usd: float
    cost_avoided_estimated_usd: float
```

Metrics to report:

- `baseline_input_tokens`: naive/full-context prompt estimate.
- `actual_input_tokens`: context pack prompt estimate.
- `avoided_input_tokens`: baseline minus actual.
- `output_tokens`: generated tokens or fake-client output tokens.
- `ttft_ms`: time to first token.
- `decode_ms`: generation time after TTFT.
- `input_tokens_per_second`: prefill throughput estimate.
- `output_tokens_per_second`: decode throughput.
- `effective_tokens_per_second`: useful tokens served per second.
- `kv_bytes_estimated`: KV memory estimate for the actual prompt.
- `kv_bytes_avoided_estimated`: KV memory avoided by context reduction.
- `cost_avoided_estimated_usd`: configurable price-model savings.

KV formula V1:

```text
kv_bytes = input_tokens * layers * hidden_size * 2 * bytes_per_element
```

Defaults:

- `bytes_per_element = 2` for FP16/BF16.
- `layers` and `hidden_size` are configured per model.
- If model shape is unknown, set KV estimates to `null` and emit a warning.

Acceptance:

- Runs without LLM still report baseline, actual, and avoided input tokens. Current status: implemented in `ContextPack.metrics`.
- Fake-LLM runs report input/output tokens, TTFT, decode time, and tokens/s. Current status: implemented in `artifacts/runs/llm_runtime_fake/`.
- Reports show token savings and estimated KV savings. Current status: implemented for the quality/token smoke.
- Quality gates compare baseline answer quality against Highway answer quality before accepting token savings. Current status: implemented for deterministic G/H smoke, real LLM quality benchmark still needed.

## Phase 4: Memory, KV, And Cache Policy

Make the current cache hierarchy explicit as a context residency policy.

Target type:

```python
@dataclass(frozen=True)
class ContextCachePolicy:
    max_hot_blocks: int = 32
    max_warm_entries: int = 256
    max_context_tokens: int = 4096
    pin_current_turn_sources: bool = True
```

Runtime tiers:

- Hot context: active blocks, pinned sources, current entities.
- Warm context: recent proof IR, evidence pools, compiled prompts, answers.
- Cold context: out-of-core corpus on disk.
- Avoided KV: tokens not sent to the model.

Required metrics:

- `hotset_hits`
- `evictions`
- `tokens_materialized_kv`
- `tokens_avoided`
- `cache_l0_hits`
- `cache_l1_hits`
- `cache_l2_hits`
- `cache_l3_hits`
- `resident_bytes`
- `kv_bytes_avoided_estimated`

Acceptance:

- Every kept or evicted block has a reason.
- Each `ContextPack` exposes the applied policy.
- No real LLM call is needed to validate the policy.

## Phase 5: Minimal LLM API

The fake-client version is implemented. Real clients should be added only after phases 1-4 remain stable.

Target API:

```python
class HighwayLLMRuntime:
    def build_prompt(self, context_pack: ContextPack) -> str:
        ...

    def answer_with_client(self, request: ContextRequest, llm_client) -> dict:
        ...
```

LLM client contract:

- `input_tokens`
- `output_tokens`
- `ttft_ms`
- `decode_ms`
- `total_ms`
- `model_name`

V1 uses a fake deterministic client for tests and reports. Ollama is now the first real local client path. vLLM/OpenAI-style clients remain optional future integrations.

Acceptance:

- Highway still works without an LLM.
- Fake client validates token in/out and tokens/s. Current status: implemented and validating.
- Real clients can be plugged in without changing `HighwayContextEngine`.
- Reports show latency, token savings, cost estimate, and KV estimate.

## Phase 6: Real Benchmarks And Engineer Demo

Move from structured synthetic queries to dev-agent-like workloads.

Benchmark content:

- Mini codebase.
- Technical docs.
- Tickets.
- Logs.
- Multi-turn conversations.
- Ambiguous queries.
- Topic switches.
- Return to old entity/source.
- Semantic queries without marker.
- Contradictory old/new documents.

Expected artifacts:

```text
artifacts/runs/context_engine_smoke/
artifacts/runs/context_adapter_multiturn/
artifacts/runs/token_economics_smoke/
artifacts/runs/dev_agent_retrieval/
```

Gates:

- No-leak: `100%`.
- Source attribution on factual tasks: `100%`.
- G/H EM on structured workloads: `100%`.
- Structured context pack p95 below `50 ms` at `100k` blocks. Current status: validating at `14.96 ms`.
- HNSW semantic recall: first target `>= 80%`, later target `>= 95%`.
- Avoided input tokens: `>= 80%` vs full-context baseline.
- Estimated KV bytes avoided: `>= 80%`.
- No user-specific absolute paths in validating reports.

## Public Interfaces

- `HighwayContextEngine.retrieve(ContextRequest) -> ContextPack`
- `ContextAdapter.plan(ContextRequest, SessionState) -> dict`
- `TokenEconomics`
- `ContextCachePolicy`
- `HighwayLLMRuntime.build_prompt(ContextPack) -> str`
- `HighwayLLMRuntime.answer_with_client(ContextRequest, llm_client) -> dict`

Existing root wrappers stay stable:

- `build_poc234_kernel_hardening_workload.py`
- `run_poc234_kernel_hardening.py`
- `eval_pccc_benchmark.py`
- `run_ooc_scaleup_benchmark.py`

## Test Plan

Unit tests:

- `ContextPack` is JSON-serializable.
- `HighwayContextEngine.retrieve()` works without LLM.
- Marker/entity planning chooses exact pruning.
- Semantic planning chooses HNSW when available.
- Faiss absence falls back cleanly.
- Follow-up requests reuse entities/sources.
- `TokenEconomics` computes avoided input tokens.
- KV bytes are estimated correctly.
- Fake LLM client reports input/output tokens and tokens/s.

Non-regression:

```powershell
pytest tests -q
python -m compileall src tests
python -m py_compile build_poc234_kernel_hardening_workload.py run_poc234_kernel_hardening.py eval_pccc_benchmark.py run_ooc_scaleup_benchmark.py
```

Benchmarks:

- Existing no-leak smoke remains unchanged.
- Existing OOC/Faiss scale-up remains valid.
- Add context engine smoke without LLM.
- Add token economics smoke with fake LLM. Current status: implemented through `llm_runtime_fake`.
- Keep the quality/token tradeoff smoke as a mandatory gate before any real LLM demo.
- Add multi-turn benchmark without LLM.
- Add real LLM benchmark only after fake-client validation. Current status: fake-client validation is green; Ollama/qwen3 is validating on the audited 25-turn synthetic conversation and the 100-turn multi-theme conversation. RAGBench harness is implemented for Headroom-like RAG checks. SWE-bench Verified is now the active dev-agent benchmark path; the next step is stronger SWE candidate generation before any Qwen patch-planning run.

## Assumptions

- Immediate priority is foundations before a real LLM.
- Faiss is installed locally but remains optional.
- `artifacts/runs/ooc_ann_scaleup_faiss/` is the principal ANN validating run.
- The LLM consumes a `ContextPack`; it does not choose corpus data.
- Savings must be measured in latency, rows scanned, bytes read, tokens avoided, estimated KV memory, and estimated cost.
- Petabyte-scale context is the long-term direction. The next proof is a local, auditable engine that can later extend to shards, object storage, prefetching, and distributed scheduling.
