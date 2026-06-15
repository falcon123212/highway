# Local LLM Quality Benchmark

Verdict: SKIPPED
Model: `qwen2.5:0.5b`

Skip reason: `ollama_unavailable:HTTP Error 404: Not Found`

The benchmark is non-destructive: missing Ollama or missing local model produces a report instead of a crash.

| Size | Baseline EM | Highway EM | Quality delta | Source attr | Coherence | Avoided tokens | Baseline TTFT p95 | Highway TTFT p95 | Context p95 | Metrics complete |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|

## Interpretation

Qwen 0.5B style runs are integration smokes. Quality claims require a stronger local model, starting with a 1.5B class model.
Token savings are accepted only when factual quality, source attribution, and multi-turn coherence do not regress.

Metrics JSON: `artifacts/runs/local_llm_quality_smoke/metrics.json`
Records JSONL: `artifacts/runs/local_llm_quality_smoke/records.jsonl`
