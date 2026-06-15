from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from statistics import mean
from typing import Any, Dict, List, Sequence

import numpy as np

from highway.benchmarks.ooc_scaleup import SyntheticScaleupEmbedder, _read_jsonl, generate_scaleup_dataset
from highway.paths import DEFAULT_RUNS_DIR
from highway.runtime.context_engine import ContextRequest, HighwayContextEngine
from highway.runtime.llm_runtime import HighwayLLMRuntime, estimate_tokens
from highway.runtime.ollama_client import OllamaLLMClient
from highway.runtime.token_economics import ModelProfile, TokenEconomics
from highway.storage.semantic_embedder import DEFAULT_EMBEDDING_MODEL, create_semantic_embedder


DEFAULT_OUTPUT_DIR = DEFAULT_RUNS_DIR / "local_llm_quality"
DEFAULT_MODEL_PROFILE = ModelProfile(name="local_ollama_llm", layers=24, hidden_size=1024)
JSON_INSTRUCTION = (
    "Return only valid JSON with keys: reasoning, answer, sources, confidence. "
    "The sources value must be a list of source_file strings copied from the context."
)
REQUIRED_RECORD_FIELDS = {
    "baseline_answer",
    "highway_answer",
    "expected_answer",
    "baseline_is_em",
    "highway_is_em",
    "quality_delta",
    "baseline_input_tokens",
    "highway_input_tokens",
    "avoided_input_tokens_pct",
    "baseline_output_tokens",
    "highway_output_tokens",
    "baseline_ttft_ms",
    "highway_ttft_ms",
    "baseline_total_ms",
    "highway_total_ms",
    "baseline_tokens_per_second",
    "highway_tokens_per_second",
    "kv_bytes_estimated",
    "kv_bytes_avoided_estimated",
    "cost_avoided_estimated_usd",
    "context_latency_ms",
    "embedding_rows_scanned",
    "blocks_materialized",
    "bytes_read",
    "baseline_verdict",
    "highway_verdict",
}


def _display_path(path: Path) -> str:
    try:
        return path.resolve().relative_to(Path.cwd().resolve()).as_posix()
    except ValueError:
        return path.as_posix()


def _write_jsonl(path: Path, rows: Sequence[Dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def _clean_answer(value: Any) -> str:
    return " ".join(str(value or "").strip().split()).lower()


def _canonical_project_from_expected(expected_answer: str) -> str:
    match = re.match(r"\s*(Project\s+[A-Za-z0-9_-]+)", str(expected_answer or ""), flags=re.IGNORECASE)
    return match.group(1) if match else str(expected_answer or "")


def _question_requires_numeric_fact(question: str) -> bool:
    normalized = _clean_answer(question)
    numeric_terms = (
        "what budget",
        "which budget",
        "how much",
        "budget amount",
        "what is the budget",
        "include the budget",
        "with the budget",
    )
    return any(term in normalized for term in numeric_terms)


def _answer_satisfies_question(answer: str, expected_answer: str, question: str) -> bool:
    if _clean_answer(answer) == _clean_answer(expected_answer):
        return True
    normalized_question = _clean_answer(question)
    if "which project" in normalized_question and not _question_requires_numeric_fact(question):
        expected_project = _canonical_project_from_expected(expected_answer)
        return _clean_answer(answer) == _clean_answer(expected_project)
    return False


def _percentile(values: Sequence[float], pct: float) -> float:
    if not values:
        return 0.0
    return float(np.percentile(list(values), pct))


def parse_model_json(text: str) -> Dict[str, Any]:
    raw = str(text or "").strip()
    candidates = [raw]
    start = raw.find("{")
    end = raw.rfind("}")
    if start >= 0 and end > start:
        candidates.append(raw[start : end + 1])
    for candidate in candidates:
        try:
            parsed = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, dict):
            sources = parsed.get("sources", [])
            if isinstance(sources, str):
                sources = [sources]
            if not isinstance(sources, list):
                sources = []
            return {
                "parse_ok": True,
                "reasoning": str(parsed.get("reasoning", "")),
                "answer": str(parsed.get("answer", "")),
                "sources": [str(source) for source in sources],
                "confidence": parsed.get("confidence"),
                "raw_text": raw,
            }
    return {
        "parse_ok": False,
        "reasoning": "",
        "answer": "",
        "sources": [],
        "confidence": None,
        "raw_text": raw,
    }


def evaluate_quality(
    parsed: Dict[str, Any],
    expected_answer: str,
    expected_sources: Sequence[str],
    allowed_sources: Sequence[str],
    previous_entity: str | None,
    current_question: str,
) -> Dict[str, Any]:
    if not parsed.get("parse_ok", False):
        return {
            "verdict": "MODEL_PARSE_FAIL",
            "is_em": False,
            "full_exact_match": False,
            "answer_satisfies_question": False,
            "source_attribution_ok": False,
            "hallucination_flag": False,
            "coherence_ok": previous_entity is None,
        }
    answer = str(parsed.get("answer", ""))
    sources = [str(source) for source in parsed.get("sources", [])]
    expected = set(str(source) for source in expected_sources)
    allowed = set(str(source) for source in allowed_sources)
    full_exact_match = _clean_answer(answer) == _clean_answer(expected_answer)
    answer_satisfies_question = _answer_satisfies_question(answer, expected_answer, current_question)
    is_em = answer_satisfies_question
    source_ok = bool(expected & set(sources)) if expected else True
    hallucination = any(source not in allowed for source in sources)
    is_follow_up = any(term in current_question.lower() for term in ("its", "their", "that", "those", "previous"))
    coherence_ok = True
    if previous_entity and is_follow_up:
        haystack = " ".join([answer, str(parsed.get("reasoning", ""))]).lower()
        coherence_ok = previous_entity.lower() in haystack
    if not is_em:
        verdict = "QUALITY_FAIL"
    elif not source_ok or hallucination:
        verdict = "SOURCE_FAIL"
    elif not coherence_ok:
        verdict = "COHERENCE_FAIL"
    else:
        verdict = "PASS"
    return {
        "verdict": verdict,
        "is_em": is_em,
        "full_exact_match": full_exact_match,
        "answer_satisfies_question": answer_satisfies_question,
        "source_attribution_ok": source_ok and not hallucination,
        "hallucination_flag": hallucination,
        "coherence_ok": coherence_ok,
    }


def _baseline_prompt(question: str, blocks: Sequence[Dict[str, Any]]) -> str:
    lines = [
        "You are a precise reasoning assistant.",
        "Use only the context below.",
        "",
        "Context:",
    ]
    for block in blocks:
        lines.append(f"[{block.get('block_id')}] {block.get('source_file')}: {block.get('text')}")
    lines.extend(["", f"Question: {question}", JSON_INSTRUCTION])
    return "\n".join(lines)


def _highway_prompt(runtime: HighwayLLMRuntime, pack: Any) -> str:
    return runtime.build_prompt(pack) + "\n" + JSON_INSTRUCTION


def _bounded_baseline_blocks(
    all_blocks: Sequence[Dict[str, Any]],
    expected_source: str,
    limit: int,
) -> List[Dict[str, Any]]:
    expected = [block for block in all_blocks if str(block.get("source_file")) == expected_source]
    others = [block for block in all_blocks if str(block.get("source_file")) != expected_source]
    return (expected + others)[: max(1, int(limit))]


def _make_embedder(backend: str, model: str, local_files_only: bool, batch_size: int) -> Any:
    if backend == "synthetic":
        return SyntheticScaleupEmbedder()
    return create_semantic_embedder(
        backend=backend,
        model_name=model,
        local_files_only=local_files_only,
        batch_size=batch_size,
    )


def _record_complete(record: Dict[str, Any]) -> bool:
    return REQUIRED_RECORD_FIELDS.issubset(record.keys())


def _summarize_tier(size: int, records: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    if not records:
        return {
            "size": size,
            "count": 0,
            "baseline_em": 0.0,
            "highway_em": 0.0,
            "quality_delta_pp": 0.0,
            "source_attribution_rate": 0.0,
            "coherence_rate": 0.0,
            "avg_avoided_input_tokens_pct": 0.0,
            "baseline_ttft_p95_ms": 0.0,
            "highway_ttft_p95_ms": 0.0,
            "baseline_total_p95_ms": 0.0,
            "highway_total_p95_ms": 0.0,
            "context_p95_ms": 0.0,
            "metrics_complete_rate": 0.0,
        }
    baseline_em = mean(bool(record["baseline_is_em"]) for record in records) * 100.0
    highway_em = mean(bool(record["highway_is_em"]) for record in records) * 100.0
    source_rate = mean(bool(record["highway_source_attribution_ok"]) for record in records) * 100.0
    coherence_rate = mean(bool(record["highway_coherence_ok"]) for record in records) * 100.0
    return {
        "size": size,
        "count": len(records),
        "baseline_em": float(baseline_em),
        "highway_em": float(highway_em),
        "quality_delta_pp": float(highway_em - baseline_em),
        "source_attribution_rate": float(source_rate),
        "coherence_rate": float(coherence_rate),
        "avg_avoided_input_tokens_pct": float(mean(record["avoided_input_tokens_pct"] for record in records)),
        "baseline_ttft_p95_ms": _percentile([record["baseline_ttft_ms"] for record in records], 95),
        "highway_ttft_p95_ms": _percentile([record["highway_ttft_ms"] for record in records], 95),
        "baseline_total_p95_ms": _percentile([record["baseline_total_ms"] for record in records], 95),
        "highway_total_p95_ms": _percentile([record["highway_total_ms"] for record in records], 95),
        "context_p95_ms": _percentile([record["context_latency_ms"] for record in records], 95),
        "metrics_complete_rate": mean(_record_complete(record) for record in records) * 100.0,
    }


def _tier_is_validating(tier: Dict[str, Any]) -> bool:
    return (
        tier["count"] > 0
        and tier["metrics_complete_rate"] == 100.0
        and tier["avg_avoided_input_tokens_pct"] >= 80.0
        and tier["highway_ttft_p95_ms"] < tier["baseline_ttft_p95_ms"]
        and tier["highway_em"] >= tier["baseline_em"]
        and tier["highway_em"] >= 80.0
        and tier["source_attribution_rate"] >= 95.0
        and tier["coherence_rate"] >= 90.0
    )


def _write_report(path: Path, summary: Dict[str, Any], metrics_path: Path, records_path: Path) -> None:
    lines = [
        "# Local LLM Quality Benchmark",
        "",
        f"Verdict: {summary['status']}",
        f"Model: `{summary['model']}`",
        "",
    ]
    if summary.get("skip_reason"):
        lines.extend([
            f"Skip reason: `{summary['skip_reason']}`",
            "",
            "The benchmark is non-destructive: missing Ollama or missing local model produces a report instead of a crash.",
            "",
        ])
    lines.extend([
        "| Size | Baseline EM | Highway EM | Quality delta | Source attr | Coherence | Avoided tokens | Baseline TTFT p95 | Highway TTFT p95 | Context p95 | Metrics complete |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ])
    for tier in summary["tiers"]:
        lines.append(
            f"| {tier['size']:,} | {tier['baseline_em']:.2f}% | {tier['highway_em']:.2f}% | "
            f"{tier['quality_delta_pp']:.2f} pp | {tier['source_attribution_rate']:.2f}% | "
            f"{tier['coherence_rate']:.2f}% | {tier['avg_avoided_input_tokens_pct']:.2f}% | "
            f"{tier['baseline_ttft_p95_ms']:.2f} ms | {tier['highway_ttft_p95_ms']:.2f} ms | "
            f"{tier['context_p95_ms']:.2f} ms | {tier['metrics_complete_rate']:.2f}% |"
        )
    lines.extend([
        "",
        "## Interpretation",
        "",
        "Qwen 0.5B style runs are integration smokes. Quality claims require a stronger local model, starting with a 1.5B class model.",
        "Token savings are accepted only when factual quality, source attribution, and multi-turn coherence do not regress.",
        "",
        f"Metrics JSON: `{_display_path(metrics_path)}`",
        f"Records JSONL: `{_display_path(records_path)}`",
    ])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_local_llm_quality_benchmark(
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    model: str = "qwen2.5:0.5b",
    sizes: Sequence[int] = (1000, 10000),
    query_count: int = 10,
    seed: int = 42,
    strategy: str = "ooc_semantic_field_rescue",
    embedding_backend: str = "synthetic",
    embedding_model: str = DEFAULT_EMBEDDING_MODEL,
    embedding_local_files_only: bool = False,
    embedding_batch_size: int = 64,
    baseline_context_limit: int = 200,
    input_cost_per_million: float = 1.0,
    output_cost_per_million: float = 2.0,
    llm_client: Any | None = None,
) -> Dict[str, Any]:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    client = llm_client or OllamaLLMClient(model=model)
    all_records: List[Dict[str, Any]] = []
    tiers: List[Dict[str, Any]] = []
    skip_reason = ""

    for size in sizes:
        embedder = _make_embedder(embedding_backend, embedding_model, embedding_local_files_only, embedding_batch_size)
        dataset = generate_scaleup_dataset(
            output_path / f"dataset_{int(size)}",
            total_blocks=int(size),
            query_count=query_count,
            seed=seed,
            mixed_query_set="marker,entity,semantic",
            embedder=embedder,
            vector_backend="faiss_hnsw" if strategy != "ooc_marker_entity_pruned" else "none",
        )
        workload = _read_jsonl(dataset.workload_path, limit=query_count)
        all_blocks = _read_jsonl(dataset.index_dir / "blocks.jsonl")
        engine = HighwayContextEngine(
            index_dir=dataset.index_dir,
            embed_model=embedder,
            model_profile=DEFAULT_MODEL_PROFILE,
            input_cost_per_million=input_cost_per_million,
            output_cost_per_million=output_cost_per_million,
        )
        runtime = HighwayLLMRuntime(engine)
        records: List[Dict[str, Any]] = []
        previous_entity: str | None = None
        for row in workload:
            request = ContextRequest(user_turn=row["question"], session_id=f"local_llm_{size}", strategy=strategy)
            pack = engine.retrieve(request, top_k=50)
            highway_prompt = _highway_prompt(runtime, pack)
            baseline_blocks = _bounded_baseline_blocks(all_blocks, row["source_file"], baseline_context_limit)
            baseline_prompt = _baseline_prompt(row["question"], baseline_blocks)
            evidence = [runtime._block_to_evidence(block) for block in pack.blocks]
            baseline_evidence = [
                {
                    "block_id": block.get("block_id", ""),
                    "source_file": block.get("source_file", ""),
                    "text": block.get("text", ""),
                    "retrieval_score": 0.0,
                }
                for block in baseline_blocks
            ]
            baseline_response = client.answer(
                prompt=baseline_prompt,
                query_ir=pack.query_ir,
                evidence=baseline_evidence,
                expected_answer=row["expected_answer"],
                query_id=row["id"],
            )
            if baseline_response.get("available") is False:
                skip_reason = str(baseline_response.get("skip_reason", "llm_unavailable"))
                break
            highway_response = client.answer(
                prompt=highway_prompt,
                query_ir=pack.query_ir,
                evidence=evidence,
                expected_answer=row["expected_answer"],
                query_id=row["id"],
            )
            if highway_response.get("available") is False:
                skip_reason = str(highway_response.get("skip_reason", "llm_unavailable"))
                break
            baseline_parsed = parse_model_json(str(baseline_response.get("raw_text", baseline_response.get("answer", ""))))
            highway_parsed = parse_model_json(str(highway_response.get("raw_text", highway_response.get("answer", ""))))
            expected_sources = [str(row["source_file"])]
            baseline_sources = [str(block.get("source_file", "")) for block in baseline_blocks]
            highway_sources = [block.source_file for block in pack.blocks]
            baseline_quality = evaluate_quality(
                baseline_parsed,
                expected_answer=str(row["expected_answer"]),
                expected_sources=expected_sources,
                allowed_sources=baseline_sources,
                previous_entity=previous_entity,
                current_question=str(row["question"]),
            )
            highway_quality = evaluate_quality(
                highway_parsed,
                expected_answer=str(row["expected_answer"]),
                expected_sources=expected_sources,
                allowed_sources=highway_sources,
                previous_entity=previous_entity,
                current_question=str(row["question"]),
            )
            economics = TokenEconomics.from_measurements(
                baseline_input_tokens=int(baseline_response["input_tokens"]),
                actual_input_tokens=int(highway_response["input_tokens"]),
                output_tokens=int(highway_response["output_tokens"]),
                ttft_ms=float(highway_response["ttft_ms"]),
                decode_ms=float(highway_response["decode_ms"]),
                model_profile=DEFAULT_MODEL_PROFILE,
                input_cost_per_million=input_cost_per_million,
                output_cost_per_million=output_cost_per_million,
            )
            metrics = dict(pack.metrics)
            metadata = row.get("metadata", {})
            if isinstance(metadata, dict):
                previous_entity = str(metadata.get("proj_a") or metadata.get("projects", [previous_entity])[0] or previous_entity)
            records.append({
                "size": int(size),
                "id": row["id"],
                "category": row["category"],
                "question": row["question"],
                "expected_answer": row["expected_answer"],
                "expected_sources": expected_sources,
                "baseline_answer": baseline_parsed["answer"],
                "highway_answer": highway_parsed["answer"],
                "baseline_is_em": bool(baseline_quality["is_em"]),
                "highway_is_em": bool(highway_quality["is_em"]),
                "baseline_full_exact_match": bool(baseline_quality["full_exact_match"]),
                "highway_full_exact_match": bool(highway_quality["full_exact_match"]),
                "baseline_answer_satisfies_question": bool(baseline_quality["answer_satisfies_question"]),
                "highway_answer_satisfies_question": bool(highway_quality["answer_satisfies_question"]),
                "quality_delta": int(bool(highway_quality["is_em"])) - int(bool(baseline_quality["is_em"])),
                "baseline_verdict": baseline_quality["verdict"],
                "highway_verdict": highway_quality["verdict"],
                "baseline_source_attribution_ok": bool(baseline_quality["source_attribution_ok"]),
                "highway_source_attribution_ok": bool(highway_quality["source_attribution_ok"]),
                "baseline_coherence_ok": bool(baseline_quality["coherence_ok"]),
                "highway_coherence_ok": bool(highway_quality["coherence_ok"]),
                "baseline_hallucination_flag": bool(baseline_quality["hallucination_flag"]),
                "highway_hallucination_flag": bool(highway_quality["hallucination_flag"]),
                "baseline_input_tokens": int(baseline_response["input_tokens"]),
                "highway_input_tokens": int(highway_response["input_tokens"]),
                "avoided_input_tokens": economics.avoided_input_tokens,
                "avoided_input_tokens_pct": (
                    economics.avoided_input_tokens / economics.baseline_input_tokens * 100.0
                    if economics.baseline_input_tokens else 0.0
                ),
                "baseline_output_tokens": int(baseline_response["output_tokens"]),
                "highway_output_tokens": int(highway_response["output_tokens"]),
                "baseline_ttft_ms": float(baseline_response["ttft_ms"]),
                "highway_ttft_ms": float(highway_response["ttft_ms"]),
                "baseline_total_ms": float(baseline_response["total_ms"]),
                "highway_total_ms": float(highway_response["total_ms"]),
                "baseline_tokens_per_second": float(baseline_response["input_tokens_per_second"]),
                "highway_tokens_per_second": float(highway_response["input_tokens_per_second"]),
                "kv_bytes_estimated": economics.kv_bytes_estimated or 0,
                "kv_bytes_avoided_estimated": economics.kv_bytes_avoided_estimated or 0,
                "cost_avoided_estimated_usd": economics.cost_avoided_estimated_usd,
                "context_latency_ms": float(metrics.get("latency_ms", 0.0)),
                "embedding_rows_scanned": int(metrics.get("embedding_rows_scanned", 0)),
                "blocks_materialized": int(metrics.get("blocks_materialized", 0)),
                "bytes_read": int(metrics.get("bytes_read", 0)),
                "baseline_prompt_tokens_verified": estimate_tokens(baseline_prompt),
                "highway_prompt_tokens_verified": estimate_tokens(highway_prompt),
            })
        if skip_reason:
            break
        all_records.extend(records)
        tiers.append(_summarize_tier(int(size), records))

    if skip_reason:
        status = "SKIPPED"
    else:
        status = "VALIDATING" if tiers and all(_tier_is_validating(tier) for tier in tiers) else "NON_VALIDATING"
    summary = {
        "status": status,
        "model": getattr(client, "model_name", model),
        "query_count": query_count,
        "seed": seed,
        "strategy": strategy,
        "embedding_backend": embedding_backend,
        "embedding_model": embedding_model,
        "baseline_context_limit": baseline_context_limit,
        "skip_reason": skip_reason,
        "tiers": tiers,
    }
    metrics_path = output_path / "metrics.json"
    records_path = output_path / "records.jsonl"
    report_path = output_path / "report.md"
    _write_jsonl(records_path, all_records)
    metrics_path.write_text(
        json.dumps(
            {
                "summary": summary,
                "model_profile": DEFAULT_MODEL_PROFILE.to_dict(),
                "required_record_fields": sorted(REQUIRED_RECORD_FIELDS),
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    _write_report(report_path, summary, metrics_path, records_path)
    return {
        "output_dir": output_path,
        "metrics_path": metrics_path,
        "records_path": records_path,
        "report_path": report_path,
        "summary": summary,
    }


def _parse_sizes(raw: str) -> List[int]:
    return [int(part.strip()) for part in raw.split(",") if part.strip()]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--model", default="qwen2.5:0.5b")
    parser.add_argument("--sizes", default="1000,10000")
    parser.add_argument("--queries", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--strategy", default="ooc_semantic_field_rescue")
    parser.add_argument("--embedding-backend", default="synthetic", choices=["synthetic", "sentence_transformer"])
    parser.add_argument("--embedding-model", default=DEFAULT_EMBEDDING_MODEL)
    parser.add_argument("--embedding-local-files-only", action="store_true")
    parser.add_argument("--embedding-batch-size", type=int, default=64)
    parser.add_argument("--baseline-context-limit", type=int, default=200)
    parser.add_argument("--input-cost-per-million", type=float, default=1.0)
    parser.add_argument("--output-cost-per-million", type=float, default=2.0)
    args = parser.parse_args()
    result = run_local_llm_quality_benchmark(
        output_dir=args.output_dir,
        model=args.model,
        sizes=_parse_sizes(args.sizes),
        query_count=args.queries,
        seed=args.seed,
        strategy=args.strategy,
        embedding_backend=args.embedding_backend,
        embedding_model=args.embedding_model,
        embedding_local_files_only=args.embedding_local_files_only,
        embedding_batch_size=args.embedding_batch_size,
        baseline_context_limit=args.baseline_context_limit,
        input_cost_per_million=args.input_cost_per_million,
        output_cost_per_million=args.output_cost_per_million,
    )
    print(json.dumps({"output_dir": str(result["output_dir"]), "summary": result["summary"]}, indent=2))


if __name__ == "__main__":
    main()
