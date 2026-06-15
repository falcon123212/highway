import argparse
import json
import time
from pathlib import Path
from statistics import mean
from typing import Any, Dict, List, Sequence

import numpy as np

from highway.benchmarks.ooc_scaleup import SyntheticScaleupEmbedder, generate_scaleup_dataset
from highway.paths import DEFAULT_RUNS_DIR
from highway.retrieval.evidence_resolver import EvidenceResolver
from highway.runtime.context_engine import ContextRequest, HighwayContextEngine
from highway.runtime.llm_runtime import DeterministicReflectiveClient, HighwayLLMRuntime, estimate_tokens
from highway.runtime.token_economics import ModelProfile, TokenEconomics


DEFAULT_OUTPUT_DIR = DEFAULT_RUNS_DIR / "llm_runtime_fake"
DEFAULT_MODEL_PROFILE = ModelProfile(name="llm_runtime_fake_model", layers=24, hidden_size=1024)
REQUIRED_RECORD_FIELDS = {
    "baseline_answer",
    "highway_answer",
    "expected_answer",
    "baseline_is_em",
    "highway_is_em",
    "quality_delta",
    "baseline_input_tokens",
    "highway_input_tokens",
    "avoided_input_tokens",
    "avoided_input_tokens_pct",
    "baseline_output_tokens",
    "highway_output_tokens",
    "baseline_ttft_ms",
    "highway_ttft_ms",
    "baseline_input_tokens_per_second",
    "highway_input_tokens_per_second",
    "baseline_output_tokens_per_second",
    "highway_output_tokens_per_second",
    "kv_bytes_estimated",
    "kv_bytes_avoided_estimated",
    "cost_estimated_usd",
    "cost_avoided_estimated_usd",
    "context_latency_ms",
    "embedding_rows_scanned",
    "blocks_materialized",
    "bytes_read",
    "ann_used",
    "ann_backend",
}


def _display_path(path: Path) -> str:
    try:
        return path.resolve().relative_to(Path.cwd().resolve()).as_posix()
    except ValueError:
        return path.as_posix()


def _read_jsonl(path: Path, limit: int | None = None) -> List[Dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
                if limit is not None and len(rows) >= limit:
                    break
    return rows


def _write_jsonl(path: Path, rows: Sequence[Dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def _clean_answer(value: str) -> str:
    return " ".join(str(value).strip().split()).lower()


def _percentile(values: Sequence[float], pct: float) -> float:
    if not values:
        return 0.0
    return float(np.percentile(list(values), pct))


def _baseline_prompt(question: str, blocks: Sequence[Dict[str, Any]]) -> str:
    lines = [
        "You are a precise reasoning assistant.",
        "Use the full context below.",
        "",
        "Context:",
    ]
    for block in blocks:
        lines.append(f"[{block.get('block_id')}] {block.get('source_file')}: {block.get('text')}")
    lines.extend(["", f"Question: {question}", "Return: reasoning + answer."])
    return "\n".join(lines)


def _summarize_tier(size: int, records: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    baseline_em_values = [bool(record["baseline_is_em"]) for record in records]
    highway_em_values = [bool(record["highway_is_em"]) for record in records]
    baseline_em = float(mean(baseline_em_values) * 100.0) if records else 0.0
    highway_em = float(mean(highway_em_values) * 100.0) if records else 0.0
    baseline_ttft = [float(record["baseline_ttft_ms"]) for record in records]
    highway_ttft = [float(record["highway_ttft_ms"]) for record in records]
    baseline_total = [float(record["baseline_total_ms"]) for record in records]
    highway_total = [float(record["highway_total_ms"]) for record in records]
    context_latencies = [float(record["context_latency_ms"]) for record in records]
    complete = [REQUIRED_RECORD_FIELDS.issubset(record.keys()) for record in records]
    avg_baseline_tokens = float(mean(record["baseline_input_tokens"] for record in records)) if records else 0.0
    avg_highway_tokens = float(mean(record["highway_input_tokens"] for record in records)) if records else 0.0
    return {
        "size": size,
        "count": len(records),
        "baseline_em": baseline_em,
        "highway_em": highway_em,
        "quality_delta_pp": highway_em - baseline_em,
        "avg_baseline_input_tokens": avg_baseline_tokens,
        "avg_highway_input_tokens": avg_highway_tokens,
        "avg_avoided_input_tokens_pct": (
            float(mean(record["avoided_input_tokens_pct"] for record in records)) if records else 0.0
        ),
        "avg_kv_bytes_avoided_estimated": (
            float(mean(record["kv_bytes_avoided_estimated"] for record in records)) if records else 0.0
        ),
        "avg_cost_avoided_estimated_usd": (
            float(mean(record["cost_avoided_estimated_usd"] for record in records)) if records else 0.0
        ),
        "baseline_ttft_mean_ms": float(mean(baseline_ttft)) if baseline_ttft else 0.0,
        "baseline_ttft_p50_ms": _percentile(baseline_ttft, 50),
        "baseline_ttft_p95_ms": _percentile(baseline_ttft, 95),
        "highway_ttft_mean_ms": float(mean(highway_ttft)) if highway_ttft else 0.0,
        "highway_ttft_p50_ms": _percentile(highway_ttft, 50),
        "highway_ttft_p95_ms": _percentile(highway_ttft, 95),
        "baseline_total_mean_ms": float(mean(baseline_total)) if baseline_total else 0.0,
        "baseline_total_p50_ms": _percentile(baseline_total, 50),
        "baseline_total_p95_ms": _percentile(baseline_total, 95),
        "highway_total_mean_ms": float(mean(highway_total)) if highway_total else 0.0,
        "highway_total_p50_ms": _percentile(highway_total, 50),
        "highway_total_p95_ms": _percentile(highway_total, 95),
        "context_p95_ms": _percentile(context_latencies, 95),
        "avg_embedding_rows_scanned": (
            float(mean(record["embedding_rows_scanned"] for record in records)) if records else 0.0
        ),
        "avg_blocks_materialized": (
            float(mean(record["blocks_materialized"] for record in records)) if records else 0.0
        ),
        "metrics_complete_rate": float(mean(complete) * 100.0) if records else 0.0,
    }


def _tier_is_validating(tier: Dict[str, Any]) -> bool:
    return (
        tier["baseline_em"] == 100.0
        and tier["highway_em"] == 100.0
        and tier["quality_delta_pp"] == 0.0
        and tier["avg_avoided_input_tokens_pct"] >= 80.0
        and tier["context_p95_ms"] <= 100.0
        and tier["metrics_complete_rate"] == 100.0
    )


def _write_report(path: Path, summary: Dict[str, Any], metrics_path: Path, records_path: Path) -> None:
    lines = [
        f"# LLM Runtime Fake Benchmark",
        "",
        f"Verdict: {summary['status']}",
        "",
        "| Size | Baseline EM | Highway EM | Quality delta | Avoided tokens | Baseline TTFT p95 | Highway TTFT p95 | Baseline total p95 | Highway total p95 | Context p95 | Rows scanned | Blocks materialized | Metrics complete |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for tier in summary["tiers"]:
        lines.append(
            f"| {tier['size']:,} | {tier['baseline_em']:.2f}% | {tier['highway_em']:.2f}% | "
            f"{tier['quality_delta_pp']:.2f} pp | {tier['avg_avoided_input_tokens_pct']:.2f}% | "
            f"{tier['baseline_ttft_p95_ms']:.2f} ms | {tier['highway_ttft_p95_ms']:.2f} ms | "
            f"{tier['baseline_total_p95_ms']:.2f} ms | {tier['highway_total_p95_ms']:.2f} ms | "
            f"{tier['context_p95_ms']:.2f} ms | {tier['avg_embedding_rows_scanned']:.2f} | "
            f"{tier['avg_blocks_materialized']:.2f} | {tier['metrics_complete_rate']:.2f}% |"
        )
    lines.extend([
        "",
        "## Why this matters",
        "",
        "Token economy is accepted only when answer quality stays correct.",
        "The fake client isolates Highway runtime behavior from model randomness.",
        "A real local LLM should only be connected after this benchmark remains validating.",
        "",
        f"Metrics JSON: `{_display_path(metrics_path)}`",
        f"Records JSONL: `{_display_path(records_path)}`",
    ])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _run_one_tier(
    output_path: Path,
    size: int,
    query_count: int,
    seed: int,
    strategy: str,
    input_cost_per_million: float,
    output_cost_per_million: float,
) -> List[Dict[str, Any]]:
    dataset = generate_scaleup_dataset(
        output_path / f"dataset_{size}",
        total_blocks=size,
        query_count=query_count,
        seed=seed,
        mixed_query_set="marker,entity",
    )
    workload = _read_jsonl(dataset.workload_path, limit=query_count)
    all_blocks = _read_jsonl(dataset.index_dir / "blocks.jsonl")
    engine = HighwayContextEngine(
        index_dir=dataset.index_dir,
        embed_model=SyntheticScaleupEmbedder(),
        model_profile=DEFAULT_MODEL_PROFILE,
        input_cost_per_million=input_cost_per_million,
        output_cost_per_million=output_cost_per_million,
    )
    runtime = HighwayLLMRuntime(engine)
    client = DeterministicReflectiveClient()
    resolver = EvidenceResolver()
    records = []
    for row in workload:
        request = ContextRequest(
            user_turn=row["question"],
            session_id=f"llm_runtime_fake_{size}",
            strategy=strategy,
        )
        start = time.perf_counter()
        pack = engine.retrieve(request, top_k=50)
        context_latency_ms = (time.perf_counter() - start) * 1000.0
        baseline_prompt = _baseline_prompt(row["question"], all_blocks)
        baseline_active, _, _ = resolver.resolve([dict(block) for block in all_blocks], pack.query_ir)
        baseline_response = client.answer(
            prompt=baseline_prompt,
            query_ir=pack.query_ir,
            evidence=baseline_active,
            expected_answer=row["expected_answer"],
            query_id=row["id"],
        )
        highway_result = runtime.answer_with_client(
            request,
            client,
            baseline_context=all_blocks,
            expected_answer=row["expected_answer"],
            top_k=50,
        )
        highway_response = highway_result["response"]
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
        baseline_is_em = _clean_answer(baseline_response["answer"]) == _clean_answer(row["expected_answer"])
        highway_is_em = _clean_answer(highway_response["answer"]) == _clean_answer(row["expected_answer"])
        metrics = highway_result["context_pack"]["metrics"]
        records.append({
            "size": size,
            "id": row["id"],
            "category": row["category"],
            "question": row["question"],
            "baseline_answer": baseline_response["answer"],
            "highway_answer": highway_response["answer"],
            "expected_answer": row["expected_answer"],
            "baseline_is_em": baseline_is_em,
            "highway_is_em": highway_is_em,
            "quality_delta": int(highway_is_em) - int(baseline_is_em),
            "baseline_input_tokens": baseline_response["input_tokens"],
            "highway_input_tokens": highway_response["input_tokens"],
            "avoided_input_tokens": economics.avoided_input_tokens,
            "avoided_input_tokens_pct": (
                economics.avoided_input_tokens / economics.baseline_input_tokens * 100.0
                if economics.baseline_input_tokens else 0.0
            ),
            "baseline_output_tokens": baseline_response["output_tokens"],
            "highway_output_tokens": highway_response["output_tokens"],
            "baseline_ttft_ms": baseline_response["ttft_ms"],
            "highway_ttft_ms": highway_response["ttft_ms"],
            "baseline_total_ms": baseline_response["total_ms"],
            "highway_total_ms": highway_response["total_ms"],
            "baseline_input_tokens_per_second": baseline_response["input_tokens_per_second"],
            "highway_input_tokens_per_second": highway_response["input_tokens_per_second"],
            "baseline_output_tokens_per_second": baseline_response["output_tokens_per_second"],
            "highway_output_tokens_per_second": highway_response["output_tokens_per_second"],
            "kv_bytes_estimated": economics.kv_bytes_estimated or 0,
            "kv_bytes_avoided_estimated": economics.kv_bytes_avoided_estimated or 0,
            "cost_estimated_usd": economics.cost_estimated_usd,
            "cost_avoided_estimated_usd": economics.cost_avoided_estimated_usd,
            "context_latency_ms": float(metrics.get("latency_ms", context_latency_ms)),
            "embedding_rows_scanned": int(metrics.get("embedding_rows_scanned", 0)),
            "blocks_materialized": int(metrics.get("blocks_materialized", 0)),
            "bytes_read": int(metrics.get("bytes_read", 0)),
            "ann_used": bool(metrics.get("ann_used", False)),
            "ann_backend": str(metrics.get("ann_backend", "none")),
            "baseline_prompt_tokens_verified": estimate_tokens(baseline_prompt),
        })
    return records


def run_llm_runtime_fake_benchmark(
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    sizes: Sequence[int] = (1000, 10000, 100000),
    query_count: int = 20,
    seed: int = 42,
    strategy: str = "ooc_marker_entity_pruned",
    input_cost_per_million: float = 1.0,
    output_cost_per_million: float = 2.0,
) -> Dict[str, Any]:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    all_records: List[Dict[str, Any]] = []
    tiers = []
    for size in sizes:
        records = _run_one_tier(
            output_path=output_path,
            size=int(size),
            query_count=query_count,
            seed=seed,
            strategy=strategy,
            input_cost_per_million=input_cost_per_million,
            output_cost_per_million=output_cost_per_million,
        )
        all_records.extend(records)
        tiers.append(_summarize_tier(int(size), records))

    status = "VALIDATING" if tiers and all(_tier_is_validating(tier) for tier in tiers) else "NON_VALIDATING"
    summary = {
        "status": status,
        "query_count": query_count,
        "strategy": strategy,
        "seed": seed,
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
    parser.add_argument("--output-dir", type=str, default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--sizes", type=str, default="1000,10000,100000")
    parser.add_argument("--queries", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--strategy", type=str, default="ooc_marker_entity_pruned")
    parser.add_argument("--input-cost-per-million", type=float, default=1.0)
    parser.add_argument("--output-cost-per-million", type=float, default=2.0)
    args = parser.parse_args()
    result = run_llm_runtime_fake_benchmark(
        output_dir=args.output_dir,
        sizes=_parse_sizes(args.sizes),
        query_count=args.queries,
        seed=args.seed,
        strategy=args.strategy,
        input_cost_per_million=args.input_cost_per_million,
        output_cost_per_million=args.output_cost_per_million,
    )
    print(json.dumps({"output_dir": str(result["output_dir"]), "summary": result["summary"]}, indent=2))


if __name__ == "__main__":
    main()
