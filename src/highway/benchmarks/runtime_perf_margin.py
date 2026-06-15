import argparse
import json
import time
from pathlib import Path
from statistics import mean
from typing import Any, Dict, List, Sequence

import numpy as np

from highway.benchmarks.ooc_scaleup import SyntheticScaleupEmbedder, generate_scaleup_dataset
from highway.paths import DEFAULT_RUNS_DIR
from highway.runtime.context_engine import ContextRequest, HighwayContextEngine
from highway.runtime.llm_runtime import DeterministicReflectiveClient, HighwayLLMRuntime
from highway.runtime.token_economics import ModelProfile


DEFAULT_OUTPUT_DIR = DEFAULT_RUNS_DIR / "runtime_perf_margin"
DEFAULT_MODEL_PROFILE = ModelProfile(name="runtime_perf_margin_fake_model", layers=24, hidden_size=1024)


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


def _percentile(values: Sequence[float], pct: float) -> float:
    if not values:
        return 0.0
    return float(np.percentile(list(values), pct))


def _summarize_tier(size: int, records: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    context_latencies = [float(record["context_latency_ms"]) for record in records]
    runtime_latencies = [float(record["runtime_total_ms"]) for record in records]
    complete = [
        all(
            key in record
            for key in (
                "context_latency_ms",
                "runtime_total_ms",
                "embedding_rows_scanned",
                "blocks_materialized",
                "tokens_avoided",
                "kv_bytes_avoided_estimated",
                "hotset_hits",
                "workload_type",
            )
        )
        for record in records
    ]
    return {
        "size": size,
        "workload_type": "structured_exact",
        "count": len(records),
        "context_mean_ms": float(mean(context_latencies)) if context_latencies else 0.0,
        "context_p50_ms": _percentile(context_latencies, 50),
        "context_p95_ms": _percentile(context_latencies, 95),
        "runtime_mean_ms": float(mean(runtime_latencies)) if runtime_latencies else 0.0,
        "runtime_p50_ms": _percentile(runtime_latencies, 50),
        "runtime_p95_ms": _percentile(runtime_latencies, 95),
        "avg_embedding_rows_scanned": (
            float(mean(record["embedding_rows_scanned"] for record in records)) if records else 0.0
        ),
        "avg_blocks_materialized": (
            float(mean(record["blocks_materialized"] for record in records)) if records else 0.0
        ),
        "avg_hotset_hits": float(mean(record["hotset_hits"] for record in records)) if records else 0.0,
        "avg_tokens_avoided": float(mean(record["tokens_avoided"] for record in records)) if records else 0.0,
        "avg_kv_bytes_avoided_estimated": (
            float(mean(record["kv_bytes_avoided_estimated"] for record in records)) if records else 0.0
        ),
        "metrics_complete_rate": float(mean(complete) * 100.0) if records else 0.0,
    }


def _tier_is_validating(tier: Dict[str, Any]) -> bool:
    return tier["context_p95_ms"] <= 50.0 and tier["metrics_complete_rate"] == 100.0


def _write_report(path: Path, summary: Dict[str, Any], metrics_path: Path, records_path: Path) -> None:
    lines = [
        "# Runtime Perf Margin Benchmark",
        "",
        f"Verdict: {summary['status']}",
        "",
        "| Size | Workload | Context p95 | Runtime p95 | Rows scanned | Blocks materialized | Hotset hits | Tokens avoided | KV avoided | Metrics complete |",
        "|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for tier in summary["tiers"]:
        lines.append(
            f"| {tier['size']:,} | {tier['workload_type']} | {tier['context_p95_ms']:.2f} ms | "
            f"{tier['runtime_p95_ms']:.2f} ms | {tier['avg_embedding_rows_scanned']:.2f} | "
            f"{tier['avg_blocks_materialized']:.2f} | {tier['avg_hotset_hits']:.2f} | "
            f"{tier['avg_tokens_avoided']:.2f} | {tier['avg_kv_bytes_avoided_estimated']:.2f} | "
            f"{tier['metrics_complete_rate']:.2f}% |"
        )
    lines.extend([
        "",
        "This benchmark measures structured marker/entity retrieval and fake runtime response without double retrieval.",
        f"Metrics JSON: `{_display_path(metrics_path)}`",
        f"Records JSONL: `{_display_path(records_path)}`",
    ])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _run_tier(output_path: Path, size: int, query_count: int, seed: int) -> List[Dict[str, Any]]:
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
        input_cost_per_million=1.0,
        output_cost_per_million=2.0,
    )
    runtime = HighwayLLMRuntime(engine)
    client = DeterministicReflectiveClient()
    seen_sources: set[str] = set()
    records = []
    for row in workload:
        request = ContextRequest(
            user_turn=row["question"],
            session_id=f"runtime_perf_margin_{size}",
            strategy="ooc_marker_entity_pruned",
        )
        start = time.perf_counter()
        pack = engine.retrieve(request, top_k=50)
        context_latency_ms = (time.perf_counter() - start) * 1000.0
        hotset_hits = sum(1 for block in pack.blocks if block.source_file in seen_sources)
        seen_sources.update(block.source_file for block in pack.blocks)
        runtime_start = time.perf_counter()
        response = runtime.answer_context_pack(
            pack,
            client,
            baseline_context=None,
            expected_answer=row["expected_answer"],
        )
        runtime_total_ms = (time.perf_counter() - runtime_start) * 1000.0
        economics = response["token_economics"]
        records.append({
            "size": size,
            "id": row["id"],
            "category": row["category"],
            "workload_type": "structured_exact",
            "context_latency_ms": float(pack.metrics.get("latency_ms", context_latency_ms)),
            "runtime_total_ms": runtime_total_ms,
            "embedding_rows_scanned": int(pack.metrics.get("embedding_rows_scanned", 0)),
            "blocks_materialized": int(pack.metrics.get("blocks_materialized", 0)),
            "bytes_read": int(pack.metrics.get("bytes_read", 0)),
            "hotset_hits": hotset_hits,
            "tokens_avoided": int(economics.get("avoided_input_tokens", 0)),
            "kv_bytes_avoided_estimated": int(economics.get("kv_bytes_avoided_estimated") or 0),
            "ann_used": bool(pack.metrics.get("ann_used", False)),
            "ann_backend": str(pack.metrics.get("ann_backend", "none")),
        })
    return records


def run_runtime_perf_margin_benchmark(
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    sizes: Sequence[int] = (1000, 10000, 100000),
    query_count: int = 20,
    seed: int = 42,
) -> Dict[str, Any]:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    all_records: List[Dict[str, Any]] = []
    tiers = []
    for size in sizes:
        records = _run_tier(output_path, int(size), query_count, seed)
        all_records.extend(records)
        tiers.append(_summarize_tier(int(size), records))
    status = "VALIDATING" if tiers and all(_tier_is_validating(tier) for tier in tiers) else "NON_VALIDATING"
    summary = {"status": status, "query_count": query_count, "seed": seed, "tiers": tiers}
    metrics_path = output_path / "metrics.json"
    records_path = output_path / "records.jsonl"
    report_path = output_path / "report.md"
    _write_jsonl(records_path, all_records)
    metrics_path.write_text(json.dumps({"summary": summary}, indent=2, ensure_ascii=False), encoding="utf-8")
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
    args = parser.parse_args()
    result = run_runtime_perf_margin_benchmark(
        output_dir=args.output_dir,
        sizes=_parse_sizes(args.sizes),
        query_count=args.queries,
        seed=args.seed,
    )
    print(json.dumps({"output_dir": str(result["output_dir"]), "summary": result["summary"]}, indent=2))


if __name__ == "__main__":
    main()
