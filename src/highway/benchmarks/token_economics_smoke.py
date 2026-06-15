import argparse
import json
from pathlib import Path
from statistics import mean
from typing import Any, Dict, List, Sequence

import numpy as np

from highway.benchmarks.ooc_scaleup import SyntheticScaleupEmbedder, generate_scaleup_dataset
from highway.paths import DEFAULT_RUNS_DIR
from highway.runtime.context_engine import ContextRequest, HighwayContextEngine
from highway.runtime.token_economics import ModelProfile


DEFAULT_OUTPUT_DIR = DEFAULT_RUNS_DIR / "token_economics_smoke"
DEFAULT_MODEL_PROFILE = ModelProfile(name="smoke_context_model", layers=24, hidden_size=1024, bytes_per_element=2)


def _read_jsonl(path: Path, limit: int | None = None) -> List[Dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
                if limit is not None and len(rows) >= limit:
                    break
    return rows


def _percentile(values: Sequence[float], pct: float) -> float:
    if not values:
        return 0.0
    return float(np.percentile(list(values), pct))


def _average(records: Sequence[Dict[str, Any]], metric_name: str) -> float:
    values = [float(record["metrics"].get(metric_name, 0.0)) for record in records]
    return float(mean(values)) if values else 0.0


def _average_economics(records: Sequence[Dict[str, Any]], metric_name: str) -> float:
    values = [
        float(record["metrics"]["token_economics"].get(metric_name, 0.0) or 0.0)
        for record in records
    ]
    return float(mean(values)) if values else 0.0


def _summarize(records: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    latencies = [float(record["metrics"].get("latency_ms", 0.0)) for record in records]
    avg_baseline = _average_economics(records, "baseline_input_tokens")
    avg_actual = _average_economics(records, "actual_input_tokens")
    avg_avoided = _average_economics(records, "avoided_input_tokens")
    avoided_pct = (avg_avoided / avg_baseline * 100.0) if avg_baseline else 0.0
    return {
        "count": len(records),
        "avg_baseline_input_tokens": avg_baseline,
        "avg_actual_input_tokens": avg_actual,
        "avg_avoided_input_tokens": avg_avoided,
        "avg_avoided_input_tokens_pct": avoided_pct,
        "avg_kv_bytes_estimated": _average_economics(records, "kv_bytes_estimated"),
        "avg_kv_bytes_avoided_estimated": _average_economics(records, "kv_bytes_avoided_estimated"),
        "avg_cost_estimated_usd": _average_economics(records, "cost_estimated_usd"),
        "avg_cost_avoided_estimated_usd": _average_economics(records, "cost_avoided_estimated_usd"),
        "avg_bytes_read": _average(records, "bytes_read"),
        "avg_embedding_rows_scanned": _average(records, "embedding_rows_scanned"),
        "avg_blocks_materialized": _average(records, "blocks_materialized"),
        "mean_latency_ms": float(mean(latencies)) if latencies else 0.0,
        "p95_latency_ms": _percentile(latencies, 95),
    }


def _write_report(path: Path, summary: Dict[str, Any], metrics_path: Path) -> None:
    lines = [
        "# Token Economics Smoke",
        "",
        "This smoke validates the no-LLM context runtime token economics path.",
        "",
        "| Metric | Value |",
        "|---|---:|",
        f"| Queries | {summary['count']} |",
        f"| Average baseline input tokens | {summary['avg_baseline_input_tokens']:.2f} |",
        f"| Average actual input tokens | {summary['avg_actual_input_tokens']:.2f} |",
        f"| Average avoided input tokens | {summary['avg_avoided_input_tokens']:.2f} |",
        f"| Average avoided input tokens pct | {summary['avg_avoided_input_tokens_pct']:.2f}% |",
        f"| Average KV bytes estimated | {summary['avg_kv_bytes_estimated']:.2f} |",
        f"| Average KV bytes avoided estimated | {summary['avg_kv_bytes_avoided_estimated']:.2f} |",
        f"| Average cost estimated USD | {summary['avg_cost_estimated_usd']:.8f} |",
        f"| Average cost avoided estimated USD | {summary['avg_cost_avoided_estimated_usd']:.8f} |",
        f"| Average rows scanned | {summary['avg_embedding_rows_scanned']:.2f} |",
        f"| Average blocks materialized | {summary['avg_blocks_materialized']:.2f} |",
        f"| p95 context latency | {summary['p95_latency_ms']:.2f} ms |",
        "",
        f"Metrics JSON: `{metrics_path.as_posix()}`",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_token_economics_smoke(
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    total_blocks: int = 1000,
    query_count: int = 20,
    seed: int = 42,
    input_cost_per_million: float = 1.0,
    output_cost_per_million: float = 2.0,
) -> Dict[str, Any]:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    dataset = generate_scaleup_dataset(
        output_path / "dataset",
        total_blocks=total_blocks,
        query_count=query_count,
        seed=seed,
        mixed_query_set="marker,entity",
    )
    workload = _read_jsonl(dataset.workload_path, limit=query_count)
    engine = HighwayContextEngine(
        index_dir=dataset.index_dir,
        embed_model=SyntheticScaleupEmbedder(),
        model_profile=DEFAULT_MODEL_PROFILE,
        input_cost_per_million=input_cost_per_million,
        output_cost_per_million=output_cost_per_million,
    )
    records = []
    for row in workload:
        pack = engine.retrieve(
            ContextRequest(
                user_turn=row["question"],
                session_id="token_economics_smoke",
                strategy="ooc_marker_entity_pruned",
                token_budget=4096,
            ),
            top_k=50,
        )
        records.append({
            "id": row["id"],
            "category": row["category"],
            "question": row["question"],
            "expected_answer": row["expected_answer"],
            "context_block_ids": [block.block_id for block in pack.blocks],
            "context_sources": [block.source_file for block in pack.blocks],
            "metrics": pack.metrics,
            "warnings": pack.warnings,
        })

    summary = _summarize(records)
    metrics_path = output_path / "metrics.json"
    report_path = output_path / "report.md"
    metrics_path.write_text(
        json.dumps(
            {
                "summary": summary,
                "model_profile": DEFAULT_MODEL_PROFILE.to_dict(),
                "records": records,
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    _write_report(report_path, summary, metrics_path)
    return {"output_dir": output_path, "metrics_path": metrics_path, "report_path": report_path, "summary": summary}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=str, default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--total-blocks", type=int, default=1000)
    parser.add_argument("--queries", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--input-cost-per-million", type=float, default=1.0)
    parser.add_argument("--output-cost-per-million", type=float, default=2.0)
    args = parser.parse_args()
    result = run_token_economics_smoke(
        output_dir=args.output_dir,
        total_blocks=args.total_blocks,
        query_count=args.queries,
        seed=args.seed,
        input_cost_per_million=args.input_cost_per_million,
        output_cost_per_million=args.output_cost_per_million,
    )
    print(json.dumps({"output_dir": str(result["output_dir"]), "summary": result["summary"]}, indent=2))


if __name__ == "__main__":
    main()
