import argparse
import json
import time
from pathlib import Path
from statistics import mean
from typing import Any, Dict, List, Sequence

import numpy as np

from highway.benchmarks.ooc_scaleup import (
    _execute_kernel,
    _read_jsonl,
    generate_scaleup_dataset,
)
from highway.paths import DEFAULT_RUNS_DIR
from highway.retrieval.evidence_resolver import EvidenceResolver
from highway.runtime.hardware_budget import HardwareBudget
from highway.storage.out_of_core_index import OutOfCoreIndex
from highway.storage.semantic_embedder import DEFAULT_EMBEDDING_MODEL, create_semantic_embedder
from highway.storage.vector_index import build_vector_index
from highway.runners.run_poc234_kernel_hardening import clean_answer


DEFAULT_OUTPUT_DIR = DEFAULT_RUNS_DIR / "semantic_ann_quality"


def _write_jsonl(path: Path, rows: Sequence[Dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def _display_path(path: Path) -> str:
    try:
        return path.resolve().relative_to(Path.cwd().resolve()).as_posix()
    except ValueError:
        return path.as_posix()


def _percentile(values: Sequence[float], pct: float) -> float:
    if not values:
        return 0.0
    return float(np.percentile(list(values), pct))


def _search_records(
    index: OutOfCoreIndex,
    workload: Sequence[Dict[str, Any]],
    strategy: str,
    candidate_cap: int,
    lexical_k: int = 0,
    reranker_input_k: int = 0,
    reranker_output_k: int = 0,
    full_scan_ids_by_query: Dict[str, set[str]] | None = None,
) -> List[Dict[str, Any]]:
    resolver = EvidenceResolver()
    records = []
    for row in workload:
        start = time.perf_counter()
        candidates, query_ir, telemetry = index.search(row["question"], top_k=50, strategy=strategy)
        latency_ms = (time.perf_counter() - start) * 1000.0
        active, _, _ = resolver.resolve(candidates, query_ir)
        audit = _execute_kernel(row["category"], query_ir, active, q_id=row["id"])
        answer = audit.get("answer", audit.get("status", "NOT_FOUND"))
        candidate_ids = {candidate["block_id"] for candidate in candidates}
        oracle_ids = full_scan_ids_by_query.get(row["id"], set()) if full_scan_ids_by_query else candidate_ids
        recall = len(candidate_ids & oracle_ids) / len(oracle_ids) * 100.0 if oracle_ids else 100.0
        records.append({
            "id": row["id"],
            "category": row["category"],
            "question": row["question"],
            "strategy": strategy,
            "candidate_cap": candidate_cap,
            "lexical_k": lexical_k,
            "reranker_input_k": reranker_input_k,
            "reranker_output_k": reranker_output_k,
            "workload_type": "semantic_ann",
            "expected_answer": row["expected_answer"],
            "answer": answer,
            "is_em": clean_answer(answer) == clean_answer(row["expected_answer"]),
            "recall_at_k": recall,
            "latency_ms": latency_ms,
            "embedding_rows_scanned": int(telemetry.get("embedding_rows_scanned", 0)),
            "blocks_materialized": int(telemetry.get("blocks_materialized", 0)),
            "ann_used": bool(telemetry.get("ann_used", False)),
            "ann_backend": str(telemetry.get("ann_backend", "none")),
            "candidate_sources": list(telemetry.get("candidate_sources", [])),
            "semantic_lexical_rescue_used": bool(telemetry.get("semantic_lexical_rescue_used", False)),
            "semantic_rerank_rescue_used": bool(telemetry.get("semantic_rerank_rescue_used", False)),
            "semantic_field_rescue_used": bool(telemetry.get("semantic_field_rescue_used", False)),
            "semantic_cross_encoder_rescue_used": bool(telemetry.get("semantic_cross_encoder_rescue_used", False)),
            "field_posting_candidates": int(telemetry.get("field_posting_candidates", 0)),
            "reranker_backend": str(telemetry.get("reranker_backend", "none")),
            "reranker_available": bool(telemetry.get("reranker_available", False)),
            "reranker_fallback_reason": str(telemetry.get("reranker_fallback_reason", "")),
            "reranker_model": str(telemetry.get("reranker_model", "")),
            "reranker_batch_size": int(telemetry.get("reranker_batch_size", 0)),
            "reranker_candidates_in": int(telemetry.get("reranker_candidates_in", 0)),
            "reranker_candidates_out": int(telemetry.get("reranker_candidates_out", 0)),
            "reranker_latency_ms": float(telemetry.get("reranker_latency_ms", 0.0)),
            "metrics_complete": all(
                key in telemetry
                for key in ("embedding_rows_scanned", "blocks_materialized", "ann_used", "ann_backend")
            ),
            "candidate_block_ids": sorted(candidate_ids),
        })
    return records


def _summarize_strategy(
    records: Sequence[Dict[str, Any]],
    strategy: str,
    candidate_cap: int,
    lexical_k: int,
    reranker_input_k: int = 0,
    reranker_output_k: int = 0,
) -> Dict[str, Any]:
    selected = [
        record
        for record in records
        if (
            record["strategy"] == strategy
            and int(record["candidate_cap"]) == int(candidate_cap)
            and int(record.get("lexical_k", 0)) == int(lexical_k)
            and int(record.get("reranker_input_k", 0)) == int(reranker_input_k)
            and int(record.get("reranker_output_k", 0)) == int(reranker_output_k)
        )
    ]
    latencies = [float(record["latency_ms"]) for record in selected]
    reranker_latencies = [float(record.get("reranker_latency_ms", 0.0)) for record in selected]
    available_values = [bool(record.get("reranker_available", False)) for record in selected]
    fallback_reasons = sorted(
        set(str(record.get("reranker_fallback_reason", "")) for record in selected if record.get("reranker_fallback_reason"))
    )
    return {
        "strategy": strategy,
        "candidate_cap": candidate_cap,
        "lexical_k": lexical_k,
        "reranker_input_k": reranker_input_k,
        "reranker_output_k": reranker_output_k,
        "count": len(selected),
        "em": float(mean(record["is_em"] for record in selected) * 100.0) if selected else 0.0,
        "recall_at_k": float(mean(record["recall_at_k"] for record in selected)) if selected else 0.0,
        "p95_latency_ms": _percentile(latencies, 95),
        "avg_embedding_rows_scanned": (
            float(mean(record["embedding_rows_scanned"] for record in selected)) if selected else 0.0
        ),
        "avg_blocks_materialized": (
            float(mean(record["blocks_materialized"] for record in selected)) if selected else 0.0
        ),
        "avg_reranker_candidates_in": (
            float(mean(record.get("reranker_candidates_in", 0) for record in selected)) if selected else 0.0
        ),
        "avg_reranker_candidates_out": (
            float(mean(record.get("reranker_candidates_out", 0) for record in selected)) if selected else 0.0
        ),
        "p95_reranker_latency_ms": _percentile(reranker_latencies, 95),
        "reranker_available_rate": float(mean(available_values) * 100.0) if selected else 0.0,
        "reranker_fallback_reasons": fallback_reasons,
        "metrics_complete_rate": (
            float(mean(record["metrics_complete"] for record in selected) * 100.0) if selected else 0.0
        ),
    }


def _summarize_tier(
    size: int,
    records: Sequence[Dict[str, Any]],
    recall_gate: float,
    strategies: Sequence[str],
    candidate_sweep: Sequence[int],
    lexical_sweep: Sequence[int],
    reranker_input_sweep: Sequence[int],
    reranker_output_sweep: Sequence[int],
    latency_budgets_ms: Sequence[float],
    embedding_metadata: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    by_strategy = []
    for cap in candidate_sweep:
        for strategy in strategies:
            lexical_values = (
                lexical_sweep
                if strategy in {
                    "ooc_semantic_lexical_rescue",
                    "ooc_semantic_rerank_rescue",
                    "ooc_semantic_field_rescue",
                    "ooc_semantic_cross_encoder_rescue",
                }
                else [0]
            )
            for lexical_k in lexical_values:
                reranker_input_values = reranker_input_sweep if strategy == "ooc_semantic_cross_encoder_rescue" else [0]
                reranker_output_values = reranker_output_sweep if strategy == "ooc_semantic_cross_encoder_rescue" else [0]
                for reranker_input_k in reranker_input_values:
                    for reranker_output_k in reranker_output_values:
                        by_strategy.append(
                            _summarize_strategy(
                                records,
                                strategy,
                                cap,
                                int(lexical_k),
                                int(reranker_input_k),
                                int(reranker_output_k),
                            )
                        )
    hnsw = max(
        (item for item in by_strategy if item["strategy"] == "ooc_ann_hnsw"),
        key=lambda item: item["recall_at_k"],
        default={"recall_at_k": 0.0, "em": 0.0, "p95_latency_ms": 0.0},
    )
    rescue_items = [item for item in by_strategy if item["strategy"] == "ooc_semantic_rescue_hybrid"]
    best_recall = max(by_strategy, key=lambda item: item["recall_at_k"])
    best_latency = min(by_strategy, key=lambda item: item["p95_latency_ms"])
    best_tradeoff = max(
        by_strategy,
        key=lambda item: (item["recall_at_k"] / max(1.0, item["p95_latency_ms"]), item["recall_at_k"]),
    )
    best_rescue_recall = max((item["recall_at_k"] for item in rescue_items), default=0.0)
    best_under_budget: Dict[str, Dict[str, Any]] = {}
    for budget in latency_budgets_ms:
        eligible = [item for item in by_strategy if item["p95_latency_ms"] <= float(budget)]
        best_under_budget[str(int(budget))] = (
            max(eligible, key=lambda item: item["recall_at_k"])
            if eligible
            else {
                "strategy": "none",
                "candidate_cap": 0,
                "lexical_k": 0,
                "recall_at_k": 0.0,
                "p95_latency_ms": 0.0,
            }
        )
    best_lexical_recall = max(
        (item["recall_at_k"] for item in by_strategy if item["strategy"] == "ooc_semantic_lexical_rescue"),
        default=0.0,
    )
    best_field_recall = max(
        (item["recall_at_k"] for item in by_strategy if item["strategy"] == "ooc_semantic_field_rescue"),
        default=0.0,
    )
    runtime_items = [item for item in by_strategy if item["strategy"] != "ooc_full_scan"]
    runtime_best_recall = (
        max(runtime_items, key=lambda item: item["recall_at_k"])
        if runtime_items
        else {
            "strategy": "none",
            "candidate_cap": 0,
            "lexical_k": 0,
            "recall_at_k": 0.0,
            "p95_latency_ms": 0.0,
        }
    )
    runtime_under_200 = [item for item in runtime_items if item["p95_latency_ms"] <= 200.0]
    runtime_best_under_200 = (
        max(runtime_under_200, key=lambda item: item["recall_at_k"])
        if runtime_under_200
        else {
            "strategy": "none",
            "candidate_cap": 0,
            "lexical_k": 0,
            "recall_at_k": 0.0,
            "p95_latency_ms": 0.0,
        }
    )
    cross_encoder_items = [
        item for item in by_strategy if item["strategy"] == "ooc_semantic_cross_encoder_rescue" and item["count"] > 0
    ]
    cross_encoder_ready = all(item.get("reranker_available_rate", 0.0) == 100.0 for item in cross_encoder_items)
    embedding_metadata = dict(embedding_metadata or {})
    embedding_ready = not bool(embedding_metadata.get("embedding_fallback_reason"))
    complete = [item["metrics_complete_rate"] == 100.0 for item in by_strategy]
    return {
        "size": size,
        "workload_type": "semantic_ann",
        "embedding_backend": embedding_metadata.get("embedding_backend", "synthetic"),
        "embedding_model": embedding_metadata.get("embedding_model", "synthetic_scaleup"),
        "embedding_dim": int(embedding_metadata.get("embedding_dim", 0) or 0),
        "embedding_local_files_only": bool(embedding_metadata.get("embedding_local_files_only", False)),
        "embedding_batch_size": int(embedding_metadata.get("embedding_batch_size", 0) or 0),
        "embedding_latency_ms": float(embedding_metadata.get("embedding_latency_ms", 0.0) or 0.0),
        "embedding_fallback_reason": str(embedding_metadata.get("embedding_fallback_reason", "")),
        "embedding_ready": embedding_ready,
        "strategies": list(strategies),
        "candidate_sweep": list(candidate_sweep),
        "lexical_sweep": list(lexical_sweep),
        "reranker_input_sweep": list(reranker_input_sweep),
        "reranker_output_sweep": list(reranker_output_sweep),
        "strategy_summaries": by_strategy,
        "hnsw_recall_at_k": hnsw["recall_at_k"],
        "hnsw_em": hnsw["em"],
        "hnsw_p95_latency_ms": hnsw["p95_latency_ms"],
        "best_recall": best_recall,
        "best_latency": best_latency,
        "best_tradeoff": best_tradeoff,
        "best_recall_under_100ms": best_under_budget.get("100"),
        "best_recall_under_200ms": best_under_budget.get("200"),
        "best_rescue_recall_at_k": best_rescue_recall,
        "best_lexical_rescue_recall_at_k": best_lexical_recall,
        "best_field_rescue_recall_at_k": best_field_recall,
        "runtime_best_recall": runtime_best_recall,
        "runtime_best_under_200ms": runtime_best_under_200,
        "cross_encoder_ready": cross_encoder_ready,
        "recall_gate": recall_gate,
        "metrics_complete_rate": float(mean(complete) * 100.0) if complete else 0.0,
        "validating": (
            runtime_best_under_200.get("recall_at_k", 0.0) >= recall_gate
            and all(complete)
            and cross_encoder_ready
            and embedding_ready
        ),
        "quality_only": (
            runtime_best_recall.get("recall_at_k", 0.0) >= recall_gate
            and all(complete)
            and cross_encoder_ready
            and embedding_ready
        ),
    }


def _write_report(path: Path, summary: Dict[str, Any], metrics_path: Path, records_path: Path) -> None:
    lines = [
        "# Semantic ANN Quality Benchmark",
        "",
        f"Verdict: {summary['status']}",
        "",
        "| Size | Strategy | ANN cap | Lexical cap | Rerank in/out | EM | Recall@k | p95 latency | Rows scanned | Reranker p95 | Reranker avail | Blocks materialized | Metrics complete |",
        "|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for tier in summary["tiers"]:
        for item in tier["strategy_summaries"]:
            lines.append(
                f"| {tier['size']:,} | {item['strategy']} | {item['candidate_cap']} | "
                f"{item.get('lexical_k', 0)} | "
                f"{item.get('reranker_input_k', 0)}/{item.get('reranker_output_k', 0)} | "
                f"{item['em']:.2f}% | "
                f"{item['recall_at_k']:.2f}% | {item['p95_latency_ms']:.2f} ms | "
                f"{item['avg_embedding_rows_scanned']:.2f} | {item['p95_reranker_latency_ms']:.2f} ms | "
                f"{item.get('reranker_available_rate', 0.0):.2f}% | "
                f"{item['avg_blocks_materialized']:.2f} | "
                f"{item['metrics_complete_rate']:.2f}% |"
            )
    lines.extend([
        "",
        "## Best Compromises",
        "",
        "| Size | Best recall | Best latency | Best recall <100ms | Best recall <200ms | Best tradeoff |",
        "|---:|---|---|---|---|---|",
    ])
    for tier in summary["tiers"]:
        best_recall = tier["best_recall"]
        best_latency = tier["best_latency"]
        best_tradeoff = tier["best_tradeoff"]
        best_100 = tier["best_recall_under_100ms"]
        best_200 = tier["best_recall_under_200ms"]
        lines.append(
            f"| {tier['size']:,} | {best_recall['strategy']} cap {best_recall['candidate_cap']} "
            f"lex {best_recall.get('lexical_k', 0)} ({best_recall['recall_at_k']:.2f}%) | "
            f"{best_latency['strategy']} cap {best_latency['candidate_cap']} "
            f"lex {best_latency.get('lexical_k', 0)} ({best_latency['p95_latency_ms']:.2f} ms) | "
            f"{best_100['strategy']} cap {best_100['candidate_cap']} lex {best_100.get('lexical_k', 0)} "
            f"({best_100['recall_at_k']:.2f}%, {best_100['p95_latency_ms']:.2f} ms) | "
            f"{best_200['strategy']} cap {best_200['candidate_cap']} lex {best_200.get('lexical_k', 0)} "
            f"({best_200['recall_at_k']:.2f}%, {best_200['p95_latency_ms']:.2f} ms) | "
            f"{best_tradeoff['strategy']} cap {best_tradeoff['candidate_cap']} lex {best_tradeoff.get('lexical_k', 0)} "
            f"({best_tradeoff['recall_at_k']:.2f}%, {best_tradeoff['p95_latency_ms']:.2f} ms) |"
        )
    lines.extend([
        "",
        "## Embedding Backend",
        "",
        "| Size | Backend | Model | Dim | Local only | Batch | Encode latency | Fallback |",
        "|---:|---|---|---:|---:|---:|---:|---|",
    ])
    for tier in summary["tiers"]:
        fallback = tier.get("embedding_fallback_reason", "")
        compact_fallback = " ".join(str(fallback).split()) if fallback else ""
        lines.append(
            f"| {tier['size']:,} | {tier.get('embedding_backend', 'synthetic')} | "
            f"{tier.get('embedding_model', 'synthetic_scaleup')} | "
            f"{int(tier.get('embedding_dim', 0))} | "
            f"{bool(tier.get('embedding_local_files_only', False))} | "
            f"{int(tier.get('embedding_batch_size', 0))} | "
            f"{float(tier.get('embedding_latency_ms', 0.0)):.2f} ms | "
            f"{compact_fallback} |"
        )
    fallback_reasons = sorted(
        {
            reason
            for tier in summary["tiers"]
            for item in tier["strategy_summaries"]
            for reason in item.get("reranker_fallback_reasons", [])
            if reason
        }
    )
    if fallback_reasons:
        lines.extend([
            "",
            "## Reranker Fallbacks",
            "",
        ])
        for reason in fallback_reasons:
            compact = " ".join(str(reason).split())
            lines.append(f"- {compact}")
    lines.extend([
        "",
        "Validation requires the best semantic path to reach the recall gate under 200 ms p95 on the executed tiers.",
        "",
        "If the verdict is NON_VALIDATING, broad real-LLM semantic demos remain blocked. The next options are a stronger embedder, "
        "a cross-encoder/reranker stage, or a more specialized lexical candidate index before mmap rerank.",
        "",
        "This benchmark is allowed to be NON_VALIDATING. Its job is to expose semantic ANN quality risk before a real LLM is connected.",
        f"Metrics JSON: `{_display_path(metrics_path)}`",
        f"Records JSONL: `{_display_path(records_path)}`",
    ])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _run_tier(
    output_path: Path,
    size: int,
    query_count: int,
    seed: int,
    strategies: Sequence[str],
    candidate_sweep: Sequence[int],
    lexical_sweep: Sequence[int],
    reranker_backend: str,
    reranker_model: str,
    reranker_input_sweep: Sequence[int],
    reranker_output_sweep: Sequence[int],
    reranker_local_files_only: bool,
    embedding_backend: str,
    embedding_model: str,
    embedding_local_files_only: bool,
    embedding_batch_size: int,
) -> tuple[List[Dict[str, Any]], Dict[str, Any]]:
    embedder = create_semantic_embedder(
        backend=embedding_backend,
        model_name=embedding_model,
        local_files_only=embedding_local_files_only,
        batch_size=embedding_batch_size,
    )
    dataset = generate_scaleup_dataset(
        output_path / f"dataset_{size}",
        total_blocks=size,
        query_count=query_count,
        seed=seed,
        mixed_query_set="semantic",
        embedder=embedder,
    )
    embedding_metadata = (
        embedder.embedding_metadata()
        if hasattr(embedder, "embedding_metadata")
        else {
            "embedding_backend": "synthetic",
            "embedding_model": "synthetic_scaleup",
            "embedding_dim": 0,
            "embedding_local_files_only": False,
            "embedding_batch_size": 0,
            "embedding_latency_ms": 0.0,
            "embedding_fallback_reason": "",
        }
    )
    build_vector_index(
        embeddings_path=dataset.index_dir / "embeddings.npy",
        output_path=dataset.index_dir,
        backend="faiss_hnsw",
        params={"M": 32, "efConstruction": 80, "efSearch": 64},
    )
    workload = _read_jsonl(dataset.workload_path, limit=query_count)
    records = []
    for cap in candidate_sweep:
        oracle_index = OutOfCoreIndex(
            dataset.index_dir,
            embed_model=embedder,
            hardware_budget=HardwareBudget(max_candidates=max(50, int(cap))),
        )
        full_oracle = _search_records(oracle_index, workload, "ooc_full_scan", candidate_cap=int(cap))
        oracle = {record["id"]: set(record["candidate_block_ids"]) for record in full_oracle}
        for strategy in strategies:
            lexical_values = (
                lexical_sweep
                if strategy in {
                    "ooc_semantic_lexical_rescue",
                    "ooc_semantic_rerank_rescue",
                    "ooc_semantic_field_rescue",
                    "ooc_semantic_cross_encoder_rescue",
                }
                else [0]
            )
            for lexical_k in lexical_values:
                reranker_input_values = reranker_input_sweep if strategy == "ooc_semantic_cross_encoder_rescue" else [0]
                reranker_output_values = reranker_output_sweep if strategy == "ooc_semantic_cross_encoder_rescue" else [0]
                for reranker_input_k in reranker_input_values:
                    for reranker_output_k in reranker_output_values:
                        index = OutOfCoreIndex(
                            dataset.index_dir,
                            embed_model=embedder,
                            hardware_budget=HardwareBudget(
                                max_candidates=max(50, int(cap)),
                                semantic_ann_k=int(cap),
                                semantic_rerank_k=max(int(cap), int(lexical_k), int(reranker_output_k)),
                                semantic_lexical_k=max(int(lexical_k), 0),
                                semantic_reranker_backend=reranker_backend,
                                semantic_reranker_input_k=max(int(reranker_input_k), 0),
                                semantic_reranker_output_k=max(int(reranker_output_k), 0),
                                semantic_reranker_batch_size=32,
                                semantic_reranker_model=reranker_model,
                                semantic_reranker_local_files_only=reranker_local_files_only,
                            ),
                        )
                        records.extend(
                            _search_records(
                                index,
                                workload,
                                strategy,
                                candidate_cap=int(cap),
                                lexical_k=int(lexical_k),
                                reranker_input_k=int(reranker_input_k),
                                reranker_output_k=int(reranker_output_k),
                                full_scan_ids_by_query=oracle,
                            )
                        )
    return records, embedding_metadata


def run_semantic_ann_quality_benchmark(
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    sizes: Sequence[int] = (1000, 10000, 100000),
    query_count: int = 20,
    seed: int = 42,
    recall_gate: float = 80.0,
    strategies: Sequence[str] = (
        "ooc_full_scan",
        "ooc_ann_hnsw",
        "ooc_ann_pruned_hybrid",
        "ooc_semantic_rescue_hybrid",
        "ooc_semantic_lexical_rescue",
        "ooc_semantic_rerank_rescue",
        "ooc_semantic_field_rescue",
    ),
    candidate_sweep: Sequence[int] = (200, 500, 1000, 2000),
    lexical_sweep: Sequence[int] = (1000, 5000, 10000),
    reranker_backend: str = "cross_encoder",
    reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
    reranker_input_sweep: Sequence[int] = (500, 1000),
    reranker_output_sweep: Sequence[int] = (200, 500),
    reranker_local_files_only: bool = False,
    embedding_backend: str = "synthetic",
    embedding_model: str = DEFAULT_EMBEDDING_MODEL,
    embedding_local_files_only: bool = False,
    embedding_batch_size: int = 64,
    latency_budgets_ms: Sequence[float] = (100.0, 200.0),
) -> Dict[str, Any]:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    all_records: List[Dict[str, Any]] = []
    tiers = []
    for size in sizes:
        records, embedding_metadata = _run_tier(
            output_path,
            int(size),
            query_count,
            seed,
            strategies,
            candidate_sweep,
            lexical_sweep,
            reranker_backend,
            reranker_model,
            reranker_input_sweep,
            reranker_output_sweep,
            reranker_local_files_only,
            embedding_backend,
            embedding_model,
            embedding_local_files_only,
            embedding_batch_size,
        )
        all_records.extend(records)
        tiers.append(
            _summarize_tier(
                int(size),
                records,
                recall_gate,
                strategies,
                candidate_sweep,
                lexical_sweep,
                reranker_input_sweep,
                reranker_output_sweep,
                latency_budgets_ms,
                embedding_metadata,
            )
        )
    required_tiers = [tier for tier in tiers if int(tier["size"]) >= 100000] or tiers
    if required_tiers and all(tier["validating"] for tier in required_tiers):
        status = "VALIDATING"
    elif required_tiers and all(tier["quality_only"] for tier in required_tiers):
        status = "QUALITY_ONLY"
    else:
        status = "NON_VALIDATING"
    summary = {
        "status": status,
        "query_count": query_count,
        "seed": seed,
        "recall_gate": recall_gate,
        "strategies": list(strategies),
        "candidate_sweep": list(candidate_sweep),
        "lexical_sweep": list(lexical_sweep),
        "reranker_backend": reranker_backend,
        "reranker_model": reranker_model,
        "reranker_input_sweep": list(reranker_input_sweep),
        "reranker_output_sweep": list(reranker_output_sweep),
        "reranker_local_files_only": reranker_local_files_only,
        "embedding_backend": embedding_backend,
        "embedding_model": embedding_model,
        "embedding_local_files_only": embedding_local_files_only,
        "embedding_batch_size": int(embedding_batch_size),
        "latency_budgets_ms": [float(value) for value in latency_budgets_ms],
        "tiers": tiers,
    }
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


def _parse_csv(raw: str) -> List[str]:
    return [part.strip() for part in raw.split(",") if part.strip()]


def _parse_float_csv(raw: str) -> List[float]:
    return [float(part.strip()) for part in raw.split(",") if part.strip()]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=str, default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--sizes", type=str, default="1000,10000,100000")
    parser.add_argument("--queries", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--recall-gate", type=float, default=80.0)
    parser.add_argument(
        "--strategies",
        type=str,
        default=(
            "ooc_full_scan,ooc_ann_hnsw,ooc_ann_pruned_hybrid,"
            "ooc_semantic_rescue_hybrid,ooc_semantic_lexical_rescue,"
            "ooc_semantic_rerank_rescue,ooc_semantic_field_rescue,"
            "ooc_semantic_cross_encoder_rescue"
        ),
    )
    parser.add_argument("--candidate-sweep", type=str, default="200,500,1000,2000")
    parser.add_argument("--lexical-sweep", type=str, default="1000,5000,10000")
    parser.add_argument("--reranker-backend", type=str, choices=["cross_encoder", "lexical_field_reranker"], default="cross_encoder")
    parser.add_argument("--reranker-model", type=str, default="cross-encoder/ms-marco-MiniLM-L-6-v2")
    parser.add_argument("--reranker-input-sweep", type=str, default="500,1000")
    parser.add_argument("--reranker-output-sweep", type=str, default="200,500")
    parser.add_argument("--reranker-local-files-only", action="store_true")
    parser.add_argument("--embedding-backend", type=str, choices=["synthetic", "sentence_transformer"], default="synthetic")
    parser.add_argument("--embedding-model", type=str, default=DEFAULT_EMBEDDING_MODEL)
    parser.add_argument("--embedding-local-files-only", action="store_true")
    parser.add_argument("--embedding-batch-size", type=int, default=64)
    parser.add_argument("--latency-budgets-ms", type=str, default="100,200")
    args = parser.parse_args()
    result = run_semantic_ann_quality_benchmark(
        output_dir=args.output_dir,
        sizes=_parse_sizes(args.sizes),
        query_count=args.queries,
        seed=args.seed,
        recall_gate=args.recall_gate,
        strategies=_parse_csv(args.strategies),
        candidate_sweep=_parse_sizes(args.candidate_sweep),
        lexical_sweep=_parse_sizes(args.lexical_sweep),
        reranker_backend=args.reranker_backend,
        reranker_model=args.reranker_model,
        reranker_input_sweep=_parse_sizes(args.reranker_input_sweep),
        reranker_output_sweep=_parse_sizes(args.reranker_output_sweep),
        reranker_local_files_only=bool(args.reranker_local_files_only),
        embedding_backend=args.embedding_backend,
        embedding_model=args.embedding_model,
        embedding_local_files_only=bool(args.embedding_local_files_only),
        embedding_batch_size=int(args.embedding_batch_size),
        latency_budgets_ms=_parse_float_csv(args.latency_budgets_ms),
    )
    print(json.dumps({"output_dir": str(result["output_dir"]), "summary": result["summary"]}, indent=2))


if __name__ == "__main__":
    main()
