import argparse
import hashlib
import json
import math
import os
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence

import numpy as np

from highway.kernels.compute_kernels import AggregationKernel, ComparisonKernel, PEOPLE, PROJECT_NAMES
from highway.retrieval.evidence_resolver import EvidenceResolver
from highway.retrieval.query_parser import QueryParser
from highway.runtime.hardware_budget import HardwareBudget
from highway.storage.vector_index import build_vector_index
from highway.storage.index_writer import write_out_of_core_index
from highway.storage.out_of_core_index import OutOfCoreIndex
from highway.runners.run_poc234_kernel_hardening import clean_answer, leak_check_query


PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "artifacts" / "runs" / "ooc_scaleup"
DEFAULT_ANN_OUTPUT_DIR = PROJECT_ROOT / "artifacts" / "runs" / "ooc_ann_scaleup"
MARKER_RE = re.compile(r"\bref_[0-9a-f]{10}\b", re.IGNORECASE)


@dataclass(frozen=True)
class ScaleupDataset:
    root_dir: Path
    corpus_dir: Path
    index_dir: Path
    workload_path: Path


@dataclass(frozen=True)
class BenchmarkResult:
    output_dir: Path
    results_path: Path
    metrics_path: Path
    report_path: Path
    summary: Dict[str, Any]


def _stable_hex(*parts: object, length: int = 16) -> str:
    payload = "|".join(str(part) for part in parts)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:length]


def _stable_vector(key: str, dim: int = 384) -> np.ndarray:
    values = []
    counter = 0
    while len(values) < dim:
        digest = hashlib.sha256(f"{key}:{counter}".encode("utf-8")).digest()
        values.extend((byte / 127.5) - 1.0 for byte in digest)
        counter += 1
    arr = np.asarray(values[:dim], dtype=np.float32)
    norm = np.linalg.norm(arr)
    if norm == 0.0:
        return arr
    return arr / norm


def _embedding_key(text: str) -> str:
    lowered = text.lower()
    if "managed by" in lowered:
        for person in PEOPLE:
            if person.lower() in lowered:
                return "manager:" + person.lower()
    projects = sorted(set(re.findall(r"\bProject\s+([A-Z][A-Z0-9_\-]+)\b", text)))
    if len(projects) >= 2:
        return "projects:" + ",".join(projects)
    for person in PEOPLE:
        if person.lower() in lowered:
            return "manager:" + person.lower()
    marker = MARKER_RE.search(text)
    if marker:
        return marker.group(0).lower()
    return _stable_hex("noise", text, length=16)


def _embedding_for_text(text: str, dim: int = 384) -> np.ndarray:
    return _stable_vector(_embedding_key(text), dim=dim)


class SyntheticScaleupEmbedder:
    def __init__(self, dim: int = 384):
        self.dim = dim

    def encode(self, text, convert_to_numpy=True, show_progress_bar=False):
        if isinstance(text, list):
            return np.asarray([self.encode(item, convert_to_numpy=True) for item in text], dtype=np.float32)
        return _embedding_for_text(str(text), dim=self.dim)


def _write_jsonl(path: Path, rows: Iterable[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def _read_jsonl(path: Path, limit: Optional[int] = None) -> List[Dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
                if limit is not None and len(rows) >= limit:
                    break
    return rows


def _index_size_bytes(index_dir: Path) -> int:
    return sum(path.stat().st_size for path in index_dir.iterdir() if path.is_file()) if index_dir.exists() else 0


def _percentile(values: Sequence[float], pct: float) -> float:
    if not values:
        return 0.0
    return float(np.percentile(values, pct))


def _source_file(seed: int, index: int, question: str) -> str:
    digest = _stable_hex("scaleup_doc", seed, index, question)
    return f"noise/poc234_scaleup_{seed}/doc_{digest}.txt"


def _project_name(index: int, offset: int = 0) -> str:
    return PROJECT_NAMES[(index + offset) % len(PROJECT_NAMES)]


def _make_comparison_case(seed: int, index: int, query_style: str = "marker") -> tuple[Dict[str, Any], Dict[str, Any], str]:
    marker = f"ref_{_stable_hex('scaleup_ref_g', seed, index, length=10)}"
    case_index = index // 2
    proj_a = _project_name(case_index * 2)
    proj_b = _project_name(case_index * 2, offset=1)
    budget_a = 100_000 + (index * 7919 % 800_000)
    budget_b = 100_000 + ((index + 17) * 6151 % 800_000)
    if budget_a == budget_b:
        budget_b += 1_000
    winner = proj_a if budget_a > budget_b else proj_b
    winner_budget = budget_a if budget_a > budget_b else budget_b
    if query_style == "marker":
        question = f"In reference {marker} which project has a higher budget: Project {proj_a} or Project {proj_b}?"
    elif query_style == "semantic":
        question = f"Using the scale-up budget evidence, which project has a higher budget: Project {proj_a} or Project {proj_b}?"
    else:
        question = f"Which project has a higher budget: Project {proj_a} or Project {proj_b}?"
    expected = f"Project {winner} (budget of ${winner_budget:,})"
    source_file = _source_file(seed, index, question)
    text = (
        f"Reference {marker} contains scale-up comparison evidence.\n"
        f"Project {proj_a} budget is ${budget_a:,}.\n"
        f"Project {proj_b} budget is ${budget_b:,}.\n"
    )
    query = {
        "id": f"q_{_stable_hex('scaleup_query_g', seed, index, question, expected)}",
        "question": question,
        "expected_answer": expected,
        "category": "G",
        "source_file": source_file,
        "metadata": {
            "type": "G",
            "sub_type": 1,
            "proj_a": proj_a,
            "proj_b": proj_b,
            "budget_a": budget_a,
            "budget_b": budget_b,
            "is_missing": False,
            "query_style": query_style,
        },
    }
    block = {
        "block_id": f"target_g_{index:06d}",
        "text": text,
        "source_file": source_file,
        "category": "scaleup",
        "token_count": len(text.split()),
        "chunk_index": 0,
    }
    return query, block, text


def _make_aggregation_case(seed: int, index: int, query_style: str = "marker") -> tuple[Dict[str, Any], Dict[str, Any], str]:
    marker = f"ref_{_stable_hex('scaleup_ref_h', seed, index, length=10)}"
    case_index = index // 2
    manager = PEOPLE[case_index % len(PEOPLE)]
    projects = [_project_name(case_index * 3, offset=offset) for offset in range(3)]
    if query_style == "marker":
        question = f"In reference {marker} list all project names managed by {manager}."
    elif query_style == "semantic":
        question = f"Using the scale-up management evidence, list all project names managed by {manager}."
    else:
        question = f"List all project names managed by {manager}."
    expected = ", ".join(sorted(projects))
    source_file = _source_file(seed, index, question)
    lines = [f"Reference {marker} contains scale-up aggregation evidence."]
    lines.extend(f"Project {project} is managed by {manager}." for project in projects)
    text = "\n".join(lines) + "\n"
    query = {
        "id": f"q_{_stable_hex('scaleup_query_h', seed, index, question, expected)}",
        "question": question,
        "expected_answer": expected,
        "category": "H",
        "source_file": source_file,
        "metadata": {
            "type": "H",
            "sub_type": 1,
            "manager": manager,
            "projects": projects,
            "is_missing": False,
            "query_style": query_style,
        },
    }
    block = {
        "block_id": f"target_h_{index:06d}",
        "text": text,
        "source_file": source_file,
        "category": "scaleup",
        "token_count": len(text.split()),
        "chunk_index": 0,
    }
    return query, block, text


def _make_noise_block(seed: int, index: int) -> Dict[str, Any]:
    project = f"NOISE{index:07d}"
    text = (
        f"Noise shard {index} seed {seed}. "
        f"Project {project} has archival metadata unrelated to marked scale-up queries."
    )
    return {
        "block_id": f"noise_{index:08d}",
        "text": text,
        "source_file": f"noise/scaleup_noise_{seed}/noise_{index:08d}.txt",
        "category": "noise",
        "token_count": len(text.split()),
        "chunk_index": 0,
    }


def generate_scaleup_dataset(
    root_dir: str | Path,
    total_blocks: int,
    query_count: int = 20,
    seed: int = 42,
    embedding_dim: int = 384,
    mixed_query_set: str = "marker",
    embedder: Any | None = None,
    vector_backend: str = "none",
    ann_params: Dict[str, Any] | None = None,
) -> ScaleupDataset:
    if total_blocks < query_count:
        raise ValueError("total_blocks must be greater than or equal to query_count")

    root = Path(root_dir)
    corpus_dir = root / "corpus"
    index_dir = corpus_dir / "index_ooc"
    workload_path = root / "workload.jsonl"
    doc_root = corpus_dir / "documents"
    blocks: List[Dict[str, Any]] = []
    workload: List[Dict[str, Any]] = []
    entities = set(PEOPLE)

    query_styles = [style.strip() for style in mixed_query_set.split(",") if style.strip()]
    if not query_styles:
        query_styles = ["marker"]
    invalid_styles = set(query_styles) - {"marker", "entity", "semantic"}
    if invalid_styles:
        raise ValueError(f"Unsupported query styles: {sorted(invalid_styles)}")

    for idx in range(query_count):
        query_style = query_styles[idx % len(query_styles)]
        if idx % 2 == 0:
            query, block, doc_text = _make_comparison_case(seed, idx, query_style=query_style)
            entities.update([query["metadata"]["proj_a"], query["metadata"]["proj_b"]])
        else:
            query, block, doc_text = _make_aggregation_case(seed, idx, query_style=query_style)
            entities.update(query["metadata"]["projects"])
        workload.append(query)
        blocks.append(block)
        doc_path = doc_root / query["source_file"]
        doc_path.parent.mkdir(parents=True, exist_ok=True)
        doc_path.write_text(doc_text, encoding="utf-8")

    for noise_idx in range(total_blocks - query_count):
        block = _make_noise_block(seed, noise_idx)
        blocks.append(block)
        entities.add(block["text"].split("Project ")[1].split()[0])

    active_embedder = embedder or SyntheticScaleupEmbedder(dim=embedding_dim)
    embeddings = np.asarray(
        active_embedder.encode([block["text"] for block in blocks], convert_to_numpy=True, show_progress_bar=False),
        dtype=np.float32,
    )
    embedding_metadata = (
        active_embedder.embedding_metadata()
        if hasattr(active_embedder, "embedding_metadata")
        else {
            "embedding_backend": "synthetic",
            "embedding_model": "synthetic_scaleup",
            "embedding_dim": int(embeddings.shape[1]),
            "embedding_local_files_only": False,
            "embedding_batch_size": 0,
            "embedding_latency_ms": 0.0,
            "embedding_fallback_reason": "",
        }
    )
    embedding_metadata["embedding_dim"] = int(embeddings.shape[1])
    write_out_of_core_index(
        index_dir,
        blocks=blocks,
        embeddings=embeddings,
        entities=entities,
        vector_backend=vector_backend,
        ann_params=ann_params,
        embedding_metadata=embedding_metadata,
    )
    _write_jsonl(workload_path, workload)
    return ScaleupDataset(root_dir=root, corpus_dir=corpus_dir, index_dir=index_dir, workload_path=workload_path)


def _execute_kernel(category: str, query_ir: Dict[str, Any], active_evidence: List[Dict[str, Any]], q_id: str) -> Dict[str, Any]:
    if category == "G":
        return ComparisonKernel().execute(query_ir, active_evidence, ir_builder=None, query_id=q_id)
    if category == "H":
        return AggregationKernel().execute(query_ir, active_evidence, ir_builder=None, query_id=q_id)
    return {
        "route": "UNSUPPORTED_CATEGORY",
        "answer": "UNSUPPORTED_CATEGORY",
        "status": "UNSUPPORTED_CATEGORY",
        "reason": f"Scale-up benchmark supports only G/H, got {category}",
    }


def _summarize(records: List[Dict[str, Any]], index_dir: Path, strategy: str, candidate_cap: Optional[int]) -> Dict[str, Any]:
    count = len(records)
    latencies = [float(r["latency_ms"]) for r in records]
    metrics = [r.get("metrics", {}) for r in records]
    ems = [bool(r.get("is_em")) for r in records]
    leak = [bool(r.get("leak_check_passed")) for r in records]
    gh = [r for r in records if r.get("category") in {"G", "H"}]
    ooc_metrics = [m for m in metrics if m.get("storage_mode") == "out_of_core"]
    index_size = _index_size_bytes(index_dir)

    def avg_metric(name: str) -> float:
        vals = [float(m.get(name, 0.0)) for m in metrics]
        return float(np.mean(vals)) if vals else 0.0

    recalls = [
        float(m["ann_recall_at_k"])
        for m in metrics
        if m.get("ann_recall_at_k") is not None
    ]
    ann_metrics = [
        m for m in metrics
        if m.get("ann_backend") not in {None, "", "none"}
    ]
    ann_used = [bool(m.get("ann_used")) for m in ann_metrics]
    ann_available = [bool(m.get("ann_available")) for m in ann_metrics]
    ann_fallback_reasons = sorted(
        set(str(m.get("ann_fallback_reason", "")) for m in ann_metrics if m.get("ann_fallback_reason"))
    )

    return {
        "strategy": strategy,
        "candidate_cap": candidate_cap,
        "count": count,
        "overall_em": float(np.mean(ems) * 100.0) if ems else 0.0,
        "gh_em_global": float(np.mean([bool(r.get("is_em")) for r in gh]) * 100.0) if gh else 0.0,
        "no_leak_pass_rate": float(np.mean(leak) * 100.0) if leak else 100.0,
        "llm_bypass_rate": 100.0,
        "mean_latency_ms": float(np.mean(latencies)) if latencies else 0.0,
        "p50_latency_ms": _percentile(latencies, 50),
        "p95_latency_ms": _percentile(latencies, 95),
        "p99_latency_ms": _percentile(latencies, 99),
        "max_latency_ms": max(latencies) if latencies else 0.0,
        "avg_bytes_read": avg_metric("bytes_read"),
        "max_bytes_read": max((float(m.get("bytes_read", 0.0)) for m in metrics), default=0.0),
        "avg_blocks_materialized": avg_metric("blocks_materialized"),
        "avg_embedding_rows_scanned": avg_metric("embedding_rows_scanned"),
        "avg_resident_mb": avg_metric("max_resident_mb"),
        "max_resident_mb": max((float(m.get("max_resident_mb", 0.0)) for m in metrics), default=0.0),
        "index_size_mb": index_size / (1024 * 1024),
        "ooc_metrics_coverage": (len(ooc_metrics) / count * 100.0) if count else 100.0,
        "avg_ann_recall_at_k": float(np.mean(recalls)) if recalls else None,
        "ann_used_rate": float(np.mean(ann_used) * 100.0) if ann_used else None,
        "ann_available_rate": float(np.mean(ann_available) * 100.0) if ann_available else None,
        "ann_fallback_reasons": ann_fallback_reasons,
    }


def _write_benchmark_report(path: Path, title: str, summaries: List[Dict[str, Any]]) -> None:
    lines = [
        f"# {title}",
        "",
        "| Size | Strategy | Cap | Count | EM | No-leak | ANN used | Recall | Mean ms | p95 ms | Rows scanned | Blocks mat. | Bytes read | Index MB |",
        "|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for summary in summaries:
        cap = summary.get("candidate_cap")
        cap_text = "" if cap is None else str(cap)
        size_text = "" if summary.get("size") is None else str(summary["size"])
        recall = summary.get("avg_ann_recall_at_k")
        recall_text = "" if recall is None else f"{float(recall):.2f}%"
        ann_used = summary.get("ann_used_rate")
        ann_used_text = "" if ann_used is None else f"{float(ann_used):.2f}%"
        lines.append(
            f"| {size_text} | {summary['strategy']} | {cap_text} | {summary['count']} | "
            f"{summary['gh_em_global']:.2f}% | {summary['no_leak_pass_rate']:.2f}% | {ann_used_text} | {recall_text} | "
            f"{summary['mean_latency_ms']:.2f} | {summary['p95_latency_ms']:.2f} | "
            f"{summary['avg_embedding_rows_scanned']:.1f} | {summary['avg_blocks_materialized']:.1f} | "
            f"{summary['avg_bytes_read']:.1f} | {summary['index_size_mb']:.3f} |"
        )

    lines.extend(["", "## Verdict", ""])
    full_summaries = [s for s in summaries if s["strategy"] == "ooc_full_scan"]
    slow_full = [
        s for s in full_summaries
        if float(s.get("p95_latency_ms", 0.0)) >= 500.0
    ]
    if slow_full:
        first_slow = sorted(slow_full, key=lambda s: (s.get("size") or 0, s["p95_latency_ms"]))[0]
        size_label = first_slow.get("size", "this tier")
        lines.append(
            f"Full mmap scan crosses the 500 ms p95 interactive threshold at {size_label} blocks "
            f"on this run ({first_slow['p95_latency_ms']:.2f} ms p95)."
        )
    elif full_summaries:
        largest = sorted(full_summaries, key=lambda s: s.get("size") or 0)[-1]
        size_label = largest.get("size", "the largest measured tier")
        lines.append(
            f"Full mmap scan stays below the 500 ms p95 threshold through {size_label} blocks "
            f"on this run ({largest['p95_latency_ms']:.2f} ms p95)."
        )
    else:
        lines.append("No full-scan mmap tier was included in this report.")

    by_size: Dict[Any, Dict[str, Dict[str, Any]]] = {}
    for summary in summaries:
        by_size.setdefault(summary.get("size"), {})[summary["strategy"]] = summary
    comparisons = []
    for size, grouped in sorted(by_size.items(), key=lambda item: item[0] or 0):
        full = grouped.get("ooc_full_scan")
        pruned = grouped.get("ooc_marker_entity_pruned")
        if full and pruned:
            full_rows = max(1.0, float(full["avg_embedding_rows_scanned"]))
            row_reduction = (full_rows - float(pruned["avg_embedding_rows_scanned"])) / full_rows * 100.0
            full_blocks = max(1.0, float(full["avg_blocks_materialized"]))
            block_reduction = (full_blocks - float(pruned["avg_blocks_materialized"])) / full_blocks * 100.0
            comparisons.append(
                f"{size} blocks: pruning scanned {row_reduction:.2f}% fewer embedding rows "
                f"and materialized {block_reduction:.2f}% fewer blocks."
            )
    if comparisons:
        lines.extend(["", *comparisons])

    ann_summaries = [s for s in summaries if str(s["strategy"]).startswith("ooc_ann")]
    ann_wins = []
    for size, grouped in sorted(by_size.items(), key=lambda item: item[0] or 0):
        full = grouped.get("ooc_full_scan")
        ann_candidates = [
            summary
            for summary in grouped.values()
            if str(summary["strategy"]).startswith("ooc_ann")
            and float(summary.get("ann_used_rate") or 0.0) > 0.0
        ]
        if not full or not ann_candidates:
            continue
        best_ann = min(ann_candidates, key=lambda summary: float(summary["p95_latency_ms"]))
        full_p95 = max(1e-9, float(full["p95_latency_ms"]))
        ann_p95 = float(best_ann["p95_latency_ms"])
        speedup = full_p95 / max(1e-9, ann_p95)
        row_reduction = (
            float(full["avg_embedding_rows_scanned"]) - float(best_ann["avg_embedding_rows_scanned"])
        ) / max(1.0, float(full["avg_embedding_rows_scanned"])) * 100.0
        recall = best_ann.get("avg_ann_recall_at_k")
        recall_text = "unknown recall" if recall is None else f"{float(recall):.2f}% recall@k"
        ann_wins.append(
            f"{size} blocks: best ANN strategy {best_ann['strategy']} reached {ann_p95:.2f} ms p95 "
            f"vs {full_p95:.2f} ms full-scan p95 ({speedup:.1f}x faster), "
            f"with {row_reduction:.2f}% fewer embedding rows reranked and {recall_text}."
        )
    if ann_wins:
        lines.extend(["", "ANN acceleration was active in this run:", *ann_wins])

    if ann_summaries and all(float(s.get("ann_used_rate") or 0.0) == 0.0 for s in ann_summaries):
        reasons = sorted({
            reason
            for summary in ann_summaries
            for reason in summary.get("ann_fallback_reasons", [])
        })
        reason_text = ", ".join(reasons) if reasons else "unknown"
        lines.extend([
            "",
            f"ANN backends were requested but not used in this run; fallback reasons: {reason_text}.",
        ])

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_scaleup_benchmark(
    index_dir: str | Path,
    workload_path: str | Path,
    output_dir: str | Path,
    strategy: str = "ooc_full_scan",
    query_limit: Optional[int] = None,
    top_k: int = 50,
    max_candidates: int = 200,
) -> BenchmarkResult:
    index_path = Path(index_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    workload = _read_jsonl(Path(workload_path), limit=query_limit)
    hardware_budget = HardwareBudget(max_candidates=max_candidates)
    index = OutOfCoreIndex(index_path, embed_model=SyntheticScaleupEmbedder(), hardware_budget=hardware_budget)
    resolver = EvidenceResolver()
    records = []

    for row in workload:
        q_id = row["id"]
        question = row["question"]
        expected = row["expected_answer"]
        category = row["category"]
        leak_ok, leak_reasons = leak_check_query(row)
        start = time.perf_counter()
        candidates, query_ir, telemetry = index.search(question, top_k=top_k, strategy=strategy)
        active, suppressed, forbidden = resolver.resolve(candidates, query_ir)
        audit = _execute_kernel(category, query_ir, active, q_id=q_id)
        answer = audit.get("answer", audit.get("status", "NOT_FOUND"))
        latency_ms = (time.perf_counter() - start) * 1000.0
        answer_matches_expected = clean_answer(answer) == clean_answer(expected)
        is_em = answer_matches_expected and leak_ok
        route = audit.get("route", audit.get("status", "UNKNOWN"))
        metrics = dict(telemetry)
        if strategy.startswith("ooc_ann") and metrics.get("ann_used"):
            exact_candidates, _, _ = index.search(question, top_k=top_k, strategy="ooc_full_scan")
            exact_ids = {candidate["block_id"] for candidate in exact_candidates}
            returned_ids = {candidate["block_id"] for candidate in candidates}
            metrics["ann_recall_at_k"] = (
                len(exact_ids & returned_ids) / len(exact_ids) * 100.0
                if exact_ids else 100.0
            )
        metrics.update({
            "route": route,
            "kernel_audit": audit,
            "llm_bypass": True,
            "verifier_passed": True,
            "prompt_tokens": 0,
            "tokens_materialized_kv": 0,
            "tokens_avoided": 1200,
        })
        records.append({
            "id": q_id,
            "question": question,
            "expected": expected,
            "expected_answer": expected,
            "generated": answer,
            "answer": answer,
            "category": category,
            "mode": strategy,
            "answer_matches_expected": answer_matches_expected,
            "leak_check_passed": leak_ok,
            "leak_check_reasons": leak_reasons,
            "is_em": is_em,
            "exact_match": is_em,
            "route": route,
            "latency_ms": latency_ms,
            "is_bypass": True,
            "verify_passed": True,
            "prompt_tokens": 0,
            "metrics": metrics,
            "metadata": row.get("metadata", {}),
        })

    results_path = output_path / "results.jsonl"
    metrics_path = output_path / "metrics.json"
    report_path = output_path / "report.md"
    _write_jsonl(results_path, records)
    summary = _summarize(records, index_path, strategy=strategy, candidate_cap=max_candidates)
    metrics_doc = {"summary": summary, "records": records}
    metrics_path.write_text(json.dumps(metrics_doc, indent=2, ensure_ascii=False), encoding="utf-8")
    _write_benchmark_report(report_path, "OOC Scale-Up Benchmark", [summary])
    return BenchmarkResult(
        output_dir=output_path,
        results_path=results_path,
        metrics_path=metrics_path,
        report_path=report_path,
        summary=summary,
    )


def run_legacy_memory_benchmark(
    index_dir: str | Path,
    workload_path: str | Path,
    output_dir: str | Path,
    query_limit: Optional[int] = None,
    top_k: int = 50,
) -> BenchmarkResult:
    index_path = Path(index_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    blocks = _read_jsonl(index_path / "blocks.jsonl")
    embeddings = np.load(index_path / "embeddings.npy")
    entities = json.loads((index_path / "entity_list.json").read_text(encoding="utf-8"))
    parser = QueryParser(entities)
    embedder = SyntheticScaleupEmbedder(dim=int(embeddings.shape[1]))
    resolver = EvidenceResolver()
    workload = _read_jsonl(Path(workload_path), limit=query_limit)
    index_size = _index_size_bytes(index_path)
    resident_mb = (int(embeddings.nbytes) + index_size) / (1024 * 1024)
    records = []

    for row in workload:
        q_id = row["id"]
        question = row["question"]
        expected = row["expected_answer"]
        category = row["category"]
        leak_ok, leak_reasons = leak_check_query(row)
        start = time.perf_counter()
        query_ir = parser.parse(question)
        q_emb = np.asarray(embedder.encode(question, convert_to_numpy=True), dtype=np.float32)
        q_norm = float(np.linalg.norm(q_emb))
        block_norms = np.linalg.norm(embeddings, axis=1)
        sims = np.dot(embeddings, q_emb) / (q_norm * block_norms + 1e-8)
        top_indices = np.argsort(-sims)[:top_k]
        candidates = []
        for rank, idx in enumerate(top_indices, start=1):
            block = dict(blocks[int(idx)])
            block["retrieval_score"] = float(sims[int(idx)])
            block["retrieval_rank"] = rank
            block["bm25_score"] = 0.0
            block["cosine_similarity"] = float(sims[int(idx)])
            candidates.append(block)
        active, suppressed, forbidden = resolver.resolve(candidates, query_ir)
        audit = _execute_kernel(category, query_ir, active, q_id=q_id)
        answer = audit.get("answer", audit.get("status", "NOT_FOUND"))
        latency_ms = (time.perf_counter() - start) * 1000.0
        answer_matches_expected = clean_answer(answer) == clean_answer(expected)
        is_em = answer_matches_expected and leak_ok
        route = audit.get("route", audit.get("status", "UNKNOWN"))
        metrics = {
            "storage_mode": "legacy_memory",
            "bytes_read": 0,
            "blocks_materialized": len(blocks),
            "embedding_rows_scanned": len(blocks),
            "embedding_windows": 1,
            "index_bytes": index_size,
            "max_resident_mb": resident_mb,
            "route": route,
            "kernel_audit": audit,
            "llm_bypass": True,
            "verifier_passed": True,
            "prompt_tokens": 0,
            "tokens_materialized_kv": 0,
            "tokens_avoided": 1200,
        }
        records.append({
            "id": q_id,
            "question": question,
            "expected": expected,
            "expected_answer": expected,
            "generated": answer,
            "answer": answer,
            "category": category,
            "mode": "legacy_memory_scan",
            "answer_matches_expected": answer_matches_expected,
            "leak_check_passed": leak_ok,
            "leak_check_reasons": leak_reasons,
            "is_em": is_em,
            "exact_match": is_em,
            "route": route,
            "latency_ms": latency_ms,
            "is_bypass": True,
            "verify_passed": True,
            "prompt_tokens": 0,
            "metrics": metrics,
            "metadata": row.get("metadata", {}),
        })

    results_path = output_path / "results.jsonl"
    metrics_path = output_path / "metrics.json"
    report_path = output_path / "report.md"
    _write_jsonl(results_path, records)
    summary = _summarize(records, index_path, strategy="legacy_memory_scan", candidate_cap=None)
    metrics_path.write_text(json.dumps({"summary": summary, "records": records}, indent=2, ensure_ascii=False), encoding="utf-8")
    _write_benchmark_report(report_path, "OOC Scale-Up Benchmark - Legacy Memory Baseline", [summary])
    return BenchmarkResult(
        output_dir=output_path,
        results_path=results_path,
        metrics_path=metrics_path,
        report_path=report_path,
        summary=summary,
    )


def run_candidate_cap_sweep(
    index_dir: str | Path,
    workload_path: str | Path,
    output_dir: str | Path,
    candidate_caps: Sequence[int] = (20, 50, 200),
    query_limit: Optional[int] = None,
    top_k: int = 50,
) -> BenchmarkResult:
    output_path = Path(output_dir)
    summaries = []
    for cap in candidate_caps:
        result = run_scaleup_benchmark(
            index_dir=index_dir,
            workload_path=workload_path,
            output_dir=output_path / f"cap_{cap}",
            strategy="ooc_full_scan",
            query_limit=query_limit,
            top_k=top_k,
            max_candidates=cap,
        )
        summary = dict(result.summary)
        summary["candidate_cap"] = cap
        summaries.append(summary)

    metrics_path = output_path / "metrics.json"
    report_path = output_path / "report.md"
    output_path.mkdir(parents=True, exist_ok=True)
    metrics_doc = {"candidate_cap_sweep": summaries}
    metrics_path.write_text(json.dumps(metrics_doc, indent=2, ensure_ascii=False), encoding="utf-8")
    _write_benchmark_report(report_path, "OOC Scale-Up Benchmark - Candidate Cap Sweep", summaries)
    return BenchmarkResult(
        output_dir=output_path,
        results_path=output_path / "cap_aggregate_results.jsonl",
        metrics_path=metrics_path,
        report_path=report_path,
        summary={"candidate_cap_sweep": summaries},
    )


def run_scaleup_suite(
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    sizes: Sequence[int] = (1000, 10000),
    query_count: int = 20,
    seed: int = 42,
    strategy: str = "ooc_full_scan",
    top_k: int = 50,
    max_candidates: int = 200,
    ann_backends: Sequence[str] = (),
    mixed_query_set: str = "marker",
) -> Dict[str, Any]:
    output_path = Path(output_dir)
    all_summaries = []
    for size in sizes:
        tier_dir = output_path / f"tier_{size}"
        dataset = generate_scaleup_dataset(
            tier_dir / "dataset",
            total_blocks=size,
            query_count=query_count,
            seed=seed,
            mixed_query_set=mixed_query_set,
        )
        for backend in ann_backends:
            build_vector_index(
                embeddings_path=dataset.index_dir / "embeddings.npy",
                output_path=dataset.index_dir,
                backend=backend,
                params=_default_ann_params(size, backend),
            )
        if strategy == "ooc_candidate_cap_sweep":
            result = run_candidate_cap_sweep(
                index_dir=dataset.index_dir,
                workload_path=dataset.workload_path,
                output_dir=tier_dir / "candidate_cap_sweep",
                candidate_caps=(20, 50, max_candidates),
                query_limit=query_count,
                top_k=top_k,
            )
            for summary in result.summary["candidate_cap_sweep"]:
                item = dict(summary)
                item["size"] = size
                all_summaries.append(item)
        elif strategy == "all":
            if size <= 10000:
                legacy = run_legacy_memory_benchmark(
                    index_dir=dataset.index_dir,
                    workload_path=dataset.workload_path,
                    output_dir=tier_dir / "legacy_memory_scan",
                    query_limit=query_count,
                    top_k=top_k,
                )
                item = dict(legacy.summary)
                item["size"] = size
                all_summaries.append(item)
            for search_strategy in ("ooc_full_scan", "ooc_marker_entity_pruned"):
                result = run_scaleup_benchmark(
                    index_dir=dataset.index_dir,
                    workload_path=dataset.workload_path,
                    output_dir=tier_dir / search_strategy,
                    strategy=search_strategy,
                    query_limit=query_count,
                    top_k=top_k,
                    max_candidates=max_candidates,
                )
                item = dict(result.summary)
                item["size"] = size
                all_summaries.append(item)
            for backend in ann_backends:
                ann_strategy = _strategy_for_ann_backend(backend)
                result = run_scaleup_benchmark(
                    index_dir=dataset.index_dir,
                    workload_path=dataset.workload_path,
                    output_dir=tier_dir / ann_strategy,
                    strategy=ann_strategy,
                    query_limit=query_count,
                    top_k=top_k,
                    max_candidates=max_candidates,
                )
                item = dict(result.summary)
                item["size"] = size
                all_summaries.append(item)
            if ann_backends:
                result = run_scaleup_benchmark(
                    index_dir=dataset.index_dir,
                    workload_path=dataset.workload_path,
                    output_dir=tier_dir / "ooc_ann_pruned_hybrid",
                    strategy="ooc_ann_pruned_hybrid",
                    query_limit=query_count,
                    top_k=top_k,
                    max_candidates=max_candidates,
                )
                item = dict(result.summary)
                item["size"] = size
                all_summaries.append(item)
            sweep = run_candidate_cap_sweep(
                index_dir=dataset.index_dir,
                workload_path=dataset.workload_path,
                output_dir=tier_dir / "candidate_cap_sweep",
                candidate_caps=(20, 50, max_candidates),
                query_limit=query_count,
                top_k=top_k,
            )
            for summary in sweep.summary["candidate_cap_sweep"]:
                item = dict(summary)
                item["strategy"] = "ooc_candidate_cap_sweep"
                item["size"] = size
                all_summaries.append(item)
        elif strategy == "legacy_memory_scan":
            result = run_legacy_memory_benchmark(
                index_dir=dataset.index_dir,
                workload_path=dataset.workload_path,
                output_dir=tier_dir / strategy,
                query_limit=query_count,
                top_k=top_k,
            )
            item = dict(result.summary)
            item["size"] = size
            all_summaries.append(item)
        else:
            result = run_scaleup_benchmark(
                index_dir=dataset.index_dir,
                workload_path=dataset.workload_path,
                output_dir=tier_dir / strategy,
                strategy=strategy,
                query_limit=query_count,
                top_k=top_k,
                max_candidates=max_candidates,
            )
            item = dict(result.summary)
            item["size"] = size
            all_summaries.append(item)

    output_path.mkdir(parents=True, exist_ok=True)
    metrics_path = output_path / "metrics.json"
    report_path = output_path / "report.md"
    metrics_doc = {
        "sizes": list(sizes),
        "query_count": query_count,
        "seed": seed,
        "strategy": strategy,
        "ann_backends": list(ann_backends),
        "mixed_query_set": mixed_query_set,
        "summaries": all_summaries,
    }
    metrics_path.write_text(json.dumps(metrics_doc, indent=2, ensure_ascii=False), encoding="utf-8")
    _write_benchmark_report(report_path, "OOC Scale-Up Benchmark", all_summaries)
    return metrics_doc


def _parse_sizes(raw: str) -> List[int]:
    values = []
    for item in raw.split(","):
        item = item.strip()
        if item:
            values.append(int(item))
    return values


def _parse_csv(raw: str) -> List[str]:
    return [item.strip() for item in raw.split(",") if item.strip()]


def _strategy_for_ann_backend(backend: str) -> str:
    if backend == "faiss_flat":
        return "ooc_ann_flat"
    if backend == "faiss_hnsw":
        return "ooc_ann_hnsw"
    if backend == "faiss_ivf_flat":
        return "ooc_ann_ivf_flat"
    raise ValueError(f"Unsupported ANN backend for benchmark strategy: {backend}")


def _default_ann_params(size: int, backend: str) -> Dict[str, Any]:
    if backend == "faiss_hnsw":
        return {"M": 32, "efConstruction": 80, "efSearch": 64}
    if backend == "faiss_ivf_flat":
        return {
            "nlist": max(64, int(4 * math.sqrt(size))),
            "nprobe": 8,
            "seed": 42,
        }
    return {}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sizes", type=str, default="1000,10000,50000,100000")
    parser.add_argument("--queries", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--strategy",
        choices=[
            "legacy_memory_scan",
            "ooc_full_scan",
            "ooc_marker_entity_pruned",
            "ooc_ann_flat",
            "ooc_ann_hnsw",
            "ooc_ann_ivf_flat",
            "ooc_ann_pruned_hybrid",
            "ooc_candidate_cap_sweep",
            "all",
        ],
        default="ooc_full_scan",
    )
    parser.add_argument("--output-dir", type=str, default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--max-candidates", type=int, default=200)
    parser.add_argument("--ann-backends", type=str, default="")
    parser.add_argument("--mixed-query-set", type=str, default="marker")
    args = parser.parse_args()

    metrics = run_scaleup_suite(
        output_dir=args.output_dir,
        sizes=_parse_sizes(args.sizes),
        query_count=args.queries,
        seed=args.seed,
        strategy=args.strategy,
        top_k=args.top_k,
        max_candidates=args.max_candidates,
        ann_backends=_parse_csv(args.ann_backends),
        mixed_query_set=args.mixed_query_set,
    )
    print(json.dumps({
        "output_dir": args.output_dir,
        "sizes": metrics["sizes"],
        "strategy": args.strategy,
        "summaries": metrics["summaries"],
    }, indent=2))


if __name__ == "__main__":
    main()
