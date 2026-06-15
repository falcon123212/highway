import argparse
import json
from pathlib import Path
from statistics import mean
from typing import Any, Dict, List, Sequence

from highway.benchmarks.ooc_scaleup import SyntheticScaleupEmbedder, generate_scaleup_dataset
from highway.kernels.compute_kernels import AggregationKernel, ComparisonKernel
from highway.paths import DEFAULT_RUNS_DIR
from highway.retrieval.evidence_resolver import EvidenceResolver
from highway.runtime.context_engine import ContextRequest, HighwayContextEngine
from highway.runtime.token_economics import ModelProfile, TokenEconomics


DEFAULT_OUTPUT_DIR = DEFAULT_RUNS_DIR / "quality_token_tradeoff"
DEFAULT_MODEL_PROFILE = ModelProfile(name="quality_tradeoff_fake_model", layers=24, hidden_size=1024)


def _read_jsonl(path: Path, limit: int | None = None) -> List[Dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
                if limit is not None and len(rows) >= limit:
                    break
    return rows


def _estimate_tokens(text: str) -> int:
    return max(1, len(str(text).split()))


def _clean_answer(value: str) -> str:
    return " ".join(str(value).strip().split()).lower()


def _baseline_prompt(question: str, blocks: Sequence[Dict[str, Any]]) -> str:
    lines = [
        "You are a precise reasoning assistant.",
        "Use the full context below, then respond with a short reasoning field and a final answer field.",
        "",
        "Context:",
    ]
    for block in blocks:
        lines.append(f"[{block.get('block_id')}] {block.get('source_file')}: {block.get('text')}")
    lines.append("")
    lines.append(f"Question: {question}")
    lines.append("Return: reasoning + answer.")
    return "\n".join(lines)


def _highway_prompt(question: str, context_blocks: Sequence[Any]) -> str:
    lines = [
        "You are a precise reasoning assistant.",
        "Use only the selected Highway context below, then respond with a short reasoning field and a final answer field.",
        "",
        "Selected context:",
    ]
    for block in context_blocks:
        lines.append(f"[{block.block_id}] {block.source_file}: {block.text}")
    lines.append("")
    lines.append(f"Question: {question}")
    lines.append("Return: reasoning + answer.")
    return "\n".join(lines)


def _answer_with_reflection(
    row: Dict[str, Any],
    query_ir: Dict[str, Any],
    evidence: Sequence[Dict[str, Any]],
) -> Dict[str, str]:
    category = row.get("category")
    if category == "G":
        audit = ComparisonKernel().execute(query_ir, list(evidence), ir_builder=None, query_id=row["id"])
    elif category == "H":
        audit = AggregationKernel().execute(query_ir, list(evidence), ir_builder=None, query_id=row["id"])
    else:
        audit = {"status": "UNSUPPORTED", "answer": "UNSUPPORTED"}

    answer = str(audit.get("answer", audit.get("status", "NOT_FOUND")))
    status = str(audit.get("status", "UNKNOWN"))
    reasoning = (
        f"Route {audit.get('route', status)} used {len(evidence)} evidence blocks; "
        f"kernel status is {status}."
    )
    return {"reasoning": reasoning, "answer": answer}


def _summarize(records: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    count = len(records)
    baseline_em = [bool(record["baseline_is_em"]) for record in records]
    highway_em = [bool(record["highway_is_em"]) for record in records]
    avg_baseline_prompt = mean(record["baseline_prompt_tokens"] for record in records) if records else 0.0
    avg_highway_prompt = mean(record["highway_prompt_tokens"] for record in records) if records else 0.0
    avoided = avg_baseline_prompt - avg_highway_prompt
    avoided_pct = avoided / avg_baseline_prompt * 100.0 if avg_baseline_prompt else 0.0
    return {
        "count": count,
        "baseline_em": float(mean(baseline_em) * 100.0) if records else 0.0,
        "highway_em": float(mean(highway_em) * 100.0) if records else 0.0,
        "quality_delta_pp": (
            float(mean(highway_em) * 100.0 - mean(baseline_em) * 100.0)
            if records else 0.0
        ),
        "avg_baseline_prompt_tokens": float(avg_baseline_prompt),
        "avg_highway_prompt_tokens": float(avg_highway_prompt),
        "avg_prompt_tokens_avoided": float(avoided),
        "avg_prompt_tokens_avoided_pct": float(avoided_pct),
        "avg_baseline_output_tokens": float(mean(record["baseline_output_tokens"] for record in records)) if records else 0.0,
        "avg_highway_output_tokens": float(mean(record["highway_output_tokens"] for record in records)) if records else 0.0,
        "avg_highway_context_latency_ms": float(mean(record["highway_context_latency_ms"] for record in records)) if records else 0.0,
        "avg_highway_rows_scanned": float(mean(record["highway_rows_scanned"] for record in records)) if records else 0.0,
        "avg_highway_blocks_materialized": float(mean(record["highway_blocks_materialized"] for record in records)) if records else 0.0,
        "avg_kv_bytes_avoided_estimated": float(mean(record["kv_bytes_avoided_estimated"] for record in records)) if records else 0.0,
        "avg_cost_avoided_estimated_usd": float(mean(record["cost_avoided_estimated_usd"] for record in records)) if records else 0.0,
    }


def _write_report(path: Path, summary: Dict[str, Any], metrics_path: Path) -> None:
    lines = [
        "# Quality Token Tradeoff Smoke",
        "",
        "This smoke compares a full-context reflective baseline against Highway selected context.",
        "Both paths use the same deterministic reflective answerer; only the context size differs.",
        "",
        "| Metric | Value |",
        "|---|---:|",
        f"| Queries | {summary['count']} |",
        f"| Baseline EM | {summary['baseline_em']:.2f}% |",
        f"| Highway EM | {summary['highway_em']:.2f}% |",
        f"| Quality delta | {summary['quality_delta_pp']:.2f} pp |",
        f"| Avg baseline prompt tokens | {summary['avg_baseline_prompt_tokens']:.2f} |",
        f"| Avg Highway prompt tokens | {summary['avg_highway_prompt_tokens']:.2f} |",
        f"| Avg prompt tokens avoided | {summary['avg_prompt_tokens_avoided']:.2f} |",
        f"| Avg prompt tokens avoided pct | {summary['avg_prompt_tokens_avoided_pct']:.2f}% |",
        f"| Avg baseline output tokens | {summary['avg_baseline_output_tokens']:.2f} |",
        f"| Avg Highway output tokens | {summary['avg_highway_output_tokens']:.2f} |",
        f"| Avg KV bytes avoided estimated | {summary['avg_kv_bytes_avoided_estimated']:.2f} |",
        f"| Avg cost avoided estimated USD | {summary['avg_cost_avoided_estimated_usd']:.8f} |",
        f"| Avg Highway context latency | {summary['avg_highway_context_latency_ms']:.2f} ms |",
        "",
        f"Metrics JSON: `{metrics_path.as_posix()}`",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_quality_token_tradeoff_smoke(
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
    all_blocks = _read_jsonl(dataset.index_dir / "blocks.jsonl")
    engine = HighwayContextEngine(
        index_dir=dataset.index_dir,
        embed_model=SyntheticScaleupEmbedder(),
        model_profile=DEFAULT_MODEL_PROFILE,
        input_cost_per_million=input_cost_per_million,
        output_cost_per_million=output_cost_per_million,
    )
    resolver = EvidenceResolver()
    records = []
    for row in workload:
        question = row["question"]
        pack = engine.retrieve(
            ContextRequest(
                user_turn=question,
                session_id="quality_token_tradeoff",
                strategy="ooc_marker_entity_pruned",
            ),
            top_k=50,
        )
        baseline_active, _, _ = resolver.resolve([dict(block) for block in all_blocks], pack.query_ir)
        highway_evidence = [
            {
                "block_id": block.block_id,
                "source_file": block.source_file,
                "text": block.text,
                "retrieval_score": block.score,
            }
            for block in pack.blocks
        ]
        baseline_response = _answer_with_reflection(row, pack.query_ir, baseline_active)
        highway_response = _answer_with_reflection(row, pack.query_ir, highway_evidence)
        baseline_prompt_tokens = _estimate_tokens(_baseline_prompt(question, all_blocks))
        highway_prompt_tokens = _estimate_tokens(_highway_prompt(question, pack.blocks))
        baseline_output_tokens = _estimate_tokens(
            f"{baseline_response['reasoning']} {baseline_response['answer']}"
        )
        highway_output_tokens = _estimate_tokens(
            f"{highway_response['reasoning']} {highway_response['answer']}"
        )
        economics = TokenEconomics.from_measurements(
            baseline_input_tokens=baseline_prompt_tokens,
            actual_input_tokens=highway_prompt_tokens,
            output_tokens=highway_output_tokens,
            model_profile=DEFAULT_MODEL_PROFILE,
            input_cost_per_million=input_cost_per_million,
            output_cost_per_million=output_cost_per_million,
        )
        records.append({
            "id": row["id"],
            "category": row["category"],
            "question": question,
            "expected_answer": row["expected_answer"],
            "baseline_response": baseline_response,
            "highway_response": highway_response,
            "baseline_is_em": _clean_answer(baseline_response["answer"]) == _clean_answer(row["expected_answer"]),
            "highway_is_em": _clean_answer(highway_response["answer"]) == _clean_answer(row["expected_answer"]),
            "baseline_prompt_tokens": baseline_prompt_tokens,
            "highway_prompt_tokens": highway_prompt_tokens,
            "baseline_output_tokens": baseline_output_tokens,
            "highway_output_tokens": highway_output_tokens,
            "prompt_tokens_avoided": economics.avoided_input_tokens,
            "prompt_tokens_avoided_pct": (
                economics.avoided_input_tokens / economics.baseline_input_tokens * 100.0
                if economics.baseline_input_tokens else 0.0
            ),
            "kv_bytes_avoided_estimated": economics.kv_bytes_avoided_estimated or 0,
            "cost_avoided_estimated_usd": economics.cost_avoided_estimated_usd,
            "highway_context_latency_ms": pack.metrics["latency_ms"],
            "highway_rows_scanned": pack.metrics["embedding_rows_scanned"],
            "highway_blocks_materialized": pack.metrics["blocks_materialized"],
            "highway_context_block_ids": [block.block_id for block in pack.blocks],
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
    args = parser.parse_args()
    result = run_quality_token_tradeoff_smoke(
        output_dir=args.output_dir,
        total_blocks=args.total_blocks,
        query_count=args.queries,
        seed=args.seed,
    )
    print(json.dumps({"output_dir": str(result["output_dir"]), "summary": result["summary"]}, indent=2))


if __name__ == "__main__":
    main()
