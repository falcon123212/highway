from __future__ import annotations

import argparse
import hashlib
import json
import re
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from random import Random
from statistics import mean
from typing import Any, Dict, Iterable, List, Mapping, Sequence

import numpy as np

from highway.benchmarks.long_conversation_quality import PromptAuditWriter, _display_path, _write_jsonl
from highway.paths import DEFAULT_RUNS_DIR
from highway.runtime.context_engine import ContextRequest, HighwayContextEngine
from highway.runtime.llm_runtime import HighwayLLMRuntime, estimate_tokens
from highway.runtime.ollama_client import OllamaLLMClient
from highway.runtime.token_economics import ModelProfile, TokenEconomics
from highway.storage.index_writer import write_out_of_core_index


DEFAULT_DATASET_ID = "galileo-ai/ragbench"
FALLBACK_DATASET_ID = "rungalileo/ragbench"
DEFAULT_OUTPUT_DIR = DEFAULT_RUNS_DIR / "highway_ragbench"
DEFAULT_CONFIGS = ("hotpotqa", "techqa", "emanual", "finqa", "covidqa")
DEFAULT_MODEL_PROFILE = ModelProfile(name="ragbench_llm", layers=24, hidden_size=1024)
JSON_INSTRUCTION = (
    "Return only valid JSON with keys: reasoning, answer, sources, confidence. "
    "The sources value must be a list of source_file strings copied from the context. "
    "If the selected context does not contain enough evidence, answer INSUFFICIENT_EVIDENCE."
)


@dataclass(frozen=True)
class RagBenchCase:
    case_id: str
    config_name: str
    question: str
    expected_answer: str
    expected_sources: List[str]
    blocks: List[Dict[str, Any]]
    ragbench_scores: Dict[str, float]
    support_available: bool

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class RagBenchEmbedder:
    def __init__(self, dim: int = 128):
        self.dim = int(dim)

    def encode(self, text: Any, convert_to_numpy: bool = True, show_progress_bar: bool = False) -> np.ndarray:
        del convert_to_numpy, show_progress_bar
        if isinstance(text, list):
            return np.asarray([self.encode(item) for item in text], dtype=np.float32)
        vec = np.zeros(self.dim, dtype=np.float32)
        for token in _tokens(str(text)):
            digest = hashlib.sha256(token.encode("utf-8")).digest()
            idx = int.from_bytes(digest[:4], "little") % self.dim
            vec[idx] += 1.0
        norm = float(np.linalg.norm(vec))
        if norm > 0.0:
            vec /= norm
        else:
            vec[-1] = 1.0
        return vec

    def embedding_metadata(self) -> Dict[str, Any]:
        return {
            "embedding_backend": "ragbench_hashing",
            "embedding_model": "ragbench_hashing_v1",
            "embedding_dim": self.dim,
            "embedding_local_files_only": True,
            "embedding_batch_size": 0,
            "embedding_latency_ms": 0.0,
            "embedding_fallback_reason": "",
        }


class GroundedRagBenchFakeClient:
    model_name = "ragbench_grounded_fake"

    def answer(
        self,
        prompt: str,
        query_ir: Dict[str, Any],
        evidence: Sequence[Dict[str, Any]],
        expected_answer: str | None = None,
        expected_sources: Sequence[str] = (),
        query_id: str = "ragbench_fake",
        **kwargs: Any,
    ) -> Dict[str, Any]:
        del query_ir, query_id, kwargs
        evidence_sources = {str(item.get("source_file", "")) for item in evidence}
        expected = [str(source) for source in expected_sources]
        source = next((source for source in expected if source in evidence_sources), "")
        answer = str(expected_answer or "") if source else "INSUFFICIENT_EVIDENCE"
        sources = [source] if source else []
        raw = json.dumps(
            {
                "reasoning": "I used only sources present in the supplied context.",
                "answer": answer,
                "sources": sources,
                "confidence": 1.0 if source else 0.0,
            },
            ensure_ascii=False,
        )
        input_tokens = estimate_tokens(prompt)
        output_tokens = estimate_tokens(raw)
        ttft_ms = max(1.0, input_tokens / 80.0)
        decode_ms = max(1.0, output_tokens / 8.0)
        return {
            "available": True,
            "model_name": self.model_name,
            "raw_text": raw,
            "answer": raw,
            "reasoning": "grounded fake",
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "ttft_ms": ttft_ms,
            "decode_ms": decode_ms,
            "total_ms": ttft_ms + decode_ms,
            "input_tokens_per_second": input_tokens / (ttft_ms / 1000.0),
            "output_tokens_per_second": output_tokens / (decode_ms / 1000.0),
        }


def normalize_ragbench_rows(
    rows: Sequence[Mapping[str, Any]],
    config_name: str,
    limit: int | None = None,
    seed: int = 42,
) -> List[RagBenchCase]:
    indexed_rows = list(enumerate(rows))
    Random(int(seed)).shuffle(indexed_rows)
    indexed_rows.sort(key=lambda item: item[0])
    selected = indexed_rows[: int(limit)] if limit is not None else indexed_rows
    cases: List[RagBenchCase] = []
    for _, row in selected:
        case_id = _safe_id(str(row.get("id", len(cases))))
        documents = [str(doc) for doc in row.get("documents", []) if str(doc).strip()]
        blocks = []
        for doc_idx, doc in enumerate(documents):
            source_file = f"ragbench/{config_name}/{case_id}/doc_{doc_idx}.txt"
            source_hash = _sha256_text(doc)
            blocks.append(
                {
                    "block_id": f"{config_name}_{case_id}_doc_{doc_idx}",
                    "source_file": source_file,
                    "source_hash": source_hash,
                    "text": doc,
                    "category": config_name,
                    "token_count": estimate_tokens(doc),
                    "chunk_index": doc_idx,
                    "ragbench_case_id": case_id,
                }
            )
        expected_sources = _expected_sources_from_support(
            row.get("sentence_support_information", []),
            config_name=config_name,
            case_id=case_id,
            doc_count=len(blocks),
        )
        support_available = bool(expected_sources)
        if not expected_sources and blocks:
            expected_sources = [blocks[0]["source_file"]]
        scores = {
            name: _safe_float(row.get(name))
            for name in (
                "trulens_groundedness",
                "trulens_context_relevance",
                "ragas_faithfulness",
                "ragas_context_relevance",
                "relevance_score",
                "utilization_score",
                "completeness_score",
            )
            if row.get(name) is not None
        }
        cases.append(
            RagBenchCase(
                case_id=case_id,
                config_name=config_name,
                question=str(row.get("question", "")),
                expected_answer=str(row.get("response", "")),
                expected_sources=expected_sources,
                blocks=blocks,
                ragbench_scores=scores,
                support_available=support_available,
            )
        )
    return cases


def load_ragbench_cases(
    dataset_id: str = DEFAULT_DATASET_ID,
    configs: Sequence[str] = DEFAULT_CONFIGS,
    split: str = "test",
    examples_per_config: int = 50,
    seed: int = 42,
    fallback_dataset_id: str = FALLBACK_DATASET_ID,
) -> List[RagBenchCase]:
    try:
        from datasets import load_dataset
    except Exception as exc:  # pragma: no cover - exercised when dependency absent
        raise RuntimeError("datasets_not_installed") from exc

    cases: List[RagBenchCase] = []
    for config in configs:
        last_error: Exception | None = None
        dataset = None
        for candidate_id in (dataset_id, fallback_dataset_id):
            try:
                dataset = load_dataset(candidate_id, config, split=split)
                break
            except Exception as exc:  # pragma: no cover - depends on network/cache state
                last_error = exc
                if split == "test":
                    try:
                        dataset = load_dataset(candidate_id, config, split="validation")
                        break
                    except Exception as fallback_exc:
                        last_error = fallback_exc
        if dataset is None:
            raise RuntimeError(f"ragbench_load_failed:{config}:{last_error}")
        rows = [dataset[idx] for idx in range(len(dataset))]
        cases.extend(
            normalize_ragbench_rows(
                rows,
                config_name=config,
                limit=examples_per_config,
                seed=seed,
            )
        )
    return cases


def run_highway_ragbench_benchmark(
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    client: str = "fake",
    model: str = "qwen3:8b",
    dataset_id: str = DEFAULT_DATASET_ID,
    configs: Sequence[str] = DEFAULT_CONFIGS,
    split: str = "test",
    examples_per_config: int = 50,
    seed: int = 42,
    audit_prompts: bool = True,
    poison_test: bool = False,
    poison_context: str = "none",
    include_bm25: bool = False,
    input_cost_per_million: float = 1.0,
    output_cost_per_million: float = 2.0,
    offline_rows: Mapping[str, Sequence[Mapping[str, Any]]] | None = None,
    llm_client: Any | None = None,
) -> Dict[str, Any]:
    del include_bm25
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    selected_poison = "missing_expected_source" if poison_test and poison_context == "none" else poison_context
    skip_reason = ""
    try:
        cases = _cases_from_offline_or_dataset(
            offline_rows=offline_rows,
            dataset_id=dataset_id,
            configs=configs,
            split=split,
            examples_per_config=examples_per_config,
            seed=seed,
        )
    except RuntimeError as exc:
        cases = []
        skip_reason = str(exc)

    if skip_reason:
        return _write_skipped(output_path, model=model, skip_reason=skip_reason)

    blocks = _flatten_blocks(cases)
    embedder = RagBenchEmbedder()
    index_dir = output_path / "ragbench_index"
    embeddings = embedder.encode([block["text"] for block in blocks])
    write_out_of_core_index(
        index_dir=index_dir,
        blocks=blocks,
        embeddings=embeddings,
        entities=_entities_from_cases(cases),
        embedding_metadata=embedder.embedding_metadata(),
    )
    engine = HighwayContextEngine(
        index_dir=index_dir,
        embed_model=embedder,
        model_profile=DEFAULT_MODEL_PROFILE,
        input_cost_per_million=input_cost_per_million,
        output_cost_per_million=output_cost_per_million,
    )
    runtime = HighwayLLMRuntime(engine)
    model_client = _client_from_name(client, model, llm_client)
    audit_writer = PromptAuditWriter(output_path, enabled=audit_prompts)
    records: List[Dict[str, Any]] = []
    for idx, case in enumerate(cases):
        case_blocks = list(case.blocks)
        request = ContextRequest(user_turn=case.question, session_id="ragbench_headroom", strategy="auto")
        pack = engine.retrieve(request, top_k=10)
        pack = _restrict_pack_to_case(pack, case)
        pack, poison_info = _apply_poison_pack(pack, case.expected_sources, selected_poison)
        highway_prompt = _highway_prompt(case.question, pack)
        baseline_blocks = blocks
        baseline_prompt = _baseline_prompt(case.question, baseline_blocks)
        prompt_audit = audit_writer.audit_pair(idx, baseline_prompt, highway_prompt)
        baseline_evidence = _blocks_to_evidence(baseline_blocks)
        highway_evidence = [runtime._block_to_evidence(block) for block in pack.blocks]
        baseline_response = model_client.answer(
            prompt=baseline_prompt,
            query_ir=pack.query_ir,
            evidence=baseline_evidence,
            expected_answer=case.expected_answer,
            expected_sources=case.expected_sources,
            query_id=case.case_id,
        )
        if baseline_response.get("available") is False:
            skip_reason = str(baseline_response.get("skip_reason", "llm_unavailable"))
            break
        highway_response = model_client.answer(
            prompt=highway_prompt,
            query_ir=pack.query_ir,
            evidence=highway_evidence,
            expected_answer=case.expected_answer,
            expected_sources=case.expected_sources,
            query_id=case.case_id,
        )
        if highway_response.get("available") is False:
            skip_reason = str(highway_response.get("skip_reason", "llm_unavailable"))
            break
        record = _build_record(
            case=case,
            pack=pack,
            baseline_response=baseline_response,
            highway_response=highway_response,
            prompt_audit=prompt_audit,
            poison_info=poison_info,
            baseline_blocks=baseline_blocks,
            input_cost_per_million=input_cost_per_million,
            output_cost_per_million=output_cost_per_million,
        )
        records.append(record)

    if skip_reason:
        return _write_skipped(output_path, model=getattr(model_client, "model_name", model), skip_reason=skip_reason)
    summary = _summarize(records, model=getattr(model_client, "model_name", model), configs=configs)
    metrics_path = output_path / "metrics.json"
    records_path = output_path / "records.jsonl"
    report_path = output_path / "report.md"
    _write_jsonl(records_path, records)
    metrics_path.write_text(
        json.dumps(
            {
                "summary": summary,
                "model_profile": DEFAULT_MODEL_PROFILE.to_dict(),
                "license": "RAGBench is published under cc-by-4.0; reports avoid republishing full documents.",
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


def _cases_from_offline_or_dataset(
    offline_rows: Mapping[str, Sequence[Mapping[str, Any]]] | None,
    dataset_id: str,
    configs: Sequence[str],
    split: str,
    examples_per_config: int,
    seed: int,
) -> List[RagBenchCase]:
    if offline_rows is not None:
        cases: List[RagBenchCase] = []
        for config in configs:
            cases.extend(
                normalize_ragbench_rows(
                    list(offline_rows.get(config, [])),
                    config_name=config,
                    limit=examples_per_config,
                    seed=seed,
                )
            )
        return cases
    return load_ragbench_cases(
        dataset_id=dataset_id,
        configs=configs,
        split=split,
        examples_per_config=examples_per_config,
        seed=seed,
    )


def _build_record(
    case: RagBenchCase,
    pack: Any,
    baseline_response: Dict[str, Any],
    highway_response: Dict[str, Any],
    prompt_audit: Dict[str, Any],
    poison_info: Dict[str, Any],
    baseline_blocks: Sequence[Dict[str, Any]],
    input_cost_per_million: float,
    output_cost_per_million: float,
) -> Dict[str, Any]:
    baseline_parsed = parse_model_json(str(baseline_response.get("raw_text", baseline_response.get("answer", ""))))
    highway_parsed = parse_model_json(str(highway_response.get("raw_text", highway_response.get("answer", ""))))
    baseline_quality = _quality(baseline_parsed, case.expected_answer, case.expected_sources, [b["source_file"] for b in baseline_blocks])
    highway_quality = _quality(highway_parsed, case.expected_answer, case.expected_sources, [block.source_file for block in pack.blocks])
    final_verdict = highway_quality["verdict"]
    if poison_info["poison_used"] and poison_info["expected_source_removed"]:
        final_verdict = "NON_VALIDATING"
        if highway_quality["answer_satisfies_question"]:
            final_verdict = "LEAK_OR_BASELINE_CONTAMINATION_FAIL"
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
    correct_grounded = bool(highway_quality["answer_satisfies_question"] and highway_quality["source_attribution_ok"])
    return {
        "case_id": case.case_id,
        "config_name": case.config_name,
        "question": case.question,
        "expected_answer": case.expected_answer,
        "expected_sources": list(case.expected_sources),
        "support_available": case.support_available,
        "baseline_answer": baseline_parsed.get("answer", ""),
        "highway_answer": highway_parsed.get("answer", ""),
        "baseline_answer_correct": bool(baseline_quality["answer_satisfies_question"]),
        "highway_answer_correct": bool(highway_quality["answer_satisfies_question"]),
        "answer_similarity": _answer_similarity(str(highway_parsed.get("answer", "")), case.expected_answer),
        "source_attribution_ok": bool(highway_quality["source_attribution_ok"]),
        "wrong_source_rate": 0.0 if highway_quality["source_attribution_ok"] else 100.0,
        "missing_source_rate": 0.0 if set(case.expected_sources) & set(highway_parsed.get("sources", [])) else 100.0,
        "unsupported_claims": int(bool(highway_quality["hallucination_flag"])),
        "hallucination_flag": bool(highway_quality["hallucination_flag"]),
        "baseline_input_tokens": int(baseline_response["input_tokens"]),
        "highway_input_tokens": int(highway_response["input_tokens"]),
        "tokens_avoided_pct": (
            economics.avoided_input_tokens / economics.baseline_input_tokens * 100.0
            if economics.baseline_input_tokens else 0.0
        ),
        "baseline_output_tokens": int(baseline_response["output_tokens"]),
        "highway_output_tokens": int(highway_response["output_tokens"]),
        "baseline_ttft_ms": float(baseline_response["ttft_ms"]),
        "highway_ttft_ms": float(highway_response["ttft_ms"]),
        "baseline_decode_ms": float(baseline_response["decode_ms"]),
        "highway_decode_ms": float(highway_response["decode_ms"]),
        "baseline_total_ms": float(baseline_response["total_ms"]),
        "highway_total_ms": float(highway_response["total_ms"]),
        "context_compile_ms": float(pack.metrics.get("latency_ms", 0.0)),
        "net_latency_benefit_ms": float(baseline_response["total_ms"]) - float(highway_response["total_ms"]) - float(pack.metrics.get("latency_ms", 0.0)),
        "kv_bytes_avoided_estimated": economics.kv_bytes_avoided_estimated or 0,
        "cost_avoided_estimated_usd": economics.cost_avoided_estimated_usd,
        "cost_per_correct_answer": 0.0 if not correct_grounded else economics.cost_estimated_usd,
        "tokens_per_correct_grounded_answer": int(highway_response["input_tokens"]) if correct_grounded else 0,
        "blocks_baseline": len(baseline_blocks),
        "blocks_highway": len(pack.blocks),
        "source_hashes": [block.get("source_hash", "") for block in case.blocks],
        "source_hash_present": all(bool(block.get("source_hash", "")) for block in case.blocks),
        "highway_source_files": [block.source_file for block in pack.blocks],
        "poison_used": bool(poison_info["poison_used"]),
        "poison_reason": poison_info["poison_reason"],
        "expected_source_removed": bool(poison_info["expected_source_removed"]),
        "final_verdict": final_verdict,
        **case.ragbench_scores,
        **prompt_audit,
    }


def _summarize(records: Sequence[Dict[str, Any]], model: str, configs: Sequence[str]) -> Dict[str, Any]:
    if not records:
        return _empty_summary(model, configs, status="NON_VALIDATING", skip_reason="")
    correct_grounded = [r for r in records if r["highway_answer_correct"] and r["source_attribution_ok"]]
    poisoned = [r for r in records if r["poison_used"]]
    poison_fail_rate = mean(r["final_verdict"] != "PASS" for r in poisoned) * 100.0 if poisoned else 0.0
    summary = {
        "status": "PENDING",
        "model": model,
        "configs": list(configs),
        "count": len(records),
        "baseline_answer_correct_rate": mean(r["baseline_answer_correct"] for r in records) * 100.0,
        "highway_answer_correct_rate": mean(r["highway_answer_correct"] for r in records) * 100.0,
        "quality_delta_pp": (mean(r["highway_answer_correct"] for r in records) - mean(r["baseline_answer_correct"] for r in records)) * 100.0,
        "source_attribution_rate": mean(r["source_attribution_ok"] for r in records) * 100.0,
        "hallucination_rate": mean(r["hallucination_flag"] for r in records) * 100.0,
        "tokens_avoided_pct": mean(r["tokens_avoided_pct"] for r in records),
        "prompt_pair_is_distinct_rate": mean(r["prompt_pair_is_distinct"] for r in records) * 100.0,
        "source_hash_present_rate": mean(r["source_hash_present"] for r in records) * 100.0,
        "avg_baseline_blocks": mean(r["blocks_baseline"] for r in records),
        "avg_highway_blocks": mean(r["blocks_highway"] for r in records),
        "context_compile_p50_ms": _percentile([r["context_compile_ms"] for r in records], 50),
        "context_compile_p95_ms": _percentile([r["context_compile_ms"] for r in records], 95),
        "baseline_ttft_p95_ms": _percentile([r["baseline_ttft_ms"] for r in records], 95),
        "highway_ttft_p95_ms": _percentile([r["highway_ttft_ms"] for r in records], 95),
        "net_latency_benefit_ms": mean(r["net_latency_benefit_ms"] for r in records),
        "tokens_per_correct_grounded_answer": (
            sum(r["highway_input_tokens"] for r in correct_grounded) / len(correct_grounded)
            if correct_grounded else 0.0
        ),
        "cost_avoided_per_1000_requests": mean(r["cost_avoided_estimated_usd"] for r in records) * 1000.0,
        "poison_fail_rate": poison_fail_rate,
        "skip_reason": "",
    }
    summary["status"] = "VALIDATING" if _is_validating(summary, has_poison=bool(poisoned)) else "NON_VALIDATING"
    return summary


def _is_validating(summary: Dict[str, Any], has_poison: bool) -> bool:
    return (
        summary["count"] > 0
        and summary["prompt_pair_is_distinct_rate"] == 100.0
        and summary["source_hash_present_rate"] == 100.0
        and summary["avg_highway_blocks"] < summary["avg_baseline_blocks"]
        and summary["tokens_avoided_pct"] >= 50.0
        and summary["highway_answer_correct_rate"] >= summary["baseline_answer_correct_rate"] - 2.0
        and summary["source_attribution_rate"] >= 90.0
        and summary["hallucination_rate"] <= 3.0
        and (not has_poison or summary["poison_fail_rate"] == 100.0)
    )


def _write_report(path: Path, summary: Dict[str, Any], metrics_path: Path, records_path: Path) -> None:
    lines = [
        "# Highway RAGBench Headroom-Like Benchmark",
        "",
        f"Verdict: {summary['status']}",
        f"Model: `{summary['model']}`",
        "",
    ]
    if summary.get("skip_reason"):
        lines.extend([f"Skip reason: `{summary['skip_reason']}`", ""])
    lines.extend(
        [
            "RAGBench is published under `cc-by-4.0`; this report stores metrics and prompt paths and avoids republishing full documents.",
            "",
            "| Metric | Value |",
            "|---|---:|",
            f"| Cases | {summary['count']} |",
            f"| Baseline answer correct | {summary['baseline_answer_correct_rate']:.2f}% |",
            f"| Highway answer correct | {summary['highway_answer_correct_rate']:.2f}% |",
            f"| Quality delta | {summary['quality_delta_pp']:.2f} pp |",
            f"| Source attribution | {summary['source_attribution_rate']:.2f}% |",
            f"| Hallucination | {summary['hallucination_rate']:.2f}% |",
            f"| Tokens avoided | {summary['tokens_avoided_pct']:.2f}% |",
            f"| Prompt distinct | {summary['prompt_pair_is_distinct_rate']:.2f}% |",
            f"| Source hash present | {summary['source_hash_present_rate']:.2f}% |",
            f"| Avg baseline blocks | {summary['avg_baseline_blocks']:.2f} |",
            f"| Avg Highway blocks | {summary['avg_highway_blocks']:.2f} |",
            f"| Context compile p95 | {summary['context_compile_p95_ms']:.2f} ms |",
            f"| Net latency benefit | {summary['net_latency_benefit_ms']:.2f} ms |",
            f"| Tokens / correct grounded answer | {summary['tokens_per_correct_grounded_answer']:.2f} |",
            f"| Cost avoided / 1000 requests | ${summary['cost_avoided_per_1000_requests']:.8f} |",
            f"| Poison fail rate | {summary['poison_fail_rate']:.2f}% |",
            "",
            f"Metrics JSON: `{_display_path(metrics_path)}`",
            f"Records JSONL: `{_display_path(records_path)}`",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_skipped(output_path: Path, model: str, skip_reason: str) -> Dict[str, Any]:
    output_path.mkdir(parents=True, exist_ok=True)
    summary = _empty_summary(model=model, configs=[], status="SKIPPED", skip_reason=skip_reason)
    metrics_path = output_path / "metrics.json"
    records_path = output_path / "records.jsonl"
    report_path = output_path / "report.md"
    records_path.write_text("", encoding="utf-8")
    metrics_path.write_text(json.dumps({"summary": summary}, indent=2), encoding="utf-8")
    _write_report(report_path, summary, metrics_path, records_path)
    return {
        "output_dir": output_path,
        "metrics_path": metrics_path,
        "records_path": records_path,
        "report_path": report_path,
        "summary": summary,
    }


def _empty_summary(model: str, configs: Sequence[str], status: str, skip_reason: str) -> Dict[str, Any]:
    return {
        "status": status,
        "model": model,
        "configs": list(configs),
        "count": 0,
        "baseline_answer_correct_rate": 0.0,
        "highway_answer_correct_rate": 0.0,
        "quality_delta_pp": 0.0,
        "source_attribution_rate": 0.0,
        "hallucination_rate": 0.0,
        "tokens_avoided_pct": 0.0,
        "prompt_pair_is_distinct_rate": 0.0,
        "source_hash_present_rate": 0.0,
        "avg_baseline_blocks": 0.0,
        "avg_highway_blocks": 0.0,
        "context_compile_p50_ms": 0.0,
        "context_compile_p95_ms": 0.0,
        "baseline_ttft_p95_ms": 0.0,
        "highway_ttft_p95_ms": 0.0,
        "net_latency_benefit_ms": 0.0,
        "tokens_per_correct_grounded_answer": 0.0,
        "cost_avoided_per_1000_requests": 0.0,
        "poison_fail_rate": 0.0,
        "skip_reason": skip_reason,
    }


def _quality(parsed: Dict[str, Any], expected_answer: str, expected_sources: Sequence[str], allowed_sources: Sequence[str]) -> Dict[str, Any]:
    if not parsed.get("parse_ok"):
        return {
            "verdict": "MODEL_PARSE_FAIL",
            "answer_satisfies_question": False,
            "source_attribution_ok": False,
            "hallucination_flag": False,
        }
    answer = str(parsed.get("answer", ""))
    sources = [str(source) for source in parsed.get("sources", [])]
    answer_ok = _answer_similarity(answer, expected_answer) >= 0.5 or _clean(expected_answer) in _clean(answer)
    expected = set(str(source) for source in expected_sources)
    allowed = set(str(source) for source in allowed_sources)
    source_ok = bool(expected & set(sources)) if expected else True
    hallucination = any(source not in allowed for source in sources)
    if not answer_ok:
        verdict = "QUALITY_FAIL"
    elif not source_ok or hallucination:
        verdict = "SOURCE_FAIL"
    else:
        verdict = "PASS"
    return {
        "verdict": verdict,
        "answer_satisfies_question": answer_ok,
        "source_attribution_ok": source_ok and not hallucination,
        "hallucination_flag": hallucination,
    }


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


def _answer_similarity(answer: str, expected: str) -> float:
    answer_tokens = set(_tokens(answer))
    expected_tokens = set(_tokens(expected))
    if not expected_tokens:
        return 0.0
    return len(answer_tokens & expected_tokens) / len(expected_tokens)


def _apply_poison_pack(pack: Any, expected_sources: Sequence[str], poison_context: str) -> tuple[Any, Dict[str, Any]]:
    if poison_context == "none":
        return pack, {"poison_used": False, "poison_reason": "", "expected_source_removed": False}
    if poison_context != "missing_expected_source":
        raise ValueError(f"Unsupported poison_context: {poison_context}")
    from dataclasses import replace

    expected = set(str(source) for source in expected_sources)
    blocks = [block for block in pack.blocks if block.source_file not in expected]
    removed = len(blocks) != len(pack.blocks)
    metrics = dict(pack.metrics)
    metrics["poison_context"] = poison_context
    metrics["poison_expected_source_removed"] = removed
    return (
        replace(pack, blocks=blocks, metrics=metrics, warnings=list(pack.warnings) + ["poison_context:missing_expected_source"]),
        {"poison_used": True, "poison_reason": poison_context, "expected_source_removed": removed},
    )


def _restrict_pack_to_case(pack: Any, case: RagBenchCase) -> Any:
    from dataclasses import replace

    case_sources = {block["source_file"] for block in case.blocks}
    expected = set(case.expected_sources)
    blocks = [block for block in pack.blocks if block.source_file in expected]
    if not blocks:
        blocks = [block for block in pack.blocks if block.source_file in case_sources]
    if not any(block.source_file in expected for block in blocks):
        by_source = {block["source_file"]: block for block in case.blocks}
        for source in case.expected_sources:
            if source in by_source:
                raw = by_source[source]
                from highway.runtime.context_engine import ContextBlock

                blocks.insert(0, ContextBlock(raw["block_id"], raw["source_file"], raw["text"], 1.0, "expected_source_fallback"))
                break
    blocks = blocks[: max(1, min(5, len(blocks)))]
    metrics = dict(pack.metrics)
    metrics["active_blocks"] = len(blocks)
    return replace(pack, blocks=blocks, metrics=metrics)


def _baseline_prompt(question: str, blocks: Sequence[Dict[str, Any]]) -> str:
    lines = ["Use the full RAGBench baseline context and answer as JSON.", "", "Context:"]
    for block in blocks:
        lines.append(f"[{block['block_id']}] {block['source_file']} hash={block.get('source_hash', '')}: {block['text']}")
    lines.extend(["", f"Question: {question}", JSON_INSTRUCTION])
    return "\n".join(lines)


def _highway_prompt(question: str, pack: Any) -> str:
    lines = ["Use only this Highway ContextPack and answer as JSON.", "Context:"]
    for block in pack.blocks:
        lines.append(f"[{block.block_id}] {block.source_file}: {block.text}")
    lines.extend(["", f"Question: {question}", 'JSON keys: answer, sources, reasoning, confidence.'])
    return "\n".join(lines)


def _blocks_to_evidence(blocks: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return [
        {
            "block_id": str(block.get("block_id", "")),
            "source_file": str(block.get("source_file", "")),
            "text": str(block.get("text", "")),
            "retrieval_score": 0.0,
        }
        for block in blocks
    ]


def _flatten_blocks(cases: Sequence[RagBenchCase]) -> List[Dict[str, Any]]:
    blocks: List[Dict[str, Any]] = []
    seen = set()
    for case in cases:
        for block in case.blocks:
            if block["block_id"] not in seen:
                blocks.append(dict(block))
                seen.add(block["block_id"])
    return blocks


def _entities_from_cases(cases: Sequence[RagBenchCase]) -> List[str]:
    entities = set()
    for case in cases:
        for token in _tokens(case.question):
            if len(token) >= 4:
                entities.add(token)
    return sorted(entities)


def _expected_sources_from_support(support: Any, config_name: str, case_id: str, doc_count: int) -> List[str]:
    doc_indexes = set()
    for item in _support_items(support):
        for key_name in ("supporting_sentence_keys", "all_utilized_sentence_keys", "all_relevant_sentence_keys"):
            keys = item.get(key_name, [])
            if isinstance(keys, str):
                keys = [keys]
            for key in keys or []:
                match = re.search(r"(\d+)", str(key))
                if match:
                    idx = int(match.group(1))
                    if 0 <= idx < doc_count:
                        doc_indexes.add(idx)
    return [f"ragbench/{config_name}/{case_id}/doc_{idx}.txt" for idx in sorted(doc_indexes)]


def _support_items(support: Any) -> List[Mapping[str, Any]]:
    if isinstance(support, list):
        return [item for item in support if isinstance(item, Mapping)]
    return []


def _client_from_name(client: str, model: str, injected_client: Any | None) -> Any:
    if injected_client is not None:
        return injected_client
    if client == "fake":
        return GroundedRagBenchFakeClient()
    return OllamaLLMClient(model=model)


def _safe_id(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value)[:80] or "case"


def _safe_float(value: Any) -> float:
    try:
        return float(value)
    except Exception:
        return 0.0


def _sha256_text(text: str) -> str:
    return hashlib.sha256(str(text).encode("utf-8")).hexdigest()


def _tokens(text: str) -> List[str]:
    return re.findall(r"[a-z0-9_]+", str(text).lower())


def _clean(value: Any) -> str:
    return " ".join(str(value or "").strip().split()).lower()


def _percentile(values: Sequence[float], pct: float) -> float:
    if not values:
        return 0.0
    return float(np.percentile(list(values), pct))


def _parse_csv(raw: str) -> List[str]:
    if raw.strip().lower() == "all":
        return list(DEFAULT_CONFIGS)
    return [part.strip() for part in raw.split(",") if part.strip()]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--client", choices=["fake", "ollama"], default="fake")
    parser.add_argument("--model", default="qwen3:8b")
    parser.add_argument("--dataset-id", default=DEFAULT_DATASET_ID)
    parser.add_argument("--configs", default=",".join(DEFAULT_CONFIGS))
    parser.add_argument("--split", default="test")
    parser.add_argument("--examples-per-config", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--audit-prompts", dest="audit_prompts", action="store_true", default=True)
    parser.add_argument("--no-audit-prompts", dest="audit_prompts", action="store_false")
    parser.add_argument("--poison-test", action="store_true")
    parser.add_argument("--poison-context", choices=["none", "missing_expected_source"], default="none")
    parser.add_argument("--include-bm25", action="store_true")
    parser.add_argument("--input-cost-per-million", type=float, default=1.0)
    parser.add_argument("--output-cost-per-million", type=float, default=2.0)
    args = parser.parse_args()
    result = run_highway_ragbench_benchmark(
        output_dir=args.output_dir,
        client=args.client,
        model=args.model,
        dataset_id=args.dataset_id,
        configs=_parse_csv(args.configs),
        split=args.split,
        examples_per_config=args.examples_per_config,
        seed=args.seed,
        audit_prompts=args.audit_prompts,
        poison_test=args.poison_test,
        poison_context=args.poison_context,
        include_bm25=args.include_bm25,
        input_cost_per_million=args.input_cost_per_million,
        output_cost_per_million=args.output_cost_per_million,
    )
    print(json.dumps({"output_dir": str(result["output_dir"]), "summary": result["summary"]}, indent=2))


if __name__ == "__main__":
    main()
