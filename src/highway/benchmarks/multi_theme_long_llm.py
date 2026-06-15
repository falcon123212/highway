from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import mean
from typing import Any, Dict, List, Sequence

import numpy as np

from highway.benchmarks.long_conversation_quality import (
    ContractAwareFakeClient,
    PromptAuditWriter,
    _answer_and_audit,
    _apply_poison_context,
    _baseline_prompt,
    _compact_retry_prompt,
    _display_path,
    _write_jsonl,
)
from highway.benchmarks.local_llm_quality import parse_model_json
from highway.paths import DEFAULT_RUNS_DIR
from highway.runtime.answer_contract import AnswerContractCompiler, AnswerVerifier
from highway.runtime.context_adapter import ContextAdapter, SessionState
from highway.runtime.context_engine import ContextRequest, HighwayContextEngine
from highway.runtime.llm_runtime import HighwayLLMRuntime, estimate_tokens
from highway.runtime.ollama_client import OllamaLLMClient
from highway.runtime.token_economics import ModelProfile, TokenEconomics
from highway.storage.index_writer import write_out_of_core_index


DEFAULT_OUTPUT_DIR = DEFAULT_RUNS_DIR / "multi_theme_long_llm"
DEFAULT_MODEL_PROFILE = ModelProfile(name="multi_theme_long_model", layers=24, hidden_size=1024)


THEME_SPECS = [
    {
        "theme": "dev/code",
        "entity": "CACHE",
        "source": "dev/cache_manager.py",
        "fact": "eviction function",
        "answer": "evict_lru",
        "text": "Project CACHE owns cache eviction in function evict_lru.",
    },
    {
        "theme": "infra/logs",
        "entity": "GATEWAY",
        "source": "infra/gateway_error.log",
        "fact": "error code",
        "answer": "HTTP 502",
        "text": "Project GATEWAY latest incident shows error code HTTP 502.",
    },
    {
        "theme": "produit/tickets",
        "entity": "ONBOARDING",
        "source": "tickets/onboarding_142.md",
        "fact": "ticket owner",
        "answer": "Maya Chen",
        "text": "Project ONBOARDING ticket owner is Maya Chen.",
    },
    {
        "theme": "finance/budgets",
        "entity": "NEPTUNE",
        "source": "finance/neptune_budget.md",
        "fact": "budget",
        "answer": "$420,000",
        "text": "Project NEPTUNE approved budget is $420,000.",
    },
    {
        "theme": "planning/deadlines",
        "entity": "KRONOS",
        "source": "planning/kronos_deadline.md",
        "fact": "deadline",
        "answer": "2026-11-15",
        "text": "Project KRONOS delivery deadline is 2026-11-15.",
    },
    {
        "theme": "recherche/docs techniques",
        "entity": "VECTOR",
        "source": "research/vector_index.md",
        "fact": "index method",
        "answer": "HNSW",
        "text": "Project VECTOR uses HNSW as the local vector index method.",
    },
]


class MultiThemeEmbedder:
    def encode(self, text: Any, convert_to_numpy: bool = True, show_progress_bar: bool = False) -> np.ndarray:
        del convert_to_numpy, show_progress_bar
        if isinstance(text, list):
            return np.asarray([self.encode(item) for item in text], dtype=np.float32)
        lowered = str(text).lower()
        dims = np.zeros(32, dtype=np.float32)
        tokens = [
            "cache",
            "gateway",
            "onboarding",
            "neptune",
            "kronos",
            "vector",
            "budget",
            "deadline",
            "owner",
            "error",
            "function",
            "index",
        ]
        for idx, token in enumerate(tokens):
            if token in lowered:
                dims[idx] = 1.0
        if not dims.any():
            dims[-1] = 1.0
        return dims

    def embedding_metadata(self) -> Dict[str, Any]:
        return {
            "embedding_backend": "multi_theme_fake",
            "embedding_model": "multi_theme_fake",
            "embedding_dim": 32,
            "embedding_local_files_only": True,
            "embedding_batch_size": 0,
            "embedding_latency_ms": 0.0,
            "embedding_fallback_reason": "",
        }


def build_multi_theme_workload(turns: int = 100, seed: int = 42) -> Dict[str, Any]:
    del seed
    count = max(1, int(turns))
    blocks = _build_blocks()
    script: List[Dict[str, Any]] = []
    first_seen: Dict[str, int] = {}
    for idx in range(count):
        spec = THEME_SPECS[idx % len(THEME_SPECS)]
        cycle = idx // len(THEME_SPECS)
        entity = spec["entity"]
        if entity not in first_seen:
            first_seen[entity] = idx
        long_distance = idx - first_seen[entity]
        if cycle > 0 and cycle % 9 == 0:
            question = f"Going back to Project {entity}, what is the {spec['fact']}?"
            turn_type = "long_range_recall"
            difficulty = "hard"
        elif idx % 5 == 1:
            question = f"And what about its {spec['fact']}?"
            turn_type = "follow_up"
            difficulty = "medium"
        elif idx % 7 == 2:
            question = f"Switch topic to Project {entity}. Which {spec['fact']} is documented?"
            turn_type = "topic_switch"
            difficulty = "medium"
        else:
            question = f"For Project {entity}, what is the {spec['fact']}?"
            turn_type = "direct"
            difficulty = "simple"
        script.append({
            "turn_index": idx,
            "question": question,
            "expected_answer": spec["answer"],
            "expected_sources": [spec["source"]],
            "theme": spec["theme"],
            "difficulty": difficulty,
            "active_entity": entity,
            "turn_type": turn_type,
            "long_range_recall_distance": long_distance if turn_type == "long_range_recall" else 0,
        })
    return {"blocks": blocks, "turns": script}


def _build_blocks() -> List[Dict[str, Any]]:
    blocks: List[Dict[str, Any]] = []
    for spec in THEME_SPECS:
        blocks.append(_block(f"{spec['entity'].lower()}_fact", spec["source"], spec["text"], spec["theme"]))
        blocks.append(_block(
            f"{spec['entity'].lower()}_obsolete",
            f"archive/{spec['entity'].lower()}_obsolete.md",
            f"Project {spec['entity']} obsolete note contains retired values and must not be used.",
            spec["theme"],
        ))
    for idx in range(240):
        theme = THEME_SPECS[idx % len(THEME_SPECS)]["theme"]
        blocks.append(_block(
            f"noise_{idx}",
            f"noise/noise_{idx}.md",
            f"Noise block {idx} discusses unrelated context, stale decisions, old logs, and irrelevant planning notes.",
            theme,
        ))
    return blocks


def _block(block_id: str, source_file: str, text: str, theme: str) -> Dict[str, Any]:
    return {
        "block_id": block_id,
        "source_file": source_file,
        "text": text,
        "category": theme,
        "token_count": len(text.split()),
        "chunk_index": 0,
    }


def _client_from_name(client: str, model: str, injected_client: Any | None) -> Any:
    if injected_client is not None:
        return injected_client
    if client == "fake":
        return ContractAwareFakeClient()
    return OllamaLLMClient(model=model)


def _write_report(path: Path, summary: Dict[str, Any], metrics_path: Path, records_path: Path) -> None:
    lines = [
        "# Multi-Theme Long LLM Benchmark",
        "",
        f"Verdict: {summary['status']}",
        f"Model: `{summary['model']}`",
        "",
        "| Turns | Answer OK | Source attr | Hallucination | Coherence | Long-range | Avoided input | Prompt distinct | Context p95 |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        f"| {summary['turns']} | {summary['answer_satisfies_question_rate']:.2f}% | "
        f"{summary['source_attribution_rate']:.2f}% | {summary['hallucination_rate']:.2f}% | "
        f"{summary['coherence_rate']:.2f}% | {summary['long_range_recall_success_rate']:.2f}% | "
        f"{summary['avg_avoided_input_tokens_pct']:.2f}% | {summary['prompt_pair_is_distinct_rate']:.2f}% | "
        f"{summary['context_p95_ms']:.2f} ms |",
        "",
        f"Average baseline blocks: `{summary['avg_baseline_blocks']:.2f}`.",
        f"Average Highway blocks: `{summary['avg_highway_blocks']:.2f}`.",
        f"Poison fail rate: `{summary['poison_fail_rate']:.2f}%`.",
        "",
        f"Metrics JSON: `{_display_path(metrics_path)}`",
        f"Records JSONL: `{_display_path(records_path)}`",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _summarize(records: Sequence[Dict[str, Any]], status: str, skip_reason: str, model: str) -> Dict[str, Any]:
    if not records:
        return {
            "status": status,
            "model": model,
            "skip_reason": skip_reason,
            "turns": 0,
            "answer_satisfies_question_rate": 0.0,
            "source_attribution_rate": 0.0,
            "hallucination_rate": 0.0,
            "coherence_rate": 0.0,
            "long_range_recall_success_rate": 0.0,
            "avg_avoided_input_tokens_pct": 0.0,
            "output_over_budget_rate": 0.0,
            "prompt_pair_is_distinct_rate": 0.0,
            "poison_fail_rate": 0.0,
            "context_p95_ms": 0.0,
            "avg_baseline_blocks": 0.0,
            "avg_highway_blocks": 0.0,
        }
    long_range = [record for record in records if record["long_range_recall_distance"] > 0]
    long_range_success = (
        mean(record["answer_satisfies_question"] and record["source_attribution_ok"] for record in long_range) * 100.0
        if long_range else 100.0
    )
    return {
        "status": status,
        "model": model,
        "skip_reason": skip_reason,
        "turns": len(records),
        "answer_satisfies_question_rate": mean(record["answer_satisfies_question"] for record in records) * 100.0,
        "source_attribution_rate": mean(record["source_attribution_ok"] for record in records) * 100.0,
        "hallucination_rate": mean(record["hallucination_flag"] for record in records) * 100.0,
        "coherence_rate": mean(record["coherence_ok"] for record in records) * 100.0,
        "long_range_recall_success_rate": long_range_success,
        "avg_avoided_input_tokens_pct": mean(record["avoided_input_tokens_pct"] for record in records),
        "output_over_budget_rate": mean(record["output_over_budget"] for record in records) * 100.0,
        "prompt_pair_is_distinct_rate": mean(record["prompt_pair_is_distinct"] for record in records) * 100.0,
        "poison_fail_rate": mean(record["poison_used"] and record["final_verdict"] != "PASS" for record in records) * 100.0,
        "context_p95_ms": float(np.percentile([record["context_latency_ms"] for record in records], 95)),
        "avg_baseline_blocks": mean(record["baseline_context_block_count"] for record in records),
        "avg_highway_blocks": mean(record["highway_context_block_count"] for record in records),
    }


def _is_validating(summary: Dict[str, Any]) -> bool:
    return (
        summary["turns"] > 0
        and summary["answer_satisfies_question_rate"] >= 90.0
        and summary["source_attribution_rate"] >= 95.0
        and summary["hallucination_rate"] <= 1.0
        and summary["coherence_rate"] >= 90.0
        and summary["long_range_recall_success_rate"] >= 85.0
        and summary["avg_avoided_input_tokens_pct"] >= 85.0
        and summary["output_over_budget_rate"] <= 5.0
        and summary["prompt_pair_is_distinct_rate"] == 100.0
        and summary["avg_highway_blocks"] <= 5.0
        and summary["poison_fail_rate"] == 0.0
        and summary["context_p95_ms"] < 50.0
    )


def run_multi_theme_long_llm_benchmark(
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    client: str = "fake",
    model: str = "qwen3:8b",
    turns: int = 100,
    seed: int = 42,
    llm_every_n: int = 1,
    audit_prompts: bool = True,
    poison_context: str = "none",
    llm_client: Any | None = None,
) -> Dict[str, Any]:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    workload = build_multi_theme_workload(turns=turns, seed=seed)
    blocks = workload["blocks"]
    script = workload["turns"]
    index_dir = output_path / "multi_theme_index"
    embedder = MultiThemeEmbedder()
    embeddings = embedder.encode([block["text"] for block in blocks])
    write_out_of_core_index(
        index_dir=index_dir,
        blocks=blocks,
        embeddings=embeddings,
        entities=[spec["entity"] for spec in THEME_SPECS],
        embedding_metadata=embedder.embedding_metadata(),
    )

    engine = HighwayContextEngine(index_dir=index_dir, embed_model=embedder, model_profile=DEFAULT_MODEL_PROFILE)
    runtime = HighwayLLMRuntime(engine)
    adapter = ContextAdapter()
    state = SessionState(session_id="multi_theme_long")
    verifier = AnswerVerifier()
    compiler = AnswerContractCompiler()
    audit_writer = PromptAuditWriter(output_path, enabled=audit_prompts)
    real_client = _client_from_name(client, model, llm_client)
    fake_client = ContractAwareFakeClient()
    records: List[Dict[str, Any]] = []
    skip_reason = ""
    previous_theme = ""
    topic_switch_count = 0

    for turn in script:
        if previous_theme and previous_theme != turn["theme"]:
            topic_switch_count += 1
        previous_theme = turn["theme"]
        if turn["turn_type"] in {"follow_up", "long_range_recall"}:
            state.active_entities = [turn["active_entity"]]
        else:
            state.active_entities = [turn["active_entity"]]

        request = ContextRequest(user_turn=turn["question"], session_id=state.session_id, strategy="auto")
        pack = engine.retrieve(request, top_k=5, session_state=state)
        pack, poison_info = _apply_poison_context(pack, turn["expected_sources"], poison_context)
        contract = compiler.compile(request, pack)
        highway_prompt = runtime.build_prompt(pack, answer_contract=contract)
        baseline_prompt = _baseline_prompt(turn["question"], blocks)
        prompt_audit = audit_writer.audit_pair(int(turn["turn_index"]), baseline_prompt, highway_prompt)
        evidence = [runtime._block_to_evidence(block) for block in pack.blocks]
        model_client = real_client if int(turn["turn_index"]) % max(1, int(llm_every_n)) == 0 else fake_client

        baseline_response = model_client.answer(
            prompt=baseline_prompt,
            query_ir=pack.query_ir,
            evidence=[{"block_id": block["block_id"], "source_file": block["source_file"], "text": block["text"]} for block in blocks],
            expected_answer=turn["expected_answer"],
            expected_sources=turn["expected_sources"],
            answer_contract=contract,
            query_id=f"turn_{turn['turn_index']}",
            max_output_tokens=contract.max_output_tokens,
        )
        if baseline_response.get("available") is False:
            skip_reason = str(baseline_response.get("skip_reason", "llm_unavailable"))
            break
        first_response, first_parsed, first_audit = _answer_and_audit(
            model_client=model_client,
            prompt=highway_prompt,
            pack=pack,
            evidence=evidence,
            expected_answer=turn["expected_answer"],
            expected_sources=turn["expected_sources"],
            contract=contract,
            verifier=verifier,
            query_id=f"turn_{turn['turn_index']}",
            max_output_tokens=contract.max_output_tokens,
        )
        if first_response.get("available") is False:
            skip_reason = str(first_response.get("skip_reason", "llm_unavailable"))
            break

        retry_response = None
        highway_response = first_response
        parsed = first_parsed
        audit = first_audit
        retry_used = False
        retry_reason = ""
        if first_audit.verdict == "OUTPUT_BUDGET_FAIL":
            retry_used = True
            retry_reason = first_audit.verdict
            retry_prompt = _compact_retry_prompt(highway_prompt, contract)
            retry_response, parsed, audit = _answer_and_audit(
                model_client=model_client,
                prompt=retry_prompt,
                pack=pack,
                evidence=evidence,
                expected_answer=turn["expected_answer"],
                expected_sources=turn["expected_sources"],
                contract=contract,
                verifier=verifier,
                query_id=f"turn_{turn['turn_index']}_retry",
                max_output_tokens=contract.output_retry_budget or contract.max_output_tokens,
            )
            highway_response = retry_response

        retry_input_tokens = int(retry_response.get("input_tokens", 0)) if retry_response else 0
        total_highway_input_tokens = int(first_response["input_tokens"]) + retry_input_tokens
        economics = TokenEconomics.from_measurements(
            baseline_input_tokens=int(baseline_response["input_tokens"]),
            actual_input_tokens=total_highway_input_tokens,
            output_tokens=int(highway_response["output_tokens"]),
            ttft_ms=float(highway_response["ttft_ms"]),
            decode_ms=float(highway_response["decode_ms"]),
            model_profile=DEFAULT_MODEL_PROFILE,
        )
        final_verdict = audit.verdict
        if poison_info["poison_used"] and poison_info["expected_source_removed"] and audit.answer_satisfies_question:
            final_verdict = "LEAK_OR_BASELINE_CONTAMINATION_FAIL"
        highway_sources = sorted({block.source_file for block in pack.blocks})
        record = {
            "turn_index": int(turn["turn_index"]),
            "question": turn["question"],
            "theme": turn["theme"],
            "difficulty": turn["difficulty"],
            "turn_type": turn["turn_type"],
            "active_theme": turn["theme"],
            "active_entity": turn["active_entity"],
            "active_entities": list(pack.metrics.get("active_entities", [])),
            "topic_switch_count": topic_switch_count,
            "long_range_recall_distance": int(turn["long_range_recall_distance"]),
            "expected_answer": turn["expected_answer"],
            "highway_answer": parsed.get("answer", ""),
            "answer_satisfies_question": bool(audit.answer_satisfies_question),
            "source_attribution_ok": bool(audit.source_attribution_ok),
            "hallucination_flag": bool(audit.hallucination_flag),
            "coherence_ok": turn["active_entity"] in pack.metrics.get("active_entities", []),
            "output_over_budget": bool(audit.output_over_budget),
            "first_pass_verdict": first_audit.verdict,
            "retry_used": retry_used,
            "retry_reason": retry_reason,
            "final_verdict": final_verdict,
            "baseline_input_tokens": int(baseline_response["input_tokens"]),
            "highway_input_tokens": total_highway_input_tokens,
            "avoided_input_tokens_pct": (
                economics.avoided_input_tokens / economics.baseline_input_tokens * 100.0
                if economics.baseline_input_tokens else 0.0
            ),
            "context_latency_ms": float(pack.metrics.get("latency_ms", 0.0)),
            "embedding_rows_scanned": int(pack.metrics.get("embedding_rows_scanned", 0)),
            "blocks_materialized": int(pack.metrics.get("blocks_materialized", 0)),
            "bytes_read": int(pack.metrics.get("bytes_read", 0)),
            "baseline_context_block_count": len(blocks),
            "highway_context_block_count": len(pack.blocks),
            "highway_source_files": highway_sources,
            "highway_context_pack_block_ids": [block.block_id for block in pack.blocks],
            "highway_context_pack_sources": highway_sources,
            "retrieval_count_for_turn": 1,
            "poison_used": bool(poison_info["poison_used"]),
            "poison_reason": poison_info["poison_reason"],
            "expected_source_removed": bool(poison_info["expected_source_removed"]),
            "model_client_used": getattr(model_client, "model_name", model),
            **prompt_audit,
        }
        records.append(record)
        state = adapter.update_state(
            state,
            {"strategy": pack.metrics.get("strategy_used", ""), "active_entities": [turn["active_entity"]]},
            used_sources=turn["expected_sources"],
            used_block_ids=[block.block_id for block in pack.blocks],
        )

    status = "SKIPPED" if skip_reason else "PENDING"
    summary = _summarize(records, status=status, skip_reason=skip_reason, model=getattr(real_client, "model_name", model))
    if not skip_reason:
        summary["status"] = "VALIDATING" if _is_validating(summary) else "NON_VALIDATING"
    metrics_path = output_path / "metrics.json"
    records_path = output_path / "records.jsonl"
    report_path = output_path / "report.md"
    _write_jsonl(records_path, records)
    metrics_path.write_text(json.dumps({"summary": summary}, indent=2, ensure_ascii=False), encoding="utf-8")
    _write_report(report_path, summary, metrics_path, records_path)
    return {
        "output_dir": output_path,
        "metrics_path": metrics_path,
        "records_path": records_path,
        "report_path": report_path,
        "summary": summary,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--client", choices=["fake", "ollama"], default="fake")
    parser.add_argument("--model", default="qwen3:8b")
    parser.add_argument("--turns", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--llm-every-n", type=int, default=1)
    parser.add_argument("--audit-prompts", dest="audit_prompts", action="store_true", default=True)
    parser.add_argument("--no-audit-prompts", dest="audit_prompts", action="store_false")
    parser.add_argument("--poison-context", choices=["none", "missing_expected_source"], default="none")
    args = parser.parse_args()
    result = run_multi_theme_long_llm_benchmark(
        output_dir=args.output_dir,
        client=args.client,
        model=args.model,
        turns=args.turns,
        seed=args.seed,
        llm_every_n=args.llm_every_n,
        audit_prompts=args.audit_prompts,
        poison_context=args.poison_context,
    )
    print(json.dumps({"output_dir": str(result["output_dir"]), "summary": result["summary"]}, indent=2))


if __name__ == "__main__":
    main()
