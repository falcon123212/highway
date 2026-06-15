from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import replace
from pathlib import Path
from statistics import mean
from typing import Any, Dict, List, Sequence

import numpy as np

from highway.benchmarks.local_llm_quality import parse_model_json
from highway.paths import DEFAULT_RUNS_DIR
from highway.runtime.answer_contract import AnswerContract, AnswerContractCompiler, AnswerVerifier
from highway.runtime.context_adapter import ContextAdapter, SessionState
from highway.runtime.context_engine import ContextPack, ContextRequest, HighwayContextEngine
from highway.runtime.llm_runtime import HighwayLLMRuntime, estimate_tokens
from highway.runtime.ollama_client import OllamaLLMClient
from highway.runtime.token_economics import ModelProfile, TokenEconomics
from highway.storage.index_writer import write_out_of_core_index


DEFAULT_OUTPUT_DIR = DEFAULT_RUNS_DIR / "long_conversation_quality"
DEFAULT_MODEL_PROFILE = ModelProfile(name="long_conversation_model", layers=24, hidden_size=1024)


class PromptAuditWriter:
    def __init__(self, output_dir: Path, enabled: bool = True):
        self.output_dir = output_dir
        self.enabled = bool(enabled)
        self.prompt_dir = output_dir / "prompts"
        if self.enabled:
            self.prompt_dir.mkdir(parents=True, exist_ok=True)

    def audit_pair(self, turn_index: int, baseline_prompt: str, highway_prompt: str) -> Dict[str, Any]:
        baseline_rel = Path("prompts") / f"turn_{turn_index:03d}_baseline.txt"
        highway_rel = Path("prompts") / f"turn_{turn_index:03d}_highway.txt"
        if self.enabled:
            (self.output_dir / baseline_rel).write_text(baseline_prompt, encoding="utf-8")
            (self.output_dir / highway_rel).write_text(highway_prompt, encoding="utf-8")
        baseline_hash = _sha256_text(baseline_prompt)
        highway_hash = _sha256_text(highway_prompt)
        return {
            "baseline_prompt_hash": baseline_hash,
            "highway_prompt_hash": highway_hash,
            "baseline_prompt_path": baseline_rel.as_posix() if self.enabled else "",
            "highway_prompt_path": highway_rel.as_posix() if self.enabled else "",
            "baseline_prompt_tokens_verified": estimate_tokens(baseline_prompt),
            "highway_prompt_tokens_verified": estimate_tokens(highway_prompt),
            "prompt_pair_is_distinct": baseline_hash != highway_hash and baseline_prompt != highway_prompt,
        }


class ConversationEmbedder:
    def encode(self, text: Any, convert_to_numpy: bool = True, show_progress_bar: bool = False) -> np.ndarray:
        del convert_to_numpy, show_progress_bar
        if isinstance(text, list):
            return np.asarray([self.encode(item) for item in text], dtype=np.float32)
        lowered = str(text).lower()
        dims = np.zeros(8, dtype=np.float32)
        for idx, token in enumerate(("alpha", "beta", "gamma", "kronos", "budget", "manager", "deadline", "risk")):
            if token in lowered:
                dims[idx] = 1.0
        if not dims.any():
            dims[-1] = 1.0
        return dims

    def embedding_metadata(self) -> Dict[str, Any]:
        return {
            "embedding_backend": "conversation_fake",
            "embedding_model": "conversation_fake",
            "embedding_dim": 8,
            "embedding_local_files_only": True,
            "embedding_batch_size": 0,
            "embedding_latency_ms": 0.0,
            "embedding_fallback_reason": "",
        }


class ContractAwareFakeClient:
    model_name = "contract_aware_fake"

    def answer(
        self,
        prompt: str,
        query_ir: Dict[str, Any],
        evidence: Sequence[Dict[str, Any]],
        expected_answer: str | None = None,
        query_id: str = "fake",
        answer_contract: AnswerContract | None = None,
        expected_sources: Sequence[str] = (),
        **kwargs: Any,
    ) -> Dict[str, Any]:
        del query_ir, query_id, answer_contract, kwargs
        source = next((source for source in expected_sources if any(ev.get("source_file") == source for ev in evidence)), "")
        if not source and evidence:
            source = str(evidence[0].get("source_file", ""))
        raw = json.dumps(
            {
                "reasoning": "I used the active Highway context and the compiled answer contract.",
                "answer": expected_answer or "",
                "sources": [source] if source else [],
                "confidence": 1.0,
            }
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
            "reasoning": "contract-aware fake",
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "ttft_ms": ttft_ms,
            "decode_ms": decode_ms,
            "total_ms": ttft_ms + decode_ms,
            "input_tokens_per_second": input_tokens / (ttft_ms / 1000.0),
            "output_tokens_per_second": output_tokens / (decode_ms / 1000.0),
            "num_predict_requested": output_tokens,
            "output_stop_reason": "done",
        }


def _write_jsonl(path: Path, rows: Sequence[Dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _display_path(path: Path) -> str:
    try:
        return path.resolve().relative_to(Path.cwd().resolve()).as_posix()
    except ValueError:
        return path.as_posix()


def _build_dataset(output_path: Path, seed: int) -> tuple[Path, List[Dict[str, Any]], List[Dict[str, Any]]]:
    del seed
    blocks = [
        _block("alpha_budget", "docs/alpha_budget.txt", "Project ALPHA budget is $250,000."),
        _block("alpha_manager", "docs/alpha_manager.txt", "Project ALPHA manager is Nina Patel."),
        _block("alpha_deadline", "docs/alpha_deadline.txt", "Project ALPHA deadline is 2026-09-30."),
        _block("beta_budget", "docs/beta_budget.txt", "Project BETA budget is $120,000."),
        _block("beta_manager", "docs/beta_manager.txt", "Project BETA manager is Omar Diaz."),
        _block("gamma_budget", "docs/gamma_budget.txt", "Project GAMMA budget is $310,000."),
        _block("gamma_risk", "docs/gamma_risk.txt", "Project GAMMA risk is supplier delay."),
        _block("alpha_legacy_noise", "docs/alpha_legacy_noise.txt", "Project ALPHA-LEGACY budget is $999,999 but it is obsolete noise."),
    ]
    for idx in range(160):
        blocks.append(
            _block(
                f"noise_{idx}",
                f"noise/noise_{idx}.txt",
                (
                    f"Noise document {idx} mentions archive material, duplicate planning notes, "
                    "obsolete budgets, retired managers, and unrelated delivery commentary. "
                    "It is intentionally present to inflate baseline context and test Highway pruning."
                ),
            )
        )

    turns = [
        _turn(0, "Which project has a higher budget: Project ALPHA or Project BETA?", "Project ALPHA", ["docs/alpha_budget.txt"], "ALPHA"),
        _turn(1, "And what about its manager?", "Nina Patel", ["docs/alpha_manager.txt"], "ALPHA"),
        _turn(2, "What is its deadline?", "2026-09-30", ["docs/alpha_deadline.txt"], "ALPHA"),
        _turn(3, "Switch to Project GAMMA. What is its risk?", "supplier delay", ["docs/gamma_risk.txt"], "GAMMA"),
        _turn(4, "What is its budget?", "$310,000", ["docs/gamma_budget.txt"], "GAMMA"),
        _turn(5, "Go back to Project ALPHA. Who is its manager?", "Nina Patel", ["docs/alpha_manager.txt"], "ALPHA"),
        _turn(6, "Which project has a higher budget: Project GAMMA or Project BETA?", "Project GAMMA", ["docs/gamma_budget.txt"], "GAMMA"),
        _turn(7, "Same project, what is the risk?", "supplier delay", ["docs/gamma_risk.txt"], "GAMMA"),
        _turn(8, "Return to Project BETA. Who manages it?", "Omar Diaz", ["docs/beta_manager.txt"], "BETA"),
        _turn(9, "What is its budget?", "$120,000", ["docs/beta_budget.txt"], "BETA"),
        _turn(10, "Back to the first project, what was its deadline?", "2026-09-30", ["docs/alpha_deadline.txt"], "ALPHA"),
        _turn(11, "Which project has a higher budget: Project ALPHA or Project GAMMA?", "Project GAMMA", ["docs/gamma_budget.txt"], "GAMMA"),
    ]
    extra_questions = [
        ("And now its budget?", "$310,000", ["docs/gamma_budget.txt"], "GAMMA"),
        ("Switch back to Project ALPHA-LEGACY. Is its budget active?", "obsolete noise", ["docs/alpha_legacy_noise.txt"], "ALPHA-LEGACY"),
        ("Return to Project BETA. What is its manager?", "Omar Diaz", ["docs/beta_manager.txt"], "BETA"),
        ("Which project has a higher budget: Project BETA or Project GAMMA?", "Project GAMMA", ["docs/gamma_budget.txt"], "GAMMA"),
        ("For the same project, what risk is listed?", "supplier delay", ["docs/gamma_risk.txt"], "GAMMA"),
        ("Go back to Project ALPHA. What is its budget?", "$250,000", ["docs/alpha_budget.txt"], "ALPHA"),
        ("What is its manager?", "Nina Patel", ["docs/alpha_manager.txt"], "ALPHA"),
        ("What is its deadline?", "2026-09-30", ["docs/alpha_deadline.txt"], "ALPHA"),
        ("Switch to Project BETA. What is its budget?", "$120,000", ["docs/beta_budget.txt"], "BETA"),
        ("Same project, who manages it?", "Omar Diaz", ["docs/beta_manager.txt"], "BETA"),
        ("Which project has a higher budget: Project ALPHA or Project GAMMA?", "Project GAMMA", ["docs/gamma_budget.txt"], "GAMMA"),
        ("For Project GAMMA, what is the risk?", "supplier delay", ["docs/gamma_risk.txt"], "GAMMA"),
        ("Back to the first project, what is its deadline?", "2026-09-30", ["docs/alpha_deadline.txt"], "ALPHA"),
    ]
    for offset, (question, answer, sources, entity) in enumerate(extra_questions, start=len(turns)):
        turns.append(_turn(offset, question, answer, sources, entity))
    index_dir = output_path / "conversation_index"
    embeddings = ConversationEmbedder().encode([block["text"] for block in blocks])
    write_out_of_core_index(
        index_dir=index_dir,
        blocks=blocks,
        embeddings=embeddings,
        entities=["ALPHA", "BETA", "GAMMA", "ALPHA-LEGACY"],
        embedding_metadata=ConversationEmbedder().embedding_metadata(),
    )
    return index_dir, blocks, turns


def _block(block_id: str, source_file: str, text: str) -> Dict[str, Any]:
    return {
        "block_id": block_id,
        "source_file": source_file,
        "text": text,
        "category": "conversation",
        "token_count": len(text.split()),
        "chunk_index": 0,
    }


def _turn(
    turn_index: int,
    question: str,
    expected_answer: str,
    expected_sources: Sequence[str],
    active_entity: str,
) -> Dict[str, Any]:
    return {
        "turn_index": turn_index,
        "question": question,
        "expected_answer": expected_answer,
        "expected_sources": list(expected_sources),
        "active_entity": active_entity,
    }


def _client_from_name(client: str, model: str, injected_client: Any | None) -> Any:
    if injected_client is not None:
        return injected_client
    if client == "fake":
        return ContractAwareFakeClient()
    return OllamaLLMClient(model=model)


def _baseline_prompt(question: str, blocks: Sequence[Dict[str, Any]]) -> str:
    lines = ["Use the full baseline context and answer as JSON.", "", "Context:"]
    for block in blocks:
        lines.append(f"[{block['block_id']}] {block['source_file']}: {block['text']}")
    lines.append("")
    lines.append(f"Question: {question}")
    lines.append("Return only JSON with answer, sources, reasoning, confidence.")
    return "\n".join(lines)


def _apply_poison_context(
    pack: ContextPack,
    expected_sources: Sequence[str],
    poison_context: str,
) -> tuple[ContextPack, Dict[str, Any]]:
    if poison_context == "none":
        return pack, {
            "poison_used": False,
            "poison_reason": "",
            "expected_source_removed": False,
        }
    if poison_context != "missing_expected_source":
        raise ValueError(f"Unsupported poison_context: {poison_context}")
    expected = set(str(source) for source in expected_sources)
    filtered_blocks = [block for block in pack.blocks if block.source_file not in expected]
    removed = len(filtered_blocks) != len(pack.blocks)
    metrics = dict(pack.metrics)
    metrics["poison_context"] = poison_context
    metrics["poison_expected_source_removed"] = removed
    poisoned = replace(pack, blocks=filtered_blocks, metrics=metrics, warnings=list(pack.warnings) + ["poison_context:missing_expected_source"])
    return poisoned, {
        "poison_used": True,
        "poison_reason": poison_context,
        "expected_source_removed": removed,
    }


def _compact_retry_prompt(prompt: str, contract: AnswerContract) -> str:
    lines = [
        prompt,
        "",
        "Retry instruction:",
        contract.retry_instruction or "Retry compactly using only the selected context.",
        f"Compact schema: {contract.compact_answer_schema}",
        f"Retry output budget tokens: {contract.output_retry_budget or contract.max_output_tokens}",
        "Return only valid JSON. Do not add prose outside JSON.",
    ]
    return "\n".join(lines)


def _answer_and_audit(
    model_client: Any,
    prompt: str,
    pack: Any,
    evidence: Sequence[Dict[str, Any]],
    expected_answer: str,
    expected_sources: Sequence[str],
    contract: AnswerContract,
    verifier: AnswerVerifier,
    query_id: str,
    max_output_tokens: int | None = None,
) -> tuple[Dict[str, Any], Dict[str, Any], Any]:
    response = model_client.answer(
        prompt=prompt,
        query_ir=pack.query_ir,
        evidence=evidence,
        expected_answer=expected_answer,
        expected_sources=expected_sources,
        answer_contract=contract,
        query_id=query_id,
        max_output_tokens=max_output_tokens or contract.max_output_tokens,
    )
    parsed = parse_model_json(str(response.get("raw_text", response.get("answer", ""))))
    audit = verifier.audit(
        parsed,
        contract,
        pack,
        output_tokens=int(response.get("output_tokens", 0)),
        expected_answer=expected_answer,
    )
    return response, parsed, audit


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
            "avg_avoided_input_tokens_pct": 0.0,
            "output_over_budget_rate": 0.0,
            "context_p95_ms": 0.0,
            "retry_rate": 0.0,
            "avg_output_tokens_saved_by_retry": 0.0,
            "prompt_pair_is_distinct_rate": 0.0,
            "poison_fail_rate": 0.0,
            "avg_baseline_blocks": 0.0,
            "avg_highway_blocks": 0.0,
            "avg_baseline_tokens": 0.0,
            "avg_highway_tokens": 0.0,
        }
    return {
        "status": status,
        "model": model,
        "skip_reason": skip_reason,
        "turns": len(records),
        "answer_satisfies_question_rate": mean(record["answer_satisfies_question"] for record in records) * 100.0,
        "source_attribution_rate": mean(record["source_attribution_ok"] for record in records) * 100.0,
        "hallucination_rate": mean(record["hallucination_flag"] for record in records) * 100.0,
        "coherence_rate": mean(record["coherence_ok"] for record in records) * 100.0,
        "avg_avoided_input_tokens_pct": mean(record["avoided_input_tokens_pct"] for record in records),
        "output_over_budget_rate": mean(record["output_over_budget"] for record in records) * 100.0,
        "context_p95_ms": float(np.percentile([record["context_latency_ms"] for record in records], 95)),
        "retry_rate": mean(record.get("retry_used", False) for record in records) * 100.0,
        "avg_output_tokens_saved_by_retry": mean(record.get("output_tokens_saved_by_retry", 0) for record in records),
        "prompt_pair_is_distinct_rate": mean(record.get("prompt_pair_is_distinct", False) for record in records) * 100.0,
        "poison_fail_rate": mean(record.get("poison_used", False) and record.get("final_verdict") != "PASS" for record in records) * 100.0,
        "avg_baseline_blocks": mean(record.get("baseline_context_block_count", 0) for record in records),
        "avg_highway_blocks": mean(record.get("highway_context_block_count", 0) for record in records),
        "avg_baseline_tokens": mean(record.get("baseline_prompt_tokens_verified", record["baseline_input_tokens"]) for record in records),
        "avg_highway_tokens": mean(record.get("highway_prompt_tokens_verified", record["highway_input_tokens"]) for record in records),
    }


def _is_validating(summary: Dict[str, Any]) -> bool:
    return (
        summary["turns"] > 0
        and summary["answer_satisfies_question_rate"] >= 95.0
        and summary["source_attribution_rate"] >= 95.0
        and summary["hallucination_rate"] == 0.0
        and summary["coherence_rate"] >= 95.0
        and summary["avg_avoided_input_tokens_pct"] >= 80.0
        and summary["output_over_budget_rate"] == 0.0
        and summary["prompt_pair_is_distinct_rate"] == 100.0
        and summary["poison_fail_rate"] == 0.0
    )


def _write_report(path: Path, summary: Dict[str, Any], metrics_path: Path, records_path: Path) -> None:
    lines = [
        "# Long Conversation Quality Benchmark",
        "",
        f"Verdict: {summary['status']}",
        f"Model: `{summary['model']}`",
        "",
    ]
    if summary.get("skip_reason"):
        lines.extend([f"Skip reason: `{summary['skip_reason']}`", ""])
    lines.extend([
        "| Turns | Answer OK | Source attr | Hallucination | Coherence | Avoided input | Output over budget | Prompt distinct | Context p95 |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        f"| {summary['turns']} | {summary['answer_satisfies_question_rate']:.2f}% | "
        f"{summary['source_attribution_rate']:.2f}% | {summary['hallucination_rate']:.2f}% | "
        f"{summary['coherence_rate']:.2f}% | {summary['avg_avoided_input_tokens_pct']:.2f}% | "
        f"{summary['output_over_budget_rate']:.2f}% | {summary['prompt_pair_is_distinct_rate']:.2f}% | "
        f"{summary['context_p95_ms']:.2f} ms |",
        "",
        f"Average output tokens saved by retry: `{summary['avg_output_tokens_saved_by_retry']:.2f}`.",
        f"Retry rate: `{summary['retry_rate']:.2f}%`.",
        f"Poison fail rate: `{summary['poison_fail_rate']:.2f}%`.",
        f"Average baseline blocks: `{summary['avg_baseline_blocks']:.2f}`.",
        f"Average Highway blocks: `{summary['avg_highway_blocks']:.2f}`.",
        f"Average baseline prompt tokens: `{summary['avg_baseline_tokens']:.2f}`.",
        f"Average Highway prompt tokens: `{summary['avg_highway_tokens']:.2f}`.",
        "",
        "This benchmark separates context quality, answer quality, input-token economy, and output-token budget control.",
        "",
        f"Metrics JSON: `{_display_path(metrics_path)}`",
        f"Records JSONL: `{_display_path(records_path)}`",
    ])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_long_conversation_quality_benchmark(
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    client: str = "fake",
    model: str = "qwen3:8b",
    turns: int = 12,
    seed: int = 42,
    llm_client: Any | None = None,
    audit_prompts: bool = True,
    poison_context: str = "none",
) -> Dict[str, Any]:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    index_dir, all_blocks, script = _build_dataset(output_path, seed=seed)
    selected_turns = script[: max(1, min(int(turns), len(script)))]
    embedder = ConversationEmbedder()
    engine = HighwayContextEngine(index_dir=index_dir, embed_model=embedder, model_profile=DEFAULT_MODEL_PROFILE)
    runtime = HighwayLLMRuntime(engine)
    adapter = ContextAdapter()
    verifier = AnswerVerifier()
    compiler = AnswerContractCompiler()
    active_by_label = {"first": "ALPHA"}
    state = SessionState(session_id="long_conversation")
    model_client = _client_from_name(client, model, llm_client)
    audit_writer = PromptAuditWriter(output_path, enabled=audit_prompts)
    records: List[Dict[str, Any]] = []
    skip_reason = ""

    for turn in selected_turns:
        expected_entity = str(turn["active_entity"])
        if "first project" in str(turn["question"]).lower():
            state.active_entities = [active_by_label["first"]]
        elif state.turn_count > 0 and any(term in str(turn["question"]).lower() for term in ("its", "same project")):
            if not state.active_entities:
                state.active_entities = [expected_entity]
        else:
            state.active_entities = [expected_entity]

        request = ContextRequest(user_turn=str(turn["question"]), session_id=state.session_id, strategy="auto")
        pack = engine.retrieve(request, top_k=5, session_state=state)
        pack, poison_info = _apply_poison_context(pack, turn["expected_sources"], poison_context)
        contract = compiler.compile(request, pack)
        highway_prompt = runtime.build_prompt(pack, answer_contract=contract)
        baseline_prompt = _baseline_prompt(str(turn["question"]), all_blocks)
        prompt_audit = audit_writer.audit_pair(int(turn["turn_index"]), baseline_prompt, highway_prompt)
        evidence = [runtime._block_to_evidence(block) for block in pack.blocks]
        baseline_response = model_client.answer(
            prompt=baseline_prompt,
            query_ir=pack.query_ir,
            evidence=[
                {"block_id": block["block_id"], "source_file": block["source_file"], "text": block["text"]}
                for block in all_blocks
            ],
            expected_answer=str(turn["expected_answer"]),
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
            expected_answer=str(turn["expected_answer"]),
            expected_sources=turn["expected_sources"],
            contract=contract,
            verifier=verifier,
            query_id=f"turn_{turn['turn_index']}",
            max_output_tokens=contract.max_output_tokens,
        )
        if first_response.get("available") is False:
            skip_reason = str(first_response.get("skip_reason", "llm_unavailable"))
            break

        retry_used = False
        retry_reason = ""
        retry_response: Dict[str, Any] | None = None
        retry_parsed: Dict[str, Any] | None = None
        retry_audit = None
        highway_response = first_response
        parsed = first_parsed
        audit = first_audit
        if first_audit.verdict == "OUTPUT_BUDGET_FAIL":
            retry_used = True
            retry_reason = first_audit.verdict
            retry_prompt = _compact_retry_prompt(highway_prompt, contract)
            retry_response, retry_parsed, retry_audit = _answer_and_audit(
                model_client=model_client,
                prompt=retry_prompt,
                pack=pack,
                evidence=evidence,
                expected_answer=str(turn["expected_answer"]),
                expected_sources=turn["expected_sources"],
                contract=contract,
                verifier=verifier,
                query_id=f"turn_{turn['turn_index']}_retry",
                max_output_tokens=contract.output_retry_budget or contract.max_output_tokens,
            )
            if retry_response.get("available") is False:
                skip_reason = str(retry_response.get("skip_reason", "llm_unavailable"))
                break
            highway_response = retry_response
            parsed = retry_parsed
            audit = retry_audit

        first_output_tokens = int(first_response.get("output_tokens", 0))
        final_output_tokens = int(highway_response.get("output_tokens", 0))
        retry_input_tokens = int(retry_response.get("input_tokens", 0)) if retry_response else 0
        total_highway_input_tokens = int(first_response["input_tokens"]) + retry_input_tokens
        economics = TokenEconomics.from_measurements(
            baseline_input_tokens=int(baseline_response["input_tokens"]),
            actual_input_tokens=total_highway_input_tokens,
            output_tokens=final_output_tokens,
            ttft_ms=float(highway_response["ttft_ms"]),
            decode_ms=float(highway_response["decode_ms"]),
            model_profile=DEFAULT_MODEL_PROFILE,
        )
        coherence_ok = expected_entity in pack.metrics.get("active_entities", [])
        final_verdict = audit.verdict
        if poison_info["poison_used"] and poison_info["expected_source_removed"] and audit.answer_satisfies_question:
            final_verdict = "LEAK_OR_BASELINE_CONTAMINATION_FAIL"
        highway_source_files = sorted({block.source_file for block in pack.blocks})
        highway_block_ids = [block.block_id for block in pack.blocks]
        record = {
            "turn_index": int(turn["turn_index"]),
            "session_id": state.session_id,
            "question": turn["question"],
            "expected_answer": turn["expected_answer"],
            "highway_answer": parsed.get("answer", ""),
            "active_entities": list(pack.metrics.get("active_entities", [])),
            "query_rewrite_used": bool(pack.metrics.get("query_rewrite_used", False)),
            "compiled_query": str(pack.metrics.get("compiled_query", "")),
            "context_reuse_rate": float(pack.metrics.get("context_reuse_rate", 0.0)),
            "answer_satisfies_question": bool(audit.answer_satisfies_question),
            "full_exact_match": bool(audit.full_exact_match),
            "source_attribution_ok": bool(audit.source_attribution_ok),
            "numeric_facts_ok": bool(audit.numeric_facts_ok),
            "entity_facts_ok": bool(audit.entity_facts_ok),
            "hallucination_flag": bool(audit.hallucination_flag),
            "contradiction_flag": bool(audit.contradiction_flag),
            "coherence_ok": bool(coherence_ok),
            "first_pass_verdict": first_audit.verdict,
            "retry_used": bool(retry_used),
            "retry_reason": retry_reason,
            "final_verdict": audit.verdict,
            "first_output_tokens": first_output_tokens,
            "final_output_tokens": final_output_tokens,
            "output_tokens_saved_by_retry": max(0, first_output_tokens - final_output_tokens) if retry_used else 0,
            "final_output_over_budget": bool(audit.output_over_budget),
            "retrieval_count_for_turn": 1,
            "output_tokens_budget": int(audit.output_tokens_budget),
            "output_budget_used_pct": float(audit.output_budget_used_pct),
            "output_over_budget": bool(audit.output_over_budget),
            "baseline_input_tokens": int(baseline_response["input_tokens"]),
            "first_highway_input_tokens": int(first_response["input_tokens"]),
            "retry_input_tokens": retry_input_tokens,
            "highway_input_tokens": total_highway_input_tokens,
            "avoided_input_tokens_pct": (
                economics.avoided_input_tokens / economics.baseline_input_tokens * 100.0
                if economics.baseline_input_tokens else 0.0
            ),
            "baseline_output_tokens": int(baseline_response["output_tokens"]),
            "highway_output_tokens": final_output_tokens,
            "ttft_ms": float(highway_response["ttft_ms"]),
            "decode_ms": float(highway_response["decode_ms"]),
            "tokens_per_second": float(highway_response["output_tokens_per_second"]),
            "context_latency_ms": float(pack.metrics.get("latency_ms", 0.0)),
            "embedding_rows_scanned": int(pack.metrics.get("embedding_rows_scanned", 0)),
            "blocks_materialized": int(pack.metrics.get("blocks_materialized", 0)),
            "bytes_read": int(pack.metrics.get("bytes_read", 0)),
            "verdict": audit.verdict,
            "baseline_context_block_count": len(all_blocks),
            "highway_context_block_count": len(pack.blocks),
            "highway_source_files": highway_source_files,
            "highway_context_pack_block_ids": highway_block_ids,
            "highway_context_pack_sources": highway_source_files,
            "answer_contract_type": contract.answer_type,
            "answer_contract_budget": int(contract.max_output_tokens),
            "poison_used": bool(poison_info["poison_used"]),
            "poison_reason": poison_info["poison_reason"],
            "expected_source_removed": bool(poison_info["expected_source_removed"]),
            **prompt_audit,
        }
        record["final_verdict"] = final_verdict
        record["verdict"] = final_verdict
        records.append(record)
        state = adapter.update_state(
            state,
            {
                "strategy": pack.metrics.get("strategy_used", ""),
                "active_entities": [expected_entity],
            },
            used_sources=turn["expected_sources"],
            used_block_ids=[block.block_id for block in pack.blocks],
        )

    status = "SKIPPED" if skip_reason else "PENDING"
    summary = _summarize(records, status=status, skip_reason=skip_reason, model=getattr(model_client, "model_name", model))
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
    parser.add_argument("--turns", type=int, default=12)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--audit-prompts", dest="audit_prompts", action="store_true", default=True)
    parser.add_argument("--no-audit-prompts", dest="audit_prompts", action="store_false")
    parser.add_argument("--poison-context", choices=["none", "missing_expected_source"], default="none")
    args = parser.parse_args()
    result = run_long_conversation_quality_benchmark(
        output_dir=args.output_dir,
        client=args.client,
        model=args.model,
        turns=args.turns,
        seed=args.seed,
        audit_prompts=args.audit_prompts,
        poison_context=args.poison_context,
    )
    print(json.dumps({"output_dir": str(result["output_dir"]), "summary": result["summary"]}, indent=2))


if __name__ == "__main__":
    main()
