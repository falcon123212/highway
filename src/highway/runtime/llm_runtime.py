from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Dict, Sequence

from highway.kernels.compute_kernels import AggregationKernel, ComparisonKernel
from highway.runtime.answer_contract import AnswerContract
from highway.runtime.context_engine import ContextBlock, ContextPack, ContextRequest
from highway.runtime.token_economics import TokenEconomics


def estimate_tokens(text: str) -> int:
    return max(1, len(str(text).split()))


@dataclass(frozen=True)
class DeterministicReflectiveClient:
    model_name: str = "deterministic_reflective_fake"
    prefill_tokens_per_ms: float = 40.0
    decode_tokens_per_ms: float = 0.25

    def answer(
        self,
        prompt: str,
        query_ir: Dict[str, Any],
        evidence: Sequence[Dict[str, Any]],
        expected_answer: str | None = None,
        query_id: str = "fake_llm",
    ) -> Dict[str, Any]:
        if expected_answer is not None:
            answer = str(expected_answer)
            status = "EXPECTED_ANSWER"
            route = "deterministic_expected"
        else:
            category = self._infer_category(query_ir)
            if category == "G":
                audit = ComparisonKernel().execute(query_ir, list(evidence), ir_builder=None, query_id=query_id)
            elif category == "H":
                audit = AggregationKernel().execute(query_ir, list(evidence), ir_builder=None, query_id=query_id)
            else:
                audit = {"status": "UNSUPPORTED", "answer": "UNSUPPORTED", "route": "unsupported"}
            answer = str(audit.get("answer", audit.get("status", "NOT_FOUND")))
            status = str(audit.get("status", "UNKNOWN"))
            route = str(audit.get("route", status))

        reasoning = f"Route {route} used {len(evidence)} evidence blocks; status is {status}."
        input_tokens = estimate_tokens(prompt)
        output_tokens = estimate_tokens(f"{reasoning} {answer}")
        ttft_ms = max(1.0, input_tokens / max(0.000001, float(self.prefill_tokens_per_ms)))
        decode_ms = max(1.0, output_tokens / max(0.000001, float(self.decode_tokens_per_ms)))
        total_ms = ttft_ms + decode_ms

        return {
            "model_name": self.model_name,
            "reasoning": reasoning,
            "answer": answer,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "ttft_ms": ttft_ms,
            "decode_ms": decode_ms,
            "total_ms": total_ms,
            "input_tokens_per_second": input_tokens / (ttft_ms / 1000.0),
            "output_tokens_per_second": output_tokens / (decode_ms / 1000.0),
        }

    @staticmethod
    def _infer_category(query_ir: Dict[str, Any]) -> str:
        operation = str(query_ir.get("operation", "")).lower()
        if "aggregation" in operation or "sum" in operation:
            return "H"
        return "G"


class HighwayLLMRuntime:
    def __init__(self, context_engine: Any):
        self.context_engine = context_engine

    def build_prompt(self, context_pack: ContextPack, answer_contract: AnswerContract | None = None) -> str:
        lines = [
            "You are a precise reasoning assistant.",
            "Use only the selected Highway context below.",
            "",
            "Selected context:",
        ]
        for block in context_pack.blocks:
            lines.append(f"[{block.block_id}] {block.source_file}: {block.text}")
        lines.extend(["", f"Question: {context_pack.request.user_turn}"])
        if answer_contract is None:
            lines.append("Return: reasoning + answer.")
        else:
            lines.extend([
                "Answer contract:",
                f"- answer_type: {answer_contract.answer_type}",
                f"- required_facts: {', '.join(answer_contract.required_facts)}",
                f"- optional_facts: {', '.join(answer_contract.optional_facts)}",
                f"- allowed_sources: {', '.join(answer_contract.allowed_sources)}",
                f"- max_output_tokens: {answer_contract.max_output_tokens}",
                f"- compact_answer_schema: {answer_contract.compact_answer_schema}",
                "Return only valid JSON with keys: reasoning, answer, sources, confidence.",
                "Keep reasoning empty unless answer_type is short_explanation.",
                "Do not cite sources outside allowed_sources.",
                "Do not invent numbers or entities not present in the selected context.",
            ])
        return "\n".join(lines)

    def answer_with_client(
        self,
        request: ContextRequest,
        llm_client: Any,
        baseline_context: Sequence[Dict[str, Any]] | None = None,
        expected_answer: str | None = None,
        top_k: int = 50,
        session_state: Any | None = None,
    ) -> Dict[str, Any]:
        pack = self.context_engine.retrieve(request, top_k=top_k, session_state=session_state)
        return self.answer_context_pack(
            pack,
            llm_client,
            baseline_context=baseline_context,
            expected_answer=expected_answer,
        )

    def answer_context_pack(
        self,
        context_pack: ContextPack,
        llm_client: Any,
        baseline_context: Sequence[Dict[str, Any]] | None = None,
        expected_answer: str | None = None,
    ) -> Dict[str, Any]:
        pack = context_pack
        prompt = self.build_prompt(pack)
        evidence = [self._block_to_evidence(block) for block in pack.blocks]
        response = llm_client.answer(
            prompt=prompt,
            query_ir=pack.query_ir,
            evidence=evidence,
            expected_answer=expected_answer,
        )
        baseline_tokens = self._estimate_baseline_tokens(baseline_context, pack)
        economics = TokenEconomics.from_measurements(
            baseline_input_tokens=baseline_tokens,
            actual_input_tokens=int(response["input_tokens"]),
            output_tokens=int(response["output_tokens"]),
            ttft_ms=float(response["ttft_ms"]),
            decode_ms=float(response["decode_ms"]),
            model_profile=getattr(self.context_engine, "model_profile", None),
            input_cost_per_million=float(getattr(self.context_engine, "input_cost_per_million", 0.0)),
            output_cost_per_million=float(getattr(self.context_engine, "output_cost_per_million", 0.0)),
        )
        metrics = dict(pack.metrics)
        metrics.update({
            "input_tokens_per_second": response["input_tokens_per_second"],
            "output_tokens_per_second": response["output_tokens_per_second"],
            "llm_ttft_ms": response["ttft_ms"],
            "llm_decode_ms": response["decode_ms"],
            "llm_total_ms": response["total_ms"],
        })
        return {
            "request": pack.request.to_dict(),
            "prompt": prompt,
            "response": response,
            "context_pack": pack.to_dict(),
            "token_economics": economics.to_dict(),
            "metrics": metrics,
            "warnings": list(pack.warnings) + list(economics.warnings),
        }

    @staticmethod
    def _block_to_evidence(block: ContextBlock) -> Dict[str, Any]:
        return {
            "block_id": block.block_id,
            "source_file": block.source_file,
            "text": block.text,
            "retrieval_score": block.score,
        }

    @staticmethod
    def _estimate_baseline_tokens(
        baseline_context: Sequence[Dict[str, Any]] | None,
        pack: ContextPack,
    ) -> int:
        if baseline_context is None:
            return int(pack.metrics.get("baseline_input_tokens_estimated", 0))
        text = "\n".join(str(block.get("text", "")) for block in baseline_context)
        return estimate_tokens(text)
