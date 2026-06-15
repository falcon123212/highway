from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List

from highway.runtime.context_engine import ContextPack, ContextRequest


PROJECT_RE = re.compile(r"\bProject\s+[A-Z][A-Z0-9_-]*\b")
MONEY_RE = re.compile(r"\$[0-9][0-9,]*(?:\.[0-9]+)?")


@dataclass(frozen=True)
class AnswerContract:
    answer_type: str
    required_facts: List[str]
    optional_facts: List[str]
    allowed_sources: List[str]
    max_output_tokens: int
    citation_required: bool = True
    numeric_facts_allowed: List[str] = field(default_factory=list)
    entity_facts_allowed: List[str] = field(default_factory=list)
    retry_instruction: str = ""
    compact_answer_schema: str = ""
    output_retry_budget: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class AnswerAudit:
    verdict: str
    parse_ok: bool
    answer_satisfies_question: bool
    full_exact_match: bool
    source_attribution_ok: bool
    numeric_facts_ok: bool
    entity_facts_ok: bool
    hallucination_flag: bool
    contradiction_flag: bool
    output_tokens: int
    output_tokens_budget: int
    output_budget_used_pct: float
    output_over_budget: bool

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class AnswerContractCompiler:
    def compile(self, request: ContextRequest, context_pack: ContextPack) -> AnswerContract:
        question = str(request.user_turn)
        lowered = _clean(question)
        allowed_sources = sorted({block.source_file for block in context_pack.blocks if block.source_file})
        numeric_facts = sorted({match.group(0) for block in context_pack.blocks for match in MONEY_RE.finditer(block.text)})
        entity_facts = sorted({match.group(0) for block in context_pack.blocks for match in PROJECT_RE.finditer(block.text)})

        if _requires_numeric_fact(lowered):
            return AnswerContract(
                answer_type="numeric_fact",
                required_facts=["budget"],
                optional_facts=[],
                allowed_sources=allowed_sources,
                max_output_tokens=64,
                citation_required=True,
                numeric_facts_allowed=numeric_facts,
                entity_facts_allowed=entity_facts,
                retry_instruction=(
                    "Retry compactly: return only the numeric fact requested and copied source file JSON. "
                    "Leave reasoning empty."
                ),
                compact_answer_schema='{"answer":"$123,000","sources":["source_file"]}',
                output_retry_budget=32,
            )
        if "which project" in lowered:
            return AnswerContract(
                answer_type="project_name",
                required_facts=["winner_project"],
                optional_facts=["winner_budget"],
                allowed_sources=allowed_sources,
                max_output_tokens=32,
                citation_required=True,
                numeric_facts_allowed=numeric_facts,
                entity_facts_allowed=entity_facts,
                retry_instruction=(
                    "Retry compactly: return only the requested project name and copied source file JSON. "
                    "Leave reasoning empty."
                ),
                compact_answer_schema='{"answer":"Project NAME","sources":["source_file"]}',
                output_retry_budget=24,
            )
        if "why" in lowered or "explain" in lowered:
            return AnswerContract(
                answer_type="short_explanation",
                required_facts=["grounded_explanation"],
                optional_facts=["supporting_numeric_fact"],
                allowed_sources=allowed_sources,
                max_output_tokens=128,
                citation_required=True,
                numeric_facts_allowed=numeric_facts,
                entity_facts_allowed=entity_facts,
                retry_instruction=(
                    "Retry compactly: return one short grounded explanation and copied source file JSON. "
                    "Do not add facts outside the selected context."
                ),
                compact_answer_schema='{"answer":"short explanation","sources":["source_file"]}',
                output_retry_budget=96,
            )
        return AnswerContract(
            answer_type="source_grounded_json",
            required_facts=["answer"],
            optional_facts=[],
            allowed_sources=allowed_sources,
            max_output_tokens=96,
            citation_required=True,
            numeric_facts_allowed=numeric_facts,
            entity_facts_allowed=entity_facts,
            retry_instruction=(
                "Retry compactly: return only the answer and copied source file JSON. "
                "Leave reasoning empty."
            ),
            compact_answer_schema='{"answer":"value","sources":["source_file"]}',
            output_retry_budget=64,
        )


class AnswerVerifier:
    def audit(
        self,
        parsed_answer: Dict[str, Any],
        contract: AnswerContract,
        context_pack: ContextPack,
        output_tokens: int = 0,
        expected_answer: str | None = None,
    ) -> AnswerAudit:
        parse_ok = bool(parsed_answer)
        answer = str(parsed_answer.get("answer", "")) if parse_ok else ""
        sources = _as_list(parsed_answer.get("sources", [])) if parse_ok else []
        allowed_sources = set(contract.allowed_sources)
        source_ok = (not contract.citation_required) or bool(set(sources) & allowed_sources)
        source_hallucination = any(source not in allowed_sources for source in sources)
        numbers = MONEY_RE.findall(answer)
        numeric_ok = all(number in contract.numeric_facts_allowed for number in numbers)
        entities = PROJECT_RE.findall(answer)
        entity_ok = all(entity in contract.entity_facts_allowed for entity in entities)
        answer_ok = self._answer_satisfies_contract(answer, contract, context_pack, expected_answer)
        full_exact = _clean(answer) == _clean(expected_answer) if expected_answer else False
        output_budget = max(1, int(contract.max_output_tokens))
        used_pct = float(output_tokens) / float(output_budget) * 100.0
        over_budget = int(output_tokens) > output_budget
        hallucination = source_hallucination or not numeric_ok or not entity_ok
        contradiction = not answer_ok

        if not parse_ok:
            verdict = "MODEL_PARSE_FAIL"
        elif hallucination:
            verdict = "HALLUCINATION_FAIL" if not numeric_ok or not entity_ok else "SOURCE_FAIL"
        elif not source_ok:
            verdict = "SOURCE_FAIL"
        elif over_budget:
            verdict = "OUTPUT_BUDGET_FAIL"
        elif not answer_ok:
            verdict = "QUALITY_FAIL"
        else:
            verdict = "PASS"

        return AnswerAudit(
            verdict=verdict,
            parse_ok=parse_ok,
            answer_satisfies_question=answer_ok,
            full_exact_match=full_exact,
            source_attribution_ok=source_ok and not source_hallucination,
            numeric_facts_ok=numeric_ok,
            entity_facts_ok=entity_ok,
            hallucination_flag=hallucination,
            contradiction_flag=contradiction,
            output_tokens=int(output_tokens),
            output_tokens_budget=output_budget,
            output_budget_used_pct=used_pct,
            output_over_budget=over_budget,
        )

    @staticmethod
    def _answer_satisfies_contract(
        answer: str,
        contract: AnswerContract,
        context_pack: ContextPack,
        expected_answer: str | None,
    ) -> bool:
        if expected_answer and _clean(answer) == _clean(expected_answer):
            return True
        if contract.answer_type == "project_name":
            answer_projects = PROJECT_RE.findall(answer)
            if not answer_projects:
                return False
            if expected_answer:
                expected_projects = PROJECT_RE.findall(expected_answer)
                return bool(expected_projects) and _clean(answer_projects[0]) == _clean(expected_projects[0])
            return answer_projects[0] in contract.entity_facts_allowed
        if contract.answer_type == "numeric_fact":
            return any(number in answer for number in contract.numeric_facts_allowed)
        if expected_answer:
            return _clean(expected_answer) in _clean(answer) or _clean(answer) in _clean(expected_answer)
        context_text = _clean(" ".join(block.text for block in context_pack.blocks))
        return bool(answer.strip()) and _clean(answer) in context_text


def _clean(value: Any) -> str:
    return " ".join(str(value or "").strip().split()).lower()


def _requires_numeric_fact(clean_question: str) -> bool:
    return any(
        term in clean_question
        for term in (
            "what budget",
            "what is the budget",
            "how much",
            "budget amount",
            "include the budget",
            "with the budget",
        )
    )


def _as_list(value: Any) -> List[str]:
    if isinstance(value, list):
        return [str(item) for item in value]
    if isinstance(value, str):
        return [value]
    return []
