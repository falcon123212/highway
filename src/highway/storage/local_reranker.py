import time
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Tuple

from highway.storage.index_writer import tokenize_for_postings


FIELD_ALIASES = {
    "owner": ("owner", "manager", "managed", "responsible"),
    "manager": ("owner", "manager", "managed", "responsible"),
    "budget": ("budget", "amount", "cost"),
    "project": ("project", "name"),
    "project_name": ("project", "name"),
}

NOISE_TERMS = {"background", "generic", "lure", "noise", "ordinary", "unrelated"}
EVIDENCE_TERMS = {"evidence", "proof", "source", "budget", "owns", "managed"}


@dataclass(frozen=True)
class RerankedCandidate:
    block_idx: int
    score: float


class LexicalFieldReranker:
    backend = "lexical_field_reranker"

    def rerank(
        self,
        question: str,
        query_ir: Dict[str, Any],
        candidates: Iterable[Dict[str, Any]],
        limit: int,
    ) -> Tuple[List[RerankedCandidate], Dict[str, Any]]:
        start = time.perf_counter()
        ranked: List[RerankedCandidate] = []
        for candidate in candidates:
            block_idx = int(candidate["block_idx"])
            text = str(candidate.get("text", ""))
            source_file = str(candidate.get("source_file", ""))
            ranked.append(RerankedCandidate(block_idx=block_idx, score=self._score(question, query_ir, text, source_file)))

        ranked.sort(key=lambda item: item.score, reverse=True)
        output = ranked[: max(0, int(limit))]
        metrics = {
            "reranker_backend": self.backend,
            "reranker_candidates_in": len(ranked),
            "reranker_candidates_out": len(output),
            "reranker_latency_ms": (time.perf_counter() - start) * 1000.0,
        }
        return output, metrics

    def _score(self, question: str, query_ir: Dict[str, Any], text: str, source_file: str) -> float:
        lowered = f"{text} {source_file}".lower()
        tokens = tokenize_for_postings(lowered)
        token_set = set(tokens)
        score = 0.0

        for entity in query_ir.get("target_entities", []):
            entity_lower = str(entity).lower()
            entity_tokens = [
                token
                for token in tokenize_for_postings(entity_lower.replace("Project ", ""))
                if token not in {"project", "projects"}
            ]
            if entity_lower in lowered:
                score += 8.0
            matched = sum(1 for token in entity_tokens if token in token_set)
            score += matched * 3.0
            if matched and self._near_any(tokens, entity_tokens, self._field_terms(query_ir), max_distance=10):
                score += 2.0

        for field_term in self._field_terms(query_ir):
            if field_term in token_set:
                score += 1.5

        score += sum(0.8 for token in EVIDENCE_TERMS if token in token_set)
        score -= sum(3.0 for token in NOISE_TERMS if token in token_set)
        return score

    def _field_terms(self, query_ir: Dict[str, Any]) -> List[str]:
        terms: List[str] = []
        for field in query_ir.get("required_fields", []):
            terms.extend(FIELD_ALIASES.get(str(field).lower(), (str(field).lower(),)))
        return sorted(set(terms))

    def _near_any(
        self,
        tokens: List[str],
        left_terms: Iterable[str],
        right_terms: Iterable[str],
        max_distance: int,
    ) -> bool:
        left_positions = [idx for idx, token in enumerate(tokens) if token in set(left_terms)]
        right_positions = [idx for idx, token in enumerate(tokens) if token in set(right_terms)]
        return any(abs(left - right) <= max_distance for left in left_positions for right in right_positions)
