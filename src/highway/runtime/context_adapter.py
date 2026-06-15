import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Sequence

from highway.runtime.context_engine import ContextRequest


MARKER_RE = re.compile(r"\bref_[0-9a-f]{10}\b", re.IGNORECASE)
PROJECT_RE = re.compile(r"\bProject\s+([A-Z][A-Z0-9_\-]*)\b")
FOLLOW_UP_RE = re.compile(r"\b(and|also|too|its|their|that|those|same|previous|about it|what about)\b", re.IGNORECASE)


@dataclass
class SessionState:
    session_id: str
    active_entities: List[str] = field(default_factory=list)
    active_sources: List[str] = field(default_factory=list)
    pinned_block_ids: List[str] = field(default_factory=list)
    last_strategy: str | None = None
    turn_count: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "session_id": self.session_id,
            "active_entities": list(self.active_entities),
            "active_sources": list(self.active_sources),
            "pinned_block_ids": list(self.pinned_block_ids),
            "last_strategy": self.last_strategy,
            "turn_count": self.turn_count,
        }


class ContextAdapter:
    def plan(self, request: ContextRequest, state: SessionState | None = None) -> Dict[str, Any]:
        state = state or SessionState(session_id=request.session_id)
        question = request.user_turn
        detected_entities = self._detect_entities(question)
        marker = MARKER_RE.search(question)
        reasons: List[str] = []

        is_follow_up = bool(state.active_entities) and bool(FOLLOW_UP_RE.search(question))
        compiled_query = question
        query_rewrite_used = False
        if marker:
            query_type = "marker"
            strategy = "ooc_marker_entity_pruned"
            reasons.append("reference_marker_detected")
        elif detected_entities:
            query_type = "entity"
            strategy = "ooc_marker_entity_pruned"
            reasons.append("entity_detected")
        elif is_follow_up:
            query_type = "follow_up"
            strategy = "ooc_ann_pruned_hybrid"
            reasons.append("follow_up_uses_session_state")
            entity_terms = " ".join(f"Project {entity}" for entity in state.active_entities)
            compiled_query = f"{entity_terms} {question}".strip()
            query_rewrite_used = True
        else:
            query_type = "semantic"
            strategy = "ooc_ann_hnsw"
            reasons.append("semantic_query_without_strong_entity")

        if request.strategy not in {"auto", ""}:
            strategy = request.strategy
            reasons.append("explicit_strategy_override")

        active_entities = self._dedupe(detected_entities if detected_entities else state.active_entities)
        return {
            "query_type": query_type,
            "strategy": strategy,
            "reasons": reasons,
            "detected_entities": detected_entities,
            "active_entities": active_entities,
            "active_sources": list(state.active_sources),
            "pinned_block_ids": list(state.pinned_block_ids),
            "session_turn_count": state.turn_count,
            "compiled_query": compiled_query,
            "query_rewrite_used": query_rewrite_used,
        }

    def update_state(
        self,
        state: SessionState,
        plan: Dict[str, Any],
        used_sources: Sequence[str] = (),
        used_block_ids: Sequence[str] = (),
    ) -> SessionState:
        return SessionState(
            session_id=state.session_id,
            active_entities=self._dedupe(plan.get("active_entities", [])),
            active_sources=self._dedupe([*state.active_sources, *used_sources]),
            pinned_block_ids=self._dedupe([*state.pinned_block_ids, *used_block_ids]),
            last_strategy=str(plan.get("strategy") or state.last_strategy or ""),
            turn_count=state.turn_count + 1,
        )

    @staticmethod
    def _detect_entities(question: str) -> List[str]:
        return ContextAdapter._dedupe(match.group(1) for match in PROJECT_RE.finditer(question))

    @staticmethod
    def _dedupe(items: Sequence[str]) -> List[str]:
        seen = set()
        values: List[str] = []
        for item in items:
            text = str(item).strip()
            if not text or text in seen:
                continue
            seen.add(text)
            values.append(text)
        return values
