import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

from highway.retrieval.evidence_resolver import EvidenceResolver
from highway.runtime.hardware_budget import HardwareBudget
from highway.runtime.token_economics import ModelProfile, TokenEconomics
from highway.storage.out_of_core_index import OutOfCoreIndex


@dataclass(frozen=True)
class ContextRequest:
    user_turn: str
    session_id: str = "default"
    token_budget: int = 4096
    latency_budget_ms: float = 100.0
    strategy: str = "auto"

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ContextBlock:
    block_id: str
    source_file: str
    text: str
    score: float
    reason: str

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ContextPack:
    request: ContextRequest
    blocks: List[ContextBlock]
    query_ir: Dict[str, Any]
    metrics: Dict[str, Any]
    warnings: List[str]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "request": self.request.to_dict(),
            "blocks": [block.to_dict() for block in self.blocks],
            "query_ir": self.query_ir,
            "metrics": self.metrics,
            "warnings": list(self.warnings),
        }


class HighwayContextEngine:
    def __init__(
        self,
        index_dir: str | Path,
        cache_dir: Optional[str | Path] = None,
        embed_model: Optional[Any] = None,
        hardware_budget: Optional[HardwareBudget] = None,
        model_profile: Optional[ModelProfile] = None,
        input_cost_per_million: float = 0.0,
        output_cost_per_million: float = 0.0,
    ):
        self.index_dir = Path(index_dir)
        self.cache_dir = Path(cache_dir) if cache_dir is not None else None
        self.embed_model = embed_model
        self.hardware_budget = hardware_budget or HardwareBudget()
        self.model_profile = model_profile
        self.input_cost_per_million = input_cost_per_million
        self.output_cost_per_million = output_cost_per_million
        self.evidence_resolver = EvidenceResolver()

        if OutOfCoreIndex.is_out_of_core_index(self.index_dir):
            self.storage_mode = "out_of_core"
            self.out_of_core_index = OutOfCoreIndex(
                self.index_dir,
                embed_model=embed_model,
                hardware_budget=self.hardware_budget,
            )
            self.search_router = None
        else:
            from highway.retrieval.search import SearchRouter

            self.storage_mode = "legacy"
            self.out_of_core_index = None
            self.search_router = SearchRouter(
                str(self.index_dir),
                storage_mode="legacy",
                hardware_budget=self.hardware_budget,
            )

    def retrieve(self, request: ContextRequest, top_k: int = 50, session_state: Optional[Any] = None) -> ContextPack:
        start = time.perf_counter()
        plan = self._plan_request(request, session_state)
        strategy = self._resolve_strategy(str(plan["strategy"]))
        search_question = str(plan.get("compiled_query") or request.user_turn)
        warnings: List[str] = []

        if self.storage_mode == "out_of_core":
            candidates, query_ir, telemetry = self.out_of_core_index.search(
                search_question,
                top_k=top_k,
                strategy=strategy,
            )
        else:
            candidates, query_ir = self.search_router.search(request.user_turn, top_k=top_k)
            telemetry = dict(getattr(self.search_router, "last_storage_metrics", {}))

        active, suppressed, forbidden = self.evidence_resolver.resolve(candidates, query_ir)
        context_blocks = [self._context_block(block) for block in active]
        context_tokens = self._estimate_context_tokens(context_blocks)
        baseline_tokens = self._estimate_baseline_tokens()
        token_economics = TokenEconomics.from_measurements(
            baseline_input_tokens=baseline_tokens,
            actual_input_tokens=context_tokens,
            output_tokens=0,
            ttft_ms=0.0,
            decode_ms=0.0,
            model_profile=self.model_profile,
            input_cost_per_million=self.input_cost_per_million,
            output_cost_per_million=self.output_cost_per_million,
        )
        if context_tokens > request.token_budget:
            warnings.append("context_token_budget_exceeded")

        latency_ms = (time.perf_counter() - start) * 1000.0
        metrics = {
            "strategy_used": strategy,
            "storage_mode": self.storage_mode,
            "latency_ms": latency_ms,
            "context_input_tokens_estimated": context_tokens,
            "baseline_input_tokens_estimated": baseline_tokens,
            "tokens_materialized_kv": context_tokens,
            "tokens_avoided": token_economics.avoided_input_tokens,
            "token_economics": token_economics.to_dict(),
            "active_blocks": len(active),
            "suppressed_blocks": len(suppressed),
            "forbidden_matches": len(forbidden),
            "bytes_read": 0,
            "embedding_rows_scanned": 0,
            "blocks_materialized": len(candidates),
            "ann_used": False,
            "ann_backend": "none",
            "query_type": plan["query_type"],
            "strategy_reasons": list(plan["reasons"]),
            "active_entities": list(plan["active_entities"]),
            "active_sources": list(plan["active_sources"]),
            "pinned_block_ids": list(plan["pinned_block_ids"]),
            "compiled_query": search_question,
            "query_rewrite_used": bool(plan.get("query_rewrite_used", False)),
            "context_reuse_rate": 0.0,
            "active_entity_count": len(plan["active_entities"]),
            "pinned_source_count": len(plan["active_sources"]),
        }
        metrics.update(telemetry)
        metrics["strategy_used"] = strategy
        metrics["latency_ms"] = latency_ms
        metrics["context_input_tokens_estimated"] = context_tokens
        metrics["baseline_input_tokens_estimated"] = baseline_tokens
        metrics["tokens_materialized_kv"] = context_tokens
        metrics["tokens_avoided"] = token_economics.avoided_input_tokens
        metrics["token_economics"] = token_economics.to_dict()
        metrics["active_blocks"] = len(active)
        metrics["suppressed_blocks"] = len(suppressed)
        metrics["forbidden_matches"] = len(forbidden)
        metrics.setdefault("ann_backend", "none")
        metrics.setdefault("ann_used", False)
        metrics["query_type"] = plan["query_type"]
        metrics["strategy_reasons"] = list(plan["reasons"])
        metrics["active_entities"] = list(plan["active_entities"])
        metrics["active_sources"] = list(plan["active_sources"])
        metrics["pinned_block_ids"] = list(plan["pinned_block_ids"])
        metrics["compiled_query"] = search_question
        metrics["query_rewrite_used"] = bool(plan.get("query_rewrite_used", False))
        metrics["context_reuse_rate"] = self._context_reuse_rate(context_blocks, plan["active_sources"])
        metrics["active_entity_count"] = len(plan["active_entities"])
        metrics["pinned_source_count"] = len(plan["active_sources"])

        return ContextPack(
            request=request,
            blocks=context_blocks,
            query_ir=query_ir,
            metrics=metrics,
            warnings=warnings,
        )

    def _resolve_strategy(self, strategy: str) -> str:
        if self.storage_mode != "out_of_core":
            return "legacy"
        if strategy == "auto":
            return "ooc_ann_pruned_hybrid"
        return strategy

    @staticmethod
    def _plan_request(request: ContextRequest, session_state: Optional[Any]) -> Dict[str, Any]:
        from highway.runtime.context_adapter import ContextAdapter

        return ContextAdapter().plan(request, session_state)

    @staticmethod
    def _context_block(block: Dict[str, Any]) -> ContextBlock:
        return ContextBlock(
            block_id=str(block.get("block_id", "")),
            source_file=str(block.get("source_file", "")),
            text=str(block.get("text", "")),
            score=float(block.get("retrieval_score", 0.0)),
            reason="ranked_candidate",
        )

    @staticmethod
    def _estimate_context_tokens(blocks: List[ContextBlock]) -> int:
        return sum(max(1, int(len(block.text.split()))) for block in blocks)

    @staticmethod
    def _context_reuse_rate(blocks: List[ContextBlock], active_sources: List[str]) -> float:
        if not blocks:
            return 0.0
        active = set(active_sources)
        if not active:
            return 0.0
        reused = sum(1 for block in blocks if block.source_file in active)
        return reused / len(blocks) * 100.0

    def _estimate_baseline_tokens(self) -> int:
        if self.storage_mode == "out_of_core" and self.out_of_core_index is not None:
            return sum(max(0, int(meta.get("token_count", 0))) for meta in self.out_of_core_index.offsets)
        if self.search_router is not None and hasattr(self.search_router, "blocks"):
            return sum(max(0, int(block.get("token_count", 0))) for block in self.search_router.blocks)
        return 0
