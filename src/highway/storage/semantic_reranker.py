import time
from dataclasses import dataclass
from typing import Any, Callable, ClassVar, Dict, Iterable, List, Tuple

import numpy as np

from highway.storage.local_reranker import LexicalFieldReranker, RerankedCandidate


CrossEncoder = None  # type: ignore


DEFAULT_CROSS_ENCODER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"


@dataclass(frozen=True)
class SemanticRerankedCandidate:
    block_idx: int
    score: float


class SemanticReranker:
    _MODEL_CACHE: ClassVar[Dict[Tuple[str, bool], Any]] = {}
    _MODEL_FAILURES: ClassVar[Dict[Tuple[str, bool], str]] = {}

    def __init__(
        self,
        backend: str = "cross_encoder",
        model_name: str = DEFAULT_CROSS_ENCODER_MODEL,
        batch_size: int = 32,
        local_files_only: bool = False,
        cross_encoder_factory: Callable[..., Any] | None = None,
    ):
        self.backend = backend
        self.model_name = model_name
        self.batch_size = int(batch_size)
        self.local_files_only = bool(local_files_only)
        self.cross_encoder_factory = cross_encoder_factory
        self._model: Any | None = None

    def rerank(
        self,
        question: str,
        query_ir: Dict[str, Any],
        candidates: Iterable[Dict[str, Any]],
        limit: int,
    ) -> Tuple[List[SemanticRerankedCandidate], Dict[str, Any]]:
        materialized = list(candidates)
        if self.backend == "lexical_field_reranker":
            return self._lexical_fallback(question, query_ir, materialized, limit, "")
        if self.backend != "cross_encoder":
            return self._lexical_fallback(
                question,
                query_ir,
                materialized,
                limit,
                f"unsupported_reranker_backend:{self.backend}",
            )

        start = time.perf_counter()
        try:
            model = self._load_cross_encoder()
            pairs = [
                (question, f"{candidate.get('text', '')}\nSource: {candidate.get('source_file', '')}")
                for candidate in materialized
            ]
            raw_scores = model.predict(pairs, batch_size=self.batch_size) if pairs else []
            scores = np.asarray(raw_scores, dtype=np.float32).reshape(-1)
            ranked = [
                SemanticRerankedCandidate(block_idx=int(candidate["block_idx"]), score=float(scores[pos]))
                for pos, candidate in enumerate(materialized)
            ]
            ranked.sort(key=lambda item: item.score, reverse=True)
            output = ranked[: max(0, int(limit))]
            return output, {
                "reranker_backend": "cross_encoder",
                "reranker_available": True,
                "reranker_fallback_reason": "",
                "reranker_model": self.model_name,
                "reranker_candidates_in": len(materialized),
                "reranker_candidates_out": len(output),
                "reranker_latency_ms": (time.perf_counter() - start) * 1000.0,
                "reranker_batch_size": self.batch_size,
            }
        except Exception as exc:
            return self._lexical_fallback(question, query_ir, materialized, limit, str(exc))

    def _load_cross_encoder(self) -> Any:
        if self._model is not None:
            return self._model
        cache_key = (self.model_name, self.local_files_only)
        if self.cross_encoder_factory is None:
            if cache_key in self._MODEL_FAILURES:
                raise RuntimeError(self._MODEL_FAILURES[cache_key])
            if cache_key in self._MODEL_CACHE:
                self._model = self._MODEL_CACHE[cache_key]
                return self._model

        factory = self.cross_encoder_factory
        if factory is None:
            global CrossEncoder
            if CrossEncoder is None:
                try:
                    from sentence_transformers import CrossEncoder as ImportedCrossEncoder  # type: ignore
                except Exception as exc:
                    raise RuntimeError("sentence_transformers_cross_encoder_unavailable") from exc
                CrossEncoder = ImportedCrossEncoder
            factory = CrossEncoder

        try:
            self._model = factory(self.model_name, local_files_only=self.local_files_only)
        except TypeError:
            self._model = factory(self.model_name)
        except Exception as exc:
            if self.cross_encoder_factory is None:
                self._MODEL_FAILURES[cache_key] = str(exc)
            raise
        if self.cross_encoder_factory is None:
            self._MODEL_CACHE[cache_key] = self._model
        return self._model

    def _lexical_fallback(
        self,
        question: str,
        query_ir: Dict[str, Any],
        candidates: List[Dict[str, Any]],
        limit: int,
        reason: str,
    ) -> Tuple[List[SemanticRerankedCandidate], Dict[str, Any]]:
        ranked, metrics = LexicalFieldReranker().rerank(question, query_ir, candidates, limit)
        converted = [SemanticRerankedCandidate(block_idx=item.block_idx, score=item.score) for item in ranked]
        metrics.update({
            "reranker_backend": "lexical_field_reranker",
            "reranker_available": False if reason else True,
            "reranker_fallback_reason": reason,
            "reranker_model": self.model_name,
            "reranker_batch_size": self.batch_size,
        })
        return converted, metrics
