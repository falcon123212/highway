import heapq
import json
import re
import sqlite3
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np

from highway.retrieval.query_parser import QueryParser
from highway.runtime.hardware_budget import HardwareBudget
from highway.runtime.residency_manager import ResidencyManager
from highway.storage.index_writer import tokenize_for_postings
from highway.storage.local_reranker import LexicalFieldReranker
from highway.storage.semantic_reranker import SemanticReranker
from highway.storage.vector_index import load_vector_index


SEMANTIC_WEAK_TERMS = {
    "a",
    "all",
    "an",
    "and",
    "by",
    "evidence",
    "for",
    "in",
    "list",
    "managed",
    "management",
    "name",
    "names",
    "of",
    "or",
    "project",
    "projects",
    "scale",
    "semantic",
    "the",
    "to",
    "up",
    "using",
    "with",
}

FIELD_STRONG_TERMS = {
    "owner": ("owner", "manager", "managed"),
    "manager": ("owner", "manager", "managed"),
    "budget": ("budget", "amount", "cost"),
    "project": ("project", "name"),
    "project_name": ("project", "name"),
}


class OutOfCoreIndex:
    def __init__(
        self,
        index_dir: str | Path,
        embed_model: Optional[Any] = None,
        hardware_budget: Optional[HardwareBudget] = None,
    ):
        self.index_dir = Path(index_dir)
        self.manifest_path = self.index_dir / "manifest.json"
        if not self.manifest_path.exists():
            raise FileNotFoundError(f"Out-of-core manifest not found at {self.manifest_path}")

        self.manifest = json.loads(self.manifest_path.read_text(encoding="utf-8"))
        if self.manifest.get("layout") != "highway_out_of_core_v1":
            raise ValueError(f"Unsupported out-of-core layout: {self.manifest.get('layout')}")

        self.hardware_budget = hardware_budget or HardwareBudget()
        self.blocks_path = self.index_dir / self.manifest["blocks_file"]
        self.offsets_path = self.index_dir / self.manifest["offsets_file"]
        self.postings_path = self.index_dir / self.manifest["postings_file"]
        self.entities_path = self.index_dir / self.manifest["entity_file"]

        # NumPy keeps mmap-backed arrays on disk while allowing ndarray-like slicing.
        self.embeddings = np.load(self.index_dir / self.manifest["embeddings_file"], mmap_mode="r")
        self.offsets = json.loads(self.offsets_path.read_text(encoding="utf-8"))
        self.entities = json.loads(self.entities_path.read_text(encoding="utf-8"))
        self.query_parser = QueryParser(self.entities)
        self.embed_model = embed_model or self._load_default_embedder()
        self._index_bytes = self._compute_index_bytes()
        self.vector_backend = self.manifest.get("vector_backend", "none")
        self._vector_indices: Dict[str, Any] = {}

    @staticmethod
    def is_out_of_core_index(index_dir: str | Path) -> bool:
        manifest_path = Path(index_dir) / "manifest.json"
        if not manifest_path.exists():
            return False
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except Exception:
            return False
        return manifest.get("layout") == "highway_out_of_core_v1"

    def _load_default_embedder(self):
        from sentence_transformers import SentenceTransformer

        return SentenceTransformer("all-MiniLM-L6-v2")

    def _compute_index_bytes(self) -> int:
        total = 0
        for key in ("blocks_file", "offsets_file", "embeddings_file", "postings_file", "entity_file"):
            path = self.index_dir / self.manifest[key]
            if path.exists():
                total += path.stat().st_size
        return total

    def _query_terms(self, question: str, query_ir: Dict[str, Any]) -> List[str]:
        terms = tokenize_for_postings(question)
        marker = query_ir.get("constraints", {}).get("reference_marker")
        if marker:
            terms.append(marker.lower())
        for entity in query_ir.get("target_entities", []):
            terms.extend(tokenize_for_postings(entity))
            terms.extend(tokenize_for_postings(entity.replace("Project ", "")))
        return sorted(set(t for t in terms if t))

    def _semantic_strong_terms(self, question: str, query_ir: Dict[str, Any]) -> List[str]:
        terms: List[str] = []
        for entity in query_ir.get("target_entities", []):
            terms.extend(tokenize_for_postings(entity))
            terms.extend(tokenize_for_postings(entity.replace("Project ", "")))
        for field in query_ir.get("required_fields", []):
            lowered = str(field).lower()
            terms.extend(FIELD_STRONG_TERMS.get(lowered, (lowered,)))

        marker = query_ir.get("constraints", {}).get("reference_marker")
        if marker:
            terms.append(str(marker).lower())

        # Keep rare-looking query terms as a backstop, but drop instruction words
        # that create huge posting pools on synthetic semantic workloads.
        for term in tokenize_for_postings(question):
            if len(term) >= 5 and term not in SEMANTIC_WEAK_TERMS:
                terms.append(term)

        return sorted(set(term for term in terms if term and term not in SEMANTIC_WEAK_TERMS))

    def _semantic_strong_posting_scores(
        self,
        question: str,
        query_ir: Dict[str, Any],
        limit: Optional[int] = None,
    ) -> Dict[int, float]:
        entity_terms: List[str] = []
        for entity in query_ir.get("target_entities", []):
            entity_terms.extend(tokenize_for_postings(entity))
            entity_terms.extend(tokenize_for_postings(entity.replace("Project ", "")))
        primary_terms = sorted(
            set(term for term in entity_terms if term and term not in SEMANTIC_WEAK_TERMS)
        )
        if primary_terms:
            primary_scores = self._lookup_term_scores(primary_terms, limit=limit)
            if primary_scores:
                return primary_scores
        return self._lookup_term_scores(self._semantic_strong_terms(question, query_ir), limit=limit)

    def _semantic_field_posting_scores(
        self,
        question: str,
        query_ir: Dict[str, Any],
        limit: Optional[int] = None,
    ) -> Dict[int, float]:
        terms: List[str] = []
        for field in query_ir.get("required_fields", []):
            field_lower = str(field).lower()
            terms.extend(FIELD_STRONG_TERMS.get(field_lower, (field_lower,)))

        intent = str(query_ir.get("intent", "")).lower()
        if intent == "comparison":
            terms.extend(("budget", "amount", "cost", "higher", "lower"))
        elif intent == "aggregation":
            terms.extend(("managed", "manager", "owner", "responsible"))

        for term in tokenize_for_postings(question):
            if term in {"budget", "managed", "manager", "owner", "amount", "cost", "evidence"}:
                terms.append(term)

        filtered = sorted(set(term for term in terms if term and term not in {"project", "projects", "name", "names"}))
        return self._lookup_term_scores(filtered, limit=limit)

    def _lookup_term_scores(self, terms: Iterable[str], limit: Optional[int] = None) -> Dict[int, float]:
        unique_terms = sorted(set(terms))
        if not unique_terms:
            return {}
        placeholders = ",".join("?" for _ in unique_terms)
        query = (
            f"SELECT block_idx, SUM(tf) AS score FROM term_postings "
            f"WHERE term IN ({placeholders}) GROUP BY block_idx ORDER BY score DESC"
        )
        params: List[Any] = list(unique_terms)
        if limit is not None:
            query += " LIMIT ?"
            params.append(int(limit))

        with sqlite3.connect(self.postings_path) as conn:
            rows = conn.execute(query, params).fetchall()
        return {int(block_idx): float(score) for block_idx, score in rows}

    def _iter_dense_top(
        self,
        question: str,
        candidate_indices: Optional[Iterable[int]] = None,
    ) -> Tuple[Dict[int, float], Dict[str, int]]:
        q_emb = np.asarray(self.embed_model.encode(question, convert_to_numpy=True), dtype=np.float32)
        q_norm = float(np.linalg.norm(q_emb))
        if q_norm == 0.0:
            return {}, {"embedding_rows_scanned": 0, "embedding_windows": 0}

        rows = int(self.embeddings.shape[0])
        selected_indices = None
        if candidate_indices is not None:
            selected_indices = sorted(set(int(idx) for idx in candidate_indices if 0 <= int(idx) < rows))
            if not selected_indices:
                return {}, {"embedding_rows_scanned": 0, "embedding_windows": 0}

        row_width = int(self.embeddings.shape[1]) if len(self.embeddings.shape) > 1 else 1
        row_bytes = max(1, int(self.embeddings.dtype.itemsize) * row_width)
        rows_per_window = max(1, self.hardware_budget.max_index_window_bytes // row_bytes)
        rows_per_window = min(rows, rows_per_window)
        heap: List[Tuple[float, int]] = []
        rows_scanned = 0
        windows = 0

        if selected_indices is None:
            index_windows = [
                list(range(start, min(rows, start + rows_per_window)))
                for start in range(0, rows, rows_per_window)
            ]
        else:
            index_windows = [
                selected_indices[start:start + rows_per_window]
                for start in range(0, len(selected_indices), rows_per_window)
            ]

        for window_indices in index_windows:
            chunk = np.asarray(self.embeddings[window_indices], dtype=np.float32)
            chunk_norms = np.linalg.norm(chunk, axis=1)
            sims = np.dot(chunk, q_emb) / (q_norm * chunk_norms + 1e-8)
            rows_scanned += len(window_indices)
            windows += 1

            for offset, score in enumerate(sims):
                block_idx = window_indices[offset]
                item = (float(score), int(block_idx))
                if len(heap) < self.hardware_budget.max_candidates:
                    heapq.heappush(heap, item)
                elif item[0] > heap[0][0]:
                    heapq.heapreplace(heap, item)

        return {idx: score for score, idx in heap}, {
            "embedding_rows_scanned": rows_scanned,
            "embedding_windows": windows,
        }

    def _fetch_block(self, block_idx: int, residency: ResidencyManager) -> Dict[str, Any]:
        meta = self.offsets[block_idx]
        residency.admit(meta["block_id"], int(meta["byte_length"]))
        with self.blocks_path.open("rb") as f:
            f.seek(int(meta["offset"]))
            data = f.read(int(meta["byte_length"]))
        return json.loads(data.decode("utf-8"))

    def _pruned_candidate_indices(self, question: str, query_ir: Dict[str, Any], top_k: int) -> Dict[int, float]:
        marker = query_ir.get("constraints", {}).get("reference_marker")
        if marker:
            marker_scores = self._lookup_term_scores([marker.lower()], limit=None)
            if marker_scores:
                return marker_scores
        entity_terms = []
        for entity in query_ir.get("target_entities", []):
            entity_terms.extend(tokenize_for_postings(entity))
            entity_terms.extend(tokenize_for_postings(entity.replace("Project ", "")))
        entity_terms = sorted(set(term for term in entity_terms if term and term not in {"project", "projects"}))
        if entity_terms:
            entity_scores = self._lookup_term_scores(
                entity_terms,
                limit=max(self.hardware_budget.max_candidates, top_k),
            )
            if entity_scores:
                return entity_scores
        return self._lookup_term_scores(
            self._query_terms(question, query_ir),
            limit=max(self.hardware_budget.max_candidates, top_k),
        )

    def _ann_backend_for_strategy(self, strategy: str) -> str:
        if self.vector_backend == "numpy_flat" and strategy in {
            "ooc_ann_flat",
            "ooc_ann_hnsw",
            "ooc_ann_ivf_flat",
            "ooc_ann_pruned_hybrid",
            "ooc_semantic_rescue_hybrid",
            "ooc_semantic_lexical_rescue",
            "ooc_semantic_rerank_rescue",
            "ooc_semantic_field_rescue",
            "ooc_semantic_cross_encoder_rescue",
        }:
            return "numpy_flat"
        if strategy == "ooc_ann_flat":
            return "faiss_flat"
        if strategy == "ooc_ann_hnsw":
            return "faiss_hnsw"
        if strategy == "ooc_ann_ivf_flat":
            return "faiss_ivf_flat"
        if strategy in {
            "ooc_semantic_rescue_hybrid",
            "ooc_semantic_lexical_rescue",
            "ooc_semantic_rerank_rescue",
            "ooc_semantic_field_rescue",
            "ooc_semantic_cross_encoder_rescue",
        }:
            if str(self.vector_backend).startswith("faiss_"):
                return str(self.vector_backend)
            return "faiss_hnsw"
        if strategy == "ooc_ann_pruned_hybrid":
            if str(self.vector_backend).startswith("faiss_"):
                return str(self.vector_backend)
            return "faiss_hnsw"
        return "none"

    def _load_vector_index(self, backend: str):
        if backend not in self._vector_indices:
            self._vector_indices[backend] = load_vector_index(self.index_dir, backend=backend)
        return self._vector_indices[backend]

    def _ann_candidates(self, question: str, backend: str, k: int) -> Tuple[Dict[int, float], Dict[str, Any]]:
        q_emb = np.asarray(self.embed_model.encode(question, convert_to_numpy=True), dtype=np.float32)
        vector_index = self._load_vector_index(backend)
        hits, ann_metrics = vector_index.search(q_emb, k=k)
        return {idx: score for idx, score in hits}, ann_metrics

    def _is_pruned_sufficient(self, query_ir: Dict[str, Any], pruned_scores: Dict[int, float], top_k: int) -> bool:
        marker = query_ir.get("constraints", {}).get("reference_marker")
        if marker and pruned_scores:
            return True
        return len(pruned_scores) >= min(top_k, self.hardware_budget.max_candidates)

    def search(
        self,
        question: str,
        top_k: int = 50,
        strategy: str = "ooc_full_scan",
    ) -> Tuple[List[Dict[str, Any]], Dict[str, Any], Dict[str, Any]]:
        supported = {
            "ooc_full_scan",
            "ooc_marker_entity_pruned",
            "ooc_ann_flat",
            "ooc_ann_hnsw",
            "ooc_ann_ivf_flat",
            "ooc_ann_pruned_hybrid",
            "ooc_semantic_rescue_hybrid",
            "ooc_semantic_lexical_rescue",
            "ooc_semantic_rerank_rescue",
            "ooc_semantic_field_rescue",
            "ooc_semantic_cross_encoder_rescue",
        }
        if strategy not in supported:
            raise ValueError(f"Unsupported out-of-core search strategy: {strategy}")

        query_ir = self.query_parser.parse(question)
        pruned_scores = {}
        ann_scores: Dict[int, float] = {}
        ann_metrics: Dict[str, Any] = {
            "ann_backend": self._ann_backend_for_strategy(strategy),
            "ann_available": False,
            "ann_used": False,
            "ann_candidates": 0,
            "ann_recall_at_k": None,
            "ann_fallback_reason": "",
            "semantic_rescue_used": False,
            "ann_k": 0,
            "rerank_k": 0,
            "lexical_k": 0,
            "candidate_sources": [],
            "fallback_used": False,
            "semantic_lexical_rescue_used": False,
            "semantic_rerank_rescue_used": False,
            "semantic_field_rescue_used": False,
            "semantic_cross_encoder_rescue_used": False,
            "strong_posting_candidates": 0,
            "field_posting_candidates": 0,
            "reranker_backend": "none",
            "reranker_available": False,
            "reranker_fallback_reason": "",
            "reranker_model": "",
            "reranker_candidates_in": 0,
            "reranker_candidates_out": 0,
            "reranker_latency_ms": 0.0,
            "reranker_batch_size": 0,
            "reranker_input_k": 0,
            "reranker_output_k": 0,
        }
        candidate_indices = None
        reranker_scores: Dict[int, float] = {}
        if strategy == "ooc_marker_entity_pruned":
            pruned_scores = self._pruned_candidate_indices(question, query_ir, top_k)
            candidate_indices = pruned_scores.keys()
        elif strategy in {"ooc_ann_flat", "ooc_ann_hnsw", "ooc_ann_ivf_flat"}:
            ann_backend = self._ann_backend_for_strategy(strategy)
            ann_scores, ann_metrics = self._ann_candidates(
                question,
                backend=ann_backend,
                k=max(self.hardware_budget.max_candidates, top_k),
            )
            if ann_scores:
                candidate_indices = ann_scores.keys()
            else:
                ann_metrics["ann_used"] = False
                candidate_indices = None
        elif strategy == "ooc_ann_pruned_hybrid":
            pruned_scores = self._pruned_candidate_indices(question, query_ir, top_k)
            if self._is_pruned_sufficient(query_ir, pruned_scores, top_k):
                candidate_indices = pruned_scores.keys()
                ann_metrics["ann_fallback_reason"] = "pruned_candidates_sufficient"
            else:
                ann_backend = self._ann_backend_for_strategy(strategy)
                ann_scores, ann_metrics = self._ann_candidates(
                    question,
                    backend=ann_backend,
                    k=max(self.hardware_budget.max_candidates, top_k),
                )
                candidate_indices = ann_scores.keys() if ann_scores else None
        elif strategy == "ooc_semantic_rescue_hybrid":
            ann_metrics["semantic_rescue_used"] = True
            ann_backend = self._ann_backend_for_strategy(strategy)
            ann_k = max(int(self.hardware_budget.semantic_ann_k), top_k)
            rerank_k = max(int(self.hardware_budget.semantic_rerank_k), top_k)
            ann_scores, ann_metrics = self._ann_candidates(question, backend=ann_backend, k=ann_k)
            pruned_scores = self._lookup_term_scores(
                self._query_terms(question, query_ir),
                limit=rerank_k,
            )
            candidate_sources = []
            merged_indices = set()
            if ann_scores:
                candidate_sources.append("ann")
                merged_indices.update(ann_scores.keys())
            if pruned_scores:
                candidate_sources.append("postings")
                merged_indices.update(pruned_scores.keys())
            if (
                self.hardware_budget.semantic_rescue_enabled
                and len(self.offsets) <= int(self.hardware_budget.semantic_full_scan_fallback_max_blocks)
            ):
                candidate_indices = None
                candidate_sources.append("full_scan_fallback")
                ann_metrics["fallback_used"] = True
            else:
                candidate_indices = sorted(merged_indices)[:rerank_k]
            ann_metrics["semantic_rescue_used"] = True
            ann_metrics["ann_k"] = ann_k
            ann_metrics["rerank_k"] = rerank_k
            ann_metrics["candidate_sources"] = candidate_sources
            ann_metrics.setdefault("fallback_used", False)
        elif strategy == "ooc_semantic_lexical_rescue":
            ann_backend = self._ann_backend_for_strategy(strategy)
            ann_k = max(int(self.hardware_budget.semantic_ann_k), top_k)
            lexical_k = max(int(self.hardware_budget.semantic_lexical_k), top_k)
            rerank_k = max(int(self.hardware_budget.semantic_rerank_k), top_k)
            ann_scores, ann_metrics = self._ann_candidates(question, backend=ann_backend, k=ann_k)
            pruned_scores = self._semantic_strong_posting_scores(question, query_ir, limit=lexical_k)

            candidate_sources = []
            combined: Dict[int, float] = {}
            if ann_scores:
                candidate_sources.append("ann")
                for idx, score in ann_scores.items():
                    combined[idx] = max(combined.get(idx, float("-inf")), float(score))
            if pruned_scores:
                candidate_sources.append("strong_postings")
                for idx, score in pruned_scores.items():
                    combined[idx] = combined.get(idx, 0.0) + min(float(score), 20.0) * 0.05

            ranked_candidates = sorted(combined.items(), key=lambda item: item[1], reverse=True)
            candidate_indices = [idx for idx, _ in ranked_candidates[:rerank_k]] if ranked_candidates else None
            ann_metrics["semantic_rescue_used"] = True
            ann_metrics["semantic_lexical_rescue_used"] = True
            ann_metrics["ann_k"] = ann_k
            ann_metrics["lexical_k"] = lexical_k
            ann_metrics["rerank_k"] = rerank_k
            ann_metrics["candidate_sources"] = candidate_sources
            ann_metrics["fallback_used"] = False
            ann_metrics["strong_posting_candidates"] = len(pruned_scores)
        elif strategy == "ooc_semantic_field_rescue":
            ann_backend = self._ann_backend_for_strategy(strategy)
            ann_k = max(int(self.hardware_budget.semantic_ann_k), top_k)
            lexical_k = max(int(self.hardware_budget.semantic_lexical_k), top_k)
            rerank_k = max(int(self.hardware_budget.semantic_rerank_k), top_k)
            ann_scores, ann_metrics = self._ann_candidates(question, backend=ann_backend, k=ann_k)
            pruned_scores = self._semantic_strong_posting_scores(question, query_ir, limit=lexical_k)
            field_scores = self._semantic_field_posting_scores(question, query_ir, limit=lexical_k)

            candidate_sources = []
            combined: Dict[int, float] = {}
            if ann_scores:
                candidate_sources.append("ann")
                for idx, score in ann_scores.items():
                    combined[idx] = max(combined.get(idx, float("-inf")), float(score))
            if pruned_scores:
                candidate_sources.append("strong_postings")
                for idx, score in pruned_scores.items():
                    combined[idx] = combined.get(idx, 0.0) + min(float(score), 20.0) * 0.05
            if field_scores:
                candidate_sources.append("field_postings")
                for idx, score in field_scores.items():
                    combined[idx] = combined.get(idx, 0.0) + min(float(score), 20.0) * 0.03

            ranked_candidates = sorted(combined.items(), key=lambda item: item[1], reverse=True)
            candidate_indices = [idx for idx, _ in ranked_candidates[:rerank_k]] if ranked_candidates else None
            ann_metrics["semantic_rescue_used"] = True
            ann_metrics["semantic_lexical_rescue_used"] = True
            ann_metrics["semantic_field_rescue_used"] = True
            ann_metrics["ann_k"] = ann_k
            ann_metrics["lexical_k"] = lexical_k
            ann_metrics["rerank_k"] = rerank_k
            ann_metrics["candidate_sources"] = candidate_sources
            ann_metrics["fallback_used"] = False
            ann_metrics["strong_posting_candidates"] = len(pruned_scores)
            ann_metrics["field_posting_candidates"] = len(field_scores)
            for idx, score in field_scores.items():
                pruned_scores[idx] = max(pruned_scores.get(idx, 0.0), float(score))
        elif strategy == "ooc_semantic_cross_encoder_rescue":
            ann_backend = self._ann_backend_for_strategy(strategy)
            ann_k = max(int(self.hardware_budget.semantic_ann_k), top_k)
            lexical_k = max(int(self.hardware_budget.semantic_lexical_k), top_k)
            input_k = max(int(self.hardware_budget.semantic_reranker_input_k), top_k)
            output_k = max(int(self.hardware_budget.semantic_reranker_output_k), top_k)
            ann_scores, ann_metrics = self._ann_candidates(question, backend=ann_backend, k=ann_k)
            pruned_scores = self._semantic_strong_posting_scores(question, query_ir, limit=lexical_k)
            field_scores = self._semantic_field_posting_scores(question, query_ir, limit=lexical_k)

            candidate_sources = []
            combined: Dict[int, float] = {}
            if ann_scores:
                candidate_sources.append("ann")
                for idx, score in ann_scores.items():
                    combined[idx] = max(combined.get(idx, float("-inf")), float(score))
            if pruned_scores:
                candidate_sources.append("strong_postings")
                for idx, score in pruned_scores.items():
                    combined[idx] = combined.get(idx, 0.0) + min(float(score), 20.0) * 0.05
            if field_scores:
                candidate_sources.append("field_postings")
                for idx, score in field_scores.items():
                    combined[idx] = combined.get(idx, 0.0) + min(float(score), 20.0) * 0.03

            ranked_candidates = sorted(combined.items(), key=lambda item: item[1], reverse=True)
            pre_rerank_indices = [idx for idx, _ in ranked_candidates[:input_k]]
            rerank_candidates = []
            for idx in pre_rerank_indices:
                block = self._read_block_without_residency(idx)
                rerank_candidates.append({
                    "block_idx": idx,
                    "text": block.get("text", ""),
                    "source_file": block.get("source_file", ""),
                })
            reranked, reranker_metrics = SemanticReranker(
                backend=self.hardware_budget.semantic_reranker_backend,
                model_name=self.hardware_budget.semantic_reranker_model,
                batch_size=self.hardware_budget.semantic_reranker_batch_size,
                local_files_only=self.hardware_budget.semantic_reranker_local_files_only,
            ).rerank(
                question=question,
                query_ir=query_ir,
                candidates=rerank_candidates,
                limit=output_k,
            )
            reranker_scores = {item.block_idx: item.score for item in reranked}
            candidate_indices = [item.block_idx for item in reranked] if reranked else None
            candidate_sources.append(
                "cross_encoder"
                if reranker_metrics.get("reranker_backend") == "cross_encoder"
                else "lexical_field_reranker"
            )
            ann_metrics["semantic_rescue_used"] = True
            ann_metrics["semantic_lexical_rescue_used"] = True
            ann_metrics["semantic_field_rescue_used"] = True
            ann_metrics["semantic_cross_encoder_rescue_used"] = True
            ann_metrics["ann_k"] = ann_k
            ann_metrics["lexical_k"] = lexical_k
            ann_metrics["rerank_k"] = output_k
            ann_metrics["reranker_input_k"] = input_k
            ann_metrics["reranker_output_k"] = output_k
            ann_metrics["candidate_sources"] = candidate_sources
            ann_metrics["fallback_used"] = reranker_metrics.get("reranker_backend") != "cross_encoder"
            ann_metrics["strong_posting_candidates"] = len(pruned_scores)
            ann_metrics["field_posting_candidates"] = len(field_scores)
            ann_metrics.update(reranker_metrics)
            for idx, score in field_scores.items():
                pruned_scores[idx] = max(pruned_scores.get(idx, 0.0), float(score))
        elif strategy == "ooc_semantic_rerank_rescue":
            ann_backend = self._ann_backend_for_strategy(strategy)
            ann_k = max(int(self.hardware_budget.semantic_ann_k), top_k)
            lexical_k = max(int(self.hardware_budget.semantic_lexical_k), top_k)
            rerank_k = max(int(self.hardware_budget.semantic_rerank_k), top_k)
            ann_scores, ann_metrics = self._ann_candidates(question, backend=ann_backend, k=ann_k)
            pruned_scores = self._semantic_strong_posting_scores(question, query_ir, limit=lexical_k)

            candidate_sources = []
            combined: Dict[int, float] = {}
            if ann_scores:
                candidate_sources.append("ann")
                for idx, score in ann_scores.items():
                    combined[idx] = max(combined.get(idx, float("-inf")), float(score))
            if pruned_scores:
                candidate_sources.append("strong_postings")
                for idx, score in pruned_scores.items():
                    combined[idx] = combined.get(idx, 0.0) + min(float(score), 20.0) * 0.05

            ranked_candidates = sorted(combined.items(), key=lambda item: item[1], reverse=True)
            pre_rerank_indices = [idx for idx, _ in ranked_candidates[: max(rerank_k, top_k)]]
            reranker_start = time.perf_counter()
            rerank_candidates = []
            for idx in pre_rerank_indices:
                block = self._read_block_without_residency(idx)
                rerank_candidates.append({
                    "block_idx": idx,
                    "text": block.get("text", ""),
                    "source_file": block.get("source_file", ""),
                })
            reranked, reranker_metrics = LexicalFieldReranker().rerank(
                question=question,
                query_ir=query_ir,
                candidates=rerank_candidates,
                limit=rerank_k,
            )
            reranker_scores = {item.block_idx: item.score for item in reranked}
            candidate_indices = [item.block_idx for item in reranked] if reranked else None
            candidate_sources.append("reranker")
            ann_metrics["semantic_rescue_used"] = True
            ann_metrics["semantic_lexical_rescue_used"] = True
            ann_metrics["semantic_rerank_rescue_used"] = True
            ann_metrics["ann_k"] = ann_k
            ann_metrics["lexical_k"] = lexical_k
            ann_metrics["rerank_k"] = rerank_k
            ann_metrics["candidate_sources"] = candidate_sources
            ann_metrics["fallback_used"] = False
            ann_metrics["strong_posting_candidates"] = len(pruned_scores)
            ann_metrics.update(reranker_metrics)
            ann_metrics["reranker_latency_ms"] = max(
                float(ann_metrics.get("reranker_latency_ms", 0.0)),
                (time.perf_counter() - reranker_start) * 1000.0,
            )

        dense_scores, dense_metrics = self._iter_dense_top(question, candidate_indices=candidate_indices)
        dense_metrics["rerank_rows_scanned"] = dense_metrics.get("embedding_rows_scanned", 0)

        semantic_pruned_strategies = {
            "ooc_semantic_lexical_rescue",
            "ooc_semantic_rerank_rescue",
            "ooc_semantic_field_rescue",
            "ooc_semantic_cross_encoder_rescue",
        }
        if strategy in {"ooc_marker_entity_pruned", "ooc_ann_pruned_hybrid"} and pruned_scores:
            lexical_scores = dict(pruned_scores)
        elif strategy in semantic_pruned_strategies and pruned_scores:
            lexical_scores = dict(pruned_scores)
        elif strategy in {"ooc_ann_flat", "ooc_ann_hnsw", "ooc_ann_ivf_flat"} and ann_scores:
            lexical_scores = {}
        else:
            lexical_scores = self._lookup_term_scores(
                self._query_terms(question, query_ir),
                limit=max(self.hardware_budget.max_candidates, top_k),
            )

        marker = query_ir.get("constraints", {}).get("reference_marker")
        marker_scores = self._lookup_term_scores([marker.lower()], limit=None) if marker else {}

        candidate_scores: Dict[int, float] = {}
        for idx, score in dense_scores.items():
            candidate_scores[idx] = max(candidate_scores.get(idx, float("-inf")), score)
        for idx, score in ann_scores.items():
            if idx in dense_scores:
                candidate_scores[idx] = max(candidate_scores.get(idx, float("-inf")), dense_scores[idx])
            else:
                candidate_scores[idx] = max(candidate_scores.get(idx, float("-inf")), score)
        for idx, score in lexical_scores.items():
            candidate_scores[idx] = candidate_scores.get(idx, 0.0) + min(score, 20.0) * 0.05
        for idx, score in pruned_scores.items():
            candidate_scores[idx] = candidate_scores.get(idx, 0.0) + min(score, 20.0) * 0.05
        for idx, score in reranker_scores.items():
            candidate_scores[idx] = candidate_scores.get(idx, 0.0) + float(score) * 0.02
        for idx in marker_scores:
            candidate_scores[idx] = candidate_scores.get(idx, 0.0) + 10.0

        ranked = sorted(candidate_scores.items(), key=lambda item: item[1], reverse=True)[:top_k]
        residency = ResidencyManager(max_resident_bytes=self.hardware_budget.max_ram_bytes)
        results = []
        for rank, (block_idx, score) in enumerate(ranked, start=1):
            block = self._fetch_block(block_idx, residency)
            block["retrieval_score"] = float(score)
            block["retrieval_rank"] = rank
            block["bm25_score"] = float(lexical_scores.get(block_idx, 0.0))
            block["cosine_similarity"] = float(dense_scores.get(block_idx, 0.0))
            results.append(block)

        telemetry = residency.snapshot_metrics()
        telemetry.update(dense_metrics)
        telemetry.update(ann_metrics)
        telemetry.update({
            "storage_mode": "out_of_core",
            "index_bytes": self._index_bytes,
            "candidate_count": len(candidate_scores),
            "pruned_candidate_count": len(pruned_scores),
            "result_count": len(results),
            "max_candidates": self.hardware_budget.max_candidates,
            "max_index_window_mb": self.hardware_budget.max_index_window_mb,
            "search_strategy": strategy,
        })
        return results, query_ir, telemetry

    def _read_block_without_residency(self, block_idx: int) -> Dict[str, Any]:
        meta = self.offsets[block_idx]
        with self.blocks_path.open("rb") as f:
            f.seek(int(meta["offset"]))
            data = f.read(int(meta["byte_length"]))
        return json.loads(data.decode("utf-8"))
