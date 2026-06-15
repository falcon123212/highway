import importlib.util
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np


VECTOR_INDEX_META = "vector_index.json"
SUPPORTED_BACKENDS = {"none", "numpy_flat", "faiss_flat", "faiss_hnsw", "faiss_ivf_flat"}


def _faiss_available() -> bool:
    return importlib.util.find_spec("faiss") is not None


def _load_faiss():
    import faiss  # type: ignore

    return faiss


def _metadata_name(backend: str) -> str:
    return f"vector_index_{backend}.json"


def _normalize_rows(values: np.ndarray) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float32)
    if arr.ndim == 1:
        norm = np.linalg.norm(arr)
        return arr if norm == 0.0 else arr / norm
    norms = np.linalg.norm(arr, axis=1, keepdims=True)
    return arr / (norms + 1e-8)


@dataclass
class VectorCandidateIndex:
    index_dir: Path
    backend: str
    metadata: Dict[str, Any]

    @property
    def available(self) -> bool:
        return bool(self.metadata.get("ann_available", False))

    @property
    def fallback_reason(self) -> str:
        return str(self.metadata.get("ann_fallback_reason", ""))

    def search(self, query_embedding: np.ndarray, k: int) -> Tuple[List[Tuple[int, float]], Dict[str, Any]]:
        raise NotImplementedError


class UnavailableVectorCandidateIndex(VectorCandidateIndex):
    def search(self, query_embedding: np.ndarray, k: int) -> Tuple[List[Tuple[int, float]], Dict[str, Any]]:
        return [], {
            "ann_backend": self.backend,
            "ann_available": False,
            "ann_used": False,
            "ann_candidates": 0,
            "ann_fallback_reason": self.fallback_reason or "ann_unavailable",
        }


class NumpyFlatVectorCandidateIndex(VectorCandidateIndex):
    def __init__(self, index_dir: Path, backend: str, metadata: Dict[str, Any]):
        super().__init__(index_dir=index_dir, backend=backend, metadata=metadata)
        embeddings_file = metadata.get("embeddings_file", "embeddings.npy")
        self.embeddings = np.load(index_dir / embeddings_file, mmap_mode="r")

    def search(self, query_embedding: np.ndarray, k: int) -> Tuple[List[Tuple[int, float]], Dict[str, Any]]:
        q = _normalize_rows(np.asarray(query_embedding, dtype=np.float32))
        vectors = np.asarray(self.embeddings, dtype=np.float32)
        vectors = _normalize_rows(vectors)
        scores = np.dot(vectors, q)
        limit = min(int(k), int(scores.shape[0]))
        if limit <= 0:
            hits: List[Tuple[int, float]] = []
        else:
            indices = np.argsort(-scores)[:limit]
            hits = [(int(idx), float(scores[int(idx)])) for idx in indices]
        return hits, {
            "ann_backend": self.backend,
            "ann_available": True,
            "ann_used": True,
            "ann_candidates": len(hits),
            "ann_fallback_reason": "",
        }


class FaissVectorCandidateIndex(VectorCandidateIndex):
    def __init__(self, index_dir: Path, backend: str, metadata: Dict[str, Any]):
        super().__init__(index_dir=index_dir, backend=backend, metadata=metadata)
        self.faiss = _load_faiss()
        self.index = self.faiss.read_index(str(index_dir / metadata["ann_file"]))
        ef_search = metadata.get("ann_params", {}).get("efSearch")
        if ef_search is not None and hasattr(self.index, "hnsw"):
            self.index.hnsw.efSearch = int(ef_search)

    def search(self, query_embedding: np.ndarray, k: int) -> Tuple[List[Tuple[int, float]], Dict[str, Any]]:
        q = _normalize_rows(np.asarray(query_embedding, dtype=np.float32)).reshape(1, -1)
        scores, labels = self.index.search(q, int(k))
        hits = [
            (int(label), float(score))
            for label, score in zip(labels[0], scores[0])
            if int(label) >= 0
        ]
        return hits, {
            "ann_backend": self.backend,
            "ann_available": True,
            "ann_used": True,
            "ann_candidates": len(hits),
            "ann_fallback_reason": "",
        }


def build_vector_index(
    embeddings_path: str | Path,
    output_path: str | Path,
    backend: str,
    params: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    if backend not in SUPPORTED_BACKENDS:
        raise ValueError(f"Unsupported vector backend: {backend}")

    output_dir = Path(output_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    embeddings_file = Path(embeddings_path).name
    params = dict(params or {})
    metadata: Dict[str, Any] = {
        "vector_backend": backend,
        "ann_metric": "inner_product",
        "ann_params": params,
        "ann_file": None,
        "ann_available": backend in {"numpy_flat"},
        "ann_fallback_reason": "",
        "embeddings_file": embeddings_file,
    }

    if backend == "none":
        metadata["ann_available"] = False
        metadata["ann_fallback_reason"] = "vector_backend_none"
    elif backend == "numpy_flat":
        metadata["ann_available"] = True
    else:
        if not _faiss_available():
            metadata["ann_available"] = False
            metadata["ann_fallback_reason"] = "faiss_not_installed"
        else:
            faiss = _load_faiss()
            vectors = _normalize_rows(np.load(embeddings_path)).astype(np.float32)
            dim = int(vectors.shape[1])
            ann_file = f"{backend}.faiss"
            if backend == "faiss_flat":
                index = faiss.IndexFlatIP(dim)
            elif backend == "faiss_hnsw":
                m = int(params.get("M", 32))
                index = faiss.IndexHNSWFlat(dim, m, faiss.METRIC_INNER_PRODUCT)
                index.hnsw.efConstruction = int(params.get("efConstruction", 80))
                index.hnsw.efSearch = int(params.get("efSearch", 64))
            else:
                nlist = int(params.get("nlist", max(64, int(4 * math.sqrt(len(vectors))))))
                nprobe = int(params.get("nprobe", 8))
                quantizer = faiss.IndexFlatIP(dim)
                index = faiss.IndexIVFFlat(quantizer, dim, nlist, faiss.METRIC_INNER_PRODUCT)
                sample_size = min(len(vectors), int(params.get("train_sample_size", 50_000)))
                rng = np.random.default_rng(int(params.get("seed", 42)))
                sample_indices = rng.choice(len(vectors), size=sample_size, replace=False)
                index.train(vectors[sample_indices])
                index.nprobe = nprobe
            index.add(vectors)
            faiss.write_index(index, str(output_dir / ann_file))
            metadata["ann_file"] = ann_file
            metadata["ann_available"] = True

    (output_dir / _metadata_name(backend)).write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    (output_dir / VECTOR_INDEX_META).write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return metadata


def load_vector_index(index_dir: str | Path, backend: str | None = None) -> VectorCandidateIndex:
    index_path = Path(index_dir)
    selected_backend = backend or "none"
    backend_metadata_path = index_path / _metadata_name(selected_backend)
    metadata_path = backend_metadata_path if backend_metadata_path.exists() else index_path / VECTOR_INDEX_META
    if metadata_path.exists():
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    else:
        metadata = {
            "vector_backend": selected_backend,
            "ann_available": False,
            "ann_fallback_reason": "vector_index_metadata_missing",
            "embeddings_file": "embeddings.npy",
        }
    selected_backend = backend or metadata.get("vector_backend", "none")
    metadata["vector_backend"] = selected_backend

    if selected_backend == "numpy_flat":
        return NumpyFlatVectorCandidateIndex(index_path, selected_backend, metadata)
    if selected_backend.startswith("faiss_"):
        ann_file = metadata.get("ann_file")
        if not _faiss_available():
            metadata["ann_available"] = False
            metadata["ann_fallback_reason"] = "faiss_not_installed"
            return UnavailableVectorCandidateIndex(index_path, selected_backend, metadata)
        if not ann_file or not (index_path / ann_file).exists():
            metadata["ann_available"] = False
            metadata["ann_fallback_reason"] = "ann_file_missing"
            return UnavailableVectorCandidateIndex(index_path, selected_backend, metadata)
        return FaissVectorCandidateIndex(index_path, selected_backend, metadata)
    return UnavailableVectorCandidateIndex(index_path, selected_backend, metadata)
