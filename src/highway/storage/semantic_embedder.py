import hashlib
import re
import time
from typing import Any, Dict

import numpy as np

from highway.kernels.compute_kernels import PEOPLE

try:
    from sentence_transformers import SentenceTransformer  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    SentenceTransformer = None  # type: ignore


DEFAULT_EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"
DEFAULT_FALLBACK_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
MARKER_RE = re.compile(r"\bref_[0-9a-f]{10}\b", re.IGNORECASE)


def _stable_vector(key: str, dim: int = 384) -> np.ndarray:
    values = []
    counter = 0
    while len(values) < dim:
        digest = hashlib.sha256(f"{key}:{counter}".encode("utf-8")).digest()
        values.extend((byte / 127.5) - 1.0 for byte in digest)
        counter += 1
    arr = np.asarray(values[:dim], dtype=np.float32)
    norm = np.linalg.norm(arr)
    if norm == 0.0:
        return arr
    return arr / norm


def _embedding_key(text: str) -> str:
    lowered = text.lower()
    if "managed by" in lowered:
        for person in PEOPLE:
            if person.lower() in lowered:
                return "manager:" + person.lower()
    projects = sorted(set(re.findall(r"\bProject\s+([A-Z][A-Z0-9_\-]+)\b", text)))
    if len(projects) >= 2:
        return "projects:" + ",".join(projects)
    for person in PEOPLE:
        if person.lower() in lowered:
            return "manager:" + person.lower()
    marker = MARKER_RE.search(text)
    if marker:
        return marker.group(0).lower()
    return hashlib.sha256(f"noise|{text}".encode("utf-8")).hexdigest()[:16]


class SyntheticSemanticEmbedder:
    def __init__(self, dim: int = 384):
        self.dim = dim
        self.last_latency_ms = 0.0

    def encode(self, text, convert_to_numpy=True, show_progress_bar=False, batch_size: int = 64, **_: Any):
        start = time.perf_counter()
        if isinstance(text, list):
            arr = np.asarray([self.encode(item, convert_to_numpy=True) for item in text], dtype=np.float32)
        else:
            arr = _stable_vector(_embedding_key(str(text)), dim=self.dim)
        self.last_latency_ms = (time.perf_counter() - start) * 1000.0
        return arr if convert_to_numpy else arr.tolist()

    def embedding_metadata(self) -> Dict[str, Any]:
        return {
            "embedding_backend": "synthetic",
            "embedding_model": "synthetic_scaleup",
            "embedding_dim": int(self.dim),
            "embedding_local_files_only": False,
            "embedding_batch_size": 0,
            "embedding_latency_ms": float(self.last_latency_ms),
            "embedding_fallback_reason": "",
        }


class SentenceTransformerEmbedder:
    def __init__(
        self,
        model_name: str = DEFAULT_EMBEDDING_MODEL,
        fallback_model_name: str = DEFAULT_FALLBACK_MODEL,
        local_files_only: bool = False,
        batch_size: int = 64,
        sentence_transformer_factory=None,
    ):
        self.requested_model_name = model_name
        self.model_name = model_name
        self.fallback_model_name = fallback_model_name
        self.local_files_only = bool(local_files_only)
        self.batch_size = int(batch_size)
        self.last_latency_ms = 0.0
        self.fallback_reason = ""
        self._model = None
        self._factory = sentence_transformer_factory
        self._embedding_dim = 0

    def _load_model(self):
        if self._model is not None:
            return self._model
        factory = self._factory or SentenceTransformer
        if factory is None:
            self.fallback_reason = "sentence_transformers_not_installed"
            self._model = SyntheticSemanticEmbedder()
            self.model_name = "synthetic_scaleup"
            return self._model
        try:
            self._model = factory(self.requested_model_name, local_files_only=self.local_files_only)
            self.model_name = self.requested_model_name
            return self._model
        except Exception as exc:
            self.fallback_reason = str(exc)
            try:
                self._model = factory(self.fallback_model_name, local_files_only=self.local_files_only)
                self.model_name = self.fallback_model_name
                return self._model
            except Exception as fallback_exc:
                self.fallback_reason = f"{self.fallback_reason}; fallback failed: {fallback_exc}"
                self._model = SyntheticSemanticEmbedder()
                self.model_name = "synthetic_scaleup"
                return self._model

    def encode(self, text, convert_to_numpy=True, show_progress_bar=False, batch_size: int | None = None, **_: Any):
        model = self._load_model()
        effective_batch_size = int(batch_size or self.batch_size)
        start = time.perf_counter()
        if isinstance(model, SyntheticSemanticEmbedder):
            arr = model.encode(text, convert_to_numpy=True, show_progress_bar=show_progress_bar)
        else:
            arr = model.encode(
                text,
                convert_to_numpy=True,
                show_progress_bar=show_progress_bar,
                batch_size=effective_batch_size,
                normalize_embeddings=True,
            )
        self.last_latency_ms = (time.perf_counter() - start) * 1000.0
        arr = np.asarray(arr, dtype=np.float32)
        self._embedding_dim = int(arr.shape[-1]) if arr.ndim else 1
        return arr if convert_to_numpy else arr.tolist()

    def embedding_metadata(self) -> Dict[str, Any]:
        return {
            "embedding_backend": "sentence_transformer",
            "embedding_model": self.model_name,
            "embedding_dim": int(self._embedding_dim),
            "embedding_local_files_only": self.local_files_only,
            "embedding_batch_size": self.batch_size,
            "embedding_latency_ms": float(self.last_latency_ms),
            "embedding_fallback_reason": self.fallback_reason,
        }


def create_semantic_embedder(
    backend: str = "synthetic",
    model_name: str = DEFAULT_EMBEDDING_MODEL,
    fallback_model_name: str = DEFAULT_FALLBACK_MODEL,
    local_files_only: bool = False,
    batch_size: int = 64,
    dim: int = 384,
):
    if backend == "synthetic":
        return SyntheticSemanticEmbedder(dim=dim)
    if backend == "sentence_transformer":
        return SentenceTransformerEmbedder(
            model_name=model_name,
            fallback_model_name=fallback_model_name,
            local_files_only=local_files_only,
            batch_size=batch_size,
        )
    raise ValueError(f"Unsupported embedding backend: {backend}")
