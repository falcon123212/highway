import json
from pathlib import Path

import numpy as np


class FakeSentenceTransformer:
    def __init__(self, model_name, local_files_only=False):
        if model_name == "missing-model":
            raise OSError("model not found")
        self.model_name = model_name
        self.local_files_only = local_files_only

    def encode(self, texts, convert_to_numpy=True, show_progress_bar=False, batch_size=64, normalize_embeddings=True):
        if isinstance(texts, str):
            return self.encode([texts], convert_to_numpy=convert_to_numpy, batch_size=batch_size)[0]
        rows = []
        for text in texts:
            lowered = str(text).lower()
            rows.append([
                1.0 if "budget" in lowered else 0.0,
                1.0 if "managed" in lowered or "manager" in lowered else 0.0,
                float(len(str(text)) % 17) / 17.0,
            ])
        arr = np.asarray(rows, dtype=np.float32)
        norms = np.linalg.norm(arr, axis=1, keepdims=True)
        arr = arr / np.maximum(norms, 1e-8)
        return arr if convert_to_numpy else arr.tolist()


def test_synthetic_embedder_keeps_stable_vectors():
    from highway.benchmarks.ooc_scaleup import SyntheticScaleupEmbedder
    from highway.storage.semantic_embedder import create_semantic_embedder

    original = SyntheticScaleupEmbedder(dim=8).encode("Project ALPHA budget", convert_to_numpy=True)
    wrapped = create_semantic_embedder(backend="synthetic", dim=8).encode("Project ALPHA budget", convert_to_numpy=True)

    assert np.allclose(original, wrapped)


def test_sentence_transformer_embedder_reports_metadata_and_dimension(monkeypatch):
    import highway.storage.semantic_embedder as semantic_embedder

    monkeypatch.setattr(semantic_embedder, "SentenceTransformer", FakeSentenceTransformer, raising=False)
    embedder = semantic_embedder.create_semantic_embedder(
        backend="sentence_transformer",
        model_name="fake-bge",
        local_files_only=True,
        batch_size=4,
    )

    vectors = embedder.encode(["budget evidence", "manager evidence"], convert_to_numpy=True)
    metadata = embedder.embedding_metadata()

    assert vectors.shape == (2, 3)
    assert metadata["embedding_backend"] == "sentence_transformer"
    assert metadata["embedding_model"] == "fake-bge"
    assert metadata["embedding_local_files_only"] is True
    assert metadata["embedding_batch_size"] == 4
    assert metadata["embedding_dim"] == 3
    assert metadata["embedding_fallback_reason"] == ""


def test_sentence_transformer_embedder_falls_back_when_model_missing(monkeypatch):
    import highway.storage.semantic_embedder as semantic_embedder

    monkeypatch.setattr(semantic_embedder, "SentenceTransformer", FakeSentenceTransformer, raising=False)
    embedder = semantic_embedder.create_semantic_embedder(
        backend="sentence_transformer",
        model_name="missing-model",
        fallback_model_name="fake-fallback",
        local_files_only=True,
    )

    vector = embedder.encode("budget evidence", convert_to_numpy=True)
    metadata = embedder.embedding_metadata()

    assert vector.shape == (3,)
    assert metadata["embedding_model"] == "fake-fallback"
    assert "model not found" in metadata["embedding_fallback_reason"]


def test_scaleup_dataset_writes_real_embedding_dimension(tmp_path, monkeypatch):
    import highway.storage.semantic_embedder as semantic_embedder
    from highway.benchmarks.ooc_scaleup import generate_scaleup_dataset

    monkeypatch.setattr(semantic_embedder, "SentenceTransformer", FakeSentenceTransformer, raising=False)
    embedder = semantic_embedder.create_semantic_embedder(
        backend="sentence_transformer",
        model_name="fake-bge",
        local_files_only=True,
    )
    dataset = generate_scaleup_dataset(
        tmp_path,
        total_blocks=12,
        query_count=4,
        seed=7,
        embedder=embedder,
        vector_backend="numpy_flat",
    )

    manifest = json.loads((dataset.index_dir / "manifest.json").read_text(encoding="utf-8"))
    embeddings = np.load(dataset.index_dir / "embeddings.npy")

    assert embeddings.shape == (12, 3)
    assert manifest["embedding_shape"] == [12, 3]
    assert manifest["embedding_backend"] == "sentence_transformer"
    assert manifest["embedding_model"] == "fake-bge"


def test_semantic_benchmark_uses_configured_embedder(tmp_path, monkeypatch):
    import highway.storage.semantic_embedder as semantic_embedder
    from highway.benchmarks.semantic_ann_quality import run_semantic_ann_quality_benchmark

    monkeypatch.setattr(semantic_embedder, "SentenceTransformer", FakeSentenceTransformer, raising=False)
    result = run_semantic_ann_quality_benchmark(
        output_dir=tmp_path / "semantic_real_embedder",
        sizes=[20],
        query_count=4,
        seed=11,
        strategies=["ooc_full_scan", "ooc_ann_hnsw"],
        candidate_sweep=[5],
        lexical_sweep=[10],
        embedding_backend="sentence_transformer",
        embedding_model="fake-bge",
        embedding_local_files_only=True,
        embedding_batch_size=4,
    )

    tier = result["summary"]["tiers"][0]
    assert tier["embedding_backend"] == "sentence_transformer"
    assert tier["embedding_model"] == "fake-bge"
    assert tier["embedding_fallback_reason"] == ""
    assert tier["embedding_dim"] == 3
