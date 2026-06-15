import json

import numpy as np


class TinyEmbedder:
    def encode(self, text, convert_to_numpy=True, show_progress_bar=False):
        if isinstance(text, list):
            return np.asarray([self.encode(item, convert_to_numpy=True) for item in text], dtype=np.float32)
        lowered = str(text).lower()
        if "alpha" in lowered or "ref_a1b2c3d4e5" in lowered:
            return np.asarray([1.0, 0.0, 0.0], dtype=np.float32)
        if "beta" in lowered:
            return np.asarray([0.0, 1.0, 0.0], dtype=np.float32)
        return np.asarray([0.0, 0.0, 1.0], dtype=np.float32)


def _blocks():
    return [
        {
            "block_id": "b0",
            "text": "Reference ref_a1b2c3d4e5. Project ALPHA budget is $900,000.",
            "source_file": "reports/alpha.txt",
            "category": "reports",
            "token_count": 8,
            "chunk_index": 0,
        },
        {
            "block_id": "b1",
            "text": "Project BETA budget is $100,000.",
            "source_file": "reports/beta.txt",
            "category": "reports",
            "token_count": 5,
            "chunk_index": 0,
        },
        {
            "block_id": "b2",
            "text": "Unrelated gamma note.",
            "source_file": "noise/gamma.txt",
            "category": "noise",
            "token_count": 3,
            "chunk_index": 0,
        },
    ]


def _write_ooc_index(index_dir, **kwargs):
    from highway.storage.index_writer import write_out_of_core_index

    return write_out_of_core_index(
        index_dir=index_dir,
        blocks=_blocks(),
        embeddings=np.asarray(
            [
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float32,
        ),
        entities=["ALPHA", "BETA"],
        **kwargs,
    )


def test_numpy_flat_vector_candidate_index_is_always_available(tmp_path):
    from highway.storage.vector_index import build_vector_index, load_vector_index

    embeddings_path = tmp_path / "embeddings.npy"
    np.save(embeddings_path, np.asarray([[1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=np.float32))

    metadata = build_vector_index(
        embeddings_path=embeddings_path,
        output_path=tmp_path,
        backend="numpy_flat",
        params={},
    )
    index = load_vector_index(tmp_path, backend="numpy_flat")
    hits, telemetry = index.search(np.asarray([1.0, 0.0, 0.0], dtype=np.float32), k=2)

    assert metadata["vector_backend"] == "numpy_flat"
    assert metadata["ann_available"] is True
    assert hits[0][0] == 0
    assert telemetry["ann_backend"] == "numpy_flat"
    assert telemetry["ann_candidates"] == 2
    assert telemetry["ann_available"] is True


def test_faiss_backend_missing_does_not_break_ooc_search(tmp_path):
    from highway.storage.out_of_core_index import OutOfCoreIndex

    _write_ooc_index(tmp_path, vector_backend="faiss_hnsw")
    index = OutOfCoreIndex(tmp_path, embed_model=TinyEmbedder())
    results, query_ir, telemetry = index.search(
        "In reference ref_a1b2c3d4e5 which project has a higher budget: Project ALPHA or Project BETA?",
        top_k=1,
        strategy="ooc_ann_hnsw",
    )

    assert results[0]["block_id"] == "b0"
    assert telemetry["ann_backend"] == "faiss_hnsw"
    assert telemetry["ann_available"] is False
    assert telemetry["ann_fallback_reason"]
    assert telemetry["rerank_rows_scanned"] >= 1


def test_ann_pruned_hybrid_uses_marker_candidates_before_ann(tmp_path):
    from highway.storage.out_of_core_index import OutOfCoreIndex

    _write_ooc_index(tmp_path, vector_backend="faiss_hnsw")
    index = OutOfCoreIndex(tmp_path, embed_model=TinyEmbedder())
    results, query_ir, telemetry = index.search(
        "In reference ref_a1b2c3d4e5 which project has a higher budget: Project ALPHA or Project BETA?",
        top_k=1,
        strategy="ooc_ann_pruned_hybrid",
    )

    assert results[0]["block_id"] == "b0"
    assert telemetry["ann_backend"] == "faiss_hnsw"
    assert telemetry["ann_available"] is False
    assert telemetry["ann_used"] is False
    assert telemetry["ann_fallback_reason"] == "pruned_candidates_sufficient"
    assert telemetry["embedding_rows_scanned"] == 1
    assert telemetry["blocks_materialized"] == 1


def test_ooc_manifest_records_optional_vector_backend(tmp_path):
    manifest = _write_ooc_index(tmp_path, vector_backend="numpy_flat", ann_params={"probe": 3})

    saved = json.loads((tmp_path / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["vector_backend"] == "numpy_flat"
    assert saved["vector_backend"] == "numpy_flat"
    assert saved["ann_metric"] == "inner_product"
    assert saved["ann_params"] == {"probe": 3}
