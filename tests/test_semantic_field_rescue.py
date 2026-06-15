import json

import numpy as np


class FieldRescueEmbedder:
    def encode(self, text, convert_to_numpy=True, show_progress_bar=False):
        if isinstance(text, list):
            return np.asarray([self.encode(item, convert_to_numpy=True) for item in text], dtype=np.float32)
        lowered = str(text).lower()
        if "semantic lure" in lowered:
            return np.asarray([1.0, 0.0, 0.0], dtype=np.float32)
        if "target manager" in lowered or "alpha" in lowered:
            return np.asarray([0.8, 0.2, 0.0], dtype=np.float32)
        return np.asarray([0.0, 1.0, 0.0], dtype=np.float32)


def _write_field_rescue_index(index_dir, block_count=64):
    from highway.storage.index_writer import write_out_of_core_index

    blocks = []
    embeddings = []
    for idx in range(block_count):
        if idx in {20, 21, 22, 23}:
            text = f"Target Manager managed Project ALPHA-{idx} with approved budget evidence."
            emb = [0.8, 0.2, 0.0]
        elif idx in {0, 1, 2, 3}:
            text = f"Semantic lure generic note {idx} without management fields."
            emb = [1.0, 0.0, 0.0]
        elif idx in {30, 31, 32, 33}:
            text = f"Operational budget management note {idx} for a different team."
            emb = [0.4, 0.4, 0.0]
        else:
            text = f"Background note {idx} unrelated to the requested manager."
            emb = [0.0, 1.0, 0.0]
        blocks.append({
            "block_id": f"b{idx}",
            "text": text,
            "source_file": f"docs/{idx}.txt",
            "category": "semantic",
            "token_count": 10,
            "chunk_index": 0,
        })
        embeddings.append(emb)
    return write_out_of_core_index(
        index_dir,
        blocks=blocks,
        embeddings=np.asarray(embeddings, dtype=np.float32),
        entities=["Target Manager", "ALPHA"],
        vector_backend="numpy_flat",
    )


def test_ooc_semantic_field_rescue_adds_field_postings_and_reports_metrics(tmp_path):
    from highway.runtime.hardware_budget import HardwareBudget
    from highway.storage.out_of_core_index import OutOfCoreIndex

    _write_field_rescue_index(tmp_path)
    hnsw_index = OutOfCoreIndex(
        tmp_path,
        embed_model=FieldRescueEmbedder(),
        hardware_budget=HardwareBudget(max_candidates=4, semantic_ann_k=4),
    )
    field_index = OutOfCoreIndex(
        tmp_path,
        embed_model=FieldRescueEmbedder(),
        hardware_budget=HardwareBudget(
            max_candidates=4,
            semantic_ann_k=4,
            semantic_rerank_k=16,
            semantic_lexical_k=8,
        ),
    )

    question = "Using semantic budget evidence, list all project names managed by Target Manager."
    _, _, hnsw_telemetry = hnsw_index.search(question, top_k=4, strategy="ooc_ann_hnsw")
    results, _, telemetry = field_index.search(question, top_k=4, strategy="ooc_semantic_field_rescue")

    assert results
    assert any(result["block_id"] in {"b20", "b21", "b22", "b23"} for result in results)
    assert telemetry["semantic_field_rescue_used"] is True
    assert "field_postings" in telemetry["candidate_sources"]
    assert telemetry["field_posting_candidates"] > 0
    assert telemetry["embedding_rows_scanned"] > hnsw_telemetry["embedding_rows_scanned"]
    assert telemetry["embedding_rows_scanned"] < 64


def test_ooc_semantic_field_rescue_does_not_repeat_generic_query_lookup(tmp_path, monkeypatch):
    from highway.runtime.hardware_budget import HardwareBudget
    from highway.storage.out_of_core_index import OutOfCoreIndex

    _write_field_rescue_index(tmp_path)
    index = OutOfCoreIndex(
        tmp_path,
        embed_model=FieldRescueEmbedder(),
        hardware_budget=HardwareBudget(
            max_candidates=4,
            semantic_ann_k=4,
            semantic_rerank_k=16,
            semantic_lexical_k=8,
        ),
    )
    observed_terms = []
    original_lookup = index._lookup_term_scores

    def recording_lookup(terms, limit=None):
        materialized = list(terms)
        observed_terms.append(set(materialized))
        return original_lookup(materialized, limit=limit)

    monkeypatch.setattr(index, "_lookup_term_scores", recording_lookup)

    index.search(
        "Using semantic budget evidence, list all project names managed by Target Manager.",
        top_k=4,
        strategy="ooc_semantic_field_rescue",
    )

    assert observed_terms
    assert all("using" not in terms for terms in observed_terms)


def test_semantic_ann_quality_includes_field_rescue(tmp_path):
    from highway.benchmarks.semantic_ann_quality import run_semantic_ann_quality_benchmark

    result = run_semantic_ann_quality_benchmark(
        output_dir=tmp_path / "semantic_ann_quality",
        sizes=[80],
        query_count=8,
        seed=42,
        recall_gate=101.0,
        strategies=[
            "ooc_full_scan",
            "ooc_semantic_field_rescue",
        ],
        candidate_sweep=[20],
        lexical_sweep=[40],
    )

    metrics = json.loads(result["metrics_path"].read_text(encoding="utf-8"))
    tier = metrics["summary"]["tiers"][0]
    strategies = {item["strategy"] for item in tier["strategy_summaries"]}

    assert "ooc_semantic_field_rescue" in strategies
    assert metrics["summary"]["status"] == "NON_VALIDATING"
