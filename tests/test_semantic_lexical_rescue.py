import json

import numpy as np


class LexicalRescueEmbedder:
    def encode(self, text, convert_to_numpy=True, show_progress_bar=False):
        if isinstance(text, list):
            return np.asarray([self.encode(item, convert_to_numpy=True) for item in text], dtype=np.float32)
        lowered = str(text).lower()
        if "target manager" in lowered or "alpha" in lowered:
            return np.asarray([1.0, 0.0, 0.0], dtype=np.float32)
        if "semantic distractor" in lowered:
            return np.asarray([0.95, 0.05, 0.0], dtype=np.float32)
        return np.asarray([0.0, 1.0, 0.0], dtype=np.float32)


def _write_lexical_rescue_index(index_dir, block_count=24):
    from highway.storage.index_writer import write_out_of_core_index

    blocks = []
    embeddings = []
    for idx in range(block_count):
        if idx in {5, 6, 7, 8}:
            text = f"Target Manager owns Project ALPHA-{idx} with budget evidence."
            emb = [0.0, 1.0, 0.0]
        elif idx in {0, 1}:
            text = f"Semantic distractor block {idx} unrelated evidence."
            emb = [1.0, 0.0, 0.0]
        else:
            text = f"Background block {idx} with ordinary notes."
            emb = [0.0, 1.0, 0.0]
        blocks.append({
            "block_id": f"b{idx}",
            "text": text,
            "source_file": f"docs/{idx}.txt",
            "category": "semantic",
            "token_count": 8,
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


def test_semantic_strong_terms_ignore_weak_query_words(tmp_path):
    from highway.storage.out_of_core_index import OutOfCoreIndex

    _write_lexical_rescue_index(tmp_path)
    index = OutOfCoreIndex(tmp_path, embed_model=LexicalRescueEmbedder())
    query_ir = {
        "target_entities": ["Target Manager", "Project ALPHA"],
        "required_fields": ["owner", "budget"],
        "constraints": {},
    }

    terms = index._semantic_strong_terms(
        "Using the scale-up management evidence, list all project names managed by Target Manager.",
        query_ir,
    )

    assert {"target", "manager", "alpha", "owner", "budget"} <= set(terms)
    assert "using" not in terms
    assert "scale" not in terms
    assert "up" not in terms
    assert "evidence" not in terms
    assert "project" not in terms
    assert "list" not in terms


def test_ooc_semantic_lexical_rescue_reports_strong_posting_sources(tmp_path):
    from highway.runtime.hardware_budget import HardwareBudget
    from highway.storage.out_of_core_index import OutOfCoreIndex

    _write_lexical_rescue_index(tmp_path)
    index = OutOfCoreIndex(
        tmp_path,
        embed_model=LexicalRescueEmbedder(),
        hardware_budget=HardwareBudget(
            max_candidates=2,
            semantic_ann_k=2,
            semantic_rerank_k=6,
            semantic_lexical_k=6,
            semantic_full_scan_fallback_max_blocks=0,
        ),
    )

    results, _, telemetry = index.search(
        "Using semantic evidence, list all project names managed by Target Manager.",
        top_k=2,
        strategy="ooc_semantic_lexical_rescue",
    )

    assert results
    assert telemetry["semantic_lexical_rescue_used"] is True
    assert telemetry["semantic_rescue_used"] is True
    assert telemetry["ann_k"] == 2
    assert telemetry["lexical_k"] == 6
    assert telemetry["rerank_k"] == 6
    assert telemetry["fallback_used"] is False
    assert "ann" in telemetry["candidate_sources"]
    assert "strong_postings" in telemetry["candidate_sources"]
    assert telemetry["strong_posting_candidates"] > 0


def test_semantic_lexical_rescue_scans_between_hnsw_and_full_scan(tmp_path):
    from highway.runtime.hardware_budget import HardwareBudget
    from highway.storage.out_of_core_index import OutOfCoreIndex

    _write_lexical_rescue_index(tmp_path, block_count=40)
    question = "Using semantic evidence, list all project names managed by Target Manager."
    hnsw = OutOfCoreIndex(
        tmp_path,
        embed_model=LexicalRescueEmbedder(),
        hardware_budget=HardwareBudget(max_candidates=2, semantic_ann_k=2),
    )
    lexical = OutOfCoreIndex(
        tmp_path,
        embed_model=LexicalRescueEmbedder(),
        hardware_budget=HardwareBudget(
            max_candidates=2,
            semantic_ann_k=2,
            semantic_rerank_k=8,
            semantic_lexical_k=8,
        ),
    )
    full = OutOfCoreIndex(
        tmp_path,
        embed_model=LexicalRescueEmbedder(),
        hardware_budget=HardwareBudget(max_candidates=2),
    )

    _, _, hnsw_metrics = hnsw.search(question, top_k=2, strategy="ooc_ann_hnsw")
    _, _, lexical_metrics = lexical.search(question, top_k=2, strategy="ooc_semantic_lexical_rescue")
    _, _, full_metrics = full.search(question, top_k=2, strategy="ooc_full_scan")

    assert hnsw_metrics["rerank_rows_scanned"] < lexical_metrics["rerank_rows_scanned"]
    assert lexical_metrics["rerank_rows_scanned"] < full_metrics["rerank_rows_scanned"]


def test_semantic_ann_quality_reports_lexical_rescue_latency_budgets(tmp_path):
    from highway.benchmarks.semantic_ann_quality import run_semantic_ann_quality_benchmark

    result = run_semantic_ann_quality_benchmark(
        output_dir=tmp_path / "semantic_ann_quality",
        sizes=[80],
        query_count=8,
        seed=42,
        recall_gate=0.0,
        candidate_sweep=[20],
        lexical_sweep=[20],
    )

    metrics = json.loads(result["metrics_path"].read_text(encoding="utf-8"))
    tier = metrics["summary"]["tiers"][0]
    strategies = {item["strategy"] for item in tier["strategy_summaries"]}

    assert "ooc_semantic_lexical_rescue" in strategies
    assert metrics["summary"]["latency_budgets_ms"] == [100.0, 200.0]
    assert tier["best_recall_under_100ms"]["strategy"] in strategies
    assert tier["best_recall_under_200ms"]["strategy"] in strategies
