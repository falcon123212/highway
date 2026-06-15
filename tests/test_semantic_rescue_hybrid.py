import json

import numpy as np


class RescueEmbedder:
    def encode(self, text, convert_to_numpy=True, show_progress_bar=False):
        if isinstance(text, list):
            return np.asarray([self.encode(item, convert_to_numpy=True) for item in text], dtype=np.float32)
        lowered = str(text).lower()
        if "alpha" in lowered:
            return np.asarray([1.0, 0.0, 0.0], dtype=np.float32)
        if "beta" in lowered:
            return np.asarray([0.0, 1.0, 0.0], dtype=np.float32)
        return np.asarray([0.0, 0.0, 1.0], dtype=np.float32)


def _write_rescue_index(index_dir):
    from highway.storage.index_writer import write_out_of_core_index

    blocks = [
        {
            "block_id": "b0",
            "text": "Project ALPHA budget evidence.",
            "source_file": "docs/alpha.txt",
            "category": "semantic",
            "token_count": 4,
            "chunk_index": 0,
        },
        {
            "block_id": "b1",
            "text": "Project BETA budget evidence.",
            "source_file": "docs/beta.txt",
            "category": "semantic",
            "token_count": 4,
            "chunk_index": 0,
        },
        {
            "block_id": "b2",
            "text": "Project GAMMA unrelated note.",
            "source_file": "docs/gamma.txt",
            "category": "semantic",
            "token_count": 4,
            "chunk_index": 0,
        },
    ]
    embeddings = np.asarray(
        [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
        dtype=np.float32,
    )
    return write_out_of_core_index(
        index_dir,
        blocks=blocks,
        embeddings=embeddings,
        entities=["ALPHA", "BETA", "GAMMA"],
        vector_backend="numpy_flat",
    )


def test_ooc_semantic_rescue_hybrid_is_accepted_and_reports_rescue_telemetry(tmp_path):
    from highway.runtime.hardware_budget import HardwareBudget
    from highway.storage.out_of_core_index import OutOfCoreIndex

    _write_rescue_index(tmp_path)
    index = OutOfCoreIndex(
        tmp_path,
        embed_model=RescueEmbedder(),
        hardware_budget=HardwareBudget(
            max_candidates=2,
            semantic_ann_k=3,
            semantic_rerank_k=3,
            semantic_full_scan_fallback_max_blocks=0,
        ),
    )

    results, query_ir, telemetry = index.search(
        "Using semantic evidence, compare Project ALPHA and Project BETA.",
        top_k=2,
        strategy="ooc_semantic_rescue_hybrid",
    )

    assert results
    assert telemetry["semantic_rescue_used"] is True
    assert telemetry["ann_k"] == 3
    assert telemetry["rerank_k"] == 3
    assert telemetry["ann_backend"] == "numpy_flat"
    assert telemetry["ann_candidates"] == 3
    assert "ann" in telemetry["candidate_sources"]
    assert "postings" in telemetry["candidate_sources"]
    assert telemetry["fallback_used"] is False


def test_semantic_rescue_scans_more_candidates_than_standard_hnsw(tmp_path):
    from highway.runtime.hardware_budget import HardwareBudget
    from highway.storage.out_of_core_index import OutOfCoreIndex

    _write_rescue_index(tmp_path)
    standard = OutOfCoreIndex(
        tmp_path,
        embed_model=RescueEmbedder(),
        hardware_budget=HardwareBudget(
            max_candidates=1,
            semantic_ann_k=3,
            semantic_rerank_k=3,
            semantic_full_scan_fallback_max_blocks=0,
        ),
    )
    rescue = OutOfCoreIndex(
        tmp_path,
        embed_model=RescueEmbedder(),
        hardware_budget=HardwareBudget(
            max_candidates=1,
            semantic_ann_k=3,
            semantic_rerank_k=3,
            semantic_full_scan_fallback_max_blocks=0,
        ),
    )

    _, _, standard_telemetry = standard.search(
        "Using semantic evidence, compare Project ALPHA and Project BETA.",
        top_k=1,
        strategy="ooc_ann_hnsw",
    )
    _, _, rescue_telemetry = rescue.search(
        "Using semantic evidence, compare Project ALPHA and Project BETA.",
        top_k=1,
        strategy="ooc_semantic_rescue_hybrid",
    )

    assert rescue_telemetry["rerank_rows_scanned"] > standard_telemetry["rerank_rows_scanned"]
    assert rescue_telemetry["ann_k"] > standard_telemetry.get("ann_candidates", 0)


def test_semantic_ann_quality_benchmark_includes_rescue_and_candidate_sweep(tmp_path):
    from highway.benchmarks.semantic_ann_quality import run_semantic_ann_quality_benchmark

    result = run_semantic_ann_quality_benchmark(
        output_dir=tmp_path / "semantic_ann_quality",
        sizes=[80],
        query_count=8,
        seed=42,
        recall_gate=101.0,
        candidate_sweep=[20, 50],
    )

    metrics = json.loads(result["metrics_path"].read_text(encoding="utf-8"))
    tier = metrics["summary"]["tiers"][0]
    strategies = {item["strategy"] for item in tier["strategy_summaries"]}

    assert "ooc_semantic_rescue_hybrid" in strategies
    assert tier["best_recall"]["strategy"] in strategies
    assert tier["best_latency"]["strategy"] in strategies
    assert tier["best_tradeoff"]["strategy"] in strategies
    assert 20 in tier["candidate_sweep"]
    assert 50 in tier["candidate_sweep"]
    assert metrics["summary"]["status"] == "NON_VALIDATING"


def test_semantic_ann_quality_can_validate_when_gate_is_met(tmp_path):
    from highway.benchmarks.semantic_ann_quality import run_semantic_ann_quality_benchmark

    result = run_semantic_ann_quality_benchmark(
        output_dir=tmp_path / "semantic_ann_quality",
        sizes=[80],
        query_count=8,
        seed=42,
        recall_gate=0.0,
        candidate_sweep=[20],
    )

    metrics = json.loads(result["metrics_path"].read_text(encoding="utf-8"))

    assert metrics["summary"]["status"] == "VALIDATING"
