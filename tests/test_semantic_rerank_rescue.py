import json

import numpy as np


class RerankRescueEmbedder:
    def encode(self, text, convert_to_numpy=True, show_progress_bar=False):
        if isinstance(text, list):
            return np.asarray([self.encode(item, convert_to_numpy=True) for item in text], dtype=np.float32)
        lowered = str(text).lower()
        if "semantic lure" in lowered:
            return np.asarray([1.0, 0.0, 0.0], dtype=np.float32)
        if "target manager" in lowered or "alpha" in lowered:
            return np.asarray([0.8, 0.2, 0.0], dtype=np.float32)
        return np.asarray([0.0, 1.0, 0.0], dtype=np.float32)


def _write_rerank_index(index_dir, block_count=32):
    from highway.storage.index_writer import write_out_of_core_index

    blocks = []
    embeddings = []
    for idx in range(block_count):
        if idx in {8, 9, 10, 11}:
            text = f"Target Manager owns Project ALPHA-{idx} with budget evidence and source proof."
            emb = [0.8, 0.2, 0.0]
        elif idx in {0, 1, 2, 3}:
            text = f"Semantic lure noise block {idx} unrelated generic evidence."
            emb = [1.0, 0.0, 0.0]
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


def test_local_reranker_boosts_entities_and_penalizes_noise():
    from highway.storage.local_reranker import LexicalFieldReranker

    query_ir = {
        "target_entities": ["Target Manager", "Project ALPHA"],
        "required_fields": ["owner", "budget"],
        "constraints": {},
    }
    candidates = [
        {
            "block_idx": 0,
            "text": "Semantic lure noise block with generic evidence.",
            "source_file": "docs/noise.txt",
        },
        {
            "block_idx": 1,
            "text": "Target Manager owns Project ALPHA with budget evidence and source proof.",
            "source_file": "docs/alpha.txt",
        },
    ]

    ranked, metrics = LexicalFieldReranker().rerank(
        question="Which projects are managed by Target Manager?",
        query_ir=query_ir,
        candidates=candidates,
        limit=2,
    )

    assert ranked[0].block_idx == 1
    assert ranked[0].score > ranked[1].score
    assert metrics["reranker_backend"] == "lexical_field_reranker"
    assert metrics["reranker_candidates_in"] == 2
    assert metrics["reranker_candidates_out"] == 2
    assert metrics["reranker_latency_ms"] >= 0.0


def test_ooc_semantic_rerank_rescue_is_accepted_and_reports_metrics(tmp_path):
    from highway.runtime.hardware_budget import HardwareBudget
    from highway.storage.out_of_core_index import OutOfCoreIndex

    _write_rerank_index(tmp_path)
    index = OutOfCoreIndex(
        tmp_path,
        embed_model=RerankRescueEmbedder(),
        hardware_budget=HardwareBudget(
            max_candidates=4,
            semantic_ann_k=4,
            semantic_rerank_k=8,
            semantic_lexical_k=8,
            semantic_full_scan_fallback_max_blocks=0,
        ),
    )

    results, _, telemetry = index.search(
        "Using semantic evidence, list all project names managed by Target Manager.",
        top_k=4,
        strategy="ooc_semantic_rerank_rescue",
    )

    assert results
    assert results[0]["block_id"] in {"b8", "b9", "b10", "b11"}
    assert telemetry["semantic_rerank_rescue_used"] is True
    assert telemetry["reranker_backend"] == "lexical_field_reranker"
    assert telemetry["reranker_candidates_in"] >= telemetry["reranker_candidates_out"] >= 4
    assert telemetry["reranker_latency_ms"] >= 0.0
    assert "reranker" in telemetry["candidate_sources"]


def test_semantic_ann_quality_includes_rerank_rescue_and_can_be_non_validating(tmp_path):
    from highway.benchmarks.semantic_ann_quality import run_semantic_ann_quality_benchmark

    result = run_semantic_ann_quality_benchmark(
        output_dir=tmp_path / "semantic_ann_quality",
        sizes=[80],
        query_count=8,
        seed=42,
        recall_gate=101.0,
        strategies=[
            "ooc_full_scan",
            "ooc_ann_hnsw",
            "ooc_semantic_lexical_rescue",
            "ooc_semantic_rerank_rescue",
        ],
        candidate_sweep=[20],
        lexical_sweep=[20],
    )

    metrics = json.loads(result["metrics_path"].read_text(encoding="utf-8"))
    tier = metrics["summary"]["tiers"][0]
    strategies = {item["strategy"] for item in tier["strategy_summaries"]}

    assert "ooc_semantic_rerank_rescue" in strategies
    assert metrics["summary"]["status"] == "NON_VALIDATING"
