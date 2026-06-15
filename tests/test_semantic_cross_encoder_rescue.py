import json

import numpy as np


class CrossEncoderRescueEmbedder:
    def encode(self, text, convert_to_numpy=True, show_progress_bar=False):
        if isinstance(text, list):
            return np.asarray([self.encode(item, convert_to_numpy=True) for item in text], dtype=np.float32)
        lowered = str(text).lower()
        if "semantic lure" in lowered:
            return np.asarray([1.0, 0.0, 0.0], dtype=np.float32)
        if "target manager" in lowered or "alpha" in lowered:
            return np.asarray([0.8, 0.2, 0.0], dtype=np.float32)
        return np.asarray([0.0, 1.0, 0.0], dtype=np.float32)


class FakeCrossEncoder:
    def __init__(self, model_name, local_files_only=False):
        self.model_name = model_name
        self.local_files_only = local_files_only

    def predict(self, pairs, batch_size=32):
        scores = []
        for _, text in pairs:
            lowered = str(text).lower()
            score = 0.0
            if "target manager" in lowered:
                score += 4.0
            if "alpha" in lowered:
                score += 3.0
            if "budget" in lowered or "managed" in lowered:
                score += 1.0
            if "semantic lure" in lowered or "noise" in lowered:
                score -= 4.0
            scores.append(score)
        return np.asarray(scores, dtype=np.float32)


def _write_cross_encoder_index(index_dir, block_count=64):
    from highway.storage.index_writer import write_out_of_core_index

    blocks = []
    embeddings = []
    for idx in range(block_count):
        if idx in {20, 21, 22, 23}:
            text = f"Target Manager managed Project ALPHA-{idx} with budget evidence."
            emb = [0.8, 0.2, 0.0]
        elif idx in {0, 1, 2, 3}:
            text = f"Semantic lure noise block {idx} with generic evidence."
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


def test_cross_encoder_reranker_scores_semantic_candidate_above_noise():
    from highway.storage.semantic_reranker import SemanticReranker

    reranker = SemanticReranker(
        backend="cross_encoder",
        model_name="fake-cross-encoder",
        batch_size=2,
        cross_encoder_factory=FakeCrossEncoder,
    )
    ranked, metrics = reranker.rerank(
        question="Which projects are managed by Target Manager?",
        query_ir={"target_entities": ["Target Manager"], "required_fields": ["owner"]},
        candidates=[
            {"block_idx": 0, "text": "Semantic lure noise with generic evidence.", "source_file": "noise.txt"},
            {"block_idx": 1, "text": "Target Manager managed Project ALPHA with budget evidence.", "source_file": "alpha.txt"},
        ],
        limit=2,
    )

    assert ranked[0].block_idx == 1
    assert metrics["reranker_backend"] == "cross_encoder"
    assert metrics["reranker_available"] is True
    assert metrics["reranker_model"] == "fake-cross-encoder"
    assert metrics["reranker_batch_size"] == 2
    assert metrics["reranker_candidates_in"] == 2
    assert metrics["reranker_candidates_out"] == 2


def test_cross_encoder_reranker_falls_back_to_lexical_when_model_fails():
    from highway.storage.semantic_reranker import SemanticReranker

    def failing_factory(model_name, local_files_only=False):
        raise RuntimeError("model unavailable")

    reranker = SemanticReranker(
        backend="cross_encoder",
        model_name="missing-cross-encoder",
        cross_encoder_factory=failing_factory,
    )
    ranked, metrics = reranker.rerank(
        question="Which projects are managed by Target Manager?",
        query_ir={"target_entities": ["Target Manager"], "required_fields": ["owner"]},
        candidates=[
            {"block_idx": 0, "text": "Semantic lure noise with generic evidence.", "source_file": "noise.txt"},
            {"block_idx": 1, "text": "Target Manager managed Project ALPHA with budget evidence.", "source_file": "alpha.txt"},
        ],
        limit=2,
    )

    assert ranked[0].block_idx == 1
    assert metrics["reranker_backend"] == "lexical_field_reranker"
    assert metrics["reranker_available"] is False
    assert "model unavailable" in metrics["reranker_fallback_reason"]
    assert metrics["reranker_model"] == "missing-cross-encoder"


def test_ooc_semantic_cross_encoder_rescue_reports_cross_encoder_metrics(tmp_path, monkeypatch):
    from highway.runtime.hardware_budget import HardwareBudget
    from highway.storage.out_of_core_index import OutOfCoreIndex
    import highway.storage.semantic_reranker as semantic_reranker

    monkeypatch.setattr(semantic_reranker, "CrossEncoder", FakeCrossEncoder, raising=False)
    _write_cross_encoder_index(tmp_path)
    index = OutOfCoreIndex(
        tmp_path,
        embed_model=CrossEncoderRescueEmbedder(),
        hardware_budget=HardwareBudget(
            max_candidates=4,
            semantic_ann_k=4,
            semantic_lexical_k=8,
            semantic_rerank_k=8,
            semantic_reranker_input_k=12,
            semantic_reranker_output_k=8,
            semantic_reranker_model="fake-cross-encoder",
            semantic_reranker_batch_size=2,
        ),
    )

    results, _, telemetry = index.search(
        "Using semantic budget evidence, list all project names managed by Target Manager.",
        top_k=4,
        strategy="ooc_semantic_cross_encoder_rescue",
    )

    assert results
    assert results[0]["block_id"] in {"b20", "b21", "b22", "b23"}
    assert telemetry["semantic_cross_encoder_rescue_used"] is True
    assert telemetry["reranker_backend"] == "cross_encoder"
    assert telemetry["reranker_available"] is True
    assert "ann" in telemetry["candidate_sources"]
    assert "strong_postings" in telemetry["candidate_sources"]
    assert "field_postings" in telemetry["candidate_sources"]
    assert "cross_encoder" in telemetry["candidate_sources"]
    assert telemetry["embedding_rows_scanned"] < 64


def test_semantic_ann_quality_includes_cross_encoder_strategy(tmp_path):
    from highway.benchmarks.semantic_ann_quality import run_semantic_ann_quality_benchmark

    result = run_semantic_ann_quality_benchmark(
        output_dir=tmp_path / "semantic_ann_quality",
        sizes=[80],
        query_count=8,
        seed=42,
        recall_gate=101.0,
        strategies=[
            "ooc_full_scan",
            "ooc_semantic_cross_encoder_rescue",
        ],
        candidate_sweep=[20],
        lexical_sweep=[40],
        reranker_input_sweep=[40],
        reranker_output_sweep=[20],
        reranker_local_files_only=True,
    )

    metrics = json.loads(result["metrics_path"].read_text(encoding="utf-8"))
    tier = metrics["summary"]["tiers"][0]
    strategies = {item["strategy"] for item in tier["strategy_summaries"]}

    assert "ooc_semantic_cross_encoder_rescue" in strategies
    assert metrics["summary"]["status"] == "NON_VALIDATING"
