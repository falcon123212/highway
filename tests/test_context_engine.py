import json
import sys
import types

import numpy as np


class FakeEmbedder:
    def encode(self, text, convert_to_numpy=True, show_progress_bar=False):
        if isinstance(text, list):
            return np.array([self.encode(item, convert_to_numpy=True) for item in text], dtype=np.float32)
        lowered = str(text).lower()
        if "alpha" in lowered or "ref_a1b2c3d4e5" in lowered:
            return np.array([1.0, 0.0, 0.0], dtype=np.float32)
        if "beta" in lowered:
            return np.array([0.0, 1.0, 0.0], dtype=np.float32)
        return np.array([0.0, 0.0, 1.0], dtype=np.float32)


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
            "text": "Unrelated operational note.",
            "source_file": "noise/note.txt",
            "category": "noise",
            "token_count": 3,
            "chunk_index": 0,
        },
    ]


def _write_ooc_index(index_dir):
    from highway.storage.index_writer import write_out_of_core_index

    write_out_of_core_index(
        index_dir=index_dir,
        blocks=_blocks(),
        embeddings=np.array(
            [
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float32,
        ),
        entities=["ALPHA", "BETA"],
    )


def test_context_pack_is_json_serializable():
    from highway.runtime.context_engine import ContextBlock, ContextPack, ContextRequest

    request = ContextRequest(user_turn="Where is Project ALPHA?")
    pack = ContextPack(
        request=request,
        blocks=[
            ContextBlock(
                block_id="b0",
                source_file="reports/alpha.txt",
                text="Project ALPHA budget is $900,000.",
                score=1.0,
                reason="ranked_candidate",
            )
        ],
        query_ir={"question": request.user_turn},
        metrics={"latency_ms": 1.5},
        warnings=[],
    )

    encoded = json.dumps(pack.to_dict(), sort_keys=True)
    decoded = json.loads(encoded)

    assert decoded["request"]["user_turn"] == "Where is Project ALPHA?"
    assert decoded["blocks"][0]["block_id"] == "b0"
    assert decoded["metrics"]["latency_ms"] == 1.5


def test_context_engine_retrieves_context_without_llm(tmp_path, monkeypatch):
    monkeypatch.setitem(
        sys.modules,
        "sentence_transformers",
        types.SimpleNamespace(SentenceTransformer=lambda _name: FakeEmbedder()),
    )

    from highway.runtime.context_engine import ContextRequest, HighwayContextEngine

    index_dir = tmp_path / "index_ooc"
    _write_ooc_index(index_dir)

    engine = HighwayContextEngine(index_dir=str(index_dir), embed_model=FakeEmbedder())
    pack = engine.retrieve(
        ContextRequest(
            user_turn="In reference ref_a1b2c3d4e5 which project has a higher budget: Project ALPHA or Project BETA?",
            strategy="ooc_marker_entity_pruned",
            token_budget=512,
        ),
        top_k=2,
    )

    assert pack.request.session_id == "default"
    assert pack.blocks[0].block_id == "b0"
    assert pack.blocks[0].reason == "ranked_candidate"
    assert pack.query_ir["constraints"]["reference_marker"] == "ref_a1b2c3d4e5"
    assert pack.metrics["strategy_used"] == "ooc_marker_entity_pruned"
    assert pack.metrics["storage_mode"] == "out_of_core"
    assert pack.metrics["blocks_materialized"] == 1
    assert pack.metrics["embedding_rows_scanned"] < 3
    assert pack.metrics["context_input_tokens_estimated"] > 0
    assert pack.metrics["latency_ms"] >= 0.0
    assert pack.metrics["ann_backend"] in {"none", "faiss_hnsw", "faiss_flat", "faiss_ivf_flat"}
    assert pack.warnings == []


def test_context_engine_warns_when_context_exceeds_budget(tmp_path, monkeypatch):
    monkeypatch.setitem(
        sys.modules,
        "sentence_transformers",
        types.SimpleNamespace(SentenceTransformer=lambda _name: FakeEmbedder()),
    )

    from highway.runtime.context_engine import ContextRequest, HighwayContextEngine

    index_dir = tmp_path / "index_ooc"
    _write_ooc_index(index_dir)

    engine = HighwayContextEngine(index_dir=str(index_dir), embed_model=FakeEmbedder())
    pack = engine.retrieve(
        ContextRequest(
            user_turn="Which project has a higher budget: Project ALPHA or Project BETA?",
            strategy="ooc_full_scan",
            token_budget=1,
        ),
        top_k=2,
    )

    assert pack.metrics["context_input_tokens_estimated"] > pack.request.token_budget
    assert "context_token_budget_exceeded" in pack.warnings
