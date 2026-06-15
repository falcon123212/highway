import sys
import types

import numpy as np


def test_token_economics_computes_savings_and_kv_bytes():
    from highway.runtime.token_economics import ModelProfile, TokenEconomics

    economics = TokenEconomics.from_measurements(
        baseline_input_tokens=10_000,
        actual_input_tokens=1_000,
        output_tokens=250,
        ttft_ms=100.0,
        decode_ms=400.0,
        model_profile=ModelProfile(name="tiny", layers=2, hidden_size=4, bytes_per_element=2),
        input_cost_per_million=1.0,
        output_cost_per_million=2.0,
    )

    assert economics.avoided_input_tokens == 9_000
    assert economics.input_tokens_per_second == 10_000.0
    assert economics.output_tokens_per_second == 625.0
    assert economics.effective_tokens_per_second == 20_000.0
    assert economics.kv_bytes_estimated == 32_000
    assert economics.kv_bytes_avoided_estimated == 288_000
    assert economics.cost_estimated_usd == 0.0015
    assert economics.cost_avoided_estimated_usd == 0.009
    assert economics.warnings == []


def test_token_economics_warns_when_model_shape_is_unknown():
    from highway.runtime.token_economics import TokenEconomics

    economics = TokenEconomics.from_measurements(
        baseline_input_tokens=100,
        actual_input_tokens=40,
        output_tokens=10,
        ttft_ms=0.0,
        decode_ms=0.0,
        model_profile=None,
    )

    assert economics.kv_bytes_estimated is None
    assert economics.kv_bytes_avoided_estimated is None
    assert "model_profile_missing" in economics.warnings
    assert economics.input_tokens_per_second == 0.0
    assert economics.output_tokens_per_second == 0.0


def test_context_engine_reports_token_economics_without_llm(tmp_path, monkeypatch):
    from highway.storage.index_writer import write_out_of_core_index

    class FakeEmbedder:
        def encode(self, text, convert_to_numpy=True, show_progress_bar=False):
            if isinstance(text, list):
                return np.array([self.encode(item, convert_to_numpy=True) for item in text], dtype=np.float32)
            lowered = str(text).lower()
            if "alpha" in lowered:
                return np.array([1.0, 0.0, 0.0], dtype=np.float32)
            return np.array([0.0, 1.0, 0.0], dtype=np.float32)

    monkeypatch.setitem(
        sys.modules,
        "sentence_transformers",
        types.SimpleNamespace(SentenceTransformer=lambda _name: FakeEmbedder()),
    )

    blocks = [
        {
            "block_id": "b0",
            "text": "Project ALPHA budget is $900,000.",
            "source_file": "reports/alpha.txt",
            "category": "reports",
            "token_count": 5,
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
            "text": "Large cold document that should not be in the context pack.",
            "source_file": "noise/cold.txt",
            "category": "noise",
            "token_count": 10,
            "chunk_index": 0,
        },
    ]
    index_dir = tmp_path / "index_ooc"
    write_out_of_core_index(
        index_dir=index_dir,
        blocks=blocks,
        embeddings=np.array(
            [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 1.0, 0.0]],
            dtype=np.float32,
        ),
        entities=["ALPHA", "BETA"],
    )

    from highway.runtime.context_engine import ContextRequest, HighwayContextEngine
    from highway.runtime.token_economics import ModelProfile

    engine = HighwayContextEngine(
        index_dir=index_dir,
        embed_model=FakeEmbedder(),
        model_profile=ModelProfile(name="tiny", layers=2, hidden_size=4, bytes_per_element=2),
    )
    pack = engine.retrieve(
        ContextRequest(
            user_turn="Which project has a higher budget: Project ALPHA or Project BETA?",
            strategy="ooc_marker_entity_pruned",
        ),
        top_k=1,
    )

    economics = pack.metrics["token_economics"]
    assert economics["baseline_input_tokens"] == 20
    assert economics["actual_input_tokens"] == pack.metrics["context_input_tokens_estimated"]
    assert economics["avoided_input_tokens"] == 15
    assert economics["output_tokens"] == 0
    assert economics["kv_bytes_estimated"] == 160
    assert economics["kv_bytes_avoided_estimated"] == 480
    assert pack.metrics["tokens_avoided"] == 15
    assert pack.metrics["tokens_materialized_kv"] == 5
