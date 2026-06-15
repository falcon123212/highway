def test_context_adapter_routes_marker_query_to_exact_pruning():
    from highway.runtime.context_adapter import ContextAdapter, SessionState
    from highway.runtime.context_engine import ContextRequest

    adapter = ContextAdapter()
    state = SessionState(session_id="s1")
    plan = adapter.plan(
        ContextRequest(
            user_turn="In reference ref_a1b2c3d4e5 which project has a higher budget: Project ALPHA or Project BETA?",
            session_id="s1",
        ),
        state,
    )

    assert plan["query_type"] == "marker"
    assert plan["strategy"] == "ooc_marker_entity_pruned"
    assert "reference_marker_detected" in plan["reasons"]
    assert "ALPHA" in plan["detected_entities"]
    assert "BETA" in plan["detected_entities"]


def test_context_adapter_routes_semantic_query_to_hnsw():
    from highway.runtime.context_adapter import ContextAdapter, SessionState
    from highway.runtime.context_engine import ContextRequest

    adapter = ContextAdapter()
    state = SessionState(session_id="s1")
    plan = adapter.plan(
        ContextRequest(
            user_turn="Find the operational evidence that explains the budget risk in the deployment notes.",
            session_id="s1",
        ),
        state,
    )

    assert plan["query_type"] == "semantic"
    assert plan["strategy"] == "ooc_ann_hnsw"
    assert "semantic_query_without_strong_entity" in plan["reasons"]


def test_context_adapter_uses_session_state_for_follow_up():
    from highway.runtime.context_adapter import ContextAdapter, SessionState
    from highway.runtime.context_engine import ContextRequest

    adapter = ContextAdapter()
    state = SessionState(
        session_id="s1",
        active_entities=["ALPHA"],
        active_sources=["reports/alpha.txt"],
        pinned_block_ids=["b0"],
        last_strategy="ooc_marker_entity_pruned",
        turn_count=1,
    )
    plan = adapter.plan(ContextRequest(user_turn="And what about its manager?", session_id="s1"), state)
    updated = adapter.update_state(
        state,
        plan,
        used_sources=["reports/alpha.txt"],
        used_block_ids=["b0", "b1"],
    )

    assert plan["query_type"] == "follow_up"
    assert plan["strategy"] == "ooc_ann_pruned_hybrid"
    assert plan["active_entities"] == ["ALPHA"]
    assert "follow_up_uses_session_state" in plan["reasons"]
    assert updated.turn_count == 2
    assert updated.active_entities == ["ALPHA"]
    assert updated.active_sources == ["reports/alpha.txt"]
    assert updated.pinned_block_ids == ["b0", "b1"]


def test_context_adapter_compiles_follow_up_with_active_entity():
    from highway.runtime.context_adapter import ContextAdapter, SessionState
    from highway.runtime.context_engine import ContextRequest

    plan = ContextAdapter().plan(
        ContextRequest(user_turn="And what about its manager?", session_id="s1"),
        SessionState(session_id="s1", active_entities=["ALPHA"], turn_count=3),
    )

    assert plan["query_type"] == "follow_up"
    assert plan["query_rewrite_used"] is True
    assert "Project ALPHA" in plan["compiled_query"]
    assert "manager" in plan["compiled_query"].lower()


def test_context_adapter_explicit_entity_switch_beats_follow_up_state():
    from highway.runtime.context_adapter import ContextAdapter, SessionState
    from highway.runtime.context_engine import ContextRequest

    plan = ContextAdapter().plan(
        ContextRequest(user_turn="Switch to Project GAMMA. What is its risk?", session_id="s1"),
        SessionState(session_id="s1", active_entities=["ALPHA"], turn_count=3),
    )

    assert plan["query_type"] == "entity"
    assert plan["query_rewrite_used"] is False
    assert plan["compiled_query"] == "Switch to Project GAMMA. What is its risk?"
    assert plan["active_entities"] == ["GAMMA"]


def test_context_engine_uses_compiled_query_for_follow_up_search():
    from highway.retrieval.evidence_resolver import EvidenceResolver
    from highway.runtime.context_adapter import SessionState
    from highway.runtime.context_engine import ContextRequest, HighwayContextEngine

    class FakeOOC:
        def __init__(self):
            self.questions = []
            self.offsets = [{"token_count": 10}]

        def search(self, question, top_k=50, strategy="ooc_full_scan"):
            self.questions.append(question)
            return (
                [
                    {
                        "block_id": "b1",
                        "source_file": "docs/alpha.txt",
                        "text": "Project ALPHA is managed by Nina Patel.",
                        "retrieval_score": 1.0,
                    }
                ],
                {"intent": "lookup", "target_entities": ["ALPHA"]},
                {"embedding_rows_scanned": 1, "blocks_materialized": 1, "bytes_read": 64},
            )

    fake_ooc = FakeOOC()
    engine = object.__new__(HighwayContextEngine)
    engine.storage_mode = "out_of_core"
    engine.out_of_core_index = fake_ooc
    engine.search_router = None
    engine.evidence_resolver = EvidenceResolver()
    engine.model_profile = None
    engine.input_cost_per_million = 0.0
    engine.output_cost_per_million = 0.0

    pack = engine.retrieve(
        ContextRequest(user_turn="And what about its manager?", session_id="s1", strategy="auto"),
        session_state=SessionState(session_id="s1", active_entities=["ALPHA"], turn_count=1),
        top_k=1,
    )

    assert fake_ooc.questions
    assert fake_ooc.questions[0] == pack.metrics["compiled_query"]
    assert "Project ALPHA" in fake_ooc.questions[0]
    assert pack.request.user_turn == "And what about its manager?"
    assert pack.metrics["query_rewrite_used"] is True
    assert pack.metrics["active_entity_count"] == 1


def test_context_engine_auto_uses_adapter_plan_in_metrics(tmp_path, monkeypatch):
    import sys
    import types

    import numpy as np

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
            "text": "Project ALPHA is managed by Nina Patel.",
            "source_file": "reports/alpha.txt",
            "category": "reports",
            "token_count": 6,
            "chunk_index": 0,
        },
        {
            "block_id": "b1",
            "text": "Unrelated note.",
            "source_file": "noise/note.txt",
            "category": "noise",
            "token_count": 2,
            "chunk_index": 0,
        },
    ]
    index_dir = tmp_path / "index_ooc"
    write_out_of_core_index(
        index_dir=index_dir,
        blocks=blocks,
        embeddings=np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float32),
        entities=["ALPHA"],
    )

    from highway.runtime.context_adapter import SessionState
    from highway.runtime.context_engine import ContextRequest, HighwayContextEngine

    engine = HighwayContextEngine(index_dir=index_dir, embed_model=FakeEmbedder())
    pack = engine.retrieve(
        ContextRequest(user_turn="And what about its manager?", session_id="s1", strategy="auto"),
        session_state=SessionState(session_id="s1", active_entities=["ALPHA"], turn_count=1),
        top_k=1,
    )

    assert pack.metrics["query_type"] == "follow_up"
    assert pack.metrics["strategy_used"] == "ooc_ann_pruned_hybrid"
    assert "follow_up_uses_session_state" in pack.metrics["strategy_reasons"]
