import json
import pickle
import sys
import types

import numpy as np


class FakeEmbedder:
    def encode(self, text, convert_to_numpy=True, show_progress_bar=False):
        if isinstance(text, list):
            return np.array([self.encode(t, convert_to_numpy=True) for t in text], dtype=np.float32)
        lowered = str(text).lower()
        if "alpha" in lowered or "ref_alpha" in lowered:
            return np.array([1.0, 0.0, 0.0], dtype=np.float32)
        if "beta" in lowered:
            return np.array([0.0, 1.0, 0.0], dtype=np.float32)
        return np.array([0.0, 0.0, 1.0], dtype=np.float32)


class TinyBM25:
    def __init__(self, corpus_tokens):
        self.corpus_tokens = corpus_tokens

    def get_scores(self, query_tokens):
        scores = []
        query = set(query_tokens)
        for tokens in self.corpus_tokens:
            scores.append(float(sum(1 for token in tokens if token in query)))
        return np.array(scores, dtype=np.float32)


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


def _write_legacy_index(index_dir):
    index_dir.mkdir(parents=True)
    blocks = _blocks()
    (index_dir / "blocks.jsonl").write_text(
        "\n".join(json.dumps(block) for block in blocks) + "\n",
        encoding="utf-8",
    )
    np.save(index_dir / "embeddings.npy", np.array([
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
    ], dtype=np.float32))
    with (index_dir / "bm25.pkl").open("wb") as f:
        pickle.dump(TinyBM25([block["text"].lower().split() for block in blocks]), f)
    (index_dir / "entity_list.json").write_text(json.dumps(["ALPHA", "BETA"]), encoding="utf-8")


def test_out_of_core_index_uses_memmap_and_fetches_blocks_lazily(tmp_path):
    from highway.runtime.hardware_budget import HardwareBudget
    from highway.storage.index_writer import write_out_of_core_index
    from highway.storage.out_of_core_index import OutOfCoreIndex

    index_dir = tmp_path / "index_ooc"
    write_out_of_core_index(
        index_dir=index_dir,
        blocks=_blocks(),
        embeddings=np.array([
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ], dtype=np.float32),
        entities=["ALPHA", "BETA"],
    )

    index = OutOfCoreIndex(index_dir, embed_model=FakeEmbedder(), hardware_budget=HardwareBudget(max_candidates=2))
    results, query_ir, telemetry = index.search(
        "In reference ref_a1b2c3d4e5 which project has a higher budget: Project ALPHA or Project BETA?",
        top_k=1,
    )

    assert isinstance(index.embeddings, np.memmap)
    assert query_ir["constraints"]["reference_marker"] == "ref_a1b2c3d4e5"
    assert [result["block_id"] for result in results] == ["b0"]
    assert telemetry["storage_mode"] == "out_of_core"
    assert telemetry["embedding_rows_scanned"] == 3
    assert telemetry["blocks_materialized"] == 1
    assert telemetry["blocks_materialized"] < 3
    assert telemetry["bytes_read"] > 0


def test_search_router_auto_preserves_top_result_between_legacy_and_out_of_core(tmp_path, monkeypatch):
    monkeypatch.setitem(sys.modules, "rank_bm25", types.SimpleNamespace(BM25Okapi=TinyBM25))
    monkeypatch.setitem(
        sys.modules,
        "sentence_transformers",
        types.SimpleNamespace(SentenceTransformer=lambda _name: FakeEmbedder()),
    )

    from highway.retrieval import search as search_module
    from highway.retrieval.search import SearchRouter
    from highway.storage.index_writer import write_out_of_core_index

    monkeypatch.setattr(search_module, "SentenceTransformer", lambda _name: FakeEmbedder())

    legacy_dir = tmp_path / "legacy"
    ooc_dir = tmp_path / "index_ooc"
    _write_legacy_index(legacy_dir)
    write_out_of_core_index(
        index_dir=ooc_dir,
        blocks=_blocks(),
        embeddings=np.array([
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ], dtype=np.float32),
        entities=["ALPHA", "BETA"],
    )

    legacy_results, _ = SearchRouter(str(legacy_dir)).search(
        "In reference ref_a1b2c3d4e5 which project has a higher budget: Project ALPHA or Project BETA?",
        top_k=1,
    )
    ooc_router = SearchRouter(str(ooc_dir))
    ooc_results, _ = ooc_router.search(
        "In reference ref_a1b2c3d4e5 which project has a higher budget: Project ALPHA or Project BETA?",
        top_k=1,
    )

    assert legacy_results[0]["block_id"] == ooc_results[0]["block_id"] == "b0"
    assert ooc_router.storage_mode == "out_of_core"
    assert ooc_router.last_storage_metrics["blocks_materialized"] == 1


def test_ingest_corpus_can_write_out_of_core_layout(tmp_path, monkeypatch):
    monkeypatch.setitem(sys.modules, "rank_bm25", types.SimpleNamespace(BM25Okapi=TinyBM25))
    monkeypatch.setitem(
        sys.modules,
        "sentence_transformers",
        types.SimpleNamespace(SentenceTransformer=lambda _name: FakeEmbedder()),
    )

    from highway.ingestion import ingest as ingest_module

    corpus_dir = tmp_path / "corpus"
    doc_dir = corpus_dir / "documents" / "reports"
    doc_dir.mkdir(parents=True)
    (doc_dir / "alpha.txt").write_text(
        "Reference ref_a1b2c3d4e5. Project ALPHA budget is $900,000.",
        encoding="utf-8",
    )
    index_dir = tmp_path / "index_ooc"

    ingest_module.ingest_corpus(str(corpus_dir), str(index_dir), layout="out_of_core")

    assert (index_dir / "manifest.json").exists()
    assert (index_dir / "postings.sqlite").exists()
    assert (index_dir / "block_offsets.json").exists()
    assert not (index_dir / "bm25.pkl").exists()


def test_scheduler_surfaces_storage_telemetry_after_search(tmp_path, monkeypatch):
    monkeypatch.setitem(sys.modules, "rank_bm25", types.SimpleNamespace(BM25Okapi=TinyBM25))
    monkeypatch.setitem(
        sys.modules,
        "sentence_transformers",
        types.SimpleNamespace(SentenceTransformer=lambda _name: FakeEmbedder()),
    )

    from highway.runtime import scheduler as scheduler_module

    class FakeParser:
        def parse(self, question):
            return {"question": question, "target_entities": [], "intent": "single_fact_lookup"}

        def canonical_hash(self, query_ir):
            return "hash"

    class FakeSearchRouter:
        def __init__(self, index_dir):
            self.query_parser = FakeParser()
            self.last_storage_metrics = {}

        def search(self, question, top_k=50):
            self.last_storage_metrics = {
                "storage_mode": "out_of_core",
                "bytes_read": 123,
                "blocks_materialized": 1,
            }
            return ([{"block_id": "b0", "text": "No answer.", "source_file": "reports/a.txt"}], self.query_parser.parse(question))

    class FakeCache:
        def __init__(self, cache_dir):
            pass

        def get_proof_ir(self, query_ir_hash):
            return None

        def get_evidence_pool(self, query_ir_hash, search_config_hash):
            return None

        def set_evidence_pool(self, query_ir_hash, search_config_hash, evidence_pool):
            pass

        def set_proof_ir(self, query_ir_hash, proof_ir):
            pass

    class FakeResolver:
        def resolve(self, evidence_pool, query_ir):
            return evidence_pool, [], []

    class FakeIRBuilder:
        def build_ir(self, query_ir, active, suppressed, forbidden):
            return {
                "query": query_ir,
                "proof": {"status": "COMPLETE"},
                "guard_decision": {"action": "BYPASS_LLM", "answer": "NOT_FOUND"},
                "output_schema": {},
                "evidence": active,
            }

    monkeypatch.setattr(scheduler_module, "SearchRouter", FakeSearchRouter)
    monkeypatch.setattr(scheduler_module, "CacheManager", FakeCache)
    monkeypatch.setattr(scheduler_module, "EvidenceResolver", FakeResolver)
    monkeypatch.setattr(scheduler_module, "IRBuilder", FakeIRBuilder)

    scheduler = scheduler_module.ExecutionScheduler(str(tmp_path / "index"), str(tmp_path / "cache"))
    result = scheduler.answer("Where is Project ALPHA?", use_cache=False)

    assert result["answer"] == "NOT_FOUND"
    assert result["metrics"]["storage_mode"] == "out_of_core"
    assert result["metrics"]["bytes_read"] == 123
    assert result["metrics"]["blocks_materialized"] == 1


def test_poc234_workload_reingestion_preserves_out_of_core_layout(tmp_path, monkeypatch):
    from highway.workloads import build_poc234_kernel_hardening_workload as workload_module
    from highway.ingestion import ingest as ingest_module

    corpus_dir = tmp_path / "corpus"
    index_dir = corpus_dir / "index_ooc"
    index_dir.mkdir(parents=True)
    (index_dir / "manifest.json").write_text(
        json.dumps({"layout": "highway_out_of_core_v1"}),
        encoding="utf-8",
    )

    calls = []

    def fake_ingest_corpus(corpus_arg, index_arg, **kwargs):
        calls.append((corpus_arg, index_arg, kwargs))

    monkeypatch.setattr(ingest_module, "ingest_corpus", fake_ingest_corpus)

    workload_module._run_ingestion(str(corpus_dir), str(index_dir))

    assert calls == [(str(corpus_dir), str(index_dir), {"layout": "out_of_core"})]
