from highway.ingestion.ingest import SimpleWordTokenizer, chunk_text
from highway.retrieval.query_parser import QueryParser
from highway.retrieval.retrievers import BM25Retriever


def test_canonical_ingestion_retrieval_extraction_generation_pipeline(tmp_path, monkeypatch):
    import highway.runtime.scheduler as scheduler_module
    from highway.runtime.scheduler import ExecutionScheduler

    tokenizer = SimpleWordTokenizer()
    source_text = "Project ORION approved budget is $123,000. Project ORION is managed by Alice Martin."
    chunks = chunk_text(source_text, tokenizer, block_size=64, overlap=0)
    blocks = [
        {
            "block_id": f"block_{idx}",
            "text": chunk["text"],
            "source_file": "canonical/orion.txt",
            "category": "canonical",
            "token_count": chunk["token_count"],
            "chunk_index": idx,
        }
        for idx, chunk in enumerate(chunks)
    ]

    class CanonicalSearchRouter:
        def __init__(self, index_dir):
            self.query_parser = QueryParser(["Project ORION"])
            self.last_storage_metrics = {}

        def search(self, question, top_k=50):
            query_ir = self.query_parser.parse(question)
            retrieved = BM25Retriever(blocks).retrieve(question, top_k=top_k)
            self.last_storage_metrics = {
                "storage_mode": "canonical_test",
                "blocks_materialized": len(blocks),
            }
            return [dict(item["item"], retrieval_score=item["score"]) for item in retrieved], query_ir

    generated_prompts = []

    def fake_llm(self, prompt):
        generated_prompts.append(prompt)
        return "$123,000"

    monkeypatch.setattr(scheduler_module, "SearchRouter", CanonicalSearchRouter)
    monkeypatch.setattr(ExecutionScheduler, "_call_llm", fake_llm)

    scheduler = ExecutionScheduler(str(tmp_path / "index"), str(tmp_path / "cache"))
    result = scheduler.answer("What is the budget of Project ORION?", use_cache=False, force_llm=True)

    assert result["answer"] == "$123,000"
    assert result["route"] == "LLM_COMPILED"
    assert result["metrics"]["storage_mode"] == "canonical_test"
    assert result["metrics"]["verifier_passed"] is True
    assert result["proof_ir"]["proof"]["status"] == "COMPLETE"
    assert "Project ORION approved budget is $123,000" in generated_prompts[0]
