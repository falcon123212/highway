import json


def test_answer_context_pack_does_not_call_retrieve():
    from highway.runtime.context_engine import ContextBlock, ContextPack, ContextRequest
    from highway.runtime.llm_runtime import DeterministicReflectiveClient, HighwayLLMRuntime
    from highway.runtime.token_economics import ModelProfile

    class ExplodingEngine:
        model_profile = ModelProfile(name="fake", layers=2, hidden_size=8)
        input_cost_per_million = 1.0
        output_cost_per_million = 2.0

        def retrieve(self, *args, **kwargs):
            raise AssertionError("retrieve must not be called")

    pack = ContextPack(
        request=ContextRequest(user_turn="Which project wins?"),
        blocks=[
            ContextBlock(
                block_id="b1",
                source_file="docs/a.txt",
                text="Project X has the largest budget.",
                score=1.0,
                reason="test",
            )
        ],
        query_ir={"operation": "comparison"},
        metrics={
            "baseline_input_tokens_estimated": 100,
            "latency_ms": 3.0,
            "embedding_rows_scanned": 1,
            "blocks_materialized": 1,
        },
        warnings=[],
    )

    runtime = HighwayLLMRuntime(ExplodingEngine())
    result = runtime.answer_context_pack(
        pack,
        DeterministicReflectiveClient(),
        baseline_context=[{"text": "x " * 200}],
        expected_answer="Project X",
    )

    assert result["response"]["answer"] == "Project X"
    assert result["context_pack"]["blocks"][0]["block_id"] == "b1"
    assert result["token_economics"]["avoided_input_tokens"] > 0


def test_runtime_perf_margin_benchmark_reports_structured_exact_metrics(tmp_path):
    from highway.benchmarks.runtime_perf_margin import run_runtime_perf_margin_benchmark

    result = run_runtime_perf_margin_benchmark(
        output_dir=tmp_path / "runtime_perf_margin",
        sizes=[80],
        query_count=8,
        seed=42,
    )

    metrics = json.loads(result["metrics_path"].read_text(encoding="utf-8"))
    tier = metrics["summary"]["tiers"][0]

    assert result["report_path"].exists()
    assert metrics["summary"]["status"] == "VALIDATING"
    assert tier["workload_type"] == "structured_exact"
    assert tier["context_p95_ms"] <= 50.0
    assert tier["runtime_p95_ms"] > 0.0
    assert tier["metrics_complete_rate"] == 100.0
    assert tier["avg_hotset_hits"] >= 0.0
    assert tier["avg_tokens_avoided"] > 0.0
    assert tier["avg_kv_bytes_avoided_estimated"] > 0.0


def test_semantic_ann_quality_can_mark_non_validating_when_recall_gate_is_not_met(tmp_path):
    from highway.benchmarks.semantic_ann_quality import run_semantic_ann_quality_benchmark

    result = run_semantic_ann_quality_benchmark(
        output_dir=tmp_path / "semantic_ann_quality",
        sizes=[80],
        query_count=8,
        seed=42,
        recall_gate=101.0,
    )

    metrics = json.loads(result["metrics_path"].read_text(encoding="utf-8"))
    summary = metrics["summary"]
    tier = summary["tiers"][0]

    assert result["report_path"].exists()
    assert summary["status"] == "NON_VALIDATING"
    assert tier["workload_type"] == "semantic_ann"
    assert {"ooc_full_scan", "ooc_ann_hnsw", "ooc_ann_pruned_hybrid"}.issubset(tier["strategies"])
    assert tier["hnsw_recall_at_k"] < 101.0
    assert tier["metrics_complete_rate"] == 100.0
    for strategy_summary in tier["strategy_summaries"]:
        assert strategy_summary["p95_latency_ms"] > 0.0

    records = [
        json.loads(line)
        for line in result["records_path"].read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert records
    assert all(record["workload_type"] == "semantic_ann" for record in records)
