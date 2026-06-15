import json


def test_highway_llm_runtime_build_prompt_uses_only_context_pack_blocks():
    from highway.runtime.context_engine import ContextBlock, ContextPack, ContextRequest
    from highway.runtime.llm_runtime import HighwayLLMRuntime

    pack = ContextPack(
        request=ContextRequest(user_turn="Where is Project ALPHA?"),
        blocks=[
            ContextBlock(
                block_id="block_keep",
                source_file="docs/keep.txt",
                text="Project ALPHA is in Paris.",
                score=1.0,
                reason="test",
            )
        ],
        query_ir={"intent": "lookup"},
        metrics={"latency_ms": 2.0},
        warnings=[],
    )

    prompt = HighwayLLMRuntime(None).build_prompt(pack)

    assert "Project ALPHA is in Paris." in prompt
    assert "Where is Project ALPHA?" in prompt
    assert "block_keep" in prompt
    assert "SHOULD_NOT_APPEAR" not in prompt


def test_deterministic_reflective_client_reports_tokens_latency_and_answer():
    from highway.runtime.llm_runtime import DeterministicReflectiveClient

    client = DeterministicReflectiveClient(prefill_tokens_per_ms=10.0, decode_tokens_per_ms=0.5)
    result = client.answer(
        prompt="one two three four",
        query_ir={"operation": "comparison"},
        evidence=[],
        expected_answer="Project X",
    )

    assert result["answer"] == "Project X"
    assert result["reasoning"]
    assert result["input_tokens"] == 4
    assert result["output_tokens"] > 0
    assert result["ttft_ms"] == 1.0
    assert result["decode_ms"] >= 1.0
    assert result["total_ms"] == result["ttft_ms"] + result["decode_ms"]
    assert result["input_tokens_per_second"] > 0
    assert result["output_tokens_per_second"] > 0
    assert result["model_name"] == "deterministic_reflective_fake"


def test_highway_llm_runtime_answer_with_client_returns_pack_and_economics(tmp_path, monkeypatch):
    from highway.runtime.context_engine import ContextBlock, ContextPack, ContextRequest
    from highway.runtime.llm_runtime import DeterministicReflectiveClient, HighwayLLMRuntime
    from highway.runtime.token_economics import ModelProfile

    class FakeEngine:
        model_profile = ModelProfile(name="fake", layers=2, hidden_size=8)
        input_cost_per_million = 1.0
        output_cost_per_million = 2.0

        def retrieve(self, request, top_k=50, session_state=None):
            return ContextPack(
                request=request,
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
                metrics={"latency_ms": 3.0, "embedding_rows_scanned": 1, "blocks_materialized": 1},
                warnings=[],
            )

    runtime = HighwayLLMRuntime(FakeEngine())
    result = runtime.answer_with_client(
        ContextRequest(user_turn="Which project has the largest budget?"),
        DeterministicReflectiveClient(),
        baseline_context=[{"block_id": "b0", "source_file": "docs/full.txt", "text": "x " * 100}],
        expected_answer="Project X",
    )

    assert result["response"]["answer"] == "Project X"
    assert result["context_pack"]["blocks"][0]["block_id"] == "b1"
    assert result["token_economics"]["baseline_input_tokens"] > result["token_economics"]["actual_input_tokens"]
    assert result["token_economics"]["avoided_input_tokens"] > 0
    assert result["token_economics"]["output_tokens"] == result["response"]["output_tokens"]
    assert result["metrics"]["input_tokens_per_second"] > 0
    assert result["metrics"]["output_tokens_per_second"] > 0


def test_llm_runtime_fake_benchmark_keeps_quality_while_avoiding_tokens(tmp_path):
    from highway.benchmarks.llm_runtime_fake import run_llm_runtime_fake_benchmark

    result = run_llm_runtime_fake_benchmark(
        output_dir=tmp_path / "llm_runtime_fake",
        sizes=[80],
        query_count=8,
        seed=42,
        strategy="ooc_marker_entity_pruned",
    )

    assert result["report_path"].exists()
    assert result["metrics_path"].exists()
    assert result["records_path"].exists()

    metrics = json.loads(result["metrics_path"].read_text(encoding="utf-8"))
    summary = metrics["summary"]
    tier = summary["tiers"][0]

    assert summary["status"] == "VALIDATING"
    assert tier["baseline_em"] == 100.0
    assert tier["highway_em"] == 100.0
    assert tier["quality_delta_pp"] == 0.0
    assert tier["avg_avoided_input_tokens_pct"] >= 80.0
    assert tier["metrics_complete_rate"] == 100.0
    assert tier["context_p95_ms"] <= 100.0

    records = [
        json.loads(line)
        for line in result["records_path"].read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert records
    required = {
        "baseline_answer",
        "highway_answer",
        "expected_answer",
        "baseline_is_em",
        "highway_is_em",
        "quality_delta",
        "baseline_input_tokens",
        "highway_input_tokens",
        "avoided_input_tokens",
        "avoided_input_tokens_pct",
        "baseline_output_tokens",
        "highway_output_tokens",
        "baseline_ttft_ms",
        "highway_ttft_ms",
        "baseline_input_tokens_per_second",
        "highway_input_tokens_per_second",
        "baseline_output_tokens_per_second",
        "highway_output_tokens_per_second",
        "kv_bytes_estimated",
        "kv_bytes_avoided_estimated",
        "cost_estimated_usd",
        "cost_avoided_estimated_usd",
        "context_latency_ms",
        "embedding_rows_scanned",
        "blocks_materialized",
        "bytes_read",
        "ann_used",
        "ann_backend",
    }
    assert all(required.issubset(record) for record in records)

    report = result["report_path"].read_text(encoding="utf-8")
    assert "Verdict: VALIDATING" in report
    assert "Why this matters" in report
