import json


def test_long_conversation_quality_benchmark_fake_writes_validating_artifacts(tmp_path):
    from highway.benchmarks.long_conversation_quality import run_long_conversation_quality_benchmark

    result = run_long_conversation_quality_benchmark(
        output_dir=tmp_path / "long_conversation_quality",
        client="fake",
        turns=8,
        seed=42,
    )

    assert result["report_path"].exists()
    assert result["metrics_path"].exists()
    assert result["records_path"].exists()

    metrics = json.loads(result["metrics_path"].read_text(encoding="utf-8"))
    summary = metrics["summary"]

    assert summary["status"] == "VALIDATING"
    assert summary["answer_satisfies_question_rate"] >= 95.0
    assert summary["source_attribution_rate"] >= 95.0
    assert summary["hallucination_rate"] == 0.0
    assert summary["coherence_rate"] >= 95.0
    assert summary["avg_avoided_input_tokens_pct"] >= 80.0
    assert summary["output_over_budget_rate"] == 0.0

    records = [
        json.loads(line)
        for line in result["records_path"].read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert records
    assert any(record["query_rewrite_used"] for record in records)
    required = {
        "turn_index",
        "session_id",
        "active_entities",
        "query_rewrite_used",
        "context_reuse_rate",
        "answer_satisfies_question",
        "full_exact_match",
        "source_attribution_ok",
        "numeric_facts_ok",
        "entity_facts_ok",
        "hallucination_flag",
        "contradiction_flag",
        "output_tokens_budget",
        "output_budget_used_pct",
        "output_over_budget",
        "baseline_input_tokens",
        "highway_input_tokens",
        "avoided_input_tokens_pct",
        "context_latency_ms",
        "embedding_rows_scanned",
        "blocks_materialized",
        "bytes_read",
    }
    assert all(required.issubset(record) for record in records)


def test_long_conversation_retry_compacts_without_second_retrieval(tmp_path):
    from highway.benchmarks.long_conversation_quality import run_long_conversation_quality_benchmark

    class VerboseThenCompactClient:
        model_name = "verbose_then_compact"

        def __init__(self):
            self.calls = 0

        def answer(self, prompt, query_ir, evidence, expected_answer=None, expected_sources=(), answer_contract=None, **kwargs):
            self.calls += 1
            source = expected_sources[0] if expected_sources else evidence[0]["source_file"]
            raw = '{"answer":"%s","sources":["%s"],"reasoning":"%s","confidence":1.0}' % (
                expected_answer,
                source,
                "very long reasoning " * (20 if self.calls == 2 else 1),
            )
            input_tokens = max(1, len(prompt.split()))
            output_tokens = answer_contract.max_output_tokens + 5 if self.calls == 2 else 8
            return {
                "available": True,
                "model_name": self.model_name,
                "raw_text": raw,
                "answer": raw,
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "ttft_ms": 1.0,
                "decode_ms": 1.0,
                "total_ms": 2.0,
                "input_tokens_per_second": 1000.0,
                "output_tokens_per_second": 1000.0,
            }

    client = VerboseThenCompactClient()
    result = run_long_conversation_quality_benchmark(
        output_dir=tmp_path / "retry",
        client="fake",
        turns=1,
        seed=42,
        llm_client=client,
    )
    records = [
        json.loads(line)
        for line in result["records_path"].read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]

    assert result["summary"]["status"] == "VALIDATING"
    assert records[0]["first_pass_verdict"] == "OUTPUT_BUDGET_FAIL"
    assert records[0]["retry_used"] is True
    assert records[0]["final_verdict"] == "PASS"
    assert records[0]["retrieval_count_for_turn"] == 1
    assert records[0]["output_tokens_saved_by_retry"] > 0


def test_long_conversation_retry_hallucination_stays_non_validating(tmp_path):
    from highway.benchmarks.long_conversation_quality import run_long_conversation_quality_benchmark

    class HallucinatingRetryClient:
        model_name = "hallucinating_retry"

        def __init__(self):
            self.calls = 0

        def answer(self, prompt, query_ir, evidence, expected_answer=None, expected_sources=(), answer_contract=None, **kwargs):
            self.calls += 1
            if self.calls == 3:
                raw = '{"answer":"Project MADEUP","sources":["docs/missing.txt"],"reasoning":"","confidence":1.0}'
                output_tokens = 8
            else:
                source = expected_sources[0] if expected_sources else evidence[0]["source_file"]
                raw = '{"answer":"%s","sources":["%s"],"reasoning":"too long","confidence":1.0}' % (
                    expected_answer,
                    source,
                )
                output_tokens = answer_contract.max_output_tokens + 5
            input_tokens = max(1, len(prompt.split()))
            return {
                "available": True,
                "model_name": self.model_name,
                "raw_text": raw,
                "answer": raw,
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "ttft_ms": 1.0,
                "decode_ms": 1.0,
                "total_ms": 2.0,
                "input_tokens_per_second": 1000.0,
                "output_tokens_per_second": 1000.0,
            }

    result = run_long_conversation_quality_benchmark(
        output_dir=tmp_path / "hallucinating_retry",
        client="fake",
        turns=1,
        seed=42,
        llm_client=HallucinatingRetryClient(),
    )
    records = [
        json.loads(line)
        for line in result["records_path"].read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]

    assert result["summary"]["status"] == "NON_VALIDATING"
    assert records[0]["retry_used"] is True
    assert records[0]["final_verdict"] in {"HALLUCINATION_FAIL", "SOURCE_FAIL", "QUALITY_FAIL"}


def test_prompt_audit_writes_distinct_baseline_and_highway_prompts(tmp_path):
    from highway.benchmarks.long_conversation_quality import run_long_conversation_quality_benchmark

    result = run_long_conversation_quality_benchmark(
        output_dir=tmp_path / "audit",
        client="fake",
        turns=2,
        seed=42,
        audit_prompts=True,
    )
    records = [
        json.loads(line)
        for line in result["records_path"].read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]

    assert records
    for record in records:
        baseline_path = result["output_dir"] / record["baseline_prompt_path"]
        highway_path = result["output_dir"] / record["highway_prompt_path"]
        assert baseline_path.exists()
        assert highway_path.exists()
        assert baseline_path.read_text(encoding="utf-8") != highway_path.read_text(encoding="utf-8")
        assert record["baseline_prompt_hash"] != record["highway_prompt_hash"]
        assert record["prompt_pair_is_distinct"] is True


def test_records_include_prompt_hashes_and_block_counts(tmp_path):
    from highway.benchmarks.long_conversation_quality import run_long_conversation_quality_benchmark

    result = run_long_conversation_quality_benchmark(
        output_dir=tmp_path / "record_fields",
        client="fake",
        turns=1,
        seed=42,
        audit_prompts=True,
    )
    record = json.loads(result["records_path"].read_text(encoding="utf-8").splitlines()[0])

    assert record["baseline_context_block_count"] > record["highway_context_block_count"]
    assert record["baseline_prompt_tokens_verified"] > record["highway_prompt_tokens_verified"]
    assert record["highway_source_files"]
    assert record["highway_context_pack_block_ids"]
    assert record["highway_context_pack_sources"] == record["highway_source_files"]
    assert record["answer_contract_type"]
    assert record["answer_contract_budget"] > 0


def test_poison_missing_expected_source_makes_run_non_validating(tmp_path):
    from highway.benchmarks.long_conversation_quality import run_long_conversation_quality_benchmark

    result = run_long_conversation_quality_benchmark(
        output_dir=tmp_path / "poison",
        client="fake",
        turns=1,
        seed=42,
        audit_prompts=True,
        poison_context="missing_expected_source",
    )
    record = json.loads(result["records_path"].read_text(encoding="utf-8").splitlines()[0])

    assert result["summary"]["status"] == "NON_VALIDATING"
    assert record["poison_used"] is True
    assert record["poison_reason"] == "missing_expected_source"
    assert record["expected_source_removed"] is True
    assert record["final_verdict"] in {"SOURCE_FAIL", "QUALITY_FAIL", "LEAK_OR_BASELINE_CONTAMINATION_FAIL"}


def test_poison_does_not_call_retrieval_twice(tmp_path):
    from highway.benchmarks.long_conversation_quality import run_long_conversation_quality_benchmark

    result = run_long_conversation_quality_benchmark(
        output_dir=tmp_path / "poison_retrieval",
        client="fake",
        turns=2,
        seed=42,
        poison_context="missing_expected_source",
    )
    records = [
        json.loads(line)
        for line in result["records_path"].read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]

    assert records
    assert all(record["retrieval_count_for_turn"] == 1 for record in records)


def test_qwen_style_fake_long_12_turns_stays_validating(tmp_path):
    from highway.benchmarks.long_conversation_quality import run_long_conversation_quality_benchmark

    result = run_long_conversation_quality_benchmark(
        output_dir=tmp_path / "fake_12",
        client="fake",
        turns=12,
        seed=42,
        audit_prompts=True,
    )

    assert result["summary"]["status"] == "VALIDATING"
    assert result["summary"]["prompt_pair_is_distinct_rate"] == 100.0
    assert result["summary"]["avg_highway_blocks"] < result["summary"]["avg_baseline_blocks"]
