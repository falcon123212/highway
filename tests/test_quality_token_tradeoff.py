import json


def test_quality_token_tradeoff_smoke_compares_baseline_and_highway(tmp_path):
    from highway.benchmarks.quality_token_tradeoff import run_quality_token_tradeoff_smoke

    result = run_quality_token_tradeoff_smoke(
        output_dir=tmp_path / "quality_token_tradeoff",
        total_blocks=80,
        query_count=8,
        seed=42,
    )

    assert result["report_path"].exists()
    assert result["metrics_path"].exists()
    metrics = json.loads(result["metrics_path"].read_text(encoding="utf-8"))
    summary = metrics["summary"]

    assert summary["count"] == 8
    assert summary["baseline_em"] == 100.0
    assert summary["highway_em"] == 100.0
    assert summary["quality_delta_pp"] == 0.0
    assert summary["avg_highway_prompt_tokens"] < summary["avg_baseline_prompt_tokens"]
    assert summary["avg_prompt_tokens_avoided_pct"] > 0.0
    assert summary["avg_highway_output_tokens"] > 0
    assert summary["avg_baseline_output_tokens"] > 0

    for record in metrics["records"]:
        assert "reasoning" in record["baseline_response"]
        assert "answer" in record["baseline_response"]
        assert "reasoning" in record["highway_response"]
        assert "answer" in record["highway_response"]
        assert record["baseline_is_em"] is True
        assert record["highway_is_em"] is True

    report = result["report_path"].read_text(encoding="utf-8")
    assert "Quality Token Tradeoff Smoke" in report
    assert "Quality delta" in report
