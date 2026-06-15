import json


def test_token_economics_smoke_writes_report_and_metrics(tmp_path):
    from highway.benchmarks.token_economics_smoke import run_token_economics_smoke

    result = run_token_economics_smoke(
        output_dir=tmp_path / "token_economics_smoke",
        total_blocks=80,
        query_count=8,
        seed=42,
    )

    assert result["report_path"].exists()
    assert result["metrics_path"].exists()
    metrics = json.loads(result["metrics_path"].read_text(encoding="utf-8"))
    summary = metrics["summary"]

    assert summary["count"] == 8
    assert summary["avg_actual_input_tokens"] > 0
    assert summary["avg_baseline_input_tokens"] > summary["avg_actual_input_tokens"]
    assert summary["avg_avoided_input_tokens"] > 0
    assert summary["avg_avoided_input_tokens_pct"] > 0.0
    assert summary["avg_kv_bytes_avoided_estimated"] > 0
    assert summary["p95_latency_ms"] >= 0.0
    assert all(record["metrics"]["token_economics"]["output_tokens"] == 0 for record in metrics["records"])

    report = result["report_path"].read_text(encoding="utf-8")
    assert "Token Economics Smoke" in report
    assert "Average avoided input tokens" in report
