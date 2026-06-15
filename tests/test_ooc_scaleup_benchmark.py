import json
import re
from pathlib import Path


def _read_jsonl(path: Path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def test_scaleup_generation_is_deterministic_and_writes_ooc_manifest(tmp_path):
    from highway.benchmarks.ooc_scaleup import generate_scaleup_dataset
    from highway.storage.out_of_core_index import OutOfCoreIndex

    first = generate_scaleup_dataset(tmp_path / "first", total_blocks=64, query_count=6, seed=42)
    second = generate_scaleup_dataset(tmp_path / "second", total_blocks=64, query_count=6, seed=42)

    first_workload = _read_jsonl(first.workload_path)
    second_workload = _read_jsonl(second.workload_path)
    assert first_workload == second_workload

    assert OutOfCoreIndex.is_out_of_core_index(first.index_dir)
    manifest = json.loads((first.index_dir / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["num_blocks"] == 64

    for query in first_workload:
        assert re.fullmatch(r"q_[0-9a-f]{16}", query["id"])
        assert "g_adv" not in query["id"]
        assert "h_adv" not in query["id"]
        source_file = query["source_file"]
        assert query["id"] not in source_file
        assert (first.corpus_dir / "documents" / source_file).exists()


def test_marker_entity_pruned_strategy_scans_fewer_rows_with_same_em(tmp_path):
    from highway.benchmarks.ooc_scaleup import generate_scaleup_dataset, run_scaleup_benchmark

    dataset = generate_scaleup_dataset(tmp_path / "dataset", total_blocks=80, query_count=8, seed=7)

    full = run_scaleup_benchmark(
        index_dir=dataset.index_dir,
        workload_path=dataset.workload_path,
        output_dir=tmp_path / "full",
        strategy="ooc_full_scan",
        query_limit=8,
        top_k=20,
    )
    pruned = run_scaleup_benchmark(
        index_dir=dataset.index_dir,
        workload_path=dataset.workload_path,
        output_dir=tmp_path / "pruned",
        strategy="ooc_marker_entity_pruned",
        query_limit=8,
        top_k=20,
    )

    assert full.summary["gh_em_global"] == 100.0
    assert pruned.summary["gh_em_global"] == 100.0
    assert pruned.summary["avg_embedding_rows_scanned"] < full.summary["avg_embedding_rows_scanned"]
    assert pruned.summary["avg_blocks_materialized"] < full.summary["avg_blocks_materialized"]
    assert pruned.summary["ooc_metrics_coverage"] == 100.0


def test_candidate_cap_sweep_writes_report_and_metrics(tmp_path):
    from highway.benchmarks.ooc_scaleup import generate_scaleup_dataset, run_candidate_cap_sweep

    dataset = generate_scaleup_dataset(tmp_path / "dataset", total_blocks=60, query_count=4, seed=11)
    result = run_candidate_cap_sweep(
        index_dir=dataset.index_dir,
        workload_path=dataset.workload_path,
        output_dir=tmp_path / "sweep",
        candidate_caps=[8, 32],
        query_limit=4,
        top_k=10,
    )

    assert result.metrics_path.exists()
    assert result.report_path.exists()
    metrics = json.loads(result.metrics_path.read_text(encoding="utf-8"))
    assert [entry["candidate_cap"] for entry in metrics["candidate_cap_sweep"]] == [8, 32]
    assert all(entry["gh_em_global"] == 100.0 for entry in metrics["candidate_cap_sweep"])
    assert "OOC Scale-Up Benchmark" in result.report_path.read_text(encoding="utf-8")


def test_legacy_memory_baseline_is_reproducible_on_small_tier(tmp_path):
    from highway.benchmarks.ooc_scaleup import generate_scaleup_dataset, run_legacy_memory_benchmark

    dataset = generate_scaleup_dataset(tmp_path / "dataset", total_blocks=40, query_count=4, seed=17)
    result = run_legacy_memory_benchmark(
        index_dir=dataset.index_dir,
        workload_path=dataset.workload_path,
        output_dir=tmp_path / "legacy",
        query_limit=4,
        top_k=10,
    )

    assert result.summary["gh_em_global"] == 100.0
    assert result.summary["no_leak_pass_rate"] == 100.0
    assert result.summary["avg_embedding_rows_scanned"] == 40.0
    assert result.summary["avg_blocks_materialized"] == 40.0


def test_mixed_query_set_includes_non_marker_queries_without_losing_em(tmp_path):
    from highway.benchmarks.ooc_scaleup import generate_scaleup_dataset, run_scaleup_benchmark

    dataset = generate_scaleup_dataset(
        tmp_path / "dataset",
        total_blocks=80,
        query_count=6,
        seed=23,
        mixed_query_set="marker,entity,semantic",
    )
    workload = _read_jsonl(dataset.workload_path)
    assert any("reference ref_" not in row["question"].lower() for row in workload)

    result = run_scaleup_benchmark(
        index_dir=dataset.index_dir,
        workload_path=dataset.workload_path,
        output_dir=tmp_path / "mixed",
        strategy="ooc_full_scan",
        query_limit=6,
        top_k=20,
    )

    assert result.summary["gh_em_global"] == 100.0
    assert result.summary["no_leak_pass_rate"] == 100.0


def test_ann_scaleup_suite_reports_fallback_when_faiss_is_absent(tmp_path):
    from highway.benchmarks.ooc_scaleup import run_scaleup_suite

    metrics = run_scaleup_suite(
        output_dir=tmp_path / "ann",
        sizes=[80],
        query_count=6,
        seed=29,
        strategy="all",
        top_k=20,
        max_candidates=20,
        ann_backends=["faiss_hnsw"],
        mixed_query_set="marker,entity,semantic",
    )

    strategies = {summary["strategy"] for summary in metrics["summaries"]}
    assert "ooc_ann_hnsw" in strategies
    assert "ooc_ann_pruned_hybrid" in strategies
    assert (tmp_path / "ann" / "report.md").exists()
    assert (tmp_path / "ann" / "metrics.json").exists()
