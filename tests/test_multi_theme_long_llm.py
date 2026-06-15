import json


def test_multi_theme_generator_is_deterministic():
    from highway.benchmarks.multi_theme_long_llm import build_multi_theme_workload

    first = build_multi_theme_workload(turns=500, seed=42)
    second = build_multi_theme_workload(turns=500, seed=42)

    assert first["turns"] == second["turns"]
    assert len(first["turns"]) == 500
    assert {turn["theme"] for turn in first["turns"]} >= {
        "dev/code",
        "infra/logs",
        "produit/tickets",
        "finance/budgets",
        "planning/deadlines",
        "recherche/docs techniques",
    }


def test_multi_theme_turns_have_required_metadata():
    from highway.benchmarks.multi_theme_long_llm import build_multi_theme_workload

    workload = build_multi_theme_workload(turns=100, seed=42)
    required = {
        "question",
        "expected_answer",
        "expected_sources",
        "theme",
        "difficulty",
        "active_entity",
        "turn_type",
    }

    assert all(required.issubset(turn) for turn in workload["turns"])
    assert all(turn["expected_sources"] for turn in workload["turns"])


def test_multi_theme_long_range_recall_distance_is_recorded(tmp_path):
    from highway.benchmarks.multi_theme_long_llm import run_multi_theme_long_llm_benchmark

    result = run_multi_theme_long_llm_benchmark(
        output_dir=tmp_path / "multi_fake",
        client="fake",
        turns=100,
        seed=42,
        audit_prompts=True,
    )
    records = [
        json.loads(line)
        for line in result["records_path"].read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]

    assert result["summary"]["status"] == "VALIDATING"
    assert any(record["long_range_recall_distance"] >= 50 for record in records)
    assert result["summary"]["long_range_recall_success_rate"] >= 85.0


def test_multi_theme_prompt_audit_and_poison(tmp_path):
    from highway.benchmarks.multi_theme_long_llm import run_multi_theme_long_llm_benchmark

    clean = run_multi_theme_long_llm_benchmark(
        output_dir=tmp_path / "clean",
        client="fake",
        turns=20,
        seed=42,
        audit_prompts=True,
    )
    clean_record = json.loads(clean["records_path"].read_text(encoding="utf-8").splitlines()[0])

    assert clean_record["prompt_pair_is_distinct"] is True
    assert (clean["output_dir"] / clean_record["baseline_prompt_path"]).exists()
    assert (clean["output_dir"] / clean_record["highway_prompt_path"]).exists()
    assert clean_record["baseline_context_block_count"] > clean_record["highway_context_block_count"]

    poison = run_multi_theme_long_llm_benchmark(
        output_dir=tmp_path / "poison",
        client="fake",
        turns=100,
        seed=42,
        audit_prompts=True,
        poison_context="missing_expected_source",
    )
    poison_records = [
        json.loads(line)
        for line in poison["records_path"].read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]

    assert poison["summary"]["status"] == "NON_VALIDATING"
    assert any(record["expected_source_removed"] for record in poison_records)
    assert all(record["retrieval_count_for_turn"] == 1 for record in poison_records)
