import json


def _fake_rows():
    return [
        {
            "id": "case_1",
            "question": "What error code did the gateway return?",
            "documents": [
                "Gateway incident report: the active error code is HTTP 502 and the resolution is retry upstream.",
                "Old gateway note: HTTP 500 was retired and must not be used.",
            ],
            "response": "HTTP 502",
            "sentence_support_information": [
                {
                    "fully_supported": True,
                    "supporting_sentence_keys": ["0:0"],
                    "all_utilized_sentence_keys": ["0:0"],
                }
            ],
            "relevance_score": 1.0,
            "utilization_score": 1.0,
            "completeness_score": 1.0,
        },
        {
            "id": "case_2",
            "question": "Which function evicts stale cache entries?",
            "documents": [
                "Cache source summary: function evict_lru removes stale cache entries.",
                "Noise source summary: function connect_db opens a database connection.",
            ],
            "response": "evict_lru",
            "sentence_support_information": [
                {
                    "fully_supported": True,
                    "supporting_sentence_keys": ["0:0"],
                    "all_utilized_sentence_keys": ["0:0"],
                }
            ],
            "relevance_score": 0.9,
            "utilization_score": 0.8,
            "completeness_score": 0.7,
        },
    ]


def test_normalizes_ragbench_rows_with_source_hashes_and_safe_source_files():
    from highway.benchmarks.ragbench_headroom_like import normalize_ragbench_rows

    cases = normalize_ragbench_rows(_fake_rows(), config_name="techqa", limit=2, seed=42)

    assert [case.question for case in cases] == [
        "What error code did the gateway return?",
        "Which function evicts stale cache entries?",
    ]
    assert cases[0].expected_answer == "HTTP 502"
    assert cases[0].expected_sources == ["ragbench/techqa/case_1/doc_0.txt"]
    assert cases[0].blocks[0]["source_hash"]
    assert "HTTP 502" not in cases[0].blocks[0]["source_file"]
    assert cases[0].ragbench_scores["relevance_score"] == 1.0


def test_normalization_is_deterministic_for_same_seed():
    from highway.benchmarks.ragbench_headroom_like import normalize_ragbench_rows

    first = normalize_ragbench_rows(_fake_rows(), config_name="techqa", limit=2, seed=7)
    second = normalize_ragbench_rows(_fake_rows(), config_name="techqa", limit=2, seed=7)

    assert [case.to_dict() for case in first] == [case.to_dict() for case in second]


def test_ragbench_fake_benchmark_is_validating_and_writes_audit(tmp_path):
    from highway.benchmarks.ragbench_headroom_like import run_highway_ragbench_benchmark

    result = run_highway_ragbench_benchmark(
        output_dir=tmp_path / "ragbench",
        client="fake",
        configs=("techqa",),
        examples_per_config=2,
        seed=42,
        audit_prompts=True,
        offline_rows={"techqa": _fake_rows()},
    )
    records = [
        json.loads(line)
        for line in result["records_path"].read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]

    assert result["summary"]["status"] == "VALIDATING"
    assert result["summary"]["prompt_pair_is_distinct_rate"] == 100.0
    assert result["summary"]["source_hash_present_rate"] == 100.0
    assert result["summary"]["tokens_per_correct_grounded_answer"] > 0
    assert result["summary"]["cost_avoided_per_1000_requests"] >= 0
    assert (result["output_dir"] / records[0]["baseline_prompt_path"]).exists()
    assert (result["output_dir"] / records[0]["highway_prompt_path"]).exists()
    assert records[0]["baseline_prompt_hash"] != records[0]["highway_prompt_hash"]
    assert records[0]["blocks_baseline"] > records[0]["blocks_highway"]


def test_ragbench_poison_missing_expected_source_is_non_validating(tmp_path):
    from highway.benchmarks.ragbench_headroom_like import run_highway_ragbench_benchmark

    result = run_highway_ragbench_benchmark(
        output_dir=tmp_path / "poison",
        client="fake",
        configs=("techqa",),
        examples_per_config=2,
        seed=42,
        audit_prompts=True,
        poison_context="missing_expected_source",
        offline_rows={"techqa": _fake_rows()},
    )
    records = [
        json.loads(line)
        for line in result["records_path"].read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]

    assert result["summary"]["status"] == "NON_VALIDATING"
    assert result["summary"]["poison_fail_rate"] == 100.0
    assert all(record["expected_source_removed"] for record in records)
