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
            "all_utilized_sentence_keys": ["0a"],
            "all_relevant_sentence_keys": ["0a"],
            "sentence_support_information": [
                {
                    "fully_supported": True,
                    "supporting_sentence_keys": ["0a"],
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
            "all_utilized_sentence_keys": ["0a"],
            "all_relevant_sentence_keys": ["0a"],
            "sentence_support_information": [
                {
                    "fully_supported": True,
                    "supporting_sentence_keys": ["0a"],
                }
            ],
            "relevance_score": 0.9,
            "utilization_score": 0.8,
            "completeness_score": 0.7,
        },
    ]


def test_normalizes_ministress_rows_properly():
    from highway.benchmarks.ragbench_ministress import normalize_ragbench_rows

    cases = normalize_ragbench_rows(_fake_rows(), config_name="techqa", limit=2, seed=42)

    assert [case.question for case in cases] == [
        "What error code did the gateway return?",
        "Which function evicts stale cache entries?",
    ]
    assert cases[0].expected_answer == "HTTP 502"
    assert cases[0].expected_sources == ["techqa/case_1/doc_0"]
    assert cases[0].utilized_sentence_keys == ["techqa/case_1/doc_0/a"]
    assert cases[0].relevant_sentence_keys == ["techqa/case_1/doc_0/a"]


def test_ministress_bm25_top_sentences():
    from highway.benchmarks.ragbench_ministress import normalize_ragbench_rows, get_bm25_top_sentences

    cases = normalize_ragbench_rows(_fake_rows(), config_name="techqa", limit=2, seed=42)
    top_sents = get_bm25_top_sentences(cases[0], top_n=1)

    assert len(top_sents) == 1
    # doc_idx, sent_idx, text
    assert top_sents[0][0] == 0
    assert "HTTP 502" in top_sents[0][2]


def test_ministress_fake_benchmark_runs_end_to_end(tmp_path):
    from highway.benchmarks.ragbench_ministress import run_ministress_benchmark

    result = run_ministress_benchmark(
        output_dir=tmp_path / "ministress",
        client="fake",
        configs=("techqa",),
        examples_per_config=2,
        seed=42,
        offline_rows={"techqa": _fake_rows()},
    )
    records = [
        json.loads(line)
        for line in result["records_path"].read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]

    assert result["summary"]["status"] == "VALIDATING"
    assert len(records) > 0
    assert "mode" in records[0]
    assert "budget" in records[0]
    assert "top_m" in records[0]
    
    # Verify Step 1 metrics
    assert "highway_local_utilized_recall" in result["summary"]
    assert result["summary"]["highway_local_utilized_recall"] == 100.0
    assert result["summary"]["bm25_local_utilized_recall"] == 100.0
    
    # Verify Step 2 metrics
    assert result["summary"]["full_local_answer_correctness"] == 100.0
    assert result["summary"]["highway_local_answer_correctness"] == 100.0
    assert result["summary"]["bm25_local_answer_correctness"] == 100.0
    
    # Verify Step 3 metrics
    assert "highway_local_poison_false_validation_rate" in result["summary"]
    assert result["summary"]["highway_local_poison_false_validation_rate"] == 0.0

    # Verify newly added metrics (prefixed & explicit)
    assert "full_local_grounded_success_rate" in result["summary"]
    assert "highway_local_grounded_success_rate" in result["summary"]
    assert "bm25_local_grounded_success_rate" in result["summary"]
    
    assert "grounded_success_rate" in result["summary"]
    assert "avg_input_tokens" in result["summary"]
    assert "avg_input_tokens_ratio" in result["summary"]
    assert "tokens_per_attempted_grounded_success" in result["summary"]
    assert "tokens_per_correct_only" in result["summary"]
    assert "poison_on_initially_valid_cases" in result["summary"]
    
    assert "ratio_of_averages" in result["summary"]
    assert "mean_of_case_ratios" in result["summary"]
    
    # Verify global retrieval metrics
    assert "highway_global_case_hit_rate" in result["summary"]
    assert "highway_global_doc_hit_rate" in result["summary"]
    assert "highway_global_support_sentence_recall" in result["summary"]
    assert "highway_global_distractor_selection_rate" in result["summary"]

    # Verify report layout
    report_text = result["report_path"].read_text(encoding="utf-8")
    assert "Performance Table Comparison" in report_text
    assert "Full local" in report_text
    assert "BM25 local" in report_text
    assert "Highway local" in report_text
    assert "ratio_of_averages" in report_text
    assert "mean_of_case_ratios" in report_text
    assert "Document Aggregation Strategy Sweep" in report_text

    # Verify highway_pruned_local mode
    assert "highway_pruned_local_grounded_success_rate" in result["summary"]
    assert "highway_pruned_local_utilized_recall" in result["summary"]
    assert "highway_pruned_local_input_tokens_avg" in result["summary"]

    # Verify highway_pruned_global mode
    assert "highway_pruned_global_grounded_success_rate" in result["summary"]
    assert "highway_pruned_global_utilized_recall" in result["summary"]
    assert "highway_pruned_global_input_tokens_avg" in result["summary"]

    # Verify highway_pruned_global_bm25_stage1 mode
    assert "highway_pruned_global_bm25_stage1_grounded_success_rate" in result["summary"]
    assert "highway_pruned_global_bm25_stage1_utilized_recall" in result["summary"]
    assert "highway_pruned_global_bm25_stage1_input_tokens_avg" in result["summary"]

    # Verify aggregation sweep summary
    assert "aggregation_sweep_summary" in result["summary"]
    assert "hybrid_sum_score" in result["summary"]["aggregation_sweep_summary"]
    assert "bm25_bm25_doc_score + max_sentence_score" in result["summary"]["aggregation_sweep_summary"]

    # Verify diagnostic gates
    assert "diagnostic_gates" in result["summary"]
    assert "grounded_success_ge_88" in result["summary"]["diagnostic_gates"]
    assert "global_grounded_success_ge_85" in result["summary"]["diagnostic_gates"]
    assert "global_bm25_stage1_grounded_success_ge_70" in result["summary"]["diagnostic_gates"]

    # Verify poison N tracking
    assert "highway_pruned_local_poison_initially_valid_n" in result["summary"]
    assert "highway_pruned_local_poison_false_validation_count" in result["summary"]
    assert "highway_pruned_global_poison_initially_valid_n" in result["summary"]
    assert "highway_pruned_global_poison_false_validation_count" in result["summary"]
    assert "highway_pruned_global_bm25_stage1_poison_initially_valid_n" in result["summary"]
    assert "highway_pruned_global_bm25_stage1_poison_false_validation_count" in result["summary"]
