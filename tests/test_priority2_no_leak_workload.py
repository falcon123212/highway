import importlib
import json
import re
import sys
from pathlib import Path

from highway.retrieval.evidence_resolver import EvidenceResolver
from highway.retrieval.query_parser import QueryParser


def _read_jsonl(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def test_workload_generation_is_reproducible_and_opaque(tmp_path):
    from highway.workloads.build_poc234_kernel_hardening_workload import generate_workload

    corpus_dir = tmp_path / "corpus"
    index_dir = corpus_dir / "index"
    output_a = tmp_path / "artifacts/runs" / "workload_a.jsonl"
    output_b = tmp_path / "artifacts/runs" / "workload_b.jsonl"

    first = generate_workload(
        corpus=str(index_dir),
        output=str(output_a),
        n_comparison=6,
        n_aggregation=6,
        seed=42,
        run_ingest=False,
        write_mixed=False,
    )
    second = generate_workload(
        corpus=str(index_dir),
        output=str(output_b),
        n_comparison=6,
        n_aggregation=6,
        seed=42,
        run_ingest=False,
        write_mixed=False,
    )

    assert first == second
    assert _read_jsonl(output_a) == _read_jsonl(output_b)

    for query in first:
        q_id = query["id"]
        source_file = query["source_file"]
        assert re.fullmatch(r"q_[0-9a-f]{16}", q_id)
        assert not re.search(r"(g_adv|h_adv|adv_doc|_\d{3}$)", q_id)
        assert q_id not in source_file
        assert not re.search(r"(g_adv|h_adv|adv_doc_\d+)", source_file)

        doc_path = corpus_dir / "documents" / Path(source_file)
        assert doc_path.exists()
        assert q_id not in doc_path.read_text(encoding="utf-8")


def test_generation_does_not_delete_existing_noise_files(tmp_path):
    from highway.workloads.build_poc234_kernel_hardening_workload import generate_workload

    corpus_dir = tmp_path / "corpus"
    index_dir = corpus_dir / "index"
    sentinel = corpus_dir / "documents" / "noise" / "keep_me.txt"
    sentinel.parent.mkdir(parents=True)
    sentinel.write_text("existing noise file must survive", encoding="utf-8")

    generate_workload(
        corpus=str(index_dir),
        output=str(tmp_path / "artifacts/runs" / "workload.jsonl"),
        n_comparison=2,
        n_aggregation=2,
        seed=7,
        run_ingest=False,
        write_mixed=False,
    )

    assert sentinel.read_text(encoding="utf-8") == "existing noise file must survive"


def test_generated_questions_use_content_marker_not_ids_for_retrieval(tmp_path):
    from highway.workloads.build_poc234_kernel_hardening_workload import generate_workload

    corpus_dir = tmp_path / "corpus"
    queries = generate_workload(
        corpus=str(corpus_dir / "index"),
        output=str(tmp_path / "artifacts/runs" / "workload.jsonl"),
        n_comparison=1,
        n_aggregation=1,
        seed=99,
        run_ingest=False,
        write_mixed=False,
    )

    for query in queries:
        marker_match = re.search(r"\bref_[0-9a-f]{10}\b", query["question"])
        assert marker_match
        marker = marker_match.group(0)
        doc_path = corpus_dir / "documents" / Path(query["source_file"])
        doc_text = doc_path.read_text(encoding="utf-8")

        assert marker in doc_text
        assert marker not in query["id"]
        assert marker not in Path(query["source_file"]).name


def test_query_parser_extracts_reference_marker_constraint():
    parser = QueryParser(["AURORA", "BEACON"])

    parsed = parser.parse("In reference ref_1234abcd56 which project has a higher budget: Project AURORA or Project BEACON?")

    assert parsed["constraints"]["reference_marker"] == "ref_1234abcd56"


def test_evidence_resolver_filters_candidates_by_reference_marker():
    resolver = EvidenceResolver()
    query_ir = {
        "question": "In reference ref_1234abcd56 which project has a higher budget: Project AURORA or Project BEACON?",
        "target_entities": ["AURORA", "BEACON"],
        "intent": "comparison",
        "constraints": {"reference_marker": "ref_1234abcd56"},
    }
    candidates = [
        {
            "block_id": "old",
            "source_file": "reports/aurora_status.txt",
            "text": "Project AURORA budget is $100,000. Project BEACON budget is $900,000.",
        },
        {
            "block_id": "marked",
            "source_file": "noise/poc234_42/doc_abcdabcdabcdabcd.txt",
            "text": "Reference ref_1234abcd56 contains evidence. Project AURORA budget is $900,000. Project BEACON budget is $100,000.",
        },
    ]

    active, suppressed, forbidden = resolver.resolve(candidates, query_ir)

    assert [b["block_id"] for b in active] == ["marked"]
    assert forbidden == []


def test_evidence_resolver_keeps_marked_aggregation_block_for_negative_evidence():
    resolver = EvidenceResolver()
    query_ir = {
        "question": "In reference ref_abcdef1234 list all project names managed by Thomas Petit.",
        "target_entities": ["Thomas Petit"],
        "intent": "aggregation",
        "constraints": {"reference_marker": "ref_abcdef1234"},
    }
    candidates = [
        {
            "block_id": "unmarked",
            "source_file": "reports/thomas.txt",
            "text": "Project AURORA is managed by Thomas Petit.",
        },
        {
            "block_id": "marked_negative",
            "source_file": "noise/poc234_42/doc_abcdabcdabcdabcd.txt",
            "text": "Reference ref_abcdef1234 contains evidence. Project BEACON is managed by Marie Dubois.",
        },
    ]

    active, suppressed, forbidden = resolver.resolve(candidates, query_ir)

    assert [b["block_id"] for b in active] == ["marked_negative"]
    assert forbidden == []


def test_g2_expected_answers_match_runtime_currency_format(tmp_path):
    from highway.workloads.build_poc234_kernel_hardening_workload import generate_workload

    queries = generate_workload(
        corpus=str(tmp_path / "corpus" / "index"),
        output=str(tmp_path / "artifacts/runs" / "workload.jsonl"),
        n_comparison=8,
        n_aggregation=0,
        seed=123,
        run_ingest=False,
        write_mixed=False,
    )

    g2_queries = [q for q in queries if q["metadata"]["type"] == "G" and q["metadata"]["sub_type"] == 2]
    assert g2_queries
    assert all(re.fullmatch(r"Project [A-Z0-9_-]+ \(budget of \$\d{1,3}(,\d{3})*\)", q["expected_answer"]) for q in g2_queries)


def test_generated_manager_aliases_are_supported_by_kernel_canonicalizer():
    from highway.workloads.build_poc234_kernel_hardening_workload import MANAGER_ALIASES
    from highway.kernels.compute_kernels import canonicalize_manager

    for manager, aliases in MANAGER_ALIASES.items():
        assert canonicalize_manager(manager) == manager
        for alias in aliases:
            assert canonicalize_manager(alias) == manager


def test_runner_flags_predictable_ids_and_oracle_source_files():
    from highway.runners.run_poc234_kernel_hardening import leak_check_query

    ok, reasons = leak_check_query({
        "id": "q_ab12cd34ef56ab78",
        "source_file": "noise/poc234_42/doc_12ab34cd56ef78ab.txt",
    })
    assert ok
    assert reasons == []

    ok, reasons = leak_check_query({
        "id": "g_adv_005",
        "source_file": "noise/g_adv_005.txt",
    })
    assert not ok
    assert "predictable_query_id" in reasons
    assert "oracle_encoded_source_file" in reasons


def test_runner_output_keeps_same_answer_for_same_question_with_different_ids(tmp_path, monkeypatch):
    runner = importlib.import_module("highway.runners.run_poc234_kernel_hardening")

    class FakeScheduler:
        def __init__(self, corpus, cache_dir):
            self.last_query_metrics = {}

        def answer(self, question, use_cache, force_llm, q_id, disable_llm_for_computable):
            return {
                "answer": "Project AURORA (budget of $900,000)",
                "route": "COMPUTE_COMPARISON",
                "metrics": {
                    "llm_bypass": True,
                    "verifier_passed": True,
                    "prompt_tokens": 0,
                    "kernel_audit": {"query_id": q_id},
                },
            }

    monkeypatch.setattr(runner, "ExecutionScheduler", FakeScheduler)

    workload = tmp_path / "workload.jsonl"
    rows = [
        {
            "id": "q_aaaaaaaaaaaaaaaa",
            "question": "Which project has a higher budget: Project AURORA or Project BEACON?",
            "expected_answer": "Project AURORA (budget of $900,000)",
            "category": "G",
            "source_file": "noise/poc234_42/doc_1111111111111111.txt",
        },
        {
            "id": "q_bbbbbbbbbbbbbbbb",
            "question": "Which project has a higher budget: Project AURORA or Project BEACON?",
            "expected_answer": "Project AURORA (budget of $900,000)",
            "category": "G",
            "source_file": "noise/poc234_42/doc_2222222222222222.txt",
        },
    ]
    workload.write_text("\n".join(json.dumps(row) for row in rows), encoding="utf-8")

    output = tmp_path / "results.jsonl"
    summary = tmp_path / "summary.md"
    monkeypatch.setattr(sys, "argv", [
        "run_poc234_kernel_hardening.py",
        "--run-name",
        "unit",
        "--corpus",
        str(tmp_path / "corpus" / "index"),
        "--workload",
        str(workload),
        "--output",
        str(output),
        "--summary",
        str(summary),
    ])

    runner.main()

    records = _read_jsonl(output)
    assert [r["answer"] for r in records] == [
        "Project AURORA (budget of $900,000)",
        "Project AURORA (budget of $900,000)",
    ]
    assert all(r["leak_check_passed"] for r in records)
    assert all(r["is_em"] for r in records)


