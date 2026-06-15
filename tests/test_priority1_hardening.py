from highway.kernels.compute_kernels import AggregationKernel, ComparisonKernel
from highway.retrieval.query_parser import QueryParser


def test_canonical_hash_includes_question_semantics():
    parser = QueryParser(["AURORA", "BEACON"])

    higher = parser.parse("Which project has a higher budget: Project AURORA or Project BEACON?")
    lower = parser.parse("Which project has a lower budget: Project AURORA or Project BEACON?")

    assert parser.canonical_hash(higher) != parser.canonical_hash(lower)


def test_canonical_hash_normalizes_question_text():
    parser = QueryParser(["AURORA", "BEACON"])

    compact = parser.parse("Which project has a higher budget: Project AURORA or Project BEACON?")
    noisy = parser.parse("  which   project HAS a higher budget: project aurora or project beacon?  ")

    assert parser.canonical_hash(compact) == parser.canonical_hash(noisy)


def test_comparison_kernel_does_not_use_query_id_to_select_evidence_file():
    query_ir = {
        "question": "Which project has a higher budget: Project AURORA or Project BEACON?",
        "target_entities": ["AURORA", "BEACON"],
    }
    evidence = [
        {
            "block_id": "b1",
            "source_file": "noise/adv_doc_0005.txt",
            "text": "Project AURORA budget is $900,000.\nProject BEACON budget is $100,000.",
        },
        {
            "block_id": "b2",
            "source_file": "noise/adv_doc_0006.txt",
            "text": "Project AURORA budget is $100,000.\nProject BEACON budget is $900,000.",
        },
    ]
    kernel = ComparisonKernel()

    first = kernel.execute(query_ir, evidence, ir_builder=None, query_id="g_adv_005")
    second = kernel.execute(query_ir, evidence, ir_builder=None, query_id="g_adv_006")

    assert first["answer"] == second["answer"]


def test_aggregation_kernel_does_not_use_query_id_to_select_evidence_file():
    query_ir = {
        "question": "List all project names managed by Jean Dupont.",
        "target_entities": ["Jean Dupont"],
    }
    evidence = [
        {
            "block_id": "b1",
            "source_file": "noise/adv_doc_0205.txt",
            "text": "Project AURORA is managed by Jean Dupont.",
        },
        {
            "block_id": "b2",
            "source_file": "noise/adv_doc_0206.txt",
            "text": "Project BEACON is managed by Jean Dupont.",
        },
    ]
    kernel = AggregationKernel()

    first = kernel.execute(query_ir, evidence, ir_builder=None, query_id="h_adv_005")
    second = kernel.execute(query_ir, evidence, ir_builder=None, query_id="h_adv_006")

    assert first["answer"] == second["answer"] == "AURORA, BEACON"


def test_aggregation_kernel_empty_evidence_status_does_not_depend_on_query_id():
    query_ir = {
        "question": "List all project names managed by Jean Dupont.",
        "target_entities": ["Jean Dupont"],
    }
    kernel = AggregationKernel()

    adversarial_id = kernel.execute(query_ir, [], ir_builder=None, query_id="h_adv_005")
    ordinary_id = kernel.execute(query_ir, [], ir_builder=None, query_id="ordinary_query")

    assert adversarial_id["status"] == ordinary_id["status"]
    assert adversarial_id["answer"] == ordinary_id["answer"]


