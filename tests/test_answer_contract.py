from highway.runtime.context_engine import ContextBlock, ContextPack, ContextRequest


def _pack(question: str) -> ContextPack:
    return ContextPack(
        request=ContextRequest(user_turn=question),
        blocks=[
            ContextBlock(
                block_id="b1",
                source_file="docs/kronos.txt",
                text="Project NEPTUNE budget is $100,000. Project KRONOS budget is $204,567.",
                score=1.0,
                reason="test",
            )
        ],
        query_ir={"intent": "comparison"},
        metrics={"latency_ms": 1.0},
        warnings=[],
    )


def test_answer_contract_project_name_for_which_project():
    from highway.runtime.answer_contract import AnswerContractCompiler

    contract = AnswerContractCompiler().compile(
        ContextRequest(user_turn="Which project has a higher budget: Project NEPTUNE or Project KRONOS?"),
        _pack("Which project has a higher budget: Project NEPTUNE or Project KRONOS?"),
    )

    assert contract.answer_type == "project_name"
    assert contract.max_output_tokens == 32
    assert contract.citation_required is True
    assert "winner_project" in contract.required_facts
    assert "winner_budget" in contract.optional_facts
    assert contract.compact_answer_schema == '{"answer":"Project NAME","sources":["source_file"]}'
    assert "compact" in contract.retry_instruction.lower()
    assert contract.output_retry_budget <= contract.max_output_tokens


def test_answer_contract_numeric_fact_for_budget_question():
    from highway.runtime.answer_contract import AnswerContractCompiler

    contract = AnswerContractCompiler().compile(
        ContextRequest(user_turn="What is the budget of Project KRONOS?"),
        _pack("What is the budget of Project KRONOS?"),
    )

    assert contract.answer_type == "numeric_fact"
    assert contract.max_output_tokens == 64
    assert "budget" in contract.required_facts
    assert contract.numeric_facts_allowed == ["$100,000", "$204,567"]


def test_answer_verifier_accepts_project_name_without_budget_when_question_asks_which_project():
    from highway.runtime.answer_contract import AnswerContractCompiler, AnswerVerifier

    pack = _pack("Which project has a higher budget: Project NEPTUNE or Project KRONOS?")
    contract = AnswerContractCompiler().compile(pack.request, pack)
    audit = AnswerVerifier().audit(
        {"answer": "Project KRONOS", "sources": ["docs/kronos.txt"], "reasoning": "KRONOS has the higher budget."},
        contract,
        pack,
        output_tokens=8,
    )

    assert audit.answer_satisfies_question is True
    assert audit.full_exact_match is False
    assert audit.source_attribution_ok is True
    assert audit.hallucination_flag is False
    assert audit.verdict == "PASS"


def test_answer_verifier_rejects_invented_budget():
    from highway.runtime.answer_contract import AnswerContractCompiler, AnswerVerifier

    pack = _pack("What is the budget of Project KRONOS?")
    contract = AnswerContractCompiler().compile(pack.request, pack)
    audit = AnswerVerifier().audit(
        {"answer": "Project KRONOS has a budget of $999,999.", "sources": ["docs/kronos.txt"]},
        contract,
        pack,
        output_tokens=10,
    )

    assert audit.numeric_facts_ok is False
    assert audit.hallucination_flag is True
    assert audit.verdict == "HALLUCINATION_FAIL"


def test_answer_verifier_rejects_source_not_in_context_pack():
    from highway.runtime.answer_contract import AnswerContractCompiler, AnswerVerifier

    pack = _pack("Which project has a higher budget: Project NEPTUNE or Project KRONOS?")
    contract = AnswerContractCompiler().compile(pack.request, pack)
    audit = AnswerVerifier().audit(
        {"answer": "Project KRONOS", "sources": ["docs/missing.txt"]},
        contract,
        pack,
        output_tokens=8,
    )

    assert audit.source_attribution_ok is False
    assert audit.verdict == "SOURCE_FAIL"


def test_answer_verifier_marks_output_budget_fail_for_long_correct_answer():
    from highway.runtime.answer_contract import AnswerContractCompiler, AnswerVerifier

    pack = _pack("Which project has a higher budget: Project NEPTUNE or Project KRONOS?")
    contract = AnswerContractCompiler().compile(pack.request, pack)
    audit = AnswerVerifier().audit(
        {
            "answer": "Project KRONOS",
            "sources": ["docs/kronos.txt"],
            "reasoning": "This is a long but grounded explanation that is unnecessary for the requested answer.",
        },
        contract,
        pack,
        output_tokens=contract.max_output_tokens + 10,
    )

    assert audit.answer_satisfies_question is True
    assert audit.output_over_budget is True
    assert audit.verdict == "OUTPUT_BUDGET_FAIL"
