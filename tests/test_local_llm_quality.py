import json


def test_ollama_client_unavailable_returns_skip_without_crashing():
    from highway.runtime.ollama_client import OllamaLLMClient

    class FailingTransport:
        def __call__(self, payload, timeout_s):
            raise OSError("connection refused")

    client = OllamaLLMClient(model="missing-model", transport=FailingTransport())
    result = client.answer(prompt="Question?", query_ir={}, evidence=[])

    assert result["available"] is False
    assert result["skip_reason"]
    assert result["answer"] == ""
    assert result["input_tokens"] > 0
    assert result["output_tokens"] == 0
    assert result["ttft_ms"] >= 0.0


def test_ollama_client_accepts_benchmark_metadata_kwargs_when_unavailable():
    from highway.runtime.ollama_client import OllamaLLMClient

    class FailingTransport:
        def __call__(self, payload, timeout_s):
            raise OSError("connection refused")

    client = OllamaLLMClient(model="missing-model", transport=FailingTransport())
    result = client.answer(
        prompt="Question?",
        query_ir={},
        evidence=[],
        expected_sources=["docs/a.txt"],
        answer_contract=None,
    )

    assert result["available"] is False
    assert result["skip_reason"]


def test_ollama_client_uses_max_output_tokens_for_num_predict():
    from highway.runtime.ollama_client import OllamaLLMClient

    captured = {}

    class OneChunkTransport:
        def __call__(self, payload, timeout_s):
            captured.update(payload)
            yield {"response": '{"answer":"Project X","sources":["docs/a.txt"]}', "done": False}
            yield {"done": True, "eval_count": 8, "prompt_eval_count": 12, "eval_duration": 1_000_000}

    client = OllamaLLMClient(model="qwen-test", transport=OneChunkTransport(), num_predict=256)
    result = client.answer(prompt="Question?", query_ir={}, evidence=[], max_output_tokens=24)

    assert result["available"] is True
    assert captured["options"]["num_predict"] == 24
    assert result["num_predict_requested"] == 24


def test_parse_model_json_accepts_clean_json_and_embedded_json():
    from highway.benchmarks.local_llm_quality import parse_model_json

    clean = parse_model_json('{"reasoning":"r","answer":"Project X","sources":["a.txt"],"confidence":0.9}')
    embedded = parse_model_json('Here is the result:\n{"answer":"Project Y","sources":["b.txt"]}\nThanks')

    assert clean["parse_ok"] is True
    assert clean["answer"] == "Project X"
    assert clean["sources"] == ["a.txt"]
    assert embedded["parse_ok"] is True
    assert embedded["answer"] == "Project Y"
    assert embedded["sources"] == ["b.txt"]


def test_parse_model_json_marks_unusable_text_as_model_parse_fail():
    from highway.benchmarks.local_llm_quality import evaluate_quality, parse_model_json

    parsed = parse_model_json("I refuse to use JSON")
    quality = evaluate_quality(
        parsed,
        expected_answer="Project X",
        expected_sources=["docs/project_x.txt"],
        allowed_sources=["docs/project_x.txt"],
        previous_entity=None,
        current_question="What is Project X?",
    )

    assert parsed["parse_ok"] is False
    assert quality["verdict"] == "MODEL_PARSE_FAIL"
    assert quality["is_em"] is False


def test_quality_fails_wrong_answer_even_when_tokens_are_saved():
    from highway.benchmarks.local_llm_quality import evaluate_quality, parse_model_json

    parsed = parse_model_json('{"reasoning":"r","answer":"Project Z","sources":["docs/project_x.txt"]}')
    quality = evaluate_quality(
        parsed,
        expected_answer="Project X",
        expected_sources=["docs/project_x.txt"],
        allowed_sources=["docs/project_x.txt"],
        previous_entity=None,
        current_question="Which project wins?",
    )

    assert quality["verdict"] == "QUALITY_FAIL"
    assert quality["is_em"] is False
    assert quality["source_attribution_ok"] is True


def test_quality_accepts_project_name_when_question_asks_which_project():
    from highway.benchmarks.local_llm_quality import evaluate_quality, parse_model_json

    parsed = parse_model_json(
        '{"reasoning":"Project KRONOS has the higher budget.","answer":"Project KRONOS","sources":["docs/project_kronos.txt"]}'
    )
    quality = evaluate_quality(
        parsed,
        expected_answer="Project KRONOS (budget of $204,567)",
        expected_sources=["docs/project_kronos.txt"],
        allowed_sources=["docs/project_kronos.txt"],
        previous_entity=None,
        current_question="Which project has a higher budget: Project NEPTUNE or Project KRONOS?",
    )

    assert quality["verdict"] == "PASS"
    assert quality["is_em"] is True
    assert quality["full_exact_match"] is False
    assert quality["answer_satisfies_question"] is True


def test_quality_fails_missing_source_attribution():
    from highway.benchmarks.local_llm_quality import evaluate_quality, parse_model_json

    parsed = parse_model_json('{"reasoning":"r","answer":"Project X","sources":[]}')
    quality = evaluate_quality(
        parsed,
        expected_answer="Project X",
        expected_sources=["docs/project_x.txt"],
        allowed_sources=["docs/project_x.txt"],
        previous_entity=None,
        current_question="Which project wins?",
    )

    assert quality["verdict"] == "SOURCE_FAIL"
    assert quality["is_em"] is True
    assert quality["source_attribution_ok"] is False


def test_quality_fails_follow_up_that_loses_previous_entity():
    from highway.benchmarks.local_llm_quality import evaluate_quality, parse_model_json

    parsed = parse_model_json('{"reasoning":"r","answer":"The manager is Alice","sources":["docs/project_x.txt"]}')
    quality = evaluate_quality(
        parsed,
        expected_answer="The manager is Alice",
        expected_sources=["docs/project_x.txt"],
        allowed_sources=["docs/project_x.txt"],
        previous_entity="Project X",
        current_question="And what about its manager?",
    )

    assert quality["verdict"] == "COHERENCE_FAIL"
    assert quality["coherence_ok"] is False


def test_local_llm_quality_benchmark_runs_with_injected_fake_client(tmp_path):
    from highway.benchmarks.local_llm_quality import run_local_llm_quality_benchmark

    class JsonEchoClient:
        model_name = "json_echo_fake"

        def answer(self, prompt, query_ir, evidence, expected_answer=None, query_id="fake"):
            source = evidence[0]["source_file"] if evidence else "unknown"
            raw = json.dumps(
                {
                    "reasoning": "I used the selected source.",
                    "answer": expected_answer,
                    "sources": [source],
                    "confidence": 1.0,
                }
            )
            input_tokens = max(1, len(prompt.split()))
            output_tokens = max(1, len(raw.split()))
            return {
                "available": True,
                "model_name": self.model_name,
                "reasoning": "I used the selected source.",
                "answer": expected_answer,
                "raw_text": raw,
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "ttft_ms": max(1.0, input_tokens / 100.0),
                "decode_ms": max(1.0, output_tokens / 10.0),
                "total_ms": max(1.0, input_tokens / 100.0) + max(1.0, output_tokens / 10.0),
                "input_tokens_per_second": input_tokens / (max(1.0, input_tokens / 100.0) / 1000.0),
                "output_tokens_per_second": output_tokens / (max(1.0, output_tokens / 10.0) / 1000.0),
            }

    result = run_local_llm_quality_benchmark(
        output_dir=tmp_path / "local_llm_quality",
        sizes=[80],
        query_count=4,
        seed=42,
        strategy="ooc_marker_entity_pruned",
        llm_client=JsonEchoClient(),
    )

    assert result["report_path"].exists()
    assert result["metrics_path"].exists()
    assert result["records_path"].exists()
    metrics = json.loads(result["metrics_path"].read_text(encoding="utf-8"))
    tier = metrics["summary"]["tiers"][0]

    assert metrics["summary"]["status"] == "VALIDATING"
    assert tier["metrics_complete_rate"] == 100.0
    assert tier["highway_em"] == 100.0
    assert tier["avg_avoided_input_tokens_pct"] >= 80.0
    assert tier["source_attribution_rate"] >= 95.0


def test_local_llm_quality_benchmark_rejects_token_savings_when_answers_are_wrong(tmp_path):
    from highway.benchmarks.local_llm_quality import run_local_llm_quality_benchmark

    class WrongJsonClient:
        model_name = "wrong_json_fake"

        def answer(self, prompt, query_ir, evidence, expected_answer=None, query_id="fake"):
            source = evidence[0]["source_file"] if evidence else "unknown"
            raw = json.dumps(
                {
                    "reasoning": "I used a source but chose the wrong answer.",
                    "answer": "WRONG ANSWER",
                    "sources": [source],
                    "confidence": 0.9,
                }
            )
            input_tokens = max(1, len(prompt.split()))
            output_tokens = max(1, len(raw.split()))
            return {
                "available": True,
                "model_name": self.model_name,
                "reasoning": "wrong",
                "answer": "WRONG ANSWER",
                "raw_text": raw,
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "ttft_ms": max(1.0, input_tokens / 100.0),
                "decode_ms": max(1.0, output_tokens / 10.0),
                "total_ms": max(1.0, input_tokens / 100.0) + max(1.0, output_tokens / 10.0),
                "input_tokens_per_second": input_tokens / (max(1.0, input_tokens / 100.0) / 1000.0),
                "output_tokens_per_second": output_tokens / (max(1.0, output_tokens / 10.0) / 1000.0),
            }

    result = run_local_llm_quality_benchmark(
        output_dir=tmp_path / "local_llm_quality_wrong",
        sizes=[80],
        query_count=4,
        seed=42,
        strategy="ooc_marker_entity_pruned",
        llm_client=WrongJsonClient(),
    )

    metrics = json.loads(result["metrics_path"].read_text(encoding="utf-8"))
    tier = metrics["summary"]["tiers"][0]

    assert metrics["summary"]["status"] == "NON_VALIDATING"
    assert tier["highway_em"] == 0.0
    assert tier["avg_avoided_input_tokens_pct"] >= 80.0
