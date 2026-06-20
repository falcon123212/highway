import importlib


def test_highway_errors_keep_legacy_execution_error_contract():
    from highway.errors import ContextOverflowError, HighwayError, LLMUnavailableError

    err = ContextOverflowError(details={"limit": 1200, "requested": 1400})

    assert isinstance(err, HighwayError)
    assert err.code == "CONTEXT_OVERFLOW"
    assert err.to_legacy_answer() == "EXECUTION_ERROR:CONTEXT_OVERFLOW"
    assert err.to_dict()["details"] == {"limit": 1200, "requested": 1400}
    assert HighwayError.from_code("LLM_UNAVAILABLE").to_legacy_answer() == LLMUnavailableError().to_legacy_answer()


def test_scheduler_records_structured_llm_error_while_returning_legacy_answer(tmp_path, monkeypatch):
    import urllib.request
    import highway.runtime.scheduler as scheduler_module
    from highway.runtime.scheduler import ExecutionScheduler

    class FakeParser:
        def parse(self, question):
            return {"question": question, "target_entities": [], "intent": "single_fact_lookup"}

        def canonical_hash(self, query_ir):
            return "hash"

    class FakeSearchRouter:
        def __init__(self, index_dir):
            self.query_parser = FakeParser()
            self.last_storage_metrics = {}

    monkeypatch.setattr(scheduler_module, "SearchRouter", FakeSearchRouter)
    scheduler = ExecutionScheduler(str(tmp_path / "index"), str(tmp_path / "cache"))
    monkeypatch.setattr(urllib.request, "urlopen", lambda *args, **kwargs: (_ for _ in ()).throw(Exception("context length exceeded 400")))

    answer = scheduler._call_llm("oversized prompt")

    assert answer == "EXECUTION_ERROR:CONTEXT_OVERFLOW"
    assert scheduler.last_query_metrics["execution_error"] == "CONTEXT_OVERFLOW"
    assert scheduler.last_query_metrics["execution_error_structured"]["error_type"] == "ContextOverflowError"


def test_unversioned_benchmark_modules_are_canonical_import_targets():
    cpu = importlib.import_module("highway.benchmarks.cpu_extractive")
    cpu_legacy = importlib.import_module("highway.benchmarks.cpu_extractive_1689b")
    scaleup = importlib.import_module("highway.benchmarks.ragbench_scaleup")
    scaleup_legacy = importlib.import_module("highway.benchmarks.ragbench_scaleup_1683")
    cli = importlib.import_module("highway.cli")

    assert cpu.classify_question_type is cpu_legacy.classify_question_type
    assert scaleup.ROLE_CONFIGS is scaleup_legacy.ROLE_CONFIGS
    assert cli.COMMAND_MAPPING["scaleup"] == "highway.benchmarks.ragbench_scaleup"


def test_config_toml_overrides_are_loaded_into_typed_config(tmp_path):
    from highway.config import load_config

    config_file = tmp_path / "config.toml"
    config_file.write_text(
        "\n".join(
            [
                "[ingestion]",
                "chunk_size = 96",
                "chunk_overlap = 24",
                "",
                "[retrieval]",
                'default_embedding_model = "local/test-embedder"',
                "default_rrf_k = 42.0",
                "",
                "[governor]",
                "cascade_budgets = [128, 256]",
                "cpu_extractive_confidence_threshold = 0.9",
            ]
        ),
        encoding="utf-8",
    )

    cfg = load_config(config_file)

    assert cfg.ingestion.chunk_size == 96
    assert cfg.ingestion.chunk_overlap == 24
    assert cfg.retrieval.default_embedding_model == "local/test-embedder"
    assert cfg.retrieval.default_rrf_k == 42.0
    assert cfg.governor.cascade_budgets == (128, 256)
    assert cfg.governor.cpu_extractive_confidence_threshold == 0.9


def test_config_toml_rejects_unknown_sections(tmp_path):
    from highway.config import ConfigurationError, load_config

    config_file = tmp_path / "config.toml"
    config_file.write_text("[surprise]\nvalue = 1\n", encoding="utf-8")

    try:
        load_config(config_file)
    except ConfigurationError as exc:
        assert exc.code == "CONFIGURATION_ERROR"
        assert "surprise" in str(exc)
    else:
        raise AssertionError("load_config should reject unknown TOML sections")
