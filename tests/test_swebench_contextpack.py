from pathlib import Path


def _fake_row():
    return {
        "instance_id": "django__django-12345",
        "repo": "django/django",
        "base_commit": "abc123",
        "problem_statement": "Fix the frobnicator path handling in utils.",
        "patch": (
            "diff --git a/django/utils/frobnicator.py b/django/utils/frobnicator.py\n"
            "--- a/django/utils/frobnicator.py\n"
            "+++ b/django/utils/frobnicator.py\n"
            "@@ -1,2 +1,2 @@\n"
            "-old\n"
            "+new\n"
            "diff --git a/tests/utils/test_frobnicator.py b/tests/utils/test_frobnicator.py\n"
            "--- a/tests/utils/test_frobnicator.py\n"
            "+++ b/tests/utils/test_frobnicator.py\n"
            "@@ -3,1 +3,1 @@\n"
            "-assert old\n"
            "+assert new\n"
        ),
        "test_patch": (
            "diff --git a/tests/utils/test_frobnicator.py b/tests/utils/test_frobnicator.py\n"
            "--- a/tests/utils/test_frobnicator.py\n"
            "+++ b/tests/utils/test_frobnicator.py\n"
        ),
        "FAIL_TO_PASS": '["tests/utils/test_frobnicator.py::test_new"]',
        "PASS_TO_PASS": '["tests/utils/test_frobnicator.py::test_existing"]',
    }


def test_extract_patch_files_ignores_dev_null_and_returns_b_paths():
    from highway.benchmarks.swebench_contextpack import extract_patch_files

    patch = (
        "diff --git a/pkg/old.py b/pkg/new.py\n"
        "--- a/pkg/old.py\n"
        "+++ b/pkg/new.py\n"
        "diff --git a/pkg/deleted.py b/pkg/deleted.py\n"
        "--- a/pkg/deleted.py\n"
        "+++ /dev/null\n"
    )

    assert extract_patch_files(patch) == ["pkg/new.py"]


def test_normalize_swebench_rows_is_seed_stable_and_safe():
    from highway.benchmarks.swebench_contextpack import normalize_swebench_rows

    rows = [_fake_row(), {**_fake_row(), "instance_id": "django__django-99999"}]
    first = normalize_swebench_rows(rows, limit=2, seed=7)
    second = normalize_swebench_rows(rows, limit=2, seed=7)

    assert [case.instance_id for case in first] == [case.instance_id for case in second]
    assert first[0].gold_files == ["django/utils/frobnicator.py", "tests/utils/test_frobnicator.py"]
    assert first[0].gold_test_files == ["tests/utils/test_frobnicator.py"]
    assert first[0].repo == "django/django"


def test_swebench_fake_repo_benchmark_writes_distinct_prompts_and_metrics(tmp_path):
    from highway.benchmarks.swebench_contextpack import run_swebench_contextpack_benchmark

    repo_root = tmp_path / "repo"
    (repo_root / "django" / "utils").mkdir(parents=True)
    (repo_root / "tests" / "utils").mkdir(parents=True)
    (repo_root / "django" / "utils" / "frobnicator.py").write_text(
        "def normalize_path(path):\n    return path.replace('\\\\', '/')\n",
        encoding="utf-8",
    )
    (repo_root / "tests" / "utils" / "test_frobnicator.py").write_text(
        "from django.utils.frobnicator import normalize_path\n",
        encoding="utf-8",
    )
    (repo_root / "django" / "utils" / "decoy.py").write_text("def unrelated(): pass\n", encoding="utf-8")
    for idx in range(8):
        (repo_root / "django" / "utils" / f"noise_{idx}.py").write_text(
            (f"def unrelated_{idx}():\n    return {idx}\n" + "# unrelated implementation detail\n" * 80),
            encoding="utf-8",
        )

    result = run_swebench_contextpack_benchmark(
        output_dir=tmp_path / "swebench",
        rows=[_fake_row()],
        repo_overrides={"django/django": repo_root},
        modes=["bm25_topk", "highway_contextpack"],
        audit_prompts=True,
        seed=42,
    )

    assert result["status"] == "VALIDATING"
    assert result["metrics"]["prompt_pair_is_distinct_rate"] == 100.0
    assert result["metrics"]["highway_contextpack"]["file_recall_at_5"] == 100.0
    assert result["metrics"]["highway_contextpack"]["avg_highway_blocks"] < result["metrics"]["avg_baseline_blocks"]
    assert (tmp_path / "swebench" / "records.jsonl").exists()
    assert (tmp_path / "swebench" / "prompts" / "turn_000_baseline.txt").exists()


def test_swebench_poison_missing_gold_file_is_non_validating(tmp_path):
    from highway.benchmarks.swebench_contextpack import run_swebench_contextpack_benchmark

    repo_root = tmp_path / "repo"
    (repo_root / "django" / "utils").mkdir(parents=True)
    (repo_root / "tests" / "utils").mkdir(parents=True)
    (repo_root / "django" / "utils" / "frobnicator.py").write_text("def normalize_path(path): pass\n", encoding="utf-8")
    (repo_root / "tests" / "utils" / "test_frobnicator.py").write_text("def test_new(): pass\n", encoding="utf-8")

    result = run_swebench_contextpack_benchmark(
        output_dir=tmp_path / "swebench_poison",
        rows=[_fake_row()],
        repo_overrides={"django/django": repo_root},
        modes=["highway_contextpack"],
        poison_context="missing_gold_file",
        audit_prompts=True,
        seed=42,
    )

    assert result["status"] == "NON_VALIDATING"
    assert result["metrics"]["poison_fail_rate"] == 100.0
    record = result["records"][0]
    assert record["poison_used"] is True
    assert record["gold_file_removed"] is True


def test_extract_patch_hunks_tracks_new_file_line_ranges():
    from highway.benchmarks.swebench_contextpack import extract_patch_hunks

    patch = (
        "diff --git a/pkg/service.py b/pkg/service.py\n"
        "--- a/pkg/service.py\n"
        "+++ b/pkg/service.py\n"
        "@@ -10,3 +20,4 @@\n"
        " context\n"
        "-old\n"
        "+new\n"
        "+another\n"
        "diff --git a/pkg/other.py b/pkg/other.py\n"
        "--- a/pkg/other.py\n"
        "+++ b/pkg/other.py\n"
        "@@ -1,1 +1,1 @@\n"
        "+created\n"
    )

    hunks = extract_patch_hunks(patch)

    assert [h.file_path for h in hunks] == ["pkg/service.py", "pkg/other.py"]
    assert hunks[0].changed_lines == [21, 22]
    assert hunks[0].start_line == 20
    assert hunks[0].end_line == 23


def test_map_hunk_to_python_function_and_method_symbols():
    from highway.benchmarks.swebench_contextpack import HunkRange, map_hunks_to_symbols

    source = (
        "class Runner:\n"
        "    def execute(self):\n"
        "        setup = True\n"
        "        return setup\n"
        "\n"
        "def helper():\n"
        "    return 1\n"
    )
    hunks = [
        HunkRange(file_path="pkg/service.py", start_line=3, end_line=3, changed_lines=[3]),
        HunkRange(file_path="pkg/service.py", start_line=7, end_line=7, changed_lines=[7]),
    ]

    symbols = map_hunks_to_symbols("pkg/service.py", source, hunks)

    assert symbols[0].symbol_name == "Runner.execute"
    assert symbols[0].mapping_status == "symbol"
    assert symbols[1].symbol_name == "helper"


def test_map_hunk_falls_back_to_line_range_without_symbol():
    from highway.benchmarks.swebench_contextpack import HunkRange, map_hunks_to_symbols

    symbols = map_hunks_to_symbols(
        "pkg/config.py",
        "SETTING = True\nOTHER = False\n",
        [HunkRange(file_path="pkg/config.py", start_line=1, end_line=1, changed_lines=[1])],
    )

    assert symbols[0].symbol_name == "pkg/config.py:1-1"
    assert symbols[0].mapping_status == "line_range_fallback"


def test_swebench_symbol_metrics_are_written_when_eval_symbols_enabled(tmp_path):
    from highway.benchmarks.swebench_contextpack import run_swebench_contextpack_benchmark

    row = {
        **_fake_row(),
        "problem_statement": "Fix normalize_path in django/utils/frobnicator.py.",
        "patch": (
            "diff --git a/django/utils/frobnicator.py b/django/utils/frobnicator.py\n"
            "--- a/django/utils/frobnicator.py\n"
            "+++ b/django/utils/frobnicator.py\n"
            "@@ -1,3 +1,3 @@\n"
            " def normalize_path(path):\n"
            "-    return path\n"
            "+    return path.replace('\\\\', '/')\n"
        ),
    }
    repo_root = tmp_path / "repo"
    (repo_root / "django" / "utils").mkdir(parents=True)
    (repo_root / "django" / "utils" / "frobnicator.py").write_text(
        "def normalize_path(path):\n    return path\n\n"
        "def unrelated():\n    return None\n",
        encoding="utf-8",
    )
    for idx in range(8):
        (repo_root / "django" / "utils" / f"noise_{idx}.py").write_text(
            "# unrelated file\n" * 100,
            encoding="utf-8",
        )

    result = run_swebench_contextpack_benchmark(
        output_dir=tmp_path / "swebench_symbols",
        rows=[row],
        repo_overrides={"django/django": repo_root},
        modes=["highway_contextpack"],
        audit_prompts=True,
        eval_symbols=True,
        seed=42,
    )

    assert result["status"] == "VALIDATING"
    record = result["records"][0]
    assert record["gold_symbols"] == ["normalize_path"]
    assert record["gold_symbols_present"] == ["normalize_path"]
    assert record["symbol_recall_at_5"] == 100.0
    assert record["hunk_area_recall"] == 100.0
    assert record["tokens_per_relevant_line"] > 0.0
    assert record["context_line_ranges_sent"][0]["snippet_reason"] in {"symbol_overlap", "selected_file"}


def test_swebench_symbol_poison_forces_zero_symbol_recall(tmp_path):
    from highway.benchmarks.swebench_contextpack import run_swebench_contextpack_benchmark

    row = {
        **_fake_row(),
        "problem_statement": "Fix normalize_path in django/utils/frobnicator.py.",
        "patch": (
            "diff --git a/django/utils/frobnicator.py b/django/utils/frobnicator.py\n"
            "--- a/django/utils/frobnicator.py\n"
            "+++ b/django/utils/frobnicator.py\n"
            "@@ -1,2 +1,2 @@\n"
            " def normalize_path(path):\n"
            "+    return path.replace('\\\\', '/')\n"
        ),
    }
    repo_root = tmp_path / "repo"
    (repo_root / "django" / "utils").mkdir(parents=True)
    (repo_root / "django" / "utils" / "frobnicator.py").write_text(
        "def normalize_path(path):\n    return path\n",
        encoding="utf-8",
    )
    (repo_root / "django" / "utils" / "decoy.py").write_text("def normalize_path(path):\n    return None\n", encoding="utf-8")

    result = run_swebench_contextpack_benchmark(
        output_dir=tmp_path / "swebench_symbol_poison",
        rows=[row],
        repo_overrides={"django/django": repo_root},
        modes=["highway_contextpack"],
        poison_context="missing_gold_file",
        audit_prompts=True,
        eval_symbols=True,
        seed=42,
    )

    assert result["status"] == "NON_VALIDATING"
    record = result["records"][0]
    assert record["symbol_recall_at_5"] == 0.0
    assert record["hunk_area_recall"] == 0.0


def test_extract_code_paths_from_traceback_and_issue_text():
    from highway.benchmarks.swebench_contextpack import extract_code_paths

    issue = (
        'File "/tmp/project/django/db/models/fields/reverse_related.py", line 140, in __hash__\n'
        "Please also inspect django/core/checks/model_checks.py and docs/topics/db/models.txt"
    )

    assert extract_code_paths(issue) == [
        "django/db/models/fields/reverse_related.py",
        "django/core/checks/model_checks.py",
        "docs/topics/db/models.txt",
    ]


def test_extract_code_symbols_from_issue_text():
    from highway.benchmarks.swebench_contextpack import extract_code_symbols

    issue = "Missing make_hashable call on through_fields in ManyToManyRel.__hash__."

    symbols = extract_code_symbols(issue)

    assert "make_hashable" in symbols
    assert "through_fields" in symbols
    assert "ManyToManyRel" in symbols
    assert "__hash__" in symbols


def test_build_symbol_index_maps_classes_and_functions_to_files():
    from highway.benchmarks.swebench_contextpack import CodeBlock, build_symbol_index

    blocks = [
        CodeBlock("a", "django/db/models/fields/reverse_related.py", "class ManyToManyRel:\n    def __hash__(self):\n        pass\n", 10),
        CodeBlock("b", "django/utils/hashable.py", "def make_hashable(value):\n    return tuple(value)\n", 10),
    ]

    index = build_symbol_index(blocks)

    assert "django/db/models/fields/reverse_related.py" in index["ManyToManyRel"]
    assert "django/db/models/fields/reverse_related.py" in index["__hash__"]
    assert "django/utils/hashable.py" in index["make_hashable"]


def test_repo_index_cache_round_trips_blocks(tmp_path):
    from highway.benchmarks.swebench_contextpack import load_or_build_repo_index

    repo_root = tmp_path / "repo"
    (repo_root / "pkg").mkdir(parents=True)
    (repo_root / "pkg" / "service.py").write_text("def run():\n    return 1\n", encoding="utf-8")
    cache_dir = tmp_path / "cache"

    first_blocks, first_metrics = load_or_build_repo_index(repo_root, cache_dir, "owner/repo", "abc123")
    second_blocks, second_metrics = load_or_build_repo_index(repo_root, cache_dir, "owner/repo", "abc123")

    assert first_metrics["repo_index_cache_hit"] is False
    assert second_metrics["repo_index_cache_hit"] is True
    assert [block.to_dict() for block in first_blocks] == [block.to_dict() for block in second_blocks]


def test_highway_code_contextpack_v2_uses_traceback_path_to_recover_gold_file(tmp_path):
    from highway.benchmarks.swebench_contextpack import run_swebench_contextpack_benchmark

    row = {
        **_fake_row(),
        "instance_id": "django__django-14672",
        "problem_statement": (
            "Missing call make_hashable on through_fields in ManyToManyRel.\n"
            'File "/tmp/site-packages/django/db/models/fields/reverse_related.py", line 140, in __hash__\n'
            "TypeError: unhashable type: 'list'"
        ),
        "patch": (
            "diff --git a/django/db/models/fields/reverse_related.py b/django/db/models/fields/reverse_related.py\n"
            "--- a/django/db/models/fields/reverse_related.py\n"
            "+++ b/django/db/models/fields/reverse_related.py\n"
            "@@ -137,4 +137,4 @@\n"
            " class ManyToManyRel:\n"
            "     def __hash__(self):\n"
            "-        return hash(self.identity)\n"
            "+        return hash(make_hashable(self.identity))\n"
        ),
    }
    repo_root = tmp_path / "repo"
    (repo_root / "django" / "db" / "models" / "fields").mkdir(parents=True)
    (repo_root / "docs" / "topics" / "db").mkdir(parents=True)
    (repo_root / "django" / "db" / "models" / "fields" / "reverse_related.py").write_text(
        "class ManyToManyRel:\n"
        "    @property\n"
        "    def identity(self):\n"
        "        return self.through_fields\n"
        "    def __hash__(self):\n"
        "        return hash(self.identity)\n",
        encoding="utf-8",
    )
    (repo_root / "docs" / "topics" / "db" / "models.txt").write_text("ManyToManyRel through_fields docs\n" * 100, encoding="utf-8")
    for idx in range(8):
        (repo_root / "django" / "db" / "models" / "fields" / f"noise_{idx}.py").write_text(
            "class ManyToManyRelDocumentation:\n    pass\n" * 50,
            encoding="utf-8",
        )

    result = run_swebench_contextpack_benchmark(
        output_dir=tmp_path / "swebench_code_v2",
        rows=[row],
        repo_overrides={"django/django": repo_root},
        modes=["highway_code_contextpack_v2"],
        audit_prompts=True,
        eval_symbols=True,
        seed=42,
    )

    assert result["status"] == "VALIDATING"
    record = result["records"][0]
    assert record["source_files_sent"][0] == "django/db/models/fields/reverse_related.py"
    assert "explicit_path" in record["candidate_sources"]["django/db/models/fields/reverse_related.py"]
    assert record["repo_index_cache_hit"] is False
