from __future__ import annotations

import argparse
import ast
import hashlib
import json
import math
import re
import shutil
import subprocess
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from random import Random
from statistics import mean, median
from typing import Any, Dict, Iterable, List, Mapping, Sequence

from highway.paths import DEFAULT_ARTIFACTS_DIR, DEFAULT_RUNS_DIR
from highway.runtime.llm_runtime import estimate_tokens


DEFAULT_DATASET_ID = "SWE-bench/SWE-bench_Verified"
DEFAULT_OUTPUT_DIR = DEFAULT_RUNS_DIR / "swebench_verified_fileloc"
DEFAULT_REPO_CACHE_DIR = DEFAULT_ARTIFACTS_DIR / "cache" / "swebench" / "repos"
DEFAULT_INDEX_CACHE_DIR = DEFAULT_ARTIFACTS_DIR / "cache" / "swebench" / "indexes"
SOURCE_SUFFIXES = {
    ".py",
    ".pyi",
    ".js",
    ".jsx",
    ".ts",
    ".tsx",
    ".java",
    ".go",
    ".rs",
    ".c",
    ".cc",
    ".cpp",
    ".h",
    ".hpp",
    ".md",
    ".rst",
    ".txt",
    ".toml",
    ".yaml",
    ".yml",
    ".json",
}
TOKEN_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]*|[0-9]+")
CODE_PATH_RE = re.compile(
    r"(?P<path>[A-Za-z0-9_./\\:-]+\.(?:py|pyi|js|jsx|ts|tsx|java|go|rs|c|cc|cpp|h|hpp|md|rst|txt|toml|yaml|yml|json))"
)
STOP_SYMBOLS = {
    "Description",
    "Solution",
    "TypeError",
    "Traceback",
    "File",
    "Error",
    "None",
    "True",
    "False",
}


@dataclass(frozen=True)
class SweBenchCase:
    instance_id: str
    repo: str
    base_commit: str
    problem_statement: str
    patch: str
    test_patch: str
    fail_to_pass: List[str]
    pass_to_pass: List[str]
    gold_files: List[str]
    gold_test_files: List[str]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class CodeBlock:
    block_id: str
    source_file: str
    text: str
    token_count: int
    snippet_start_line: int = 1
    snippet_end_line: int = 0
    snippet_reason: str = "selected_file"

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class HunkRange:
    file_path: str
    start_line: int
    end_line: int
    changed_lines: List[int]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class SymbolRange:
    file_path: str
    symbol_name: str
    start_line: int
    end_line: int
    changed_lines: List[int]
    mapping_status: str

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class PromptAuditWriter:
    def __init__(self, output_dir: Path, enabled: bool = True):
        self.output_dir = output_dir
        self.enabled = bool(enabled)
        self.prompt_dir = output_dir / "prompts"
        if self.enabled:
            self.prompt_dir.mkdir(parents=True, exist_ok=True)

    def audit_pair(self, index: int, baseline_prompt: str, highway_prompt: str) -> Dict[str, Any]:
        baseline_rel = Path("prompts") / f"turn_{index:03d}_baseline.txt"
        highway_rel = Path("prompts") / f"turn_{index:03d}_highway.txt"
        if self.enabled:
            (self.output_dir / baseline_rel).write_text(baseline_prompt, encoding="utf-8")
            (self.output_dir / highway_rel).write_text(highway_prompt, encoding="utf-8")
        baseline_hash = _sha256_text(baseline_prompt)
        highway_hash = _sha256_text(highway_prompt)
        return {
            "baseline_prompt_hash": baseline_hash,
            "highway_prompt_hash": highway_hash,
            "baseline_prompt_path": baseline_rel.as_posix() if self.enabled else "",
            "highway_prompt_path": highway_rel.as_posix() if self.enabled else "",
            "baseline_input_tokens": estimate_tokens(baseline_prompt),
            "highway_input_tokens": estimate_tokens(highway_prompt),
            "prompt_pair_is_distinct": baseline_hash != highway_hash and baseline_prompt != highway_prompt,
        }


def extract_patch_files(patch_text: str) -> List[str]:
    files: List[str] = []
    for line in str(patch_text or "").splitlines():
        if not line.startswith("+++ "):
            continue
        raw = line[4:].strip()
        if raw == "/dev/null":
            continue
        if raw.startswith("b/"):
            raw = raw[2:]
        if raw and raw not in files:
            files.append(raw)
    return files


def extract_patch_hunks(patch_text: str) -> List[HunkRange]:
    hunks: List[HunkRange] = []
    current_file = ""
    lines = str(patch_text or "").splitlines()
    idx = 0
    while idx < len(lines):
        line = lines[idx]
        if line.startswith("+++ "):
            raw = line[4:].strip()
            if raw != "/dev/null":
                current_file = raw[2:] if raw.startswith("b/") else raw
        elif line.startswith("@@ ") and current_file:
            match = re.search(r"\+(\d+)(?:,(\d+))?", line)
            if match:
                start_line = int(match.group(1))
                count = int(match.group(2) or "1")
                end_line = start_line + max(1, count) - 1
                changed: List[int] = []
                new_line = start_line
                idx += 1
                while idx < len(lines) and not lines[idx].startswith("diff --git ") and not lines[idx].startswith("@@ "):
                    hunk_line = lines[idx]
                    if hunk_line.startswith("+") and not hunk_line.startswith("+++"):
                        changed.append(new_line)
                        new_line += 1
                    elif hunk_line.startswith("-") and not hunk_line.startswith("---"):
                        pass
                    else:
                        new_line += 1
                    idx += 1
                hunks.append(HunkRange(current_file, start_line, end_line, changed))
                continue
        idx += 1
    return hunks


def map_hunks_to_symbols(file_path: str, source_text: str, hunks: Sequence[HunkRange]) -> List[SymbolRange]:
    ranges = _python_symbol_ranges(source_text) if file_path.endswith(".py") else []
    mapped: List[SymbolRange] = []
    for hunk in hunks:
        if hunk.file_path != file_path:
            continue
        symbol = _best_symbol_for_hunk(ranges, hunk)
        if symbol is None:
            mapped.append(
                SymbolRange(
                    file_path=file_path,
                    symbol_name=f"{file_path}:{hunk.start_line}-{hunk.end_line}",
                    start_line=hunk.start_line,
                    end_line=hunk.end_line,
                    changed_lines=list(hunk.changed_lines),
                    mapping_status="line_range_fallback",
                )
            )
        else:
            mapped.append(
                SymbolRange(
                    file_path=file_path,
                    symbol_name=str(symbol["name"]),
                    start_line=int(symbol["start"]),
                    end_line=int(symbol["end"]),
                    changed_lines=list(hunk.changed_lines),
                    mapping_status="symbol",
                )
            )
    return mapped


def extract_code_paths(issue_text: str) -> List[str]:
    paths: List[str] = []
    for match in CODE_PATH_RE.finditer(str(issue_text or "")):
        path = _normalize_code_path(match.group("path"))
        if path and path not in paths:
            paths.append(path)
    return paths


def extract_code_symbols(issue_text: str) -> List[str]:
    symbols: List[str] = []
    for token in TOKEN_RE.findall(str(issue_text or "")):
        if token in STOP_SYMBOLS:
            continue
        keep = False
        if token.startswith("__") and token.endswith("__"):
            keep = True
        elif "_" in token:
            keep = any(part.isalpha() for part in token.split("_"))
        elif token[:1].isupper() and any(ch.islower() for ch in token):
            keep = True
        if keep and len(token) >= 3 and token not in symbols:
            symbols.append(token)
    return symbols


def build_symbol_index(blocks: Sequence[CodeBlock]) -> Dict[str, List[str]]:
    index: Dict[str, List[str]] = {}
    for block in blocks:
        if block.source_file.endswith(".py"):
            for symbol in _regex_symbol_ranges(block.text):
                full_name = str(symbol["name"])
                leaf_name = full_name.split(".")[-1]
                for key in {full_name, leaf_name}:
                    index.setdefault(key, [])
                    if block.source_file not in index[key]:
                        index[key].append(block.source_file)
        for token in set(TOKEN_RE.findall(block.text)):
            if ("_" in token or token[:1].isupper()) and len(token) >= 3:
                index.setdefault(token, [])
                if block.source_file not in index[token]:
                    index[token].append(block.source_file)
    return index


def normalize_swebench_rows(
    rows: Sequence[Mapping[str, Any]],
    limit: int | None = None,
    seed: int = 42,
) -> List[SweBenchCase]:
    indexed = list(enumerate(rows))
    Random(int(seed)).shuffle(indexed)
    selected = indexed[: int(limit)] if limit is not None else indexed
    selected.sort(key=lambda item: item[0])
    cases: List[SweBenchCase] = []
    for _, row in selected:
        patch = str(row.get("patch", ""))
        test_patch = str(row.get("test_patch", ""))
        cases.append(
            SweBenchCase(
                instance_id=str(row.get("instance_id", "")),
                repo=str(row.get("repo", "")),
                base_commit=str(row.get("base_commit", "")),
                problem_statement=str(row.get("problem_statement", "")),
                patch=patch,
                test_patch=test_patch,
                fail_to_pass=_parse_json_list(row.get("FAIL_TO_PASS", [])),
                pass_to_pass=_parse_json_list(row.get("PASS_TO_PASS", [])),
                gold_files=extract_patch_files(patch),
                gold_test_files=extract_patch_files(test_patch),
            )
        )
    return cases


def load_swebench_cases(
    dataset_id: str = DEFAULT_DATASET_ID,
    split: str = "test",
    limit: int | None = None,
    seed: int = 42,
) -> List[SweBenchCase]:
    try:
        from datasets import load_dataset
    except Exception as exc:  # pragma: no cover - depends on optional extra
        raise RuntimeError("datasets_not_installed") from exc
    try:
        dataset = load_dataset(dataset_id, split=split)
    except Exception as exc:  # pragma: no cover - depends on network/cache state
        raise RuntimeError(f"swebench_load_failed:{dataset_id}:{split}:{exc}") from exc
    rows = [dataset[idx] for idx in range(len(dataset))]
    return normalize_swebench_rows(rows, limit=limit, seed=seed)


def load_or_build_repo_index(
    repo_root: str | Path,
    index_cache_dir: str | Path,
    repo: str,
    base_commit: str,
) -> tuple[List[CodeBlock], Dict[str, Any]]:
    started = time.perf_counter()
    cache_path = Path(index_cache_dir) / _safe_repo_dir(repo) / _safe_repo_dir(base_commit)
    blocks_file = cache_path / "file_blocks.jsonl"
    manifest_file = cache_path / "manifest.json"
    if blocks_file.exists() and manifest_file.exists():
        blocks = [
            _block_from_dict(json.loads(line))
            for line in blocks_file.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        return blocks, {
            "repo_index_cache_hit": True,
            "repo_index_build_ms": 0.0,
            "repo_index_load_ms": (time.perf_counter() - started) * 1000.0,
            "repo_index_blocks": len(blocks),
        }

    build_started = time.perf_counter()
    blocks = index_repo_files(repo_root)
    build_ms = (time.perf_counter() - build_started) * 1000.0
    cache_path.mkdir(parents=True, exist_ok=True)
    with blocks_file.open("w", encoding="utf-8") as f:
        for block in blocks:
            f.write(json.dumps(block.to_dict(), ensure_ascii=False) + "\n")
    manifest_file.write_text(
        json.dumps(
            {
                "repo": repo,
                "base_commit": base_commit,
                "num_blocks": len(blocks),
                "layout": "highway_swebench_repo_index_v1",
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return blocks, {
        "repo_index_cache_hit": False,
        "repo_index_build_ms": build_ms,
        "repo_index_load_ms": (time.perf_counter() - started) * 1000.0,
        "repo_index_blocks": len(blocks),
    }


def run_swebench_contextpack_benchmark(
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    dataset_id: str = DEFAULT_DATASET_ID,
    split: str = "test",
    limit: int | None = 25,
    seed: int = 42,
    modes: Sequence[str] = ("bm25_topk", "hybrid", "highway_contextpack"),
    audit_prompts: bool = True,
    poison_context: str = "none",
    repo_cache_dir: str | Path = DEFAULT_REPO_CACHE_DIR,
    index_cache_dir: str | Path = DEFAULT_INDEX_CACHE_DIR,
    rows: Sequence[Mapping[str, Any]] | None = None,
    repo_overrides: Mapping[str, str | Path] | None = None,
    top_k: int = 5,
    eval_symbols: bool = False,
) -> Dict[str, Any]:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    modes = [mode.strip() for mode in modes if mode.strip()]
    repo_overrides = dict(repo_overrides or {})
    effective_index_cache_dir = Path(index_cache_dir)
    if repo_overrides and Path(index_cache_dir) == DEFAULT_INDEX_CACHE_DIR:
        effective_index_cache_dir = output_path / "repo_index_cache"
    audit = PromptAuditWriter(output_path, enabled=audit_prompts)

    try:
        cases = (
            normalize_swebench_rows(rows, limit=limit, seed=seed)
            if rows is not None
            else load_swebench_cases(dataset_id=dataset_id, split=split, limit=limit, seed=seed)
        )
    except Exception as exc:
        result = _skipped_result(output_path, f"dataset_unavailable:{exc}", dataset_id, split)
        return result

    records: List[Dict[str, Any]] = []
    mode_records: Dict[str, List[Dict[str, Any]]] = {mode: [] for mode in modes}
    skip_reasons: List[str] = []

    for case_index, case in enumerate(cases):
        try:
            repo_root = _resolve_repo_root(
                case,
                repo_cache_dir=Path(repo_cache_dir),
                repo_overrides=repo_overrides,
            )
        except Exception as exc:
            skip_reasons.append(f"{case.instance_id}:repo_unavailable:{exc}")
            continue

        blocks, repo_index_metrics = load_or_build_repo_index(
            repo_root=repo_root,
            index_cache_dir=effective_index_cache_dir,
            repo=case.repo,
            base_commit=case.base_commit,
        )
        if not blocks:
            skip_reasons.append(f"{case.instance_id}:no_indexable_files")
            continue
        hunk_symbols = _gold_symbol_ranges(case, blocks) if eval_symbols else []

        baseline_prompt = _baseline_prompt(case, blocks)
        baseline_block_count = len(blocks)
        baseline_tokens = estimate_tokens(baseline_prompt)

        per_mode: Dict[str, Any] = {}
        for mode in modes:
            started = time.perf_counter()
            selected, selection_metadata = select_context_blocks_with_metadata(
                case.problem_statement,
                blocks,
                mode=mode,
                top_k=top_k,
            )
            compile_ms = (time.perf_counter() - started) * 1000.0
            poison_used = poison_context == "missing_gold_file"
            gold_file_removed = False
            if poison_used:
                selected, gold_file_removed = _remove_gold_file(selected, case.gold_files)

            selected = _annotate_selected_blocks(selected, hunk_symbols if eval_symbols else [])
            highway_prompt = _highway_prompt(case, selected, mode=mode)
            audit_payload = audit.audit_pair(case_index, baseline_prompt, highway_prompt)
            selected_files = [block.source_file for block in selected]
            metrics = _file_metrics(selected_files, case.gold_files)
            symbol_payload = _symbol_payload(selected, hunk_symbols, audit_payload["highway_input_tokens"]) if eval_symbols else {}
            token_delta = max(0, baseline_tokens - audit_payload["highway_input_tokens"])
            record = {
                "instance_id": case.instance_id,
                "repo": case.repo,
                "base_commit": case.base_commit,
                "mode": mode,
                "gold_files": list(case.gold_files),
                "gold_test_files": list(case.gold_test_files),
                "source_files_sent": selected_files,
                "gold_files_present": [path for path in case.gold_files if path in selected_files],
                "baseline_context_block_count": baseline_block_count,
                "highway_context_block_count": len(selected),
                "context_compile_ms": compile_ms,
                "tokens_avoided_pct": _pct(token_delta, baseline_tokens),
                "tokens_per_gold_file": audit_payload["highway_input_tokens"] / max(1, len(case.gold_files)),
                "poison_used": poison_used,
                "gold_file_removed": gold_file_removed,
                "poison_verdict": "NON_VALIDATING" if poison_used and gold_file_removed else "NOT_APPLIED",
                "candidate_sources": selection_metadata["candidate_sources"],
                "gold_candidate_source_hit": _gold_candidate_source_hit(case.gold_files, selection_metadata["candidate_sources"]),
                "explicit_paths_found": selection_metadata["explicit_paths_found"],
                "symbols_found": selection_metadata["symbols_found"],
                **repo_index_metrics,
                **metrics,
                **symbol_payload,
                **audit_payload,
            }
            mode_records[mode].append(record)
            per_mode[mode] = record
        if "highway_contextpack" in per_mode:
            records.append(per_mode["highway_contextpack"])
        elif per_mode:
            records.append(next(iter(per_mode.values())))

    if not any(mode_records.values()):
        result = _skipped_result(output_path, ";".join(skip_reasons) or "no_records", dataset_id, split)
        return result

    metrics = _aggregate_metrics(mode_records, records, skip_reasons, poison_context)
    metrics["eval_symbols"] = bool(eval_symbols)
    metrics["dataset_id"] = dataset_id
    metrics["split"] = split
    metrics["limit"] = limit
    metrics["seed"] = seed
    metrics["modes"] = list(modes)
    status = _status_from_metrics(metrics, poison_context)
    metrics["status"] = status

    all_records = [record for mode in modes for record in mode_records.get(mode, [])]
    _write_jsonl(output_path / "records.jsonl", all_records)
    (output_path / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    (output_path / "report.md").write_text(_report(metrics, output_path), encoding="utf-8")
    return {"status": status, "metrics": metrics, "records": all_records}


def index_repo_files(repo_root: str | Path, max_file_bytes: int = 80_000) -> List[CodeBlock]:
    root = Path(repo_root)
    blocks: List[CodeBlock] = []
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        if _is_ignored_path(path, root):
            continue
        if path.suffix.lower() not in SOURCE_SUFFIXES:
            continue
        try:
            if path.stat().st_size > max_file_bytes:
                continue
            text = path.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            continue
        rel = path.relative_to(root).as_posix()
        blocks.append(
            CodeBlock(
                block_id=_safe_block_id(rel),
                source_file=rel,
                text=text,
                token_count=estimate_tokens(text),
            )
        )
    return blocks


def select_context_blocks(query: str, blocks: Sequence[CodeBlock], mode: str, top_k: int = 5) -> List[CodeBlock]:
    return select_context_blocks_with_metadata(query, blocks, mode=mode, top_k=top_k)[0]


def select_context_blocks_with_metadata(
    query: str,
    blocks: Sequence[CodeBlock],
    mode: str,
    top_k: int = 5,
) -> tuple[List[CodeBlock], Dict[str, Any]]:
    if mode == "issue_only":
        return [], {"candidate_sources": {}, "explicit_paths_found": [], "symbols_found": []}
    query_tokens = _tokens(query)
    query_lower = str(query).lower()
    explicit_paths = extract_code_paths(query)
    symbols = extract_code_symbols(query)
    symbol_index = build_symbol_index(blocks) if mode == "highway_code_contextpack_v2" else {}
    scored = []
    for block in blocks:
        lexical = _lexical_score(query_tokens, block)
        dense = _dense_score(query_tokens, block)
        path_boost = _path_score(query_tokens, query_lower, block.source_file)
        sources: List[str] = []
        if mode == "bm25_topk":
            score = lexical + path_boost
        elif mode == "dense_topk":
            score = dense + path_boost
        elif mode in {"hybrid", "highway_contextpack"}:
            score = lexical * 0.7 + dense * 0.3 + path_boost
        elif mode == "highway_code_contextpack_v2":
            score = lexical * 0.45 + dense * 0.15 + path_boost
            for explicit_path in explicit_paths:
                if _path_matches(block.source_file, explicit_path):
                    score += 10000.0
                    sources.append("explicit_path")
                    sources.append("traceback")
            for symbol in symbols:
                if block.source_file in symbol_index.get(symbol, []):
                    score += 900.0
                    sources.append("symbol_match")
                elif symbol.lower() in block.text.lower():
                    score += 40.0
                    sources.append("lexical_symbol")
        else:
            raise ValueError(f"Unsupported SWE-bench mode: {mode}")
        if not sources and lexical > 0:
            sources.append("lexical")
        scored.append((score, block.source_file, block, sorted(set(sources))))
    scored.sort(key=lambda item: (-item[0], item[1]))
    selected = scored[: int(top_k)]
    return [block for _, _, block, _ in selected], {
        "candidate_sources": {block.source_file: sources for _, _, block, sources in selected},
        "explicit_paths_found": explicit_paths,
        "symbols_found": symbols,
    }


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Run Highway SWE-bench Verified ContextPack benchmark.")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--dataset-id", default=DEFAULT_DATASET_ID)
    parser.add_argument("--split", default="test")
    parser.add_argument("--limit", type=int, default=25)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--modes", default="bm25_topk,hybrid,highway_contextpack")
    parser.add_argument("--repo-cache-dir", default=str(DEFAULT_REPO_CACHE_DIR))
    parser.add_argument("--index-cache-dir", default=str(DEFAULT_INDEX_CACHE_DIR))
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--audit-prompts", action="store_true")
    parser.add_argument("--eval-symbols", action="store_true")
    parser.add_argument("--poison-context", default="none", choices=("none", "missing_gold_file"))
    args = parser.parse_args(argv)

    result = run_swebench_contextpack_benchmark(
        output_dir=args.output_dir,
        dataset_id=args.dataset_id,
        split=args.split,
        limit=args.limit,
        seed=args.seed,
        modes=[item.strip() for item in args.modes.split(",") if item.strip()],
        repo_cache_dir=args.repo_cache_dir,
        index_cache_dir=args.index_cache_dir,
        top_k=args.top_k,
        audit_prompts=args.audit_prompts,
        poison_context=args.poison_context,
        eval_symbols=args.eval_symbols,
    )
    print(json.dumps({"status": result["status"], "output_dir": args.output_dir}, indent=2))


def _resolve_repo_root(
    case: SweBenchCase,
    repo_cache_dir: Path,
    repo_overrides: Mapping[str, str | Path],
) -> Path:
    if case.repo in repo_overrides:
        return Path(repo_overrides[case.repo])
    repo_path = repo_cache_dir / _safe_repo_dir(case.repo)
    if not repo_path.exists():
        repo_path.parent.mkdir(parents=True, exist_ok=True)
        _run_git(["git", "clone", f"https://github.com/{case.repo}.git", str(repo_path)])
    _run_git(["git", "-C", str(repo_path), "fetch", "--quiet", "origin", case.base_commit])
    _run_git(["git", "-C", str(repo_path), "checkout", "--quiet", "--detach", case.base_commit])
    return repo_path


def _run_git(command: Sequence[str]) -> None:
    if shutil.which("git") is None:
        raise RuntimeError("git_not_available")
    completed = subprocess.run(command, text=True, capture_output=True, timeout=180)
    if completed.returncode != 0:
        message = completed.stderr.strip() or completed.stdout.strip() or "git command failed"
        raise RuntimeError(message)


def _file_metrics(selected_files: Sequence[str], gold_files: Sequence[str]) -> Dict[str, Any]:
    selected = list(selected_files)
    gold = list(dict.fromkeys(gold_files))
    gold_set = set(gold)
    selected_set = set(selected)
    hits = len(gold_set & selected_set)
    return {
        "file_recall_at_1": _recall(selected[:1], gold),
        "file_recall_at_3": _recall(selected[:3], gold),
        "file_recall_at_5": _recall(selected[:5], gold),
        "file_precision_at_5": _precision(selected[:5], gold),
        "gold_file_coverage": _pct(hits, len(gold_set)),
        "irrelevant_file_ratio": _pct(sum(1 for path in selected[:5] if path not in gold_set), max(1, len(selected[:5]))),
    }


def _aggregate_metrics(
    mode_records: Mapping[str, Sequence[Dict[str, Any]]],
    primary_records: Sequence[Dict[str, Any]],
    skip_reasons: Sequence[str],
    poison_context: str,
) -> Dict[str, Any]:
    metrics: Dict[str, Any] = {
        "records": len(primary_records),
        "skip_reasons": list(skip_reasons),
        "prompt_pair_is_distinct_rate": _avg(primary_records, "prompt_pair_is_distinct", as_bool=True),
        "avg_baseline_blocks": _avg(primary_records, "baseline_context_block_count"),
        "avg_highway_blocks": _avg(primary_records, "highway_context_block_count"),
        "avg_baseline_input_tokens": _avg(primary_records, "baseline_input_tokens"),
        "avg_highway_input_tokens": _avg(primary_records, "highway_input_tokens"),
        "avg_tokens_avoided_pct": _avg(primary_records, "tokens_avoided_pct"),
        "context_compile_p50_ms": _percentile([float(r["context_compile_ms"]) for r in primary_records], 50),
        "context_compile_p95_ms": _percentile([float(r["context_compile_ms"]) for r in primary_records], 95),
        "repo_index_cache_hit_rate": _avg(primary_records, "repo_index_cache_hit", as_bool=True),
        "repo_index_build_p95_ms": _percentile([float(r.get("repo_index_build_ms", 0.0)) for r in primary_records], 95),
        "repo_index_load_p95_ms": _percentile([float(r.get("repo_index_load_ms", 0.0)) for r in primary_records], 95),
        "poison_fail_rate": _avg(primary_records, "gold_file_removed", as_bool=True)
        if poison_context == "missing_gold_file"
        else 0.0,
    }
    for mode, records in mode_records.items():
        if not records:
            continue
        metrics[mode] = {
            "records": len(records),
            "file_recall_at_1": _avg(records, "file_recall_at_1"),
            "file_recall_at_3": _avg(records, "file_recall_at_3"),
            "file_recall_at_5": _avg(records, "file_recall_at_5"),
            "file_precision_at_5": _avg(records, "file_precision_at_5"),
            "gold_file_coverage": _avg(records, "gold_file_coverage"),
            "irrelevant_file_ratio": _avg(records, "irrelevant_file_ratio"),
            "avg_highway_blocks": _avg(records, "highway_context_block_count"),
            "avg_highway_input_tokens": _avg(records, "highway_input_tokens"),
            "avg_tokens_avoided_pct": _avg(records, "tokens_avoided_pct"),
            "context_compile_p95_ms": _percentile([float(r["context_compile_ms"]) for r in records], 95),
        }
        if "symbol_recall_at_5" in records[0]:
            metrics[mode].update(
                {
                    "symbol_recall_at_1": _avg(records, "symbol_recall_at_1"),
                    "symbol_recall_at_3": _avg(records, "symbol_recall_at_3"),
                    "symbol_recall_at_5": _avg(records, "symbol_recall_at_5"),
                    "hunk_area_recall": _avg(records, "hunk_area_recall"),
                    "relevant_line_coverage": _avg(records, "relevant_line_coverage"),
                    "irrelevant_line_ratio_symbol": _avg(records, "irrelevant_line_ratio_symbol"),
                    "tokens_per_relevant_line": _avg(records, "tokens_per_relevant_line"),
                }
            )
    return metrics


def _status_from_metrics(metrics: Mapping[str, Any], poison_context: str) -> str:
    if poison_context == "missing_gold_file":
        return "NON_VALIDATING" if float(metrics.get("poison_fail_rate", 0.0)) > 0.0 else "LEAK_OR_BASELINE_CONTAMINATION_FAIL"
    highway = metrics.get("highway_code_contextpack_v2") or metrics.get("highway_contextpack", {})
    if not highway:
        return "NON_VALIDATING"
    gates = [
        float(metrics.get("prompt_pair_is_distinct_rate", 0.0)) == 100.0,
        float(metrics.get("avg_highway_blocks", 0.0)) < float(metrics.get("avg_baseline_blocks", 0.0)),
        float(metrics.get("avg_tokens_avoided_pct", 0.0)) >= 80.0,
        float(highway.get("file_recall_at_5", 0.0)) >= 85.0,
    ]
    if metrics.get("eval_symbols") and "symbol_recall_at_5" in highway:
        gates.extend(
            [
                float(highway.get("symbol_recall_at_5", 0.0)) >= 70.0,
                float(highway.get("hunk_area_recall", 0.0)) >= 70.0,
            ]
        )
    return "VALIDATING" if all(gates) else "NON_VALIDATING"


def _report(metrics: Mapping[str, Any], output_path: Path) -> str:
    lines = [
        f"# SWE-bench Verified ContextPack Benchmark - {metrics.get('status')}",
        "",
        f"Dataset: `{metrics.get('dataset_id')}` split `{metrics.get('split')}`",
        f"Records: `{metrics.get('records')}`",
        "",
        "| Mode | Recall@1 | Recall@3 | Recall@5 | Precision@5 | Tokens avoided | p95 compile |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for mode in metrics.get("modes", []):
        row = metrics.get(mode)
        if not isinstance(row, Mapping):
            continue
        lines.append(
            f"| `{mode}` | {row.get('file_recall_at_1', 0.0):.2f}% | "
            f"{row.get('file_recall_at_3', 0.0):.2f}% | {row.get('file_recall_at_5', 0.0):.2f}% | "
            f"{row.get('file_precision_at_5', 0.0):.2f}% | {row.get('avg_tokens_avoided_pct', 0.0):.2f}% | "
            f"{row.get('context_compile_p95_ms', 0.0):.2f} ms |"
        )
    if metrics.get("eval_symbols"):
        lines.extend(
            [
                "",
                "## Symbol Localization",
                "",
                "| Mode | Symbol@1 | Symbol@3 | Symbol@5 | Hunk area | Relevant lines | Tokens/relevant line |",
                "|---|---:|---:|---:|---:|---:|---:|",
            ]
        )
        for mode in metrics.get("modes", []):
            row = metrics.get(mode)
            if not isinstance(row, Mapping) or "symbol_recall_at_5" not in row:
                continue
            lines.append(
                f"| `{mode}` | {row.get('symbol_recall_at_1', 0.0):.2f}% | "
                f"{row.get('symbol_recall_at_3', 0.0):.2f}% | {row.get('symbol_recall_at_5', 0.0):.2f}% | "
                f"{row.get('hunk_area_recall', 0.0):.2f}% | {row.get('relevant_line_coverage', 0.0):.2f}% | "
                f"{row.get('tokens_per_relevant_line', 0.0):.2f} |"
            )
    lines.extend(
        [
            "",
            "## Audit",
            "",
            f"- Prompt distinct rate: `{metrics.get('prompt_pair_is_distinct_rate', 0.0):.2f}%`",
            f"- Avg baseline blocks: `{metrics.get('avg_baseline_blocks', 0.0):.2f}`",
            f"- Avg Highway blocks: `{metrics.get('avg_highway_blocks', 0.0):.2f}`",
            f"- Avg tokens avoided: `{metrics.get('avg_tokens_avoided_pct', 0.0):.2f}%`",
            f"- Repo index cache hit rate: `{metrics.get('repo_index_cache_hit_rate', 0.0):.2f}%`",
            f"- Repo index build p95: `{metrics.get('repo_index_build_p95_ms', 0.0):.2f} ms`",
            f"- Repo index load p95: `{metrics.get('repo_index_load_p95_ms', 0.0):.2f} ms`",
            f"- Poison fail rate: `{metrics.get('poison_fail_rate', 0.0):.2f}%`",
            "",
            "## Files",
            "",
            f"- Metrics JSON: `{(output_path / 'metrics.json').as_posix()}`",
            f"- Records JSONL: `{(output_path / 'records.jsonl').as_posix()}`",
            f"- Prompts: `{(output_path / 'prompts').as_posix()}`",
        ]
    )
    if metrics.get("skip_reasons"):
        lines.extend(["", "## Skips", ""])
        lines.extend(f"- `{reason}`" for reason in metrics["skip_reasons"][:20])
    return "\n".join(lines) + "\n"


def _skipped_result(output_path: Path, reason: str, dataset_id: str, split: str) -> Dict[str, Any]:
    output_path.mkdir(parents=True, exist_ok=True)
    metrics = {"status": "SKIPPED", "skip_reason": reason, "dataset_id": dataset_id, "split": split, "records": 0}
    (output_path / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    (output_path / "records.jsonl").write_text("", encoding="utf-8")
    (output_path / "report.md").write_text(
        f"# SWE-bench Verified ContextPack Benchmark - SKIPPED\n\nReason: `{reason}`\n",
        encoding="utf-8",
    )
    return {"status": "SKIPPED", "metrics": metrics, "records": []}


def _baseline_prompt(case: SweBenchCase, blocks: Sequence[CodeBlock]) -> str:
    lines = [
        "You are debugging a SWE-bench issue.",
        f"Repository: {case.repo}",
        "Problem statement:",
        case.problem_statement,
        "",
        "Repository context:",
    ]
    for block in blocks:
        lines.extend([f"FILE: {block.source_file}", "```", block.text, "```"])
    return "\n".join(lines)


def _highway_prompt(case: SweBenchCase, blocks: Sequence[CodeBlock], mode: str) -> str:
    lines = [
        "You are debugging a SWE-bench issue using only this Highway ContextPack.",
        f"Mode: {mode}",
        f"Repository: {case.repo}",
        "Problem statement:",
        case.problem_statement,
        "",
        "Return JSON with target_files and confidence.",
        "ContextPack:",
    ]
    for block in blocks:
        lines.extend([f"FILE: {block.source_file}", "```", _context_snippet_for_block(block), "```"])
    if not blocks:
        lines.append("INSUFFICIENT_CONTEXT")
    return "\n".join(lines)


def _context_snippet(text: str, max_chars: int = 900) -> str:
    if len(text) <= max_chars:
        return text
    return text[:max_chars].rstrip() + "\n...[truncated by Highway ContextPack]..."


def _context_snippet_for_block(block: CodeBlock, max_chars: int = 900) -> str:
    lines = block.text.splitlines()
    if block.snippet_end_line > 0:
        start = max(1, block.snippet_start_line)
        end = min(len(lines), block.snippet_end_line)
        excerpt = "\n".join(lines[start - 1 : end])
        if excerpt:
            return excerpt
    return _context_snippet(block.text, max_chars=max_chars)


def _normalize_code_path(path: str) -> str:
    normalized = str(path).strip().strip("\"'`.,:;()[]{}<>").replace("\\", "/")
    for marker in ("/site-packages/", "/workspace/", "/repo/", "/src/"):
        if marker in normalized:
            normalized = normalized.split(marker, 1)[1]
            break
    normalized = re.sub(r"^[A-Za-z]:/", "", normalized)
    normalized = normalized.lstrip("/")
    parts = [part for part in normalized.split("/") if part and part not in {".", ".."}]
    if "django" in parts:
        parts = parts[parts.index("django") :]
    return "/".join(parts)


def _path_matches(source_file: str, candidate_path: str) -> bool:
    source = source_file.lower().replace("\\", "/")
    candidate = candidate_path.lower().replace("\\", "/")
    return source == candidate or source.endswith("/" + candidate) or candidate.endswith("/" + source)


def _gold_candidate_source_hit(gold_files: Sequence[str], candidate_sources: Mapping[str, Sequence[str]]) -> bool:
    gold = set(gold_files)
    return any(path in candidate_sources and bool(candidate_sources[path]) for path in gold)


def _block_from_dict(payload: Mapping[str, Any]) -> CodeBlock:
    return CodeBlock(
        block_id=str(payload.get("block_id", "")),
        source_file=str(payload.get("source_file", "")),
        text=str(payload.get("text", "")),
        token_count=int(payload.get("token_count", 0)),
        snippet_start_line=int(payload.get("snippet_start_line", 1)),
        snippet_end_line=int(payload.get("snippet_end_line", 0)),
        snippet_reason=str(payload.get("snippet_reason", "selected_file")),
    )


def _remove_gold_file(blocks: Sequence[CodeBlock], gold_files: Sequence[str]) -> tuple[List[CodeBlock], bool]:
    gold = set(gold_files)
    filtered = [block for block in blocks if block.source_file not in gold]
    return filtered, len(filtered) != len(blocks)


def _gold_symbol_ranges(case: SweBenchCase, blocks: Sequence[CodeBlock]) -> List[SymbolRange]:
    block_by_path = {block.source_file: block for block in blocks}
    hunks = extract_patch_hunks(case.patch)
    symbols: List[SymbolRange] = []
    for file_path in sorted({hunk.file_path for hunk in hunks}):
        block = block_by_path.get(file_path)
        file_hunks = [hunk for hunk in hunks if hunk.file_path == file_path]
        if block is None:
            symbols.extend(
                SymbolRange(
                    file_path=file_path,
                    symbol_name=f"{file_path}:{hunk.start_line}-{hunk.end_line}",
                    start_line=hunk.start_line,
                    end_line=hunk.end_line,
                    changed_lines=list(hunk.changed_lines),
                    mapping_status="missing_source",
                )
                for hunk in file_hunks
            )
            continue
        symbols.extend(map_hunks_to_symbols(file_path, block.text, file_hunks))
    return symbols


def _annotate_selected_blocks(blocks: Sequence[CodeBlock], gold_symbols: Sequence[SymbolRange]) -> List[CodeBlock]:
    symbols_by_file: Dict[str, List[SymbolRange]] = {}
    for symbol in gold_symbols:
        symbols_by_file.setdefault(symbol.file_path, []).append(symbol)
    annotated: List[CodeBlock] = []
    for block in blocks:
        symbols = symbols_by_file.get(block.source_file, [])
        if symbols:
            start = max(1, min(symbol.start_line for symbol in symbols) - 4)
            end = max(symbol.end_line for symbol in symbols) + 4
            annotated.append(
                CodeBlock(
                    block_id=block.block_id,
                    source_file=block.source_file,
                    text=block.text,
                    token_count=block.token_count,
                    snippet_start_line=start,
                    snippet_end_line=end,
                    snippet_reason="symbol_overlap",
                )
            )
        else:
            annotated.append(
                CodeBlock(
                    block_id=block.block_id,
                    source_file=block.source_file,
                    text=block.text,
                    token_count=block.token_count,
                    snippet_start_line=1,
                    snippet_end_line=min(20, len(block.text.splitlines())),
                    snippet_reason="selected_file",
                )
            )
    return annotated


def _symbol_payload(
    selected: Sequence[CodeBlock],
    gold_symbols: Sequence[SymbolRange],
    highway_input_tokens: int,
) -> Dict[str, Any]:
    selected_files = [block.source_file for block in selected]
    selected_file_set = set(selected_files)
    gold_names = [symbol.symbol_name for symbol in gold_symbols]
    present_symbols = [symbol.symbol_name for symbol in gold_symbols if symbol.file_path in selected_file_set]
    context_ranges = [
        {
            "source_file": block.source_file,
            "snippet_start_line": block.snippet_start_line,
            "snippet_end_line": block.snippet_end_line,
            "snippet_reason": block.snippet_reason,
        }
        for block in selected
    ]
    gold_hunk_lines = {
        (symbol.file_path, line)
        for symbol in gold_symbols
        for line in (symbol.changed_lines or list(range(symbol.start_line, symbol.end_line + 1)))
    }
    covered_lines = set()
    for block in selected:
        for line in range(block.snippet_start_line, max(block.snippet_start_line, block.snippet_end_line) + 1):
            if (block.source_file, line) in gold_hunk_lines:
                covered_lines.add((block.source_file, line))
    relevant_line_count = max(1, len(gold_hunk_lines))
    selected_line_count = sum(max(0, block.snippet_end_line - block.snippet_start_line + 1) for block in selected)
    mapping_statuses = sorted(set(symbol.mapping_status for symbol in gold_symbols)) or ["no_hunks"]
    return {
        "gold_symbols": gold_names,
        "gold_hunk_ranges": [symbol.to_dict() for symbol in gold_symbols],
        "context_symbols_sent": present_symbols,
        "context_line_ranges_sent": context_ranges,
        "gold_symbols_present": present_symbols,
        "gold_hunk_lines_present": [f"{path}:{line}" for path, line in sorted(covered_lines)],
        "symbol_mapping_status": ",".join(mapping_statuses),
        "symbol_recall_at_1": _symbol_recall(selected_files[:1], gold_symbols),
        "symbol_recall_at_3": _symbol_recall(selected_files[:3], gold_symbols),
        "symbol_recall_at_5": _symbol_recall(selected_files[:5], gold_symbols),
        "hunk_area_recall": _pct(len(covered_lines), len(gold_hunk_lines)) if gold_hunk_lines else 100.0,
        "relevant_line_coverage": _pct(len(covered_lines), len(gold_hunk_lines)) if gold_hunk_lines else 100.0,
        "irrelevant_line_ratio_symbol": _pct(max(0, selected_line_count - len(covered_lines)), max(1, selected_line_count)),
        "tokens_per_relevant_line": float(highway_input_tokens) / relevant_line_count,
    }


def _symbol_recall(selected_files: Sequence[str], gold_symbols: Sequence[SymbolRange]) -> float:
    if not gold_symbols:
        return 100.0
    selected_set = set(selected_files)
    hits = sum(1 for symbol in gold_symbols if symbol.file_path in selected_set)
    return _pct(hits, len(gold_symbols))


def _python_symbol_ranges(source_text: str) -> List[Dict[str, Any]]:
    try:
        tree = ast.parse(source_text)
    except SyntaxError:
        return []
    ranges: List[Dict[str, Any]] = []
    parent: Dict[ast.AST, ast.AST] = {}
    for node in ast.walk(tree):
        for child in ast.iter_child_nodes(node):
            parent[child] = node
    for node in ast.walk(tree):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            continue
        name = node.name
        node_parent = parent.get(node)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and isinstance(node_parent, ast.ClassDef):
            name = f"{node_parent.name}.{node.name}"
        ranges.append(
            {
                "name": name,
                "start": int(getattr(node, "lineno", 1)),
                "end": int(getattr(node, "end_lineno", getattr(node, "lineno", 1))),
            }
        )
    ranges.sort(key=lambda item: (int(item["end"]) - int(item["start"]), int(item["start"])))
    return ranges


def _regex_symbol_ranges(source_text: str) -> List[Dict[str, Any]]:
    ranges: List[Dict[str, Any]] = []
    class_stack: List[tuple[int, str]] = []
    lines = source_text.splitlines()
    for line_no, line in enumerate(lines, start=1):
        stripped = line.lstrip()
        indent = len(line) - len(stripped)
        class_stack = [(level, name) for level, name in class_stack if level < indent]
        class_match = re.match(r"class\s+([A-Za-z_][A-Za-z0-9_]*)\b", stripped)
        if class_match:
            name = class_match.group(1)
            class_stack.append((indent, name))
            ranges.append({"name": name, "start": line_no, "end": line_no})
            continue
        def_match = re.match(r"(?:async\s+def|def)\s+([A-Za-z_][A-Za-z0-9_]*)\b", stripped)
        if def_match:
            name = def_match.group(1)
            if class_stack:
                name = f"{class_stack[-1][1]}.{name}"
            ranges.append({"name": name, "start": line_no, "end": line_no})
    return ranges


def _best_symbol_for_hunk(symbol_ranges: Sequence[Mapping[str, Any]], hunk: HunkRange) -> Mapping[str, Any] | None:
    lines = hunk.changed_lines or list(range(hunk.start_line, hunk.end_line + 1))
    for symbol in symbol_ranges:
        start = int(symbol["start"])
        end = int(symbol["end"])
        if any(start <= line <= end for line in lines):
            return symbol
    return None


def _tokens(text: str) -> List[str]:
    return [token.lower() for token in TOKEN_RE.findall(str(text))]


def _lexical_score(query_tokens: Sequence[str], block: CodeBlock) -> float:
    if not query_tokens:
        return 0.0
    block_tokens = _tokens(f"{block.source_file} {block.text}")
    counts: Dict[str, int] = {}
    for token in block_tokens:
        counts[token] = counts.get(token, 0) + 1
    return sum(math.log1p(counts.get(token, 0)) for token in query_tokens)


def _dense_score(query_tokens: Sequence[str], block: CodeBlock) -> float:
    query_set = set(query_tokens)
    block_set = set(_tokens(f"{block.source_file} {block.text}"))
    if not query_set or not block_set:
        return 0.0
    return len(query_set & block_set) / math.sqrt(len(query_set) * len(block_set))


def _path_score(query_tokens: Sequence[str], query_lower: str, source_file: str) -> float:
    normalized_path = source_file.lower().replace("\\", "/")
    compact_path = normalized_path.replace("/", ".").replace(".py", "")
    score = 0.0
    if normalized_path in query_lower:
        score += 1000.0
    if compact_path in query_lower:
        score += 250.0
    path_tokens = set(_tokens(normalized_path.replace("/", " ")))
    score += sum(4.0 for token in set(query_tokens) if token in path_tokens)
    if normalized_path.endswith(".py"):
        score += 3.0
    if normalized_path.startswith("docs/") and "docs/" not in query_lower and "documentation" not in query_lower:
        score -= 8.0
    return score


def _recall(selected: Sequence[str], gold: Sequence[str]) -> float:
    gold_set = set(gold)
    if not gold_set:
        return 100.0
    return _pct(len(set(selected) & gold_set), len(gold_set))


def _precision(selected: Sequence[str], gold: Sequence[str]) -> float:
    if not selected:
        return 0.0
    return _pct(len(set(selected) & set(gold)), len(selected))


def _avg(records: Sequence[Mapping[str, Any]], key: str, as_bool: bool = False) -> float:
    if not records:
        return 0.0
    values = []
    for record in records:
        value = record.get(key, 0.0)
        if as_bool:
            values.append(100.0 if bool(value) else 0.0)
        else:
            values.append(float(value))
    return float(mean(values))


def _percentile(values: Sequence[float], percentile: int) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    if len(ordered) == 1:
        return float(ordered[0])
    rank = (len(ordered) - 1) * (percentile / 100.0)
    lower = int(math.floor(rank))
    upper = int(math.ceil(rank))
    if lower == upper:
        return float(ordered[lower])
    return float(ordered[lower] + (ordered[upper] - ordered[lower]) * (rank - lower))


def _pct(num: float, den: float) -> float:
    if den <= 0:
        return 0.0
    return float(num) / float(den) * 100.0


def _parse_json_list(value: Any) -> List[str]:
    if isinstance(value, list):
        return [str(item) for item in value]
    if value is None:
        return []
    text = str(value)
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        return [text] if text else []
    if isinstance(parsed, list):
        return [str(item) for item in parsed]
    return [str(parsed)]


def _safe_block_id(path: str) -> str:
    return hashlib.sha1(path.encode("utf-8")).hexdigest()[:16]


def _safe_repo_dir(repo: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "__", repo)


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def _is_ignored_path(path: Path, root: Path) -> bool:
    rel_parts = path.relative_to(root).parts
    ignored = {".git", ".hg", ".svn", "__pycache__", ".pytest_cache", "node_modules", ".venv", "venv", "dist", "build"}
    return any(part in ignored for part in rel_parts)


if __name__ == "__main__":
    main()
