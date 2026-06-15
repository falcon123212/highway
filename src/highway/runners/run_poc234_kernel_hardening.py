import argparse
import json
import os
import random
import re
import time
import unicodedata
from typing import Dict, List, Tuple

import numpy as np



ExecutionScheduler = None


PREDICTABLE_ID_RE = re.compile(r"(^[gh]_adv_\d+$|^q_\d+$|g_adv_|h_adv_)", re.IGNORECASE)
ORACLE_SOURCE_RE = re.compile(r"(g_adv_|h_adv_|adv_doc_\d+)", re.IGNORECASE)
OPAQUE_ID_RE = re.compile(r"^q_[0-9a-f]{16}$")


def _scheduler_class():
    global ExecutionScheduler
    if ExecutionScheduler is None:
        from highway.runtime.scheduler import ExecutionScheduler as Scheduler

        ExecutionScheduler = Scheduler
    return ExecutionScheduler


def clean_answer(text: object) -> str:
    text_clean = str(text).lower().strip().replace("$", "").replace(",", "")
    text_clean = text_clean.replace("project ", "").replace("project", "")
    nfkd_form = unicodedata.normalize("NFKD", text_clean)
    text_clean = "".join(c for c in nfkd_form if not unicodedata.combining(c))
    parts = sorted(p.strip() for p in re.split(r"[\s,]+", text_clean) if p.strip())
    return " ".join(parts)


def leak_check_query(query: Dict[str, object]) -> Tuple[bool, List[str]]:
    category = str(query.get("category", ""))
    if category and category not in {"G", "H"}:
        return True, []

    reasons = []
    q_id = str(query.get("id", ""))
    source_file = str(query.get("source_file", "")).replace("\\", "/")

    if not OPAQUE_ID_RE.fullmatch(q_id) or PREDICTABLE_ID_RE.search(q_id):
        reasons.append("predictable_query_id")

    if q_id and q_id in source_file:
        reasons.append("query_id_embedded_in_source_file")

    if ORACLE_SOURCE_RE.search(source_file):
        reasons.append("oracle_encoded_source_file")

    if source_file and not re.fullmatch(r"noise/poc234_[A-Za-z0-9_]+/doc_[0-9a-f]{16}\.txt", source_file):
        reasons.append("non_opaque_source_file")

    return len(reasons) == 0, reasons


def _read_workload(path: str) -> List[Dict[str, object]]:
    queries = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                queries.append(json.loads(line))
    return queries


def _ensure_parent(path: str) -> None:
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)


def _write_summary(path: str, run_name: str, output: str, records: List[Dict[str, object]]) -> None:
    _ensure_parent(path)
    total = len(records)
    leak_failures = [r for r in records if not r.get("leak_check_passed", False)]
    em = sum(1 for r in records if r.get("is_em", False))
    em_rate = (em / total * 100.0) if total else 0.0
    leak_rate = ((total - len(leak_failures)) / total * 100.0) if total else 100.0

    status = "VALIDATING" if not leak_failures else "HISTORICAL_NON_VALIDATING"
    warning = ""
    if leak_failures:
        warning = (
            "\nWARNING: This run is historical/non-validating because at least one "
            "record failed the no-leak workload checks.\n"
        )

    summary = (
        "# POC 2.3.4/2.3.5 No-Leak Kernel Hardening Summary\n\n"
        f"- Run name: {run_name}\n"
        f"- Output: {output}\n"
        f"- Validation status: {status}\n"
        f"- Records: {total}\n"
        f"- No-leak pass rate: {leak_rate:.2f}%\n"
        f"- Exact match rate after leak gate: {em_rate:.2f}%\n"
        f"{warning}"
    )
    with open(path, "w", encoding="utf-8") as f:
        f.write(summary)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-name", type=str, required=True)
    parser.add_argument("--corpus", type=str, required=True)
    parser.add_argument("--workload", type=str, required=True)
    parser.add_argument("--modes", type=str, default="pccc_compute_kernels")
    parser.add_argument("--enable-comparison-kernel", type=str, default="true")
    parser.add_argument("--enable-aggregation-kernel", type=str, default="true")
    parser.add_argument("--enable-kernel-verifier", type=str, default="true")
    parser.add_argument("--enable-entity-canonicalizer", type=str, default="true")
    parser.add_argument("--enable-budget-normalizer", type=str, default="true")
    parser.add_argument("--enable-active-evidence-filter", type=str, default="true")
    parser.add_argument("--disable-llm-for-computable", type=str, default="true")
    parser.add_argument("--timeout", type=int, default=60)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--summary", type=str, required=True)
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    queries = _read_workload(args.workload)
    print(f"Loaded {len(queries)} queries from workload.")

    modes = [m.strip() for m in args.modes.split(",") if m.strip()]
    print(f"Modes to run: {modes}")

    corpus_dir = os.path.dirname(args.corpus.rstrip("/\\"))
    cache_dir = os.path.join(corpus_dir, "cache")
    os.makedirs(cache_dir, exist_ok=True)

    Scheduler = _scheduler_class()
    scheduler = Scheduler(args.corpus, cache_dir)
    disable_llm = args.disable_llm_for_computable.lower() == "true"

    _ensure_parent(args.output)
    records = []

    with open(args.output, "w", encoding="utf-8") as out_file:
        for idx, q in enumerate(queries):
            q_id = q["id"]
            question = q["question"]
            expected = q["expected_answer"]
            cat = q["category"]
            leak_ok, leak_reasons = leak_check_query(q)

            for mode in modes:
                t_start = time.time()
                res = scheduler.answer(
                    question,
                    use_cache=False,
                    force_llm=False,
                    q_id=q_id,
                    disable_llm_for_computable=disable_llm,
                )

                answer = res["answer"]
                route = res["route"]
                metrics = res["metrics"]
                answer_matches_expected = clean_answer(answer) == clean_answer(expected)
                is_em = answer_matches_expected and leak_ok

                record = {
                    "id": q_id,
                    "question": question,
                    "expected": expected,
                    "expected_answer": expected,
                    "generated": answer,
                    "answer": answer,
                    "category": cat,
                    "mode": mode,
                    "answer_matches_expected": answer_matches_expected,
                    "leak_check_passed": leak_ok,
                    "leak_check_reasons": leak_reasons,
                    "is_em": is_em,
                    "exact_match": is_em,
                    "route": route,
                    "latency_ms": (time.time() - t_start) * 1000.0,
                    "is_bypass": metrics.get("llm_bypass", True),
                    "verify_passed": metrics.get("verifier_passed", True),
                    "prompt_tokens": metrics.get("prompt_tokens", 0),
                    "metrics": metrics,
                    "metadata": q.get("metadata"),
                }

                records.append(record)
                out_file.write(json.dumps(record, ensure_ascii=False) + "\n")
                out_file.flush()

            if (idx + 1) % 50 == 0 or (idx + 1) == len(queries):
                print(f"Processed {idx + 1}/{len(queries)} queries...")

    _write_summary(args.summary, args.run_name, args.output, records)
    print("Benchmark run completed successfully.")


if __name__ == "__main__":
    main()



