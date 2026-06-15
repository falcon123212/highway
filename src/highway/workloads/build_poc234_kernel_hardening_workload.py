import argparse
import hashlib
import json
import os
import random
from typing import Dict, List, Tuple


PROJECT_NAMES = [
    "NEPTUNE", "KRONOS", "ECLIPSE", "FALCON", "IRIS", "JUPITER", "METEOR", "NEXUS",
    "ORION", "PHOENIX", "QUASAR", "SIRIUS", "TITAN", "VEGA", "ZENITH", "AURORA",
    "BEACON", "CHRONOS", "DAWN", "GENESIS", "HELIOS", "LUNA", "PEGASE", "SOLARIS",
]

PEOPLE = [
    "Jean Dupont", "Alice Martin", "Pierre Leroy", "Marie Dubois", "Thomas Petit",
    "Sophie Richard", "Michel Bernard", "Julie Thomas", "Nicolas Durand", "Emma Michel",
]

MANAGER_ALIASES = {
    "Emma Michel": ["emma michel", "Emma M.", "EMMA MICHEL"],
    "Jean Dupont": ["jean dupont", "Jean D.", "JEAN DUPONT"],
    "Alice Martin": ["alice martin", "Alice M.", "ALICE MARTIN"],
    "Pierre Leroy": ["pierre leroy", "Pierre L.", "PIERRE LEROY"],
    "Marie Dubois": ["marie dubois", "Marie D.", "MARIE DUBOIS"],
    "Thomas Petit": ["thomas petit", "Thomas P.", "THOMAS PETIT"],
    "Sophie Richard": ["sophie richard", "Sophie R.", "SOPHIE RICHARD"],
    "Michel Bernard": ["michel bernard", "Michel B.", "MICHEL BERNARD"],
    "Julie Thomas": ["julie thomas", "Julie T.", "JULIE THOMAS"],
    "Nicolas Durand": ["nicolas durand", "Nicolas D.", "N. Durand", "NICOLAS DURAND"],
}


def _stable_hex(*parts: object, length: int = 16) -> str:
    payload = "|".join(str(part) for part in parts)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:length]


def _make_query_id(seed: int, category: str, index: int, question: str, expected_answer: str) -> str:
    return f"q_{_stable_hex('query', seed, category, index, question, expected_answer)}"


def _make_doc_filename(seed: int, category: str, index: int, question: str) -> str:
    return f"doc_{_stable_hex('document', seed, category, index, question)}.txt"


def _make_reference_marker(seed: int, category: str, index: int) -> str:
    return f"ref_{_stable_hex('reference', seed, category, index, length=10)}"


def _format_budget(value: int, variant: int) -> str:
    formats = [
        lambda val: f"${val:,}",
        lambda val: f"{val:,} USD",
        lambda val: f"USD {val // 1000}k",
        lambda val: f"{val // 1000} 000 dollars",
        lambda val: f"{val / 1000000:.3f}M",
        lambda val: f"{val}",
    ]
    return formats[variant % len(formats)](value)


def _comparison_case(rng: random.Random, seed: int, index: int) -> Tuple[Dict[str, object], str]:
    sub_type = index % 6 + 1
    marker = _make_reference_marker(seed, "G", index)
    proj_a, proj_b = rng.sample(PROJECT_NAMES, 2)
    budget_a = rng.randint(100, 999) * 1000
    budget_b = rng.randint(100, 999) * 1000
    while budget_a == budget_b:
        budget_b = rng.randint(100, 999) * 1000

    doc_lines = []
    question = f"In reference {marker} which project has a higher budget: Project {proj_a} or Project {proj_b}?"
    is_missing = False

    if sub_type == 1:
        doc_lines.append(f"The budget for Project {proj_a} is ${budget_a:,}.")
        doc_lines.append(f"Approved Budget for Project {proj_b}: ${budget_b:,}.")

    elif sub_type == 2:
        doc_lines.append(f"Project {proj_a} budget: {_format_budget(budget_a, rng.randint(1, 5))}.")
        doc_lines.append(f"Project {proj_b} budget: {_format_budget(budget_b, rng.randint(1, 5))}.")

    elif sub_type == 3:
        old_budget_a = rng.randint(100, 999) * 1000
        doc_lines.append(f"Old memo: Project {proj_a} budget was ${old_budget_a:,}.")
        doc_lines.append(f"Latest approved budget: Project {proj_a} = ${budget_a:,}.")
        doc_lines.append(f"Project {proj_b} budget is ${budget_b:,}.")

    elif sub_type == 4:
        budget_b = budget_a + 1000
        doc_lines.append(f"Project {proj_a} budget is ${budget_a:,}.")
        doc_lines.append(f"Project {proj_b} budget is ${budget_b:,}.")

    elif sub_type == 5:
        alias_a = rng.choice([proj_a.lower(), f"project {proj_a.lower()}", proj_a.capitalize()])
        alias_b = rng.choice([proj_b.lower(), f"Project {proj_b.capitalize()}", proj_b])
        doc_lines.append(f"{alias_a} has a budget of ${budget_a:,}.")
        doc_lines.append(f"{alias_b} budget: ${budget_b:,}.")

    else:
        is_missing = True
        if rng.choice([True, False]):
            doc_lines.append(f"Project {proj_a} budget is not defined yet.")
            doc_lines.append(f"Project {proj_b} budget: ${budget_b:,}.")
        else:
            doc_lines.append(f"Project {proj_a} budget: ${budget_a:,}.")
            doc_lines.append(f"Budget for Project {proj_b} remains ambiguous.")

    if is_missing:
        expected_answer = "KERNEL_MISSING_FIELD"
    else:
        winner = proj_a if budget_a > budget_b else proj_b
        winner_value = budget_a if budget_a > budget_b else budget_b
        expected_answer = f"Project {winner} (budget of ${winner_value:,})"

    q_id = _make_query_id(seed, "G", index, question, expected_answer)
    doc_filename = _make_doc_filename(seed, "G", index, question)
    doc_text = f"Reference {marker} contains synthetic comparison evidence.\n" + "\n".join(doc_lines) + "\n"

    query = {
        "id": q_id,
        "question": question,
        "expected_answer": expected_answer,
        "category": "G",
        "difficulty": "hard",
        "evidence_quote": "Adversarial simulated evidence block.",
        "reasoning": f"Hardened comparison category {sub_type} testing robust kernel.",
        "metadata": {
            "type": "G",
            "sub_type": sub_type,
            "proj_a": proj_a,
            "proj_b": proj_b,
            "budget_a": budget_a,
            "budget_b": budget_b,
            "is_missing": is_missing,
        },
        "_doc_filename": doc_filename,
    }
    return query, doc_text


def _aggregation_case(rng: random.Random, seed: int, index: int) -> Tuple[Dict[str, object], str]:
    sub_type = index % 6 + 1
    marker = _make_reference_marker(seed, "H", index)
    manager = rng.choice(PEOPLE)
    assigned_projects = rng.sample(PROJECT_NAMES, rng.randint(2, 4))
    expected_projects = list(assigned_projects)
    doc_lines = []
    question = f"In reference {marker} list all project names managed by {manager}."
    is_missing = False

    if sub_type == 1:
        for proj in assigned_projects:
            doc_lines.append(f"Project {proj} is managed by {manager}.")

    elif sub_type == 2:
        aliases = MANAGER_ALIASES[manager]
        for idx, proj in enumerate(assigned_projects):
            doc_lines.append(f"Project {proj} is led by {aliases[idx % len(aliases)]}.")

    elif sub_type == 3:
        for proj in assigned_projects:
            doc_lines.append(f"Project {proj} is managed by {manager}.")
            doc_lines.append(f"Contract indicates Project {proj} is managed by {manager}.")

    elif sub_type == 4:
        obsolete_proj = rng.choice([p for p in PROJECT_NAMES if p not in assigned_projects])
        other_mgr = rng.choice([p for p in PEOPLE if p != manager])
        doc_lines.append(f"Old record: {manager} managed Project {obsolete_proj}.")
        doc_lines.append(f"Current record: Project {obsolete_proj} is reassigned to {other_mgr}.")
        for proj in assigned_projects:
            doc_lines.append(f"Project {proj} is currently managed by {manager}.")

    elif sub_type == 5:
        renamed_index = rng.randint(0, len(assigned_projects) - 1)
        old_proj = assigned_projects[renamed_index]
        new_proj = f"{old_proj}-R"
        assigned_projects[renamed_index] = new_proj
        expected_projects = list(assigned_projects)
        doc_lines.append(f"Project {old_proj} was renamed to Project {new_proj}.")
        doc_lines.append(f"Project {new_proj} is managed by {manager}.")
        for idx, proj in enumerate(assigned_projects):
            if idx != renamed_index:
                doc_lines.append(f"Project {proj} is managed by {manager}.")

    else:
        is_missing = True
        other_mgr = rng.choice([p for p in PEOPLE if p != manager])
        for proj in assigned_projects:
            doc_lines.append(f"Project {proj} is managed by {other_mgr}.")

    expected_answer = "NOT_FOUND" if is_missing else ", ".join(sorted(expected_projects))
    q_id = _make_query_id(seed, "H", index, question, expected_answer)
    doc_filename = _make_doc_filename(seed, "H", index, question)
    doc_text = f"Reference {marker} contains synthetic aggregation evidence.\n" + "\n".join(doc_lines) + "\n"

    query = {
        "id": q_id,
        "question": question,
        "expected_answer": expected_answer,
        "category": "H",
        "difficulty": "hard",
        "evidence_quote": "Adversarial simulated evidence block.",
        "reasoning": f"Hardened aggregation category {sub_type} testing robust kernel.",
        "metadata": {
            "type": "H",
            "sub_type": sub_type,
            "manager": manager,
            "projects": expected_projects,
            "is_missing": is_missing,
        },
        "_doc_filename": doc_filename,
    }
    return query, doc_text


def _write_jsonl(path: str, records: List[Dict[str, object]]) -> None:
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def _run_ingestion(corpus_dir: str, index_dir: str) -> None:
    from highway.ingestion.ingest import ingest_corpus
    from highway.storage.out_of_core_index import OutOfCoreIndex

    layout = "out_of_core" if OutOfCoreIndex.is_out_of_core_index(index_dir) else "legacy"
    ingest_corpus(corpus_dir, index_dir, layout=layout)


def _mixed_output_path(output: str) -> str:
    if output.endswith("_400.jsonl"):
        return output.replace("_400.jsonl", "_full_mixed_800.jsonl")
    return os.path.join(os.path.dirname(output), "poc234_full_mixed_800.jsonl")


def generate_workload(
    corpus: str,
    output: str,
    n_comparison: int = 200,
    n_aggregation: int = 200,
    seed: int = 42,
    run_ingest: bool = True,
    write_mixed: bool = True,
) -> List[Dict[str, object]]:
    rng = random.Random(seed)
    corpus_dir = os.path.dirname(corpus.rstrip("/\\"))
    namespace = f"poc234_{seed}_g{n_comparison}_h{n_aggregation}"
    generated_doc_dir = os.path.join(corpus_dir, "documents", "noise", namespace)
    os.makedirs(generated_doc_dir, exist_ok=True)

    queries = []
    generated_docs = []

    for idx in range(n_comparison):
        query, doc_text = _comparison_case(rng, seed, idx)
        queries.append(query)
        generated_docs.append((query["_doc_filename"], doc_text))

    for idx in range(n_aggregation):
        query, doc_text = _aggregation_case(rng, seed, idx)
        queries.append(query)
        generated_docs.append((query["_doc_filename"], doc_text))

    for query, (doc_filename, doc_text) in zip(queries, generated_docs):
        doc_path = os.path.join(generated_doc_dir, doc_filename)
        with open(doc_path, "w", encoding="utf-8") as f:
            f.write(doc_text)
        query["source_file"] = f"noise/{namespace}/{doc_filename}"
        del query["_doc_filename"]

    if run_ingest:
        print("Re-running ingestion with the current Python interpreter...")
        _run_ingestion(corpus_dir, corpus)
        print("Ingestion succeeded.")

    _write_jsonl(output, queries)
    print(f"Saved targeted no-leak workload of {len(queries)} queries to {output}.")

    if write_mixed:
        gold_qa_path = os.path.join(corpus_dir, "questions", "qa_gold.json")
        if os.path.exists(gold_qa_path):
            with open(gold_qa_path, "r", encoding="utf-8") as f:
                gold_qa = json.load(f)
            gold_af = [q for q in gold_qa if q.get("category") in ["A", "B", "C", "D", "E", "F"]]
            sampled_af = rng.sample(gold_af, min(400, len(gold_af)))
            combined = queries + sampled_af
            rng.shuffle(combined)
            combined_path = _mixed_output_path(output)
            _write_jsonl(combined_path, combined)
            print(f"Saved combined workload of {len(combined)} queries to {combined_path}.")
        else:
            print(f"Skipping mixed workload: {gold_qa_path} not found.")

    return queries


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--n-comparison", type=int, default=200)
    parser.add_argument("--n-aggregation", type=int, default=200)
    parser.add_argument("--include-budget-format-noise", type=str, default="true")
    parser.add_argument("--include-obsolete-values", type=str, default="true")
    parser.add_argument("--include-duplicates", type=str, default="true")
    parser.add_argument("--include-aliases", type=str, default="true")
    parser.add_argument("--include-missing-fields", type=str, default="true")
    parser.add_argument("--include-renamed-projects", type=str, default="true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--skip-ingest", action="store_true")
    args = parser.parse_args()

    generate_workload(
        corpus=args.corpus,
        output=args.output,
        n_comparison=args.n_comparison,
        n_aggregation=args.n_aggregation,
        seed=args.seed,
        run_ingest=not args.skip_ingest,
        write_mixed=True,
    )


if __name__ == "__main__":
    main()


