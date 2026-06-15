import os
import json
import argparse
import random
import re
import subprocess
from build_poc24_true_synthesis_workload import CLEAN_PROJECTS, PROJECT_NAMES, PEOPLE, DEPARTMENTS

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--n-deterministic", type=int, default=300)
    parser.add_argument("--n-comparison", type=int, default=200)
    parser.add_argument("--n-aggregation", type=int, default=200)
    parser.add_argument("--n-true-synthesis", type=int, default=300)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)

    corpus_dir = os.path.dirname(args.corpus.rstrip("/\\"))
    noise_doc_dir = os.path.join(corpus_dir, "documents", "noise")
    os.makedirs(noise_doc_dir, exist_ok=True)

    # 1. Load gold QA for A-F
    gold_qa_path = os.path.join(corpus_dir, "questions", "qa_gold.json")
    with open(gold_qa_path, "r", encoding="utf-8") as f:
        gold_qa = json.load(f)
    gold_af = [q for q in gold_qa if q["category"] in ["A", "B", "C", "D", "E", "F"]]
    
    # Sample 300 deterministic queries
    sampled_af = random.sample(gold_af, min(args.n_deterministic, len(gold_af)))
    for q in sampled_af:
        q["route"] = "DETERMINISTIC" if q["category"] != "E" else "NOT_FOUND"

    mixed_queries = []
    mixed_queries.extend(sampled_af)

    noise_docs_content = []

    # 2. Generate G comparison queries (200)
    formats = [
        lambda val: f"${val:,}",
        lambda val: f"{val:,} USD",
        lambda val: f"USD {val//1000}k",
        lambda val: f"{val//1000} 000 dollars",
        lambda val: f"{val/1000000:.3f}M",
        lambda val: f"{val}"
    ]
    for i in range(args.n_comparison):
        q_id = f"g_mixed_{i:03d}"
        sub_type = i % 6 + 1
        proj_a, proj_b = random.sample(PROJECT_NAMES, 2)
        budget_a = random.randint(100, 999) * 1000
        budget_b = random.randint(100, 999) * 1000
        while budget_a == budget_b:
            budget_b = random.randint(100, 999) * 1000

        doc_lines = []
        question = f"Which project has a higher budget: Project {proj_a} or Project {proj_b}?"
        expected_winner = proj_a if budget_a > budget_b else proj_b
        expected_winner_val = budget_a if budget_a > budget_b else budget_b
        val_a_str = f"${budget_a:,}"
        val_b_str = f"${budget_b:,}"

        if sub_type == 1:
            doc_lines.append(f"The budget for Project {proj_a} is {val_a_str}.")
            doc_lines.append(f"Approved Budget for Project {proj_b}: {val_b_str}.")
            expected_answer = f"Project {expected_winner} (budget of ${expected_winner_val:,})"
        elif sub_type == 2:
            str_a = random.choice(formats[1:])(budget_a)
            str_b = random.choice(formats[1:])(budget_b)
            doc_lines.append(f"Project {proj_a} budget: {str_a}.")
            doc_lines.append(f"Project {proj_b} budget: {str_b}.")
            expected_answer = f"Project {expected_winner} (budget of {expected_winner_val})"
        elif sub_type == 3:
            old_budget_a = random.randint(100, 999) * 1000
            doc_lines.append(f"Old memo: Project {proj_a} budget was ${old_budget_a:,}.")
            doc_lines.append(f"Latest approved budget: Project {proj_a} = {val_a_str}.")
            doc_lines.append(f"Project {proj_b} budget is {val_b_str}.")
            expected_answer = f"Project {expected_winner} (budget of ${expected_winner_val:,})"
        elif sub_type == 4:
            budget_b = budget_a + 1000
            val_b_str = f"${budget_b:,}"
            expected_winner = proj_b
            expected_winner_val = budget_b
            doc_lines.append(f"Project {proj_a} budget is {val_a_str}.")
            doc_lines.append(f"Project {proj_b} budget is {val_b_str}.")
            expected_answer = f"Project {expected_winner} (budget of ${expected_winner_val:,})"
        elif sub_type == 5:
            alias_a = random.choice([proj_a.lower(), proj_a.capitalize()])
            alias_b = random.choice([proj_b.lower(), proj_b])
            doc_lines.append(f"{alias_a} has a budget of {val_a_str}.")
            doc_lines.append(f"{alias_b} budget: {val_b_str}.")
            expected_answer = f"Project {expected_winner} (budget of ${expected_winner_val:,})"
        else:
            doc_lines.append(f"Project {proj_a} budget: {val_a_str}.")
            doc_lines.append(f"Budget for Project {proj_b} remains ambiguous.")
            expected_answer = "KERNEL_MISSING_FIELD"

        noise_docs_content.append(f"--- G {q_id} ---\n" + "\n".join(doc_lines))
        mixed_queries.append({
            "id": q_id,
            "question": question,
            "expected_answer": expected_answer,
            "category": "G",
            "route": "COMPUTE_COMPARISON",
            "metadata": {"type": "G", "sub_type": sub_type}
        })

    # 3. Generate H aggregation queries (200)
    for i in range(args.n_aggregation):
        q_id = f"h_mixed_{i:03d}"
        sub_type = i % 6 + 1
        manager = random.choice(PEOPLE)
        assigned_projects = random.sample(PROJECT_NAMES, random.randint(1, 4))
        expected_projects = list(assigned_projects)

        doc_lines = []
        question = f"List all project names managed by {manager}."

        if sub_type == 1:
            for proj in assigned_projects:
                doc_lines.append(f"Project {proj} is managed by {manager}.")
            expected_answer = ", ".join(sorted(expected_projects))
        elif sub_type == 2:
            for proj in assigned_projects:
                alias = random.choice(DEPARTMENTS) # distractor department
                doc_lines.append(f"Project {proj} is led by {manager} in {alias}.")
            expected_answer = ", ".join(sorted(expected_projects))
        elif sub_type == 3:
            for proj in assigned_projects:
                doc_lines.append(f"Project {proj} is managed by {manager}.")
                doc_lines.append(f"Contract indicates Project {proj} is managed by {manager}.")
            expected_answer = ", ".join(sorted(expected_projects))
        elif sub_type == 4:
            obsolete_proj = random.choice([p for p in PROJECT_NAMES if p not in assigned_projects])
            other_mgr = random.choice([p for p in PEOPLE if p != manager])
            doc_lines.append(f"Old record: {manager} managed Project {obsolete_proj}.")
            doc_lines.append(f"Current record: Project {obsolete_proj} is reassigned to {other_mgr}.")
            for proj in assigned_projects:
                doc_lines.append(f"Project {proj} is currently managed by {manager}.")
            expected_answer = ", ".join(sorted(expected_projects))
        elif sub_type == 5:
            renamed_index = random.randint(0, len(assigned_projects) - 1)
            old_proj = assigned_projects[renamed_index]
            new_proj = f"{old_proj}-R"
            assigned_projects[renamed_index] = new_proj
            doc_lines.append(f"Project {old_proj} was renamed to Project {new_proj}.")
            doc_lines.append(f"Project {new_proj} is managed by {manager}.")
            for idx, proj in enumerate(assigned_projects):
                if idx != renamed_index:
                    doc_lines.append(f"Project {proj} is managed by {manager}.")
            expected_answer = ", ".join(sorted(assigned_projects))
        else:
            other_mgr = random.choice([p for p in PEOPLE if p != manager])
            for proj in assigned_projects:
                doc_lines.append(f"Project {proj} is managed by {other_mgr}.")
            expected_answer = "NOT_FOUND"

        noise_docs_content.append(f"--- H {q_id} ---\n" + "\n".join(doc_lines))
        mixed_queries.append({
            "id": q_id,
            "question": question,
            "expected_answer": expected_answer,
            "category": "H",
            "route": "COMPUTE_AGGREGATION",
            "metadata": {"type": "H", "sub_type": sub_type}
        })

    # 4. Generate I true synthesis queries (300)
    # Helper to calculate manager active portfolio
    def get_portfolio(manager):
        portfolio = {}
        for proj, data in CLEAN_PROJECTS.items():
            if data["owner"] == manager:
                portfolio[proj] = data
        return portfolio

    def get_total_budget(portfolio):
        return sum(data["budget"] for data in portfolio.values())

    templates = ["I1", "I2", "I3", "I4", "I5", "I6"]
    for i in range(args.n_true_synthesis):
        q_id = f"i_mixed_{i:03d}"
        t = templates[i % len(templates)]
        
        doc_lines = []
        doc_lines.append(f"--- I {q_id} ---")
        
        if t == "I1":
            mgr_a, mgr_b = random.sample(PEOPLE, 2)
            budget_a = get_total_budget(get_portfolio(mgr_a))
            budget_b = get_total_budget(get_portfolio(mgr_b))
            if budget_a == budget_b:
                expected_conclusion = "Risk is balanced / inconclusive"
            elif budget_a > budget_b:
                expected_conclusion = f"{mgr_a} appears riskier"
            else:
                expected_conclusion = f"{mgr_b} appears riskier"
            question = f"Compare {mgr_a} and {mgr_b}â€™s project portfolios and explain which one appears riskier based on budget concentration, project status, and number of active projects."
            allowed = [f"{mgr_a} appears riskier", f"{mgr_b} appears riskier", "Risk is balanced / inconclusive"]
            category = "I_RISK_SYNTHESIS"
        elif t == "I2":
            proj_a, proj_b = random.sample(PROJECT_NAMES, 2)
            budget_a = CLEAN_PROJECTS[proj_a]["budget"]
            budget_b = CLEAN_PROJECTS[proj_b]["budget"]
            if budget_a > budget_b:
                expected_conclusion = f"Project {proj_a} should be prioritized"
                winner = proj_a
            else:
                expected_conclusion = f"Project {proj_b} should be prioritized"
                winner = proj_b
            question = f"Based on the active evidence, explain why Project {proj_a} should be prioritized over Project {proj_b}."
            allowed = [f"Project {proj_a} should be prioritized", f"Project {proj_b} should be prioritized"]
            category = "I_PRIORITIZATION"
        elif t == "I3":
            mgr_a, mgr_b = random.sample(PEOPLE, 2)
            budget_a = get_total_budget(get_portfolio(mgr_a))
            budget_b = get_total_budget(get_portfolio(mgr_b))
            if budget_a > budget_b:
                expected_conclusion = f"{mgr_a} has a larger portfolio budget"
            else:
                expected_conclusion = f"{mgr_b} has a larger portfolio budget"
            question = f"Summarize the main operational differences between {mgr_a}â€™s and {mgr_b}â€™s project portfolios."
            allowed = [f"{mgr_a} has a larger portfolio budget", f"{mgr_b} has a larger portfolio budget"]
            category = "I_PORTFOLIO_SUMMARY"
        elif t == "I4":
            proj = random.choice(["QUASAR", "SIRIUS", "TITAN", "VEGA", "ZENITH"])
            expected_conclusion = "Budget was increased"
            question = f"Explain how the latest amendments changed the risk profile of Project {proj}."
            allowed = ["Budget was increased", "Deadline was extended"]
            category = "I_CHANGE_EXPLANATION"
        elif t == "I5":
            mgrs = random.sample(PEOPLE, 3)
            budgets = {m: get_total_budget(get_portfolio(m)) for m in mgrs}
            max_mgr = max(budgets, key=budgets.get)
            expected_conclusion = f"{max_mgr}'s portfolio should receive additional oversight"
            question = f"Recommend which managerâ€™s portfolio should receive additional oversight and justify the recommendation using only active evidence."
            allowed = [f"{m}'s portfolio should receive additional oversight" for m in mgrs]
            category = "I_OVERSIGHT_RECOMMENDATION"
        else:
            proj_a, proj_b = random.sample(PROJECT_NAMES, 2)
            budget_a = CLEAN_PROJECTS[proj_a]["budget"]
            budget_b = CLEAN_PROJECTS[proj_b]["budget"]
            if budget_a > budget_b:
                expected_conclusion = f"Project {proj_a} appears more strategically important"
            else:
                expected_conclusion = f"Project {proj_b} appears more strategically important"
            question = f"Compare Project {proj_a} and Project {proj_b} across budget, status, department, and timeline, then explain which one appears more strategically important."
            allowed = [f"Project {proj_a} appears more strategically important", f"Project {proj_b} appears more strategically important"]
            category = "I_MULTIFACTOR_COMPARISON"

        noise_docs_content.append(f"--- I {q_id} ---\n" + "\n".join(doc_lines))
        mixed_queries.append({
            "id": q_id,
            "question": question,
            "expected_answer": f"Synthesis expected answer for {q_id}",
            "category": category,
            "route": "LLM_SYNTHESIS",
            "expected_conclusion": expected_conclusion,
            "allowed_conclusions": allowed,
            "metadata": {"type": "I", "sub_type": t}
        })

    # Write generated documents to the corpus
    for idx, content in enumerate(noise_docs_content):
        doc_filename = f"mixed_noise_{idx:04d}.txt"
        doc_path = os.path.join(noise_doc_dir, doc_filename)
        with open(doc_path, "w", encoding="utf-8") as f:
            f.write(content)
    print(f"Generated {len(noise_docs_content)} mixed noise files under {noise_doc_dir}.")

    # Re-run ingestion to index the new documents
    print("Re-running ingestion to index mixed documents...")
    python_bin = "/home/taurus_silver/miniconda3/envs/poseidon_wsl/bin/python"
    ingest_script = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ingest.py")
    ingest_cmd = [python_bin, ingest_script, "--corpus-dir", corpus_dir, "--output-dir", args.corpus]
    
    result = subprocess.run(ingest_cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print("Ingestion failed!")
        return
    else:
        print("Ingestion succeeded.")

    # Save workload
    random.shuffle(mixed_queries)
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        for q in mixed_queries:
            f.write(json.dumps(q, ensure_ascii=False) + "\n")
    print(f"Saved combined mixed workload of {len(mixed_queries)} queries to {args.output}.")

if __name__ == "__main__":
    main()


