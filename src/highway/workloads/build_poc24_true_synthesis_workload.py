import os
import json
import argparse
import random
import re
import subprocess

# Canonical project names and managers from the clean corpus
PROJECT_NAMES = [
    "NEPTUNE", "KRONOS", "ECLIPSE", "FALCON", "IRIS", "JUPITER", "METEOR", "NEXUS",
    "ORION", "PHOENIX", "QUASAR", "SIRIUS", "TITAN", "VEGA", "ZENITH", "AURORA",
    "BEACON", "CHRONOS", "DAWN", "GENESIS", "HELIOS", "LUNA", "PEGASE", "SOLARIS"
]

PEOPLE = [
    "Jean Dupont", "Alice Martin", "Pierre Leroy", "Marie Dubois", "Thomas Petit",
    "Sophie Richard", "Michel Bernard", "Julie Thomas", "Nicolas Durand", "Emma Michel"
]

DEPARTMENTS = ["Engineering", "Finance", "HR", "Legal", "Marketing", "Operations", "Sales"]

# Active facts parsed from the clean corpus
CLEAN_PROJECTS = {
    "AURORA": {"budget": 538000, "deadline": "12 April 2027", "owner": "Emma Michel", "department": "Finance"},
    "BEACON": {"budget": 981000, "deadline": "05 November 2027", "owner": "Jean Dupont", "department": "Finance"},
    "CHRONOS": {"budget": 494000, "deadline": "20 August 2027", "owner": "Alice Martin", "department": "Marketing"},
    "DAWN": {"budget": 798000, "deadline": "25 November 2027", "owner": "Alice Martin", "department": "HR"},
    "ECLIPSE": {"budget": 127000, "deadline": "23 November 2027", "owner": "Emma Michel", "department": "Operations"},
    "FALCON": {"budget": 703000, "deadline": "26 January 2027", "owner": "Julie Thomas", "department": "Operations"},
    "GENESIS": {"budget": 564000, "deadline": "24 December 2027", "owner": "Pierre Leroy", "department": "HR"},
    "HELIOS": {"budget": 208000, "deadline": "27 November 2027", "owner": "Nicolas Durand", "department": "Operations"},
    "IRIS": {"budget": 384000, "deadline": "07 June 2027", "owner": "Sophie Richard", "department": "Operations"},
    "JUPITER": {"budget": 967000, "deadline": "20 May 2027", "owner": "Sophie Richard", "department": "Operations"},
    "KRONOS": {"budget": 189000, "deadline": "02 January 2027", "owner": "Nicolas Durand", "department": "Operations"},
    "LUNA": {"budget": 880000, "deadline": "18 September 2027", "owner": "Sophie Richard", "department": "Operations"},
    "METEOR": {"budget": 227000, "deadline": "03 September 2027", "owner": "Nicolas Durand", "department": "Operations"},
    "NEPTUNE": {"budget": 125000, "deadline": "08 April 2027", "owner": "Alice Martin", "department": "Operations"},
    "NEXUS": {"budget": 296000, "deadline": "02 November 2027", "owner": "Emma Michel", "department": "Operations"},
    "ORION": {"budget": 987000, "deadline": "13 May 2027", "owner": "Marie Dubois", "department": "Operations"},
    "PEGASE": {"budget": 214000, "deadline": "27 May 2027", "owner": "Jean Dupont", "department": "Operations"},
    "PHOENIX": {"budget": 479000, "deadline": "07 November 2027", "owner": "Pierre Leroy", "department": "Operations"},
    "QUASAR": {"budget": 723000, "deadline": "18 December 2027", "owner": "Alice Martin", "department": "Operations"},
    "SIRIUS": {"budget": 99000, "deadline": "25 January 2026", "owner": "Thomas Petit", "department": "Operations"},
    "SOLARIS": {"budget": 187000, "deadline": "27 February 2027", "owner": "Alice Martin", "department": "Operations"},
    "TITAN": {"budget": 67000, "deadline": "03 April 2026", "owner": "Marie Dubois", "department": "Operations"},
    "VEGA": {"budget": 75000, "deadline": "21 August 2026", "owner": "Emma Michel", "department": "Operations"},
    "ZENITH": {"budget": 84000, "deadline": "09 December 2026", "owner": "Pierre Leroy", "department": "Operations"}
}

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--n-risk-comparison", type=int, default=40)
    parser.add_argument("--n-prioritization", type=int, default=40)
    parser.add_argument("--n-portfolio-summary", type=int, default=40)
    parser.add_argument("--n-change-explanation", type=int, default=30)
    parser.add_argument("--n-oversight-recommendation", type=int, default=30)
    parser.add_argument("--n-multifactor-comparison", type=int, default=20)
    parser.add_argument("--include-obsolete-facts", type=str, default="true")
    parser.add_argument("--include-distractors", type=str, default="true")
    parser.add_argument("--include-balanced-cases", type=str, default="true")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)

    corpus_dir = os.path.dirname(args.corpus.rstrip("/\\"))
    noise_doc_dir = os.path.join(corpus_dir, "documents", "noise")
    os.makedirs(noise_doc_dir, exist_ok=True)

    queries = []
    noise_docs_content = []

    # Helper: calculate manager active portfolio
    def get_portfolio(manager):
        portfolio = {}
        for proj, data in CLEAN_PROJECTS.items():
            if data["owner"] == manager:
                portfolio[proj] = data
        return portfolio

    # Helper: calculate total active budget for a manager
    def get_total_budget(portfolio):
        return sum(data["budget"] for data in portfolio.values())

    # --- I1: Risk Comparison (40) ---
    for i in range(args.n_risk_comparison):
        q_id = f"i_risk_{i:03d}"
        mgr_a, mgr_b = random.sample(PEOPLE, 2)
        
        portfolio_a = get_portfolio(mgr_a)
        portfolio_b = get_portfolio(mgr_b)
        
        doc_lines = []
        doc_lines.append(f"--- I1 {q_id} ---")
        
        # Add obsolete reassignments to test verifier
        if args.include_obsolete_facts.lower() == "true" and random.choice([True, False]):
            obsolete_proj = random.choice(PROJECT_NAMES)
            while obsolete_proj in portfolio_a or obsolete_proj in portfolio_b:
                obsolete_proj = random.choice(PROJECT_NAMES)
            doc_lines.append(f"Old record: {mgr_a} managed Project {obsolete_proj}.")
            doc_lines.append(f"Current record: Project {obsolete_proj} is reassigned to Thomas Petit.")

        # Add suffix distractors to test verifier
        if args.include_distractors.lower() == "true" and random.choice([True, False]):
            suffix_proj = f"{random.choice(PROJECT_NAMES)}-Legacy"
            doc_lines.append(f"{mgr_b} managed {suffix_proj} with a budget of $999,000.")

        # Determine winner riskier manager
        budget_a = get_total_budget(portfolio_a)
        budget_b = get_total_budget(portfolio_b)
        
        if budget_a == budget_b:
            expected_conclusion = "Risk is balanced / inconclusive"
        elif budget_a > budget_b:
            expected_conclusion = f"{mgr_a} appears riskier"
        else:
            expected_conclusion = f"{mgr_b} appears riskier"
            
        noise_docs_content.append("\n".join(doc_lines))
        
        question = f"Compare {mgr_a} and {mgr_b}â€™s project portfolios and explain which one appears riskier based on budget concentration, project status, and number of active projects."
        queries.append({
            "id": q_id,
            "question": question,
            "expected_answer": f"Based on active evidence, {expected_conclusion}.",
            "category": "I_RISK_SYNTHESIS",
            "route": "LLM_SYNTHESIS",
            "expected_conclusion": expected_conclusion,
            "allowed_conclusions": [f"{mgr_a} appears riskier", f"{mgr_b} appears riskier", "Risk is balanced / inconclusive"],
            "metadata": {
                "manager_a": mgr_a,
                "manager_b": mgr_b,
                "budget_a": budget_a,
                "budget_b": budget_b,
                "expected_conclusion": expected_conclusion
            }
        })

    # --- I2: Prioritization (40) ---
    for i in range(args.n_prioritization):
        q_id = f"i_prior_{i:03d}"
        proj_a, proj_b = random.sample(PROJECT_NAMES, 2)
        
        data_a = CLEAN_PROJECTS[proj_a]
        data_b = CLEAN_PROJECTS[proj_b]
        
        doc_lines = []
        doc_lines.append(f"--- I2 {q_id} ---")
        
        if args.include_obsolete_facts.lower() == "true" and random.choice([True, False]):
            doc_lines.append(f"Old memo: Project {proj_a} budget was ${data_a['budget'] // 2:,}.")
            
        budget_a = data_a["budget"]
        budget_b = data_b["budget"]
        
        if budget_a > budget_b:
            expected_conclusion = f"Project {proj_a} should be prioritized"
            winner = proj_a
        else:
            expected_conclusion = f"Project {proj_b} should be prioritized"
            winner = proj_b
            
        noise_docs_content.append("\n".join(doc_lines))
        
        question = f"Based on the active evidence, explain why Project {proj_a} should be prioritized over Project {proj_b}."
        queries.append({
            "id": q_id,
            "question": question,
            "expected_answer": f"Project {winner} has a higher active budget of ${CLEAN_PROJECTS[winner]['budget']:,}.",
            "category": "I_PRIORITIZATION",
            "route": "LLM_SYNTHESIS",
            "expected_conclusion": expected_conclusion,
            "allowed_conclusions": [f"Project {proj_a} should be prioritized", f"Project {proj_b} should be prioritized"],
            "metadata": {
                "project_a": proj_a,
                "project_b": proj_b,
                "budget_a": budget_a,
                "budget_b": budget_b,
                "expected_conclusion": expected_conclusion
            }
        })

    # --- I3: Portfolio Summary (40) ---
    for i in range(args.n_portfolio_summary):
        q_id = f"i_port_{i:03d}"
        mgr_a, mgr_b = random.sample(PEOPLE, 2)
        
        portfolio_a = get_portfolio(mgr_a)
        portfolio_b = get_portfolio(mgr_b)
        
        budget_a = get_total_budget(portfolio_a)
        budget_b = get_total_budget(portfolio_b)
        
        if budget_a > budget_b:
            expected_conclusion = f"{mgr_a} has a larger portfolio budget"
        else:
            expected_conclusion = f"{mgr_b} has a larger portfolio budget"
            
        question = f"Summarize the main operational differences between {mgr_a}â€™s and {mgr_b}â€™s project portfolios."
        queries.append({
            "id": q_id,
            "question": question,
            "expected_answer": f"Operational summary between {mgr_a} and {mgr_b}.",
            "category": "I_PORTFOLIO_SUMMARY",
            "route": "LLM_SYNTHESIS",
            "expected_conclusion": expected_conclusion,
            "allowed_conclusions": [f"{mgr_a} has a larger portfolio budget", f"{mgr_b} has a larger portfolio budget"],
            "metadata": {
                "manager_a": mgr_a,
                "manager_b": mgr_b,
                "expected_conclusion": expected_conclusion
            }
        })

    # --- I4: Change Explanation (30) ---
    amended_projects = ["QUASAR", "SIRIUS", "TITAN", "VEGA", "ZENITH"]
    for i in range(args.n_change_explanation):
        q_id = f"i_change_{i:03d}"
        proj = random.choice(amended_projects)
        
        expected_conclusion = "Budget was increased"
        
        question = f"Explain how the latest amendments changed the risk profile of Project {proj}."
        queries.append({
            "id": q_id,
            "question": question,
            "expected_answer": f"The amendment changed the risk profile of Project {proj}.",
            "category": "I_CHANGE_EXPLANATION",
            "route": "LLM_SYNTHESIS",
            "expected_conclusion": expected_conclusion,
            "allowed_conclusions": ["Budget was increased", "Deadline was extended"],
            "metadata": {
                "project": proj,
                "expected_conclusion": expected_conclusion
            }
        })

    # --- I5: Oversight Recommendation (30) ---
    for i in range(args.n_oversight_recommendation):
        q_id = f"i_oversight_{i:03d}"
        
        # Pick 3 managers
        mgrs = random.sample(PEOPLE, 3)
        budgets = {m: get_total_budget(get_portfolio(m)) for m in mgrs}
        max_mgr = max(budgets, key=budgets.get)
        
        expected_conclusion = f"{max_mgr}'s portfolio should receive additional oversight"
        
        question = f"Recommend which managerâ€™s portfolio should receive additional oversight and justify the recommendation using only active evidence."
        queries.append({
            "id": q_id,
            "question": question,
            "expected_answer": f"Recommend additional oversight for {max_mgr}.",
            "category": "I_OVERSIGHT_RECOMMENDATION",
            "route": "LLM_SYNTHESIS",
            "expected_conclusion": expected_conclusion,
            "allowed_conclusions": [f"{m}'s portfolio should receive additional oversight" for m in mgrs],
            "metadata": {
                "managers": mgrs,
                "max_manager": max_mgr,
                "expected_conclusion": expected_conclusion
            }
        })

    # --- I6: Multi-Factor Comparison (20) ---
    for i in range(args.n_multifactor_comparison):
        q_id = f"i_multi_{i:03d}"
        proj_a, proj_b = random.sample(PROJECT_NAMES, 2)
        
        budget_a = CLEAN_PROJECTS[proj_a]["budget"]
        budget_b = CLEAN_PROJECTS[proj_b]["budget"]
        
        if budget_a > budget_b:
            expected_conclusion = f"Project {proj_a} appears more strategically important"
            winner = proj_a
        else:
            expected_conclusion = f"Project {proj_b} appears more strategically important"
            winner = proj_b
            
        question = f"Compare Project {proj_a} and Project {proj_b} across budget, status, department, and timeline, then explain which one appears more strategically important."
        queries.append({
            "id": q_id,
            "question": question,
            "expected_answer": f"Project {winner} has a higher budget and is strategically important.",
            "category": "I_MULTIFACTOR_COMPARISON",
            "route": "LLM_SYNTHESIS",
            "expected_conclusion": expected_conclusion,
            "allowed_conclusions": [f"Project {proj_a} appears more strategically important", f"Project {proj_b} appears more strategically important"],
            "metadata": {
                "project_a": proj_a,
                "project_b": proj_b,
                "winner": winner,
                "expected_conclusion": expected_conclusion
            }
        })

    # Write noise files
    for idx, content in enumerate(noise_docs_content):
        doc_filename = f"synthesis_noise_{idx:04d}.txt"
        doc_path = os.path.join(noise_doc_dir, doc_filename)
        with open(doc_path, "w", encoding="utf-8") as f:
            f.write(content)
            
    print(f"Generated {len(noise_docs_content)} synthesis noise files.")

    # Re-run ingestion to index the new documents
    print("Re-running ingestion to index synthesis documents...")
    python_bin = "/home/taurus_silver/miniconda3/envs/poseidon_wsl/bin/python"
    ingest_script = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ingest.py")
    ingest_cmd = [python_bin, ingest_script, "--corpus-dir", corpus_dir, "--output-dir", args.corpus]
    
    result = subprocess.run(ingest_cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print("Ingestion failed!")
        return
    else:
        print("Ingestion succeeded.")

    # Save targeted workload
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        for q in queries:
            f.write(json.dumps(q, ensure_ascii=False) + "\n")
    print(f"Saved targeted synthesis workload to {args.output}.")

if __name__ == "__main__":
    main()


