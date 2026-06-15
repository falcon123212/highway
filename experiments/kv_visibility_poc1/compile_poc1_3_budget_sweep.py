import json
import os
import numpy as np
import pandas as pd

def compile_report(results_file: str, output_report: str):
    if not os.path.exists(results_file):
        print(f"Results file {results_file} does not exist.")
        return
        
    records = []
    with open(results_file, "r") as f:
        for line in f:
            if line.strip():
                try:
                    records.append(json.loads(line))
                except Exception:
                    pass
                    
    if not records:
        print("No records found.")
        return
        
    df = pd.DataFrame(records)
    
    # Self-healing: Load expected answers from answers.jsonl
    gold_answers = {}
    answers_path = r"experiments/kv_visibility_poc1\data_poc1_1\answers.jsonl"
    if os.path.exists(answers_path):
        with open(answers_path, "r") as f:
            for line in f:
                if line.strip():
                    item = json.loads(line)
                    gold_answers[item["question_id"]] = item
                    
    if "expected_answer" not in df.columns:
        df["expected_answer"] = df["sample_id"].map(lambda x: gold_answers.get(x, {}).get("expected_answer", ""))
        
    if "error_type" not in df.columns:
        def get_err_type(row):
            if row["oom"]: return "oom"
            if row["exact_match"]: return "correct"
            is_abst = gold_answers.get(row["sample_id"], {}).get("is_abstention", False)
            if is_abst and row["abstention_accuracy"] == 0:
                return "missing_project_hallucination"
            if row.get("suffix_error", False):
                return "suffix_confusion"
            if not row["numeric_preservation"]:
                return "numeric_wrong"
            return "model_failed_despite_gold"
        df["error_type"] = df.apply(get_err_type, axis=1)
        
    aggregates = []
    for mode, group in df.groupby("mode"):
        count = len(group)
        em = group["exact_match"].mean() * 100.0
        f1 = group["f1_score"].mean() * 100.0
        gold_recall = group["gold_block_recall"].mean() * 100.0
        num_pres = group["numeric_preservation"].mean() * 100.0
        
        # Suffix error rate (only for Category E)
        sub_e = group[group["category"] == "E"]
        suffix_err = sub_e["suffix_error"].mean() * 100.0 if not sub_e.empty else 0.0
        
        # Abstention accuracy
        sub_abst = group[group["expected_answer"].str.contains("cannot answer", case=False, na=False)]
        abst_acc = sub_abst["exact_match"].mean() * 100.0 if not sub_abst.empty else 0.0
        
        parse_fail = group["oom"].mean() * 100.0
        avg_blocks = group["blocks_kept"].mean()
        tok_red = group["token_reduction"].mean()
        
        # Category specific EMs
        cat_ems = {}
        for cat in ["A", "B", "C", "D", "E"]:
            sub_cat = group[group["category"] == cat]
            cat_ems[cat] = sub_cat["exact_match"].mean() * 100.0 if not sub_cat.empty else 0.0
            
        aggregates.append({
            "mode": mode,
            "sample_count": int(count),
            "exact_match": float(em),
            "f1_score": float(f1),
            "gold_block_recall": float(gold_recall),
            "numeric_preservation": float(num_pres),
            "suffix_error_rate": float(suffix_err),
            "abstention_accuracy": float(abst_acc),
            "avg_blocks_kept": float(avg_blocks),
            "avg_token_reduction": float(tok_red),
            "parse_fail": float(parse_fail),
            "EM_A": cat_ems["A"],
            "EM_B": cat_ems["B"],
            "EM_C": cat_ems["C"],
            "EM_D": cat_ems["D"],
            "EM_E": cat_ems["E"]
        })
        
    df_agg = pd.DataFrame(aggregates)
    
    # Sort logical order
    mode_order = ["max_kept_4", "max_kept_6", "max_kept_8", "max_kept_12", "max_kept_16"]
    df_agg = df_agg.set_index("mode").reindex(mode_order).reset_index().dropna(subset=["exact_match"])
    
    report = "# POC 1.3 Mini-run 3 Report â€” Budget Sweep\n\n"
    
    # 1. Quality & Efficiency Summary
    report += "## 1. Quality & Efficiency Summary (100 mixed samples, 400 blocks)\n\n"
    report += "| Budget | Overall EM | Gold Recall | Suffix Error | Abstention Acc | Parse Fail | Avg Blocks Kept | Token Reduction |\n"
    report += "|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|\n"
    
    for _, row in df_agg.iterrows():
        report += f"| **{row['mode']}** | {row['exact_match']:.1f}% | {row['gold_block_recall']:.1f}% | {row['suffix_error_rate']:.1f}% | {row['abstention_accuracy']:.1f}% | {row['parse_fail']:.1f}% | {row['avg_blocks_kept']:.2f} | {row['avg_token_reduction']:.1f}% |\n"
        
    # 2. Category specific EM
    report += "\n## 2. Category-Specific Exact Match Breakdown\n\n"
    report += "| Budget | Category A | Category B | Category C | Category D | Category E |\n"
    report += "|---|:---:|:---:|:---:|:---:|:---:|\n"
    
    for _, row in df_agg.iterrows():
        report += f"| **{row['mode']}** | {row['EM_A']:.1f}% | {row['EM_B']:.1f}% | {row['EM_C']:.1f}% | {row['EM_D']:.1f}% | {row['EM_E']:.1f}% |\n"
        
    # 3. Success Gates validation
    report += "\n## 3. Success Gates Validation\n\n"
    
    for _, row in df_agg.iterrows():
        report += f"### Budget: {row['mode']}\n\n"
        report += "| Success Gate | Target | Actual | Status |\n"
        report += "|---|---|:---:|:---:|\n"
        
        g_em = "PASS" if row["exact_match"] >= 95.0 else "FAIL"
        g_d = "PASS" if row["EM_D"] >= 90.0 else "FAIL"
        g_rec = "PASS" if row["gold_block_recall"] >= 99.0 else "FAIL"
        g_suff = "PASS" if row["suffix_error_rate"] == 0.0 else "FAIL"
        g_abst = "PASS" if row["abstention_accuracy"] == 100.0 else "FAIL"
        g_pf = "PASS" if row["parse_fail"] == 0.0 else "FAIL"
        g_red = "PASS" if row["avg_token_reduction"] >= 96.0 else "FAIL"
        
        report += f"| **Overall EM** | &ge; 95% | **{row['exact_match']:.1f}%** | **{g_em}** |\n"
        report += f"| **Category D EM** | &ge; 90% | **{row['EM_D']:.1f}%** | **{g_d}** |\n"
        report += f"| **Gold Recall** | &ge; 99% | **{row['gold_block_recall']:.1f}%** | **{g_rec}** |\n"
        report += f"| **Suffix Error (Cat E)** | = 0% | **{row['suffix_error_rate']:.1f}%** | **{g_suff}** |\n"
        report += f"| **Abstention Accuracy** | = 100% | **{row['abstention_accuracy']:.1f}%** | **{g_abst}** |\n"
        report += f"| **Parse Fail / OOM** | = 0% | **{row['parse_fail']:.1f}%** | **{g_pf}** |\n"
        report += f"| **Token Reduction** | &ge; 96% | **{row['avg_token_reduction']:.1f}%** | **{g_red}** |\n\n"
        
    # 4. Decision Matrix Check
    report += "\n## 4. Decision Matrix & Sweet Spot Recommendation\n\n"
    
    def all_gates_pass(row):
        return (row["exact_match"] >= 95.0 and 
                row["EM_D"] >= 90.0 and 
                row["gold_block_recall"] >= 99.0 and 
                row["suffix_error_rate"] == 0.0 and 
                row["abstention_accuracy"] == 100.0 and 
                row["parse_fail"] == 0.0 and 
                row["avg_token_reduction"] >= 96.0)

    passing_budgets = []
    for _, row in df_agg.iterrows():
        if all_gates_pass(row):
            passing_budgets.append(row["mode"])
            
    if "max_kept_4" in passing_budgets:
        decision = "ðŸ“Œ **DECISION: Ultra-compaction (max_kept = 4)**. Toutes les gates passent mÃªme avec seulement 4 blocs de budget maximal. C'est le choix optimal pour rÃ©duire les coÃ»ts et la latence au maximum."
    elif any(b in passing_budgets for b in ["max_kept_6", "max_kept_8"]):
        sweet_spot = [b for b in passing_budgets if b in ["max_kept_6", "max_kept_8"]][0]
        decision = f"ðŸ“Œ **DECISION: Sweet spot (max_kept = {sweet_spot.split('_')[-1]})**. Le budget de 4 blocs dÃ©grade la qualitÃ© (notamment la CatÃ©gorie D), mais {sweet_spot} permet de valider toutes les success gates avec un excellent taux de culling."
    elif "max_kept_16" in passing_budgets:
        decision = "ðŸ“Œ **DECISION: Large budget (max_kept = 16)**. Seul le budget de 16 blocs passe toutes les success gates. L'Adaptive Kernel fonctionne mais nÃ©cessite de conserver un historique contextuel plus large."
    else:
        decision = "âš ï¸ **DECISION: Aucune configuration ne valide 100% des success gates**. Une analyse plus approfondie des Ã©checs ou un ajustement des seuils de culling est nÃ©cessaire."
        
    report += f"{decision}\n"
    
    with open(output_report, "w", encoding="utf-8") as f:
        f.write(report)
    print(f"Report compiled successfully to {output_report}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Compile POC 1.3 Budget Sweep report")
    parser.add_argument("--input", type=str, default="artifacts/runs/poc_1_3_budget_sweep/results.jsonl")
    parser.add_argument("--output", type=str, default="artifacts/runs/poc_1_3_budget_sweep/report.md")
    args = parser.parse_args()
    compile_report(args.input, args.output)


