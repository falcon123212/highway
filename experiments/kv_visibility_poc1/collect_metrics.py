import os
import json
import re
import collections
import string
import numpy as np
import pandas as pd
from typing import Dict, Any, List

def parse_json_response_b(text: str) -> Dict[str, str]:
    text = text.strip()
    if text.startswith("```"):
        lines = text.split("\n")
        if lines[0].startswith("```"):
            lines = lines[1:]
        if lines[-1].strip() == "```":
            lines = lines[:-1]
        text = "\n".join(lines).strip()
    try:
        start = text.find("{")
        end = text.rfind("}")
        if start != -1 and end != -1:
            text = text[start:end+1]
        parsed = json.loads(text)
        return {
            "answer": str(parsed.get("answer", parsed.get("expected_answer", text))),
            "evidence_block_id": str(parsed.get("evidence_block_id", parsed.get("evidence_id", ""))),
            "evidence_quote": str(parsed.get("evidence_quote", ""))
        }
    except Exception:
        ans_match = re.search(r'"answer"\s*:\s*"([^"]+)"', text, re.IGNORECASE)
        doc_match = re.search(r'"evidence_(?:block_)?id"\s*:\s*"([^"]+)"', text, re.IGNORECASE)
        quote_match = re.search(r'"evidence_quote"\s*:\s*"([^"]+)"', text, re.IGNORECASE)
        return {
            "answer": ans_match.group(1) if ans_match else text,
            "evidence_block_id": doc_match.group(1) if doc_match else "",
            "evidence_quote": quote_match.group(1) if quote_match else ""
        }

def normalize_answer(val: str) -> str:
    if val is None:
        return ""
    val = str(val).strip().lower()
    val = val.replace("$", "")
    def clean_num(match):
        return match.group(0).replace(",", "").replace(".", "").replace(" ", "")
    val = re.sub(r'\b\d+([,\.\s]\d+)+\b', clean_num, val)
    val = val.translate(str.maketrans("", "", string.punctuation))
    val = " ".join(val.split())
    return val

def calculate_f1(prediction: str, ground_truth: str) -> float:
    pred_tokens = normalize_answer(prediction).split()
    gt_tokens = normalize_answer(ground_truth).split()
    if not pred_tokens or not gt_tokens:
        return 1.0 if pred_tokens == gt_tokens else 0.0
    common = collections.Counter(pred_tokens) & collections.Counter(gt_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0.0
    precision = 1.0 * num_same / len(pred_tokens)
    recall = 1.0 * num_same / len(gt_tokens)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1

def check_abstention(generated: str) -> bool:
    norm = normalize_answer(generated)
    keywords = ["cannot answer", "not mentioned", "do not have", "no information", "not found", "insufficient information", "does not state"]
    return any(kw in norm or kw.replace(" ", "") in norm.replace(" ", "") for kw in keywords)

def compile_metrics(results_file: str, output_csv_prefix: str) -> Dict[str, Any]:
    with open(results_file, "r") as f:
        results = json.load(f)
        
    processed_list = []
    
    for item in results:
        q_id = item["question_id"]
        category = item["category"]
        project = item["project"]
        expected_raw = item["expected_answer"]
        gen_raw = item["generated_text"]
        oom = item["oom"]
        
        # 1. Parse JSON response if present
        parsed = parse_json_response_b(gen_raw) if not oom else {"answer": "", "evidence_block_id": "", "evidence_quote": ""}
        extracted_answer = parsed["answer"]
        
        # 2. Normalize answers
        norm_expected = normalize_answer(expected_raw)
        norm_generated = normalize_answer(extracted_answer)
        
        # 3. Exact Match
        em = (norm_generated == norm_expected) if not oom else False
        
        # 4. F1 Score
        f1 = calculate_f1(extracted_answer, expected_raw) if not oom else 0.0
        
        # 5. Numeric Preservation
        expected_digits = re.findall(r'\d+', norm_expected)
        generated_digits = re.findall(r'\d+', norm_generated)
        num_pres = all(d in generated_digits for d in expected_digits) if expected_digits and not oom else True
        
        # 6. Suffix Error Rate (Category E)
        suffix_error = False
        if category == "E" and not oom and not em:
            # Check if answer contains a distractor date (which would be different from active_date)
            # Since distractors have different dates, any non-em answer that looks like a date is a suffix error
            if re.search(r'\d{1,2}\s+[a-zA-Z]+\s+\d{4}', norm_generated):
                suffix_error = True
                
        # 7. Abstention Accuracy
        abstention_correct = True
        if item["is_abstention"]:
            # If the question should not be answered, we expect the model to abstain
            abstention_correct = check_abstention(extracted_answer) if not oom else False
            # EM should be true if model correctly abstained or false
            em = abstention_correct
            
        processed_list.append({
            "question_id": q_id,
            "category": category,
            "project": project,
            "context_size_blocks": item["context_size_blocks"],
            "mode": item["mode"],
            "oom": oom,
            "exact_match": em,
            "f1_score": f1,
            "numeric_preservation": num_pres,
            "gold_block_recall": item["gold_block_recall"],
            "suffix_error": suffix_error,
            "abstention_correct": abstention_correct,
            "is_abstention": item["is_abstention"],
            "ttft_ms": item["ttft_ms"],
            "decode_ms": item["decode_ms"],
            "e2e_ms": item["e2e_ms"],
            "selector_latency_ms": item["selector_latency_ms"],
            "tokens_per_sec_in": item["tokens_per_sec_in"],
            "tokens_per_sec_out": item["tokens_per_sec_out"],
            "effective_context_tokens_per_sec": item["effective_context_tokens_per_sec"],
            "kept_blocks_count": item["kept_blocks_count"],
            "token_reduction_pct": item["token_reduction_pct"]
        })
        
    df = pd.DataFrame(processed_list)
    
    # Save raw CSV
    df.to_csv(f"{output_csv_prefix}_raw.csv", index=False)
    
    # Compile aggregates grouped by context size and mode
    aggregates = []
    
    for (size, mode), group in df.groupby(["context_size_blocks", "mode"]):
        # Percentages
        em_pct = group["exact_match"].mean() * 100.0
        f1_pct = group["f1_score"].mean() * 100.0
        num_pres_pct = group["numeric_preservation"].mean() * 100.0
        gold_recall_pct = group["gold_block_recall"].mean() * 100.0
        
        # Suffix error rate (only for Category E)
        cat_e_group = group[group["category"] == "E"]
        suffix_err_pct = cat_e_group["suffix_error"].mean() * 100.0 if not cat_e_group.empty else 0.0
        
        # Abstention accuracy (only for samples with is_abstention = True)
        abst_group = group[group["is_abstention"] == True]
        abst_acc_pct = abst_group["abstention_correct"].mean() * 100.0 if not abst_group.empty else 0.0
        
        # Latency percentiles
        ttft_vals = group["ttft_ms"].values
        decode_vals = group["decode_ms"].values
        e2e_vals = group["e2e_ms"].values
        sel_vals = group["selector_latency_ms"].values
        
        ttft_p50 = np.percentile(ttft_vals, 50) if len(ttft_vals) > 0 else 0.0
        ttft_p95 = np.percentile(ttft_vals, 95) if len(ttft_vals) > 0 else 0.0
        ttft_p99 = np.percentile(ttft_vals, 99) if len(ttft_vals) > 0 else 0.0
        
        e2e_p50 = np.percentile(e2e_vals, 50) if len(e2e_vals) > 0 else 0.0
        e2e_p95 = np.percentile(e2e_vals, 95) if len(e2e_vals) > 0 else 0.0
        e2e_p99 = np.percentile(e2e_vals, 99) if len(e2e_vals) > 0 else 0.0
        
        sel_p50 = np.percentile(sel_vals, 50) if len(sel_vals) > 0 else 0.0
        sel_p95 = np.percentile(sel_vals, 95) if len(sel_vals) > 0 else 0.0
        
        # Throughput
        in_tp = group["tokens_per_sec_in"].mean()
        out_tp = group["tokens_per_sec_out"].mean()
        eff_tp = group["effective_context_tokens_per_sec"].mean()
        
        # Memory/OOM
        oom_rate = group["oom"].mean() * 100.0
        
        # Cost Efficiency
        total_gpu_time_sec = group["e2e_ms"].sum() / 1000.0
        total_selector_time_sec = group["selector_latency_ms"].sum() / 1000.0
        total_time_sec = total_gpu_time_sec + total_selector_time_sec
        correct_answers = group["exact_match"].sum()
        cost_per_correct = total_time_sec / max(1, correct_answers)
        
        aggregates.append({
            "context_size_blocks": size,
            "mode": mode,
            "exact_match_pct": em_pct,
            "f1_score_pct": f1_pct,
            "numeric_preservation_pct": num_pres_pct,
            "gold_block_recall_pct": gold_recall_pct,
            "suffix_error_rate_pct": suffix_err_pct,
            "abstention_accuracy_pct": abst_acc_pct,
            "ttft_p50_ms": ttft_p50,
            "ttft_p95_ms": ttft_p95,
            "ttft_p99_ms": ttft_p99,
            "e2e_p50_ms": e2e_p50,
            "e2e_p95_ms": e2e_p95,
            "e2e_p99_ms": e2e_p99,
            "selector_p50_ms": sel_p50,
            "selector_p95_ms": sel_p95,
            "avg_kept_blocks": group["kept_blocks_count"].mean(),
            "avg_token_reduction_pct": group["token_reduction_pct"].mean(),
            "avg_tokens_per_sec_in": in_tp,
            "avg_tokens_per_sec_out": out_tp,
            "effective_context_tokens_per_sec": eff_tp,
            "oom_rate_pct": oom_rate,
            "cost_per_correct_answer_sec": cost_per_correct
        })
        
    df_agg = pd.DataFrame(aggregates)
    df_agg.to_csv(f"{output_csv_prefix}_summary.csv", index=False)
    
    # Also compile category stats (quality metrics by size, mode, category)
    cat_aggregates = []
    for (size, mode, cat), group in df.groupby(["context_size_blocks", "mode", "category"]):
        em_pct = group["exact_match"].mean() * 100.0
        f1_pct = group["f1_score"].mean() * 100.0
        gold_rec = group["gold_block_recall"].mean() * 100.0
        cat_aggregates.append({
            "context_size_blocks": size,
            "mode": mode,
            "category": cat,
            "exact_match_pct": em_pct,
            "f1_score_pct": f1_pct,
            "gold_block_recall_pct": gold_rec,
            "count": len(group)
        })
    df_cat = pd.DataFrame(cat_aggregates)
    df_cat.to_csv(f"{output_csv_prefix}_by_category.csv", index=False)
    
    print(f"Aggregated reports saved with prefix: {output_csv_prefix}")
    return df_agg.to_dict(orient="records")


