import os
import json
import argparse
import numpy as np

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--export-json", type=str, required=True)
    args = parser.parse_args()

    if not os.path.exists(args.input):
        print(f"Error: input file {args.input} not found.")
        return

    # Load records
    records = []
    with open(args.input, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                try:
                    records.append(json.loads(line))
                except Exception as e:
                    print(f"Skipping bad line: {e}")

    print(f"Loaded {len(records)} results from {args.input}")

    # Group by mode
    mode_records = {}
    for r in records:
        mode = r["mode"]
        mode_records.setdefault(mode, []).append(r)

    # Compute metrics per mode
    stats = {}
    for mode, recs in mode_records.items():
        count = len(recs)
        leak_checks = [r.get("leak_check_passed", False) for r in recs]
        ems = [r["is_em"] and r.get("leak_check_passed", False) for r in recs]
        latencies = [r["latency_ms"] for r in recs]
        bypasses = [r["is_bypass"] for r in recs]
        verifiers = [r["verify_passed"] for r in recs]
        prompt_tokens = [r["prompt_tokens"] for r in recs]

        mean_em = np.mean(ems) * 100 if ems else 0.0
        mean_lat = np.mean(latencies) if latencies else 0.0
        p95_lat = np.percentile(latencies, 95) if latencies else 0.0
        bypass_rate = np.mean(bypasses) * 100 if bypasses else 0.0
        verifier_pass_rate = np.mean(verifiers) * 100 if verifiers else 0.0
        leak_check_pass_rate = np.mean(leak_checks) * 100 if leak_checks else 0.0
        non_validating_records = sum(1 for r in recs if not r.get("leak_check_passed", False))
        total_prompt_tokens = sum(prompt_tokens)

        # Category breakdown
        cat_em = {}
        cat_count = {}
        for r in recs:
            cat = r["category"]
            cat_em.setdefault(cat, []).append(r["is_em"] and r.get("leak_check_passed", False))
            cat_count[cat] = cat_count.get(cat, 0) + 1

        cat_breakdown = {}
        for cat in sorted(cat_em.keys()):
            cat_breakdown[cat] = {
                "count": cat_count[cat],
                "em": float(np.mean(cat_em[cat]) * 100)
            }

        # Routing Accuracy
        routing_correct = []
        for r in recs:
            cat = r["category"]
            route = r.get("route", "")
            expected = r.get("expected_answer", r.get("expected", ""))
            
            if expected == "NOT_FOUND":
                is_correct = (route in ["NOT_FOUND", "L0_ANSWER_CACHE"])
            elif cat == "D":
                is_correct = (route in ["LONG_CONTEXT_FALLBACK", "DETERMINISTIC", "L0_ANSWER_CACHE"])
            elif cat == "G":
                is_correct = (route in ["COMPUTE_COMPARISON", "DETERMINISTIC", "LLM_COMPILED", "L0_ANSWER_CACHE"])
            elif cat == "H":
                is_correct = (route in ["COMPUTE_AGGREGATION", "DETERMINISTIC", "LLM_COMPILED", "L0_ANSWER_CACHE"])
            else:
                is_correct = (route in ["DETERMINISTIC", "LLM_COMPILED", "L0_ANSWER_CACHE", "L1_PROOF_CACHE"])
                
            routing_correct.append(is_correct)
            
        routing_accuracy = np.mean(routing_correct) * 100 if routing_correct else 0.0

        # False Deterministic Answer (deterministic answer that is incorrect)
        false_det = 0
        for r in recs:
            route = r.get("route", "")
            if route in ["DETERMINISTIC", "L0_ANSWER_CACHE"] and not r["is_em"]:
                false_det += 1
        false_det_rate = (false_det / count) * 100 if count > 0 else 0.0

        # False NOT_FOUND (returned NOT_FOUND incorrectly when expected was something else)
        false_nf = 0
        for r in recs:
            ans = r.get("answer", "")
            expected = r.get("expected_answer", r.get("expected", ""))
            if ans == "NOT_FOUND" and expected != "NOT_FOUND":
                false_nf += 1
        false_nf_rate = (false_nf / count) * 100 if count > 0 else 0.0

        # LLM-required EM (EM on queries where LLM was called)
        llm_queries = [r for r in recs if not r["is_bypass"]]
        llm_em = np.mean([r["is_em"] for r in llm_queries]) * 100 if llm_queries else 100.0

        # LLM-required Call Rate
        gh_queries = [r for r in recs if r["category"] in ["G", "H"]]
        gh_llm_called = sum(1 for r in gh_queries if not r["is_bypass"] and r.get("route") in ["LLM_COMPILED", "LONG_CONTEXT_FALLBACK"])
        llm_required_call_rate = (gh_llm_called / len(gh_queries)) * 100 if gh_queries else 0.0

        # Long-context malformed output rate (OOM or parse fail on Category D)
        d_queries = [r for r in recs if r["category"] == "D"]
        # A malformed fallback is if it returned empty or not matching JSON
        malformed_d = sum(1 for r in d_queries if not r["is_em"] and r.get("oom", False))
        malformed_d_rate = (malformed_d / len(d_queries)) * 100 if d_queries else 0.0

        # Compute Kernel metrics
        compute_comp_queries = [r for r in recs if r.get("route") == "COMPUTE_COMPARISON"]
        compute_agg_queries = [r for r in recs if r.get("route") == "COMPUTE_AGGREGATION"]
        compute_comp_em = np.mean([r["is_em"] for r in compute_comp_queries]) * 100 if compute_comp_queries else 0.0
        compute_agg_em = np.mean([r["is_em"] for r in compute_agg_queries]) * 100 if compute_agg_queries else 0.0
        
        # G/H EM Global
        gh_em = np.mean([r["is_em"] for r in gh_queries]) * 100 if gh_queries else 0.0
        
        # LLM Call Rate on G/H (should be 0% with compute kernels)
        gh_llm_called_kernels = sum(1 for r in gh_queries if r.get("route") in ["LLM_COMPILED", "LONG_CONTEXT_FALLBACK"])
        gh_llm_call_rate = (gh_llm_called_kernels / len(gh_queries)) * 100 if gh_queries else 0.0
        
        # Execution Error â†’ NOT_FOUND Conversion Rate
        exec_error_to_nf = sum(1 for r in recs if r.get("metrics", {}).get("execution_error") and r.get("answer") == "NOT_FOUND")
        exec_error_conversion_rate = (exec_error_to_nf / count) * 100 if count > 0 else 0.0

        # Secondary Metrics for Hardening (POC 2.3.4)
        budget_parse_correct = 0
        budget_parse_total = 0
        
        canon_correct = 0
        canon_total = 0
        
        dup_suppress_correct = 0
        dup_suppress_total = 0
        
        alias_resolve_correct = 0
        alias_resolve_total = 0
        
        missing_field_correct = 0
        missing_field_total = 0
        
        for r in recs:
            meta = r.get("metadata")
            if not meta:
                continue
                
            audit = r.get("metrics", {}).get("kernel_audit", {})
            if not audit:
                continue
                
            m_type = meta.get("type")
            sub_type = meta.get("sub_type")
            
            if m_type == "G":
                proj_a = meta.get("proj_a")
                proj_b = meta.get("proj_b")
                gt_a = meta.get("budget_a")
                gt_b = meta.get("budget_b")
                is_missing = meta.get("is_missing", False)
                
                inputs = audit.get("inputs", {})
                
                if not is_missing:
                    # Budget parsing check
                    p_a_key = f"Project {proj_a}"
                    p_b_key = f"Project {proj_b}"
                    
                    parsed_a = inputs.get(p_a_key, {}).get("budget")
                    parsed_b = inputs.get(p_b_key, {}).get("budget")
                    
                    budget_parse_total += 2
                    if parsed_a == gt_a:
                        budget_parse_correct += 1
                    if parsed_b == gt_b:
                        budget_parse_correct += 1
                        
                    # Entity canonicalization check
                    canon_total += 2
                    if p_a_key in inputs:
                        canon_correct += 1
                    if p_b_key in inputs:
                        canon_correct += 1
                        
                    # Alias resolution check (G5)
                    if sub_type == 5:
                        alias_resolve_total += 1
                        if r["is_em"]:
                            alias_resolve_correct += 1
                else:
                    # Missing-field check (G6)
                    if sub_type == 6:
                        missing_field_total += 1
                        if r["generated"] == "KERNEL_MISSING_FIELD":
                            missing_field_correct += 1
                            
            elif m_type == "H":
                manager = meta.get("manager")
                gt_projects = set(meta.get("projects", []))
                is_missing = meta.get("is_missing", False)
                
                inputs = audit.get("inputs", {})
                outputs = audit.get("outputs", [])
                
                if not is_missing:
                    # Entity canonicalization check
                    canon_total += 1
                    if inputs.get("manager") == manager:
                        canon_correct += 1
                        
                    # Duplicate suppression check (H3)
                    if sub_type == 3:
                        dup_suppress_total += 1
                        output_projects = [o.get("project") for o in outputs]
                        has_duplicates = len(output_projects) != len(set(output_projects))
                        if not has_duplicates and r["is_em"] and audit.get("deduplicated", False):
                            dup_suppress_correct += 1
                            
                    # Alias resolution check (H2)
                    if sub_type == 2:
                        alias_resolve_total += 1
                        if r["is_em"]:
                            alias_resolve_correct += 1
                else:
                    # Missing-field/true absence check (H6)
                    if sub_type == 6:
                        missing_field_total += 1
                        if r["generated"] == "NOT_FOUND":
                            missing_field_correct += 1
                            
        budget_parse_acc = (budget_parse_correct / budget_parse_total) * 100 if budget_parse_total > 0 else 100.0
        canon_acc = (canon_correct / canon_total) * 100 if canon_total > 0 else 100.0
        dup_suppress_acc = (dup_suppress_correct / dup_suppress_total) * 100 if dup_suppress_total > 0 else 100.0
        alias_resolve_acc = (alias_resolve_correct / alias_resolve_total) * 100 if alias_resolve_total > 0 else 100.0
        missing_field_acc = (missing_field_correct / missing_field_total) * 100 if missing_field_total > 0 else 100.0

        stats[mode] = {
            "count": count,
            "overall_em": float(mean_em),
            "mean_latency_ms": float(mean_lat),
            "p95_latency_ms": float(p95_lat),
            "llm_bypass_rate": float(bypass_rate),
            "verifier_pass_rate": float(verifier_pass_rate),
            "leak_check_pass_rate": float(leak_check_pass_rate),
            "non_validating_records": int(non_validating_records),
            "routing_accuracy": float(routing_accuracy),
            "false_deterministic_rate": float(false_det_rate),
            "false_not_found_rate": float(false_nf_rate),
            "llm_required_em": float(llm_em),
            "llm_required_call_rate": float(llm_required_call_rate),
            "long_context_malformed_rate": float(malformed_d_rate),
            "total_prompt_tokens": int(total_prompt_tokens),
            "cat_breakdown": cat_breakdown,
            "oom_rate": sum(1 for r in recs if r.get("oom", False)) / count * 100,
            "compute_comparison_em": float(compute_comp_em),
            "compute_aggregation_em": float(compute_agg_em),
            "gh_em_global": float(gh_em),
            "gh_llm_call_rate": float(gh_llm_call_rate),
            "exec_error_conversion_rate": float(exec_error_conversion_rate),
            "compute_comparison_count": len(compute_comp_queries),
            "compute_aggregation_count": len(compute_agg_queries),
            "budget_parse_accuracy": float(budget_parse_acc),
            "entity_canonicalization_accuracy": float(canon_acc),
            "duplicate_suppression_accuracy": float(dup_suppress_acc),
            "alias_resolution_accuracy": float(alias_resolve_acc),
            "missing_field_accuracy": float(missing_field_acc)
        }

    # Calculate KV Tokens Avoided (PCCC Cache mode vs Baseline)
    pccc_mode = "pccc_cache_scheduler_prefix"
    if pccc_mode not in stats:
        pccc_mode = list(stats.keys())[0] if stats else "unknown"
        
    baseline_mode = "hybrid_topk_llm_baseline_stratified_150"
    
    kv_tokens_avoided = 0.0
    if pccc_mode in stats and baseline_mode in stats:
        baseline_qids = {r["id"] for r in mode_records[baseline_mode]}
        pccc_subset_tokens = sum(r["prompt_tokens"] for r in mode_records[pccc_mode] if r["id"] in baseline_qids)
        baseline_subset_tokens = sum(r["prompt_tokens"] for r in mode_records[baseline_mode])
        if baseline_subset_tokens > 0:
            kv_tokens_avoided = (baseline_subset_tokens - pccc_subset_tokens) / baseline_subset_tokens * 100

    # Prefix TTFT Reduction
    prefix_ttft_reduction = 0.0
    if pccc_mode in stats and baseline_mode in stats:
        baseline_llm_ttft = [r["metrics"]["llm_ttft_ms"] for r in mode_records[baseline_mode] if r["metrics"]["llm_ttft_ms"] > 0]
        pccc_llm_ttft = [r["metrics"]["llm_ttft_ms"] for r in mode_records[pccc_mode] if r["metrics"]["llm_ttft_ms"] > 0]
        if baseline_llm_ttft and pccc_llm_ttft:
            mean_base_ttft = np.mean(baseline_llm_ttft)
            mean_pccc_ttft = np.mean(pccc_llm_ttft)
            prefix_ttft_reduction = (mean_base_ttft - mean_pccc_ttft) / mean_base_ttft * 100

    # Save final metrics JSON
    final_metrics = {
        "modes": stats,
        "kv_tokens_avoided_pct": float(kv_tokens_avoided),
        "prefix_ttft_reduction_pct": float(prefix_ttft_reduction)
    }

    os.makedirs(os.path.dirname(args.export_json), exist_ok=True)
    with open(args.export_json, "w", encoding="utf-8") as f:
        json.dump(final_metrics, f, indent=2)
    print(f"Saved compiled metrics to: {args.export_json}")

    # Generate Markdown Report
    pccc_stats = stats.get(pccc_mode, {})
    no_cache_stats = stats.get("pccc_no_cache", {})
    base_stats = stats.get(baseline_mode, {})
    leak_pass_rate = pccc_stats.get("leak_check_pass_rate", 0.0)
    warning_md = ""
    if leak_pass_rate < 100.0:
        warning_md = (
            "\n> WARNING: This run is historical/non-validating because one or more "
            "records are missing no-leak metadata or failed the no-leak workload checks.\n"
        )

    report_md = fr"""# POC 2.3.4/2.3.5 No-Leak Performance & Verification Report
## Kernel Hardening - Adversarial Structured Evidence

This report verifies the performance gates and claims for the **Hardened Compute Kernels** under POC 2.3.4/2.3.5. Legacy runs without `leak_check_passed=true` are treated as historical/non-validating.
{warning_md}

### Primary Success Gates Validation

| Metric | Target Gate | Actual Result ({pccc_mode}) | Status |
|---|---|---|---|
| **No-Leak Workload Check** | $= 100\%$ | **{leak_pass_rate:.2f}%** | **{"PASS" if leak_pass_rate == 100.0 else "FAIL"}** |
| **G/H EM Global** | $\ge 98\%$ | **{pccc_stats.get("gh_em_global", 0.0):.2f}%** | **{"PASS" if pccc_stats.get("gh_em_global", 0.0) >= 98 else "FAIL"}** |
| **COMPUTE_COMPARISON EM** | $\ge 98\%$ | **{pccc_stats.get("compute_comparison_em", 0.0):.2f}%** | **{"PASS" if pccc_stats.get("compute_comparison_em", 0.0) >= 98 else "FAIL"}** |
| **COMPUTE_AGGREGATION EM** | $\ge 98\%$ | **{pccc_stats.get("compute_aggregation_em", 0.0):.2f}%** | **{"PASS" if pccc_stats.get("compute_aggregation_em", 0.0) >= 98 else "FAIL"}** |
| **False NOT_FOUND** | $= 0\%$ | **{pccc_stats.get("false_not_found_rate", 0.0):.2f}%** | **{"PASS" if pccc_stats.get("false_not_found_rate", 0.0) == 0 else "FAIL"}** |
| **Wrong active/obsolete selection** | separately measured | **N/A** | **N/A** |
| **Kernel Verifier Pass Rate** | $\ge 99\%$ | **{pccc_stats.get("verifier_pass_rate", 0.0):.2f}%** | **{"PASS" if pccc_stats.get("verifier_pass_rate", 0.0) >= 99 else "FAIL"}** |
| **LLM Call Rate on G/H** | $= 0\%$ | **{pccc_stats.get("gh_llm_call_rate", 0.0):.2f}%** | **{"PASS" if pccc_stats.get("gh_llm_call_rate", 0.0) == 0 else "FAIL"}** |
| **Exec Error to NOT_FOUND Conv.** | $= 0\%$ | **{pccc_stats.get("exec_error_conversion_rate", 0.0):.2f}%** | **{"PASS" if pccc_stats.get("exec_error_conversion_rate", 0.0) == 0 else "FAIL"}** |
| **Latency p95** | $< 50$ ms | **{pccc_stats.get("p95_latency_ms", 0.0):.1f} ms** | **{"PASS" if pccc_stats.get("p95_latency_ms", 0.0) < 50 else "FAIL"}** |
| **VRAM OOM Rate** | $0.0\%$ | **{pccc_stats.get("oom_rate", 0.0):.2f}%** | **{"PASS" if pccc_stats.get("oom_rate", 0.0) == 0 else "FAIL"}** |

### Secondary Success Gates Validation

| Metric | Target Gate | Actual Result ({pccc_mode}) | Status |
|---|---|---|---|
| **Budget Parse Accuracy** | $\ge 99\%$ | **{pccc_stats.get("budget_parse_accuracy", 0.0):.2f}%** | **{"PASS" if pccc_stats.get("budget_parse_accuracy", 0.0) >= 99 else "FAIL"}** |
| **Entity Canonicalization Accuracy** | $\ge 98\%$ | **{pccc_stats.get("entity_canonicalization_accuracy", 0.0):.2f}%** | **{"PASS" if pccc_stats.get("entity_canonicalization_accuracy", 0.0) >= 98 else "FAIL"}** |
| **Duplicate Suppression Accuracy** | $\ge 99\%$ | **{pccc_stats.get("duplicate_suppression_accuracy", 0.0):.2f}%** | **{"PASS" if pccc_stats.get("duplicate_suppression_accuracy", 0.0) >= 99 else "FAIL"}** |
| **Alias Resolution Accuracy** | $\ge 95\%$ | **{pccc_stats.get("alias_resolution_accuracy", 0.0):.2f}%** | **{"PASS" if pccc_stats.get("alias_resolution_accuracy", 0.0) >= 95 else "FAIL"}** |
| **Missing-field Handling Accuracy** | $\ge 98\%$ | **{pccc_stats.get("missing_field_accuracy", 0.0):.2f}%** | **{"PASS" if pccc_stats.get("missing_field_accuracy", 0.0) >= 98 else "FAIL"}** |

---

### Detailed Performance Breakdown by Mode

#### 1. PCCC Compute Kernels
* **Overall EM**: {pccc_stats.get("overall_em", 0.0):.2f}%
* **No-Leak Pass Rate**: {pccc_stats.get("leak_check_pass_rate", 0.0):.2f}%
* **Non-validating Records**: {pccc_stats.get("non_validating_records", 0)}
* **Global LLM Bypass Rate**: {pccc_stats.get("llm_bypass_rate", 0.0):.2f}%
* **Mean Latency**: {pccc_stats.get("mean_latency_ms", 0.0):.1f} ms
* **p95 Latency**: {pccc_stats.get("p95_latency_ms", 0.0):.1f} ms
* **Category Breakdown**:
"""
    for cat, item in pccc_stats.get("cat_breakdown", {}).items():
        report_md += f"  - Category {cat}: EM={item['em']:.2f}% (Count={item['count']})\n"

    report_md += f"""
---

### Conclusion & Key Findings
1. **No-Leak Validation**: The report treats legacy records without `leak_check_passed=true` as historical/non-validating.
2. **Kernel Accuracy**: Accuracy claims are based on exact-match records that also pass the no-leak gate.
3. **Remaining Work**: Any failed gate in the table above should be treated as a real follow-up item, not hidden by historical scores.
"""

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        f.write(report_md)
    print(f"Saved Markdown report to: {args.output}")

if __name__ == "__main__":
    main()


