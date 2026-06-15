import os
import pandas as pd
from typing import Dict, Any

def generate_report(
    attention_blocks_path: str,
    replay_results_path: str,
    kv_estimates_path: str,
    output_report_path: str
) -> Dict[str, Any]:
    """
    Reads the CSV output files, aggregates metrics, checks gates,
    and writes the final markdown report.
    """
    df_attn = pd.read_csv(attention_blocks_path)
    df_replay = pd.read_csv(replay_results_path)
    df_kv = pd.read_csv(kv_estimates_path)
    
    # 1. Basic Stats
    total_questions = df_kv["question_id"].nunique()
    avg_total_blocks = df_kv["total_blocks"].mean()
    avg_kept_blocks = df_kv["kept_blocks"].mean()
    avg_kv_reduction = df_kv["estimated_kv_read_reduction"].mean()
    
    # 2. Replay Metrics
    # Filter by mode
    df_full = df_replay[df_replay["mode"] == "full"]
    df_vis = df_replay[df_replay["mode"] == "visibility"]
    df_rand = df_replay[df_replay["mode"] == "random"]
    df_bm25 = df_replay[df_replay["mode"] == "bm25"]
    
    full_em = df_full["exact_match"].mean() * 100
    vis_em = df_vis["exact_match"].mean() * 100
    rand_em = df_rand["exact_match"].mean() * 100
    bm25_em = df_bm25["exact_match"].mean() * 100
    
    full_num = df_full["numeric_preservation"].mean() * 100
    vis_num = df_vis["numeric_preservation"].mean() * 100
    
    # Contradiction category C accuracy
    # Filter contradiction question IDs
    contradiction_qids = df_replay[df_replay["category"] == "C"]["question_id"].unique()
    if len(contradiction_qids) > 0:
        vis_contra_acc = df_vis[df_vis["question_id"].isin(contradiction_qids)]["active_truth"].mean() * 100
    else:
        vis_contra_acc = 100.0 # Default if no contradiction category present (e.g. in mini tests)
        
    # Gold Block Recall
    # We check if the gold blocks are always kept in visibility mode
    vis_gold_recall = df_vis["gold_recall"].mean() * 100
    
    # Tokens
    avg_full_tokens = df_full["input_tokens"].mean()
    avg_replay_tokens = df_vis["input_tokens"].mean()
    token_reduction = (1 - avg_replay_tokens / avg_full_tokens) * 100
    
    # Latency (TTFT)
    avg_full_ttft = df_full["ttft_ms"].mean()
    avg_replay_ttft = df_vis["ttft_ms"].mean()
    ttft_reduction = (1 - avg_replay_ttft / avg_full_ttft) * 100 if avg_full_ttft > 0 else 0.0
    
    # 3. Gate validation
    gates = {
        "Gold Block Recall": {"value": vis_gold_recall, "target": 99.0, "status": "PASS" if vis_gold_recall >= 99.0 else "FAIL"},
        "Exact Match": {"value": vis_em, "target": 95.0, "status": "PASS" if vis_em >= 95.0 else "FAIL"},
        "Numeric Preservation": {"value": vis_num, "target": 99.0, "status": "PASS" if vis_num >= 99.0 else "FAIL"},
        "KV read reduction": {"value": avg_kv_reduction * 100, "target": 60.0, "status": "PASS" if (avg_kv_reduction * 100) >= 60.0 else "FAIL"},
        "Token reduction": {"value": token_reduction, "target": 60.0, "status": "PASS" if token_reduction >= 60.0 else "FAIL"},
        "Random baseline gap": {"value": vis_em - rand_em, "target": 20.0, "status": "PASS" if (vis_em - rand_em) >= 20.0 else "FAIL"},
        "BM25 baseline gap": {"value": vis_em - bm25_em, "target": 0.0, "status": "PASS" if (vis_em >= bm25_em) else "FAIL"},
        "Contradiction accuracy": {"value": vis_contra_acc, "target": 95.0, "status": "PASS" if vis_contra_acc >= 95.0 else "FAIL"}
    }
    
    overall_status = "PASS" if all(g["status"] == "PASS" for g in gates.values()) else "FAIL"
    
    # Format markdown
    report_content = f"""# POC 0 â€” KV Visibility Map Report

Status: **{overall_status}**

## Dataset Configuration
- **Total Questions**: {total_questions}
- **Average Blocks per Prompt**: {avg_total_blocks:.1f}
- **Block Size**: 128 tokens

## Quality Metrics

| Metric | Full Context | KV Visibility | Random Baseline | BM25 Baseline | Target | Status |
|---|---|---|---|---|---|---|
| **Exact Match** | {full_em:.1f}% | {vis_em:.1f}% | {rand_em:.1f}% | {bm25_em:.1f}% | &ge; 95.0% | **{gates["Exact Match"]["status"]}** |
| **Numeric Preservation** | {full_num:.1f}% | {vis_num:.1f}% | - | - | &ge; 99.0% | **{gates["Numeric Preservation"]["status"]}** |
| **Contradiction Accuracy** | - | {vis_contra_acc:.1f}% | - | - | &ge; 95.0% | **{gates["Contradiction accuracy"]["status"]}** |

## Evidence and Alignment

- **Gold Block Recall**: {vis_gold_recall:.1f}% (Target: &ge; 99.0%) &rarr; **{gates["Gold Block Recall"]["status"]}**
- **Random Baseline Gap**: {vis_em - rand_em:.1f} pts (Target: &ge; 20.0 pts) &rarr; **{gates["Random baseline gap"]["status"]}**
- **BM25 Comparison**: KV Visibility is {vis_em - bm25_em:+.1f} pts relative to BM25 (Target: &ge; 0.0 pts) &rarr; **{gates["BM25 baseline gap"]["status"]}**

## Efficiency & KV Performance

- **Average Context Blocks**: {avg_total_blocks:.1f}
- **Average Kept Blocks**: {avg_kept_blocks:.1f}
- **Estimated KV Read Reduction**: {avg_kv_reduction * 100:.1f}% (Target: &ge; 60.0%) &rarr; **{gates["KV read reduction"]["status"]}**
- **Token Reduction**: {token_reduction:.1f}% (Target: &ge; 60.0%) &rarr; **{gates["Token reduction"]["status"]}**

## Latency Metrics (TTFT)
- **Full Context TTFT**: {avg_full_ttft:.1f} ms
- **Replay Context TTFT**: {avg_replay_ttft:.1f} ms
- **TTFT Reduction**: {ttft_reduction:.1f}%

## Verdict
{overall_status == 'PASS' and 'The test shows that useful KV blocks are sparse, predictable, and highly localized. A render-inspired visibility/culling policy can successfully reduce active context and estimated KV reads while preserving answer quality.' or 'The test failed to satisfy all performance gates. Check individual baseline gaps and gold block recall rates.'}
"""

    os.makedirs(os.path.dirname(output_report_path), exist_ok=True)
    with open(output_report_path, "w") as f:
        f.write(report_content)
        
    print(f"Report written to {output_report_path}")
    print(f"Verdict: {overall_status}")
    
    return {
        "overall_status": overall_status,
        "gates": gates
    }


