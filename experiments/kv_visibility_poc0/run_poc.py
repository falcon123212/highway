import os
import json
import argparse
import torch
import pandas as pd
from tqdm import tqdm
from typing import List, Dict, Any

from src.build_dataset import build_dataset
from src.tokenize_blocks import tokenize_and_map_blocks
from src.run_full_attention import load_model_and_tokenizer, run_full_attention_inference
from src.select_visible_blocks import select_blocks_policy
from src.build_replay_prompt import build_replay_prompts
from src.run_replay import run_replay_inference
from src.evaluate import evaluate_sample
from src.report import generate_report

def main():
    parser = argparse.ArgumentParser(description="POC 0 â€” KV Visibility Map Orchestrator")
    parser.add_argument("--model", type=str, default="Qwen/Qwen2.5-0.5B-Instruct", help="Hugging Face model path")
    parser.add_argument("--block-size", type=int, default=128, help="KV block size in tokens")
    parser.add_argument("--num-samples", type=int, default=500, help="Number of samples to generate/evaluate")
    parser.add_argument("--top-k", type=int, default=24, help="Top K blocks to keep based on attention mass")
    parser.add_argument("--recent-n", type=int, default=2, help="N most recent blocks to always keep")
    parser.add_argument("--device", type=str, default="auto", help="cuda, cpu, or auto")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--out-dir", type=str, default="data", help="Directory for generated datasets")
    parser.add_argument("--reports-dir", type=str, default="reports", help="Directory for generated reports")
    parser.add_argument("--num-blocks", type=int, default=30, help="Number of context blocks per prompt")
    parser.add_argument("--step", type=str, choices=["all", "dataset", "run"], default="all", help="Pipeline step to run")
    
    args = parser.parse_args()
    
    # Resolve device
    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device
        
    print(f"Using device: {device}")
    
    # Fix seed
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
        
    os.makedirs(args.reports_dir, exist_ok=True)
    
    # 1. Dataset Generation Step
    if args.step in ["all", "dataset"]:
        print("\n=== [Step 1/2] Generating Dataset ===")
        build_dataset(num_samples=args.num_samples, out_dir=args.out_dir, num_blocks=args.num_blocks)
        if args.step == "dataset":
            return
            
    # 2. Inference & Evaluation Step
    print("\n=== [Step 2/2] Running Inference and Evaluation ===")
    
    # Read files
    corpus_path = os.path.join(args.out_dir, "corpus.jsonl")
    questions_path = os.path.join(args.out_dir, "questions.jsonl")
    answers_path = os.path.join(args.out_dir, "answers.jsonl")
    
    if not (os.path.exists(corpus_path) and os.path.exists(answers_path)):
        raise FileNotFoundError("Dataset files not found. Please run with --step dataset or --step all first.")
        
    corpus_samples = []
    with open(corpus_path, "r") as f:
        for line in f:
            corpus_samples.append(json.loads(line))
            
    gold_answers = {}
    with open(answers_path, "r") as f:
        for line in f:
            item = json.loads(line)
            gold_answers[item["question_id"]] = item
            
    # Load Model
    model, tokenizer = load_model_and_tokenizer(args.model, device)
    
    # Lists to hold data for CSVs
    attention_blocks_data = []
    replay_results_data = []
    kv_estimates_data = []
    
    # Process each sample
    print(f"Processing {len(corpus_samples)} samples...")
    for idx, sample in enumerate(tqdm(corpus_samples)):
        q_id = sample["question_id"]
        gold_info = gold_answers[q_id]
        
        # Tokenize and Blockify
        tokenized_sample = tokenize_and_map_blocks(
            tokenizer=tokenizer,
            sample=sample,
            gold_info=gold_info,
            block_size=args.block_size
        )
        
        # Run Full Attention Pass
        full_res = run_full_attention_inference(
            model=model,
            tokenizer=tokenizer,
            tokenized_sample=tokenized_sample,
            device=device
        )
        
        # Apply Visibility Policy
        # Project entity is in gold_info, but let's parse from project name
        project_entity = sample["project"]
        updated_blocks, kept_ids = select_blocks_policy(
            blocks=tokenized_sample["blocks"],
            block_scores=full_res["block_scores"],
            project_entity=project_entity,
            top_k=args.top_k,
            recent_n=args.recent_n
        )
        
        # Save attention block details for CSV
        for b in updated_blocks:
            # Find the rank of the block in attention scores
            ranked_scores = sorted(full_res["block_scores"].items(), key=lambda x: x[1], reverse=True)
            rank = [bid for bid, _ in ranked_scores].index(b["block_id"]) + 1
            
            attention_blocks_data.append({
                "question_id": q_id,
                "block_id": b["block_id"],
                "contains_gold_fact": str(b["contains_gold_fact"]).lower(),
                "attention_mass": f"{b['attention_mass']:.6f}",
                "rank": rank,
                "policy": b["policy"]
            })
            
        # Build Replay Prompts (visibility, random, BM25)
        replay_prompts = build_replay_prompts(
            sample=sample,
            kept_visibility_ids=kept_ids,
            tokenizer=tokenizer,
            seed=args.seed
        )
        
        # Run Replays
        replay_res = run_replay_inference(
            model=model,
            tokenizer=tokenizer,
            replay_prompts=replay_prompts,
            device=device
        )
        
        # Evaluate
        eval_res = evaluate_sample(
            sample_id=q_id,
            category=sample["category"],
            expected_answer=gold_info["expected_answer"],
            gold_block_ids=gold_info["gold_block_ids"],
            deprecated_block_ids=gold_info["deprecated_block_ids"],
            full_result=full_res,
            replay_results=replay_res,
            model_config=model.config
        )
        
        # Save evaluation details for CSVs
        # 1. replay_results.csv
        for mode in ["full", "visibility", "random", "bm25"]:
            mode_eval = eval_res[mode]
            replay_results_data.append({
                "question_id": q_id,
                "category": sample["category"],
                "mode": mode,
                "expected": gold_info["expected_answer"],
                "answer": mode_eval["answer"],
                "exact_match": str(mode_eval["exact_match"]).lower(),
                "numeric_preservation": str(mode_eval["numeric_preservation"]).lower(),
                "active_truth": str(mode_eval["active_truth"]).lower(),
                "gold_recall": str(mode_eval["gold_recall"]).lower(),
                "input_tokens": mode_eval["input_tokens"],
                "kept_blocks": mode_eval["kept_blocks"],
                "ttft_ms": f"{mode_eval['ttft_ms']:.2f}"
            })
            
        # 2. kv_estimates.csv
        total_blocks = len(tokenized_sample["blocks"])
        kept_blocks = len(kept_ids)
        dropped_blocks = total_blocks - kept_blocks
        kv_reduction = 1.0 - (kept_blocks / total_blocks)
        
        kv_estimates_data.append({
            "question_id": q_id,
            "total_blocks": total_blocks,
            "kept_blocks": kept_blocks,
            "dropped_blocks": dropped_blocks,
            "estimated_kv_read_reduction": f"{kv_reduction:.6f}"
        })
        
        # Periodic cache cleanup
        if device == "cuda":
            torch.cuda.empty_cache()
            
    # Write CSVs
    print("\nWriting output CSV files...")
    
    df_attn = pd.DataFrame(attention_blocks_data)
    attn_csv_path = os.path.join(args.reports_dir, "attention_blocks.csv")
    df_attn.to_csv(attn_csv_path, index=False)
    print(f"Saved: {attn_csv_path}")
    
    df_replay = pd.DataFrame(replay_results_data)
    replay_csv_path = os.path.join(args.reports_dir, "replay_results.csv")
    df_replay.to_csv(replay_csv_path, index=False)
    print(f"Saved: {replay_csv_path}")
    
    df_kv = pd.DataFrame(kv_estimates_data)
    kv_csv_path = os.path.join(args.reports_dir, "kv_estimates.csv")
    df_kv.to_csv(kv_csv_path, index=False)
    print(f"Saved: {kv_csv_path}")
    
    # Generate Report
    report_md_path = os.path.join(args.reports_dir, "poc0_report.md")
    generate_report(
        attention_blocks_path=attn_csv_path,
        replay_results_path=replay_csv_path,
        kv_estimates_path=kv_csv_path,
        output_report_path=report_md_path
    )
    
    print("\nPOC 0 Pipeline complete!")

if __name__ == "__main__":
    main()


