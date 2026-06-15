import os
import json
import argparse
import torch
import pandas as pd
from tqdm import tqdm
from typing import List, Dict, Any

from src.tokenize_blocks import tokenize_and_map_blocks
from src.run_full_attention import load_model_and_tokenizer, run_full_attention_inference

def main():
    parser = argparse.ArgumentParser(description="Extract Attention Weights for Training Predictor")
    parser.add_argument("--model", type=str, default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--block-size", type=int, default=128)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--data-dir", type=str, default="data")
    parser.add_argument("--reports-dir", type=str, default="reports")
    args = parser.parse_args()
    
    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device
    print(f"Using device: {device}")
    
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
        
    os.makedirs(args.reports_dir, exist_ok=True)
    
    corpus_path = os.path.join(args.data_dir, "corpus.jsonl")
    answers_path = os.path.join(args.data_dir, "answers.jsonl")
    
    if not (os.path.exists(corpus_path) and os.path.exists(answers_path)):
        raise FileNotFoundError("Dataset files not found. Please run build_dataset first.")
        
    corpus_samples = []
    with open(corpus_path, "r") as f:
        for line in f:
            corpus_samples.append(json.loads(line))
            
    gold_answers = {}
    with open(answers_path, "r") as f:
        for line in f:
            item = json.loads(line)
            gold_answers[item["question_id"]] = item
            
    model, tokenizer = load_model_and_tokenizer(args.model, device, attn_implementation="eager")
    
    attention_blocks_data = []
    
    print(f"Extracting prefill attention maps for {len(corpus_samples)} samples...")
    for sample in tqdm(corpus_samples):
        q_id = sample["question_id"]
        gold_info = gold_answers[q_id]
        
        tokenized_sample = tokenize_and_map_blocks(
            tokenizer=tokenizer,
            sample=sample,
            gold_info=gold_info,
            block_size=args.block_size
        )
        
        # Run with max_new_tokens=1 for maximum speed (we only need the prefill attention map!)
        full_res = run_full_attention_inference(
            model=model,
            tokenizer=tokenizer,
            tokenized_sample=tokenized_sample,
            device=device,
            max_new_tokens=1
        )
        
        # Aggregate attention details
        for block in tokenized_sample["blocks"]:
            bid = block["block_id"]
            score = full_res["block_scores"][bid]
            ranked_scores = sorted(full_res["block_scores"].items(), key=lambda x: x[1], reverse=True)
            rank = [k for k, _ in ranked_scores].index(bid) + 1
            
            # Simple policy classification
            if rank <= 16:
                policy = "HOT"
            elif rank <= 24:
                policy = "WARM"
            else:
                policy = "COLD"
                
            attention_blocks_data.append({
                "question_id": q_id,
                "block_id": bid,
                "contains_gold_fact": str(block["contains_gold_fact"]).lower(),
                "attention_mass": f"{score:.6f}",
                "rank": rank,
                "policy": policy
            })
            
        if device == "cuda":
            torch.cuda.empty_cache()
            
    df_attn = pd.DataFrame(attention_blocks_data)
    attn_csv_path = os.path.join(args.reports_dir, "attention_blocks.csv")
    df_attn.to_csv(attn_csv_path, index=False)
    print(f"Saved prefill attention weights to: {attn_csv_path}")

if __name__ == "__main__":
    main()


