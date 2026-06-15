import os
import json
import pickle
import argparse
import torch
import pandas as pd
import numpy as np
from tqdm import tqdm
from typing import Dict, Any, List, Tuple

from src.extract_features import extract_block_features
from src.run_full_attention import load_model_and_tokenizer
from src.build_replay_prompt import tokenize_for_bm25
from src.run_replay import run_replay_inference
from src.evaluate import evaluate_sample

def build_baseline_ids(
    sample: Dict[str, Any],
    num_kept: int,
    bm25_scores: np.ndarray,
    cos_sims: np.ndarray,
    seed: int = 42
) -> Dict[str, List[int]]:
    """Builds block IDs to keep for Random, Dense, and Hybrid baselines."""
    documents = sample["documents"]
    total_docs = len(documents)
    
    # 1. Random
    import random
    rng = random.Random(seed + hash(sample["question_id"]))
    random_ids = sorted(rng.sample(range(total_docs), min(num_kept, total_docs)))
    
    # 2. Dense (top num_kept by MiniLM cosine similarity)
    dense_indices = np.argsort(cos_sims)[::-1]
    dense_ids = sorted(list(dense_indices[:num_kept]))
    
    # 3. Hybrid (top num_kept by 0.5 * BM25 + 0.5 * CosSim)
    max_bm25 = max(bm25_scores) if len(bm25_scores) > 0 and max(bm25_scores) > 0 else 1.0
    norm_bm25 = bm25_scores / max_bm25
    hybrid_scores = 0.5 * norm_bm25 + 0.5 * cos_sims
    hybrid_indices = np.argsort(hybrid_scores)[::-1]
    hybrid_ids = sorted(list(hybrid_indices[:num_kept]))
    
    return {
        "random": random_ids,
        "dense": dense_ids,
        "hybrid": hybrid_ids
    }

def main():
    parser = argparse.ArgumentParser(description="POC 0.1 â€” Pre-Prefill Visibility Predictor Pipeline")
    parser.add_argument("--model", type=str, default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--predictor-path", type=str, default="models/visibility_predictor.pkl")
    parser.add_argument("--data-dir", type=str, default="data")
    parser.add_argument("--reports-dir", type=str, default="reports")
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    
    # Resolve device
    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device
    print(f"Using device: {device}")
    
    # Load Predictor Model
    if not os.path.exists(args.predictor_path):
        raise FileNotFoundError(f"Predictor model {args.predictor_path} not found. Please run training first.")
        
    with open(args.predictor_path, "rb") as f:
        predictor_data = pickle.load(f)
    clf = predictor_data["model"]
    
    # Load Dataset
    corpus_path = os.path.join(args.data_dir, "corpus.jsonl")
    answers_path = os.path.join(args.data_dir, "answers.jsonl")
    poc0_replay_path = os.path.join(args.reports_dir, "replay_results.csv")
    
    if not (os.path.exists(corpus_path) and os.path.exists(answers_path)):
        raise FileNotFoundError("Dataset files not found.")
        
    corpus_samples = []
    with open(corpus_path, "r") as f:
        for line in f:
            corpus_samples.append(json.loads(line))
            
    gold_answers = {}
    with open(answers_path, "r") as f:
        for line in f:
            item = json.loads(line)
            gold_answers[item["question_id"]] = item
            
    # Load Oracle results from POC 0 for comparison
    oracle_df = None
    if os.path.exists(poc0_replay_path):
        print("Loading POC 0 Oracle results...")
        oracle_df = pd.read_csv(poc0_replay_path)
        # Keep only Oracle/visibility mode
        oracle_df = oracle_df[oracle_df["mode"] == "visibility"]
        oracle_map = {row["question_id"]: row["exact_match"] for _, row in oracle_df.iterrows()}
    else:
        print("WARNING: POC 0 Oracle results not found. Oracle comparison will be skipped.")
        oracle_map = {}
        
    # Split: we only validate on the test split (last 20%, 100 samples)
    num_samples = len(corpus_samples)
    test_start_idx = int(num_samples * 0.8)
    test_samples = corpus_samples[test_start_idx:]
    print(f"Running evaluation on {len(test_samples)} test split samples...")
    
    # Load LLM
    model, tokenizer = load_model_and_tokenizer(args.model, device)
    
    results_data = []
    
    # Process test split
    for sample in tqdm(test_samples):
        q_id = sample["question_id"]
        gold_info = gold_answers[q_id]
        question = sample["question"]
        project = sample["project"]
        documents = sample["documents"]
        
        # 1. Extract features
        features = extract_block_features(
            question=question,
            blocks=documents,
            project_entity=project
        )
        
        # 2. Run predictor with optimized threshold 0.70 (Gold Recall = 100%, Avg Kept Blocks = 11.7)
        probs = clf.predict_proba(features)[:, 1]
        kept_predictor_ids = [i for i, p in enumerate(probs) if p >= 0.70]
        if len(kept_predictor_ids) < 4:
            ranked_indices = np.argsort(probs)[::-1]
            kept_predictor_ids = sorted(list(ranked_indices[:4]))
            
        num_kept = len(kept_predictor_ids)
        
        # Extract individual features for BM25 and CosSim scores
        bm25_scores = features[:, 0]
        cos_sims = features[:, 1]
        
        # 3. Build baseline block IDs (with the same num_kept to be fair!)
        baselines = build_baseline_ids(
            sample=sample,
            num_kept=num_kept,
            bm25_scores=bm25_scores,
            cos_sims=cos_sims,
            seed=args.seed
        )
        
        # 4. Assemble prompts
        def assemble_prompt(kept_ids: List[int]) -> str:
            system_text = "<|im_start|>system\nYou are a helpful assistant. Answer the question based on the provided context. Be concise and precise.<|im_end|>\n<|im_start|>user\nContext:\n"
            context_parts = []
            last_id = -2
            for idx in kept_ids:
                if last_id != -2 and idx != last_id + 1:
                    context_parts.append("[...]")
                context_parts.append(documents[idx]["text"])
                last_id = idx
            context_text = "\n\n".join(context_parts)
            question_text = f"\n\nQuestion: {question}<|im_end|>\n<|im_start|>assistant\n"
            return system_text + context_text + question_text

        prompt_modes = {
            "predictor": kept_predictor_ids,
            "random": baselines["random"],
            "dense": baselines["dense"],
            "hybrid": baselines["hybrid"]
        }
        
        replay_prompts = {}
        for mode, ids in prompt_modes.items():
            prompt_text = assemble_prompt(ids)
            replay_prompts[mode] = {
                "text": prompt_text,
                "ids": ids,
                "token_ids": tokenizer.encode(prompt_text)
            }
            
        # 5. Run Replays
        import time
        mode_answers = {}
        for mode, prompt_data in replay_prompts.items():
            input_ids = prompt_data["token_ids"]
            prompt_len = len(input_ids)
            input_tensor = torch.tensor([input_ids], device=device)
            t0 = time.perf_counter()
            with torch.no_grad():
                outputs = model.generate(
                    input_tensor,
                    max_new_tokens=64,
                    output_attentions=False,
                    return_dict_in_generate=True,
                    use_cache=True,
                    pad_token_id=tokenizer.eos_token_id
                )
            duration_ms = (time.perf_counter() - t0) * 1000.0
            generated_ids = outputs.sequences[0][prompt_len:]
            answer = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
            
            mode_answers[mode] = {
                "answer": answer,
                "ttft_ms": duration_ms,
                "input_tokens": prompt_len,
                "output_tokens": len(generated_ids),
                "kept_ids": prompt_data["ids"]
            }
        
        # 6. Evaluate each mode
        for mode, res in mode_answers.items():
            # Mock full result for evaluate_sample
            mock_full = {"answer": "", "input_tokens": 100, "ttft_ms": 0.0}
            mock_replay = {
                "visibility": {
                    "answer": res["answer"],
                    "input_tokens": res["input_tokens"],
                    "ttft_ms": res["ttft_ms"],
                    "kept_ids": res["kept_ids"]
                },
                "random": {"answer": "", "input_tokens": 0, "ttft_ms": 0.0, "kept_ids": []},
                "bm25": {"answer": "", "input_tokens": 0, "ttft_ms": 0.0, "kept_ids": []}
            }
            eval_res = evaluate_sample(
                sample_id=q_id,
                category=sample["category"],
                expected_answer=gold_info["expected_answer"],
                gold_block_ids=gold_info["gold_block_ids"],
                deprecated_block_ids=gold_info["deprecated_block_ids"],
                full_result=mock_full,
                replay_results=mock_replay,
                model_config=model.config
            )
            
            mode_eval = eval_res["visibility"]
            oracle_em = str(oracle_map.get(q_id, False)).lower()
            
            results_data.append({
                "question_id": q_id,
                "category": sample["category"],
                "mode": mode,
                "expected": gold_info["expected_answer"],
                "answer": mode_eval["answer"],
                "exact_match": str(mode_eval["exact_match"]).lower(),
                "numeric_preservation": str(mode_eval["numeric_preservation"]).lower(),
                "gold_recall": str(mode_eval["gold_recall"]).lower(),
                "kept_blocks": mode_eval["kept_blocks"],
                "input_tokens": mode_eval["input_tokens"],
                "ttft_ms": f"{mode_eval['ttft_ms']:.2f}",
                "oracle_exact_match": oracle_em
            })
            
    # Write CSV
    df_results = pd.DataFrame(results_data)
    results_path = os.path.join(args.reports_dir, "poc01_results.csv")
    df_results.to_csv(results_path, index=False)
    print(f"Saved: {results_path}")
    
    # 7. Generate final report
    generate_poc01_report(df_results, os.path.join(args.reports_dir, "poc01_report.md"))

def generate_poc01_report(df: pd.DataFrame, output_path: str):
    """Aggregates results and generates a markdown report."""
    total_samples = df["question_id"].nunique()
    
    # Cast float columns before groupby
    df = df.copy()
    df["ttft_ms"] = df["ttft_ms"].astype(float)
    
    # Cast boolean columns to bool
    df["exact_match"] = df["exact_match"].astype(str).str.lower() == "true"
    df["numeric_preservation"] = df["numeric_preservation"].astype(str).str.lower() == "true"
    df["gold_recall"] = df["gold_recall"].astype(str).str.lower() == "true"
    df["oracle_exact_match"] = df["oracle_exact_match"].astype(str).str.lower() == "true"
    
    # Group by mode and calculate averages
    grouped = df.groupby("mode")
    
    em = grouped["exact_match"].mean() * 100
    num_pres = grouped["numeric_preservation"].mean() * 100
    recall = grouped["gold_recall"].mean() * 100
    kept_blocks = grouped["kept_blocks"].mean()
    tokens = grouped["input_tokens"].mean()
    ttft = grouped["ttft_ms"].mean()
    
    # Oracle EM (since oracle is constant per question, we look at the saved column)
    oracle_em_val = df[df["mode"] == "predictor"]["oracle_exact_match"].mean() * 100
    
    # Token reduction
    # Max tokens represents the full context (~50 blocks * 128 tokens = 6400 tokens + metadata = ~6550)
    full_context_tokens = 6550.0
    token_reduction = (1.0 - tokens["predictor"] / full_context_tokens) * 100
    
    # Gates check
    predictor_em = em["predictor"]
    dense_em = em["dense"]
    bm25_target = dense_em + 5.0 # baseline target
    
    gates = {
        "Gold Block Recall": {"value": recall["predictor"], "target": 99.0, "status": "PASS" if recall["predictor"] >= 99.0 else "FAIL"},
        "Kept Blocks": {"value": kept_blocks["predictor"], "target": 20.0, "status": "PASS" if kept_blocks["predictor"] <= 20.0 else "FAIL"},
        "Token Reduction": {"value": token_reduction, "target": 60.0, "status": "PASS" if token_reduction >= 60.0 else "FAIL"},
        "Exact Match": {"value": predictor_em, "target": bm25_target, "status": "PASS" if predictor_em >= bm25_target else "FAIL"}
    }
    
    overall_status = "PASS" if all(g["status"] == "PASS" for g in gates.values()) else "FAIL"
    
    report_content = f"""# POC 0.1 â€” Pre-Prefill Visibility Predictor Report

Status: **{overall_status}**

## Configuration
- **Validation Test Samples**: {total_samples}
- **Average Blocks per Prompt**: 50.0
- **Block Size**: 128 tokens
- **No Full-Context LLM Pass**: **TRUE** (No prefill attention pass required!)

## Quality & Accuracy Comparison

| Metric | Predictor (POC 0.1) | Oracle (POC 0) | Dense (MiniLM) | Hybrid | Random |
|---|---|---|---|---|---|
| **Exact Match** | {em["predictor"]:.1f}% | {oracle_em_val:.1f}% | {em["dense"]:.1f}% | {em["hybrid"]:.1f}% | {em["random"]:.1f}% |
| **Numeric Preservation** | {num_pres["predictor"]:.1f}% | - | {num_pres["dense"]:.1f}% | {num_pres["hybrid"]:.1f}% | - |

## Evidence and Culling Performance

- **Gold Block Recall**: {recall["predictor"]:.1f}% (Target: &ge; 99.0%) &rarr; **{gates["Gold Block Recall"]["status"]}**
- **Average Kept Blocks**: {kept_blocks["predictor"]:.1f} / 50 (Target: &le; 20.0) &rarr; **{gates["Kept Blocks"]["status"]}**
- **Token Reduction**: {token_reduction:.1f}% (Target: &ge; 60.0%) &rarr; **{gates["Token Reduction"]["status"]}**
- **Exact Match vs Dense**: Predictor is {predictor_em - dense_em:+.1f} pts relative to Dense (Target: &ge; +5.0 pts) &rarr; **{gates["Exact Match"]["status"]}**

## Latency Metrics (TTFT)
- **Predictor TTFT**: {ttft["predictor"]:.1f} ms
- **Hybrid TTFT**: {ttft["hybrid"]:.1f} ms

## Verdict
{overall_status == 'PASS' and 'The visibility predictor successfully culls the KV context prior to LLM prefill with high recall, achieving significant token reduction and satisfying all gates.' or 'The predictor failed to satisfy all validation gates. Review classifier accuracy and recall.'}
"""

    with open(output_path, "w") as f:
        f.write(report_content)
    print(f"Report written to {output_path}")
    print(f"Verdict: {overall_status}")

if __name__ == "__main__":
    main()


