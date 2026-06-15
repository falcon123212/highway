import os
import json
import pickle
import time
import argparse
import pandas as pd
import numpy as np
import random
from tqdm import tqdm
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

from src.extract_features import extract_block_features

def main():
    parser = argparse.ArgumentParser(description="Train Pre-Prefill Visibility Predictor")
    parser.add_argument("--data-dir", type=str, default="data")
    parser.add_argument("--reports-dir", type=str, default="reports")
    parser.add_argument("--model-dir", type=str, default="models")
    parser.add_argument("--top-k", type=int, default=16, help="Top K blocks oracle kept")
    parser.add_argument("--recent-n", type=int, default=2, help="N recent blocks oracle kept")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--ablation-mode", type=str, default="full", choices=["full", "no_position", "semantic_only"])
    parser.add_argument("--split-mode", type=str, default="standard", choices=["standard", "abcd_e", "abe_cd"])
    args = parser.parse_args()
    
    os.makedirs(args.model_dir, exist_ok=True)
    
    corpus_path = os.path.join(args.data_dir, "corpus.jsonl")
    attn_csv_path = os.path.join(args.reports_dir, "attention_blocks.csv")
    
    if not (os.path.exists(corpus_path) and os.path.exists(attn_csv_path)):
        raise FileNotFoundError("POC corpus.jsonl or attention_blocks.csv not found. Please run build_dataset and extract_attention_dataset first.")
        
    print("Loading attention logs...")
    df_attn = pd.read_csv(attn_csv_path)
    
    # Lookup mapping: (question_id, block_id) -> rank
    rank_map = {}
    for idx, row in df_attn.iterrows():
        key = (row["question_id"], int(row["block_id"]))
        rank_map[key] = {
            "rank": int(row["rank"]),
            "contains_gold": str(row["contains_gold_fact"]).lower() == "true"
        }
        
    print("Loading corpus samples...")
    samples = []
    with open(corpus_path, "r") as f:
        for line in f:
            samples.append(json.loads(line))
            
    print(f"Loaded {len(samples)} samples. Extracting features ({args.ablation_mode}) and labels...")
    
    # We will process sample by sample and store features and labels grouped by sample
    sample_data = []
    
    for s_idx, sample in enumerate(tqdm(samples)):
        q_id = sample["question_id"]
        project = sample["project"]
        question = sample["question"]
        blocks = sample["documents"]
        num_blocks = len(blocks)
        
        # 1. Feature Extraction with ablation support
        features = extract_block_features(
            question=question,
            blocks=blocks,
            project_entity=project,
            ablation_mode=args.ablation_mode
        )
        
        # 2. Label Generation
        labels = []
        recent_ids = set(range(max(0, num_blocks - args.recent_n), num_blocks))
        
        for b_idx, block in enumerate(blocks):
            key = (q_id, b_idx)
            info = rank_map.get(key, {"rank": 999, "contains_gold": False})
            
            is_top_k = info["rank"] <= args.top_k
            is_recent = b_idx in recent_ids
            entity_matched = project.lower() in block["text"].lower()
            
            keep = is_top_k or is_recent or entity_matched
            labels.append(1 if keep else 0)
            
        sample_data.append({
            "question_id": q_id,
            "category": sample["category"],
            "features": features, # (num_blocks, num_features)
            "labels": np.array(labels, dtype=np.int32)
        })
        
    # Split training/testing according to split_mode
    print(f"Splitting dataset using split mode: {args.split_mode}...")
    
    train_samples = []
    test_samples = []
    
    if args.split_mode == "standard":
        # Mixed split: reproducibly shuffle sample list, then split 80/20
        random.seed(args.seed)
        shuffled_data = list(sample_data)
        random.shuffle(shuffled_data)
        train_samples = shuffled_data[:400]
        test_samples = shuffled_data[400:]
    elif args.split_mode == "abcd_e":
        # Train on A, B, C, D (first 400 samples)
        # Test on E (last 100 samples)
        train_samples = sample_data[:400]
        test_samples = sample_data[400:]
    elif args.split_mode == "abe_cd":
        # Train on A, B, E (first 200 and last 100 samples)
        # Test on C, D (middle 200 samples)
        train_samples = sample_data[:200] + sample_data[400:]
        test_samples = sample_data[200:400]
        
    # Concatenate features and labels
    X_train = np.vstack([s["features"] for s in train_samples])
    y_train = np.concatenate([s["labels"] for s in train_samples])
    
    X_test = np.vstack([s["features"] for s in test_samples])
    y_test = np.concatenate([s["labels"] for s in test_samples])
    
    print(f"Train size: {X_train.shape[0]} blocks ({len(train_samples)} samples)")
    print(f"Test size: {X_test.shape[0]} blocks ({len(test_samples)} samples)")
    
    # 3. Model Training
    print(f"Training Random Forest classifier (split: {args.split_mode}, ablation: {args.ablation_mode})...")
    clf = RandomForestClassifier(n_estimators=100, class_weight="balanced", random_state=args.seed, max_depth=8)
    
    t0 = time.perf_counter()
    clf.fit(X_train, y_train)
    train_time = time.perf_counter() - t0
    print(f"Training completed in {train_time:.4f} seconds.")
    
    # Evaluate
    y_pred = clf.predict(X_test)
    print("\n=== Classifier Evaluation on Test Set ===")
    print(classification_report(y_test, y_pred))
    
    # Measure Latency on CPU
    print("\nMeasuring CPU latency per sample...")
    test_sample = samples[-1]
    latencies = []
    for _ in range(50):
        t_start = time.perf_counter()
        feats = extract_block_features(
            question=test_sample["question"],
            blocks=test_sample["documents"],
            project_entity=test_sample["project"],
            ablation_mode=args.ablation_mode
        )
        preds = clf.predict(feats)
        latencies.append((time.perf_counter() - t_start) * 1000.0)
        
    avg_latency = np.mean(latencies)
    print(f"Average CPU latency per sample: {avg_latency:.2f} ms")
    
    # Feature names based on ablation_mode
    if args.ablation_mode == "full":
        feat_names = ["bm25", "semantic", "rel_pos", "recency", "entity_match", "num_match", "status_active"]
    elif args.ablation_mode == "no_position":
        feat_names = ["bm25", "semantic", "entity_match", "num_match", "status_active"]
    else: # semantic_only
        feat_names = ["semantic", "entity_match"]
        
    # Save the custom model file
    model_name = f"visibility_predictor_{args.split_mode}_{args.ablation_mode}.pkl"
    model_path = os.path.join(args.model_dir, model_name)
    with open(model_path, "wb") as f:
        pickle.dump({
            "model": clf,
            "feature_names": feat_names,
            "ablation_mode": args.ablation_mode,
            "split_mode": args.split_mode
        }, f)
    print(f"Model saved to: {model_path}")
    
    # Also save to default path for backwards compatibility if running standard full
    if args.split_mode == "standard" and args.ablation_mode == "full":
        default_path = os.path.join(args.model_dir, "visibility_predictor.pkl")
        with open(default_path, "wb") as f:
            pickle.dump({
                "model": clf,
                "feature_names": feat_names,
                "ablation_mode": args.ablation_mode,
                "split_mode": args.split_mode
            }, f)
        print(f"Saved default model backup to: {default_path}")

if __name__ == "__main__":
    main()


