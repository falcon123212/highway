import os
import json
import pickle
import time
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.metrics import classification_report, recall_score, precision_score

from src.extract_features import extract_block_features

def main():
    corpus_path = "data/corpus.jsonl"
    attn_csv_path = "reports/attention_blocks.csv"
    answers_path = "data/answers.jsonl"
    
    if not (os.path.exists(corpus_path) and os.path.exists(attn_csv_path)):
        print("Data files not found.")
        return
        
    print("Loading attention logs and dataset...")
    df_attn = pd.read_csv(attn_csv_path)
    rank_map = {}
    for idx, row in df_attn.iterrows():
        key = (row["question_id"], int(row["block_id"]))
        rank_map[key] = {
            "rank": int(row["rank"]),
            "contains_gold": str(row["contains_gold_fact"]).lower() == "true"
        }
        
    samples = [json.loads(line) for line in open(corpus_path)]
    answers = {item["question_id"]: item for item in [json.loads(line) for line in open(answers_path)]}
    
    print(f"Extracting features for {len(samples)} samples...")
    X_list = []
    y_list = []
    
    for sample in tqdm(samples):
        q_id = sample["question_id"]
        project = sample["project"]
        question = sample["question"]
        blocks = sample["documents"]
        num_blocks = len(blocks)
        
        features = extract_block_features(question, blocks, project)
        
        labels = []
        recent_ids = set(range(max(0, num_blocks - 2), num_blocks))
        
        for b_idx, block in enumerate(blocks):
            key = (q_id, b_idx)
            info = rank_map.get(key, {"rank": 999, "contains_gold": False})
            
            is_top_k = info["rank"] <= 16
            is_recent = b_idx in recent_ids
            entity_matched = project.lower() in block["text"].lower()
            
            keep = is_top_k or is_recent or entity_matched
            labels.append(1 if keep else 0)
            
        X_list.append(features)
        y_list.extend(labels)
        
    X = np.vstack(X_list)
    y = np.array(y_list)
    
    # Train/Test Split
    train_count = 400
    X_train, y_train = X[:train_count * 50], y[:train_count * 50]
    X_test, y_test = X[train_count * 50:], y[train_count * 50:]
    
    test_samples = samples[train_count:]
    
    models = {
        "Logistic Regression": LogisticRegression(class_weight="balanced", random_state=42, max_iter=1000),
        "Random Forest": RandomForestClassifier(n_estimators=100, class_weight="balanced", random_state=42, max_depth=8),
        "HistGradientBoosting": HistGradientBoostingClassifier(random_state=42, max_leaf_nodes=15)
    }
    
    for name, clf in models.items():
        print(f"\n--- Training {name} ---")
        t0 = time.perf_counter()
        clf.fit(X_train, y_train)
        fit_time = time.perf_counter() - t0
        
        y_pred = clf.predict(X_test)
        print(f"Fit time: {fit_time:.4f}s")
        print(classification_report(y_test, y_pred))
        
        # CPU Latency & Kept Blocks & Gold Recall
        latencies = []
        recalls = []
        kept_counts = []
        
        # Let's test different thresholds for each model
        thresholds = [0.5, 0.6, 0.7, 0.8]
        if name == "HistGradientBoosting":
            # HistGradientBoosting doesn't support class_weight directly, so threshold tuning is key
            thresholds = [0.2, 0.3, 0.4, 0.5]
            
        for thresh in thresholds:
            recalls = []
            kept_counts = []
            
            # Measure latency on 10 runs of a test sample
            t_lat = []
            for _ in range(10):
                t_start = time.perf_counter()
                feats = extract_block_features(test_samples[0]["question"], test_samples[0]["documents"], test_samples[0]["project"])
                probs = clf.predict_proba(feats)[:, 1]
                t_lat.append((time.perf_counter() - t_start) * 1000.0)
                
            avg_lat = np.mean(t_lat)
            
            # Evaluate threshold performance across all test samples
            for s_idx, sample in enumerate(test_samples):
                gold_ids = answers[sample["question_id"]]["gold_block_ids"]
                # Get the features for this test sample
                sample_feats = X_test[s_idx * 50 : (s_idx + 1) * 50]
                probs = clf.predict_proba(sample_feats)[:, 1]
                
                kept = [i for i, p in enumerate(probs) if p >= thresh]
                if len(kept) < 4:
                    kept = sorted(list(np.argsort(probs)[::-1][:4]))
                    
                kept_counts.append(len(kept))
                recalls.append(all(gid in kept for gid in gold_ids))
                
            print(f"Thresh: {thresh:.2f} | Gold Recall: {np.mean(recalls)*100:.1f}% | Avg Kept Blocks: {np.mean(kept_counts):.2f} | Latency: {avg_lat:.2f} ms")

if __name__ == "__main__":
    main()


