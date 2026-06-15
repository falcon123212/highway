import os
import json
import time
import argparse
import pickle
import numpy as np
import random
from typing import Dict, Any, List, Tuple

import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from src.extract_features import (
    extract_block_features, get_block_embeddings, get_embedding_model, tokenize_for_bm25, clear_embedding_cache
)

# Prompt compilers
def assemble_prompt(kept_ids: List[int], question: str, documents: List[Dict[str, Any]]) -> str:
    system_text = (
        "<|im_start|>system\n"
        "You are a helpful assistant. Answer the question based on the provided context.\n"
        "You MUST respond with a strict JSON object containing the keys 'answer', 'evidence_block_id', and 'evidence_quote'.\n"
        "Rules:\n"
        "- Copy numbers, dates, IDs exactly from the evidence.\n"
        "- Do not paraphrase numeric values.\n"
        "- Do not explain.\n"
        "- Match the requested project name EXACTLY. Do not answer using a project that has an extra suffix. "
        "For example, if asked for 'Project X', do NOT match it to 'Project X-Legacy', 'Project X-A', or 'Project X-B'. Only match 'Project X' exactly.\n"
        "- If the answer is a date, return only the date string from the evidence.\n"
        "- If the answer is a budget, return only the exact budget value from the evidence.\n"
        "- If the question asks for both a date and a budget, return both formatted exactly as 'DATE and BUDGET' (for example: '15 May 2027 and $150,000').\n"
        "Output ONLY the raw JSON block. Do not include markdown code block formatting or explanation.\n"
        "Example format:\n"
        '{\n  "answer": "15 May 2027",\n  "evidence_block_id": "DOC_0012",\n  "evidence_quote": "Project: X Active delivery date: 15 May 2027"\n}\n'
        "<|im_end|>\n"
        "<|im_start|>user\nContext:\n"
    )
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

def scale_sample(sample: Dict[str, Any], num_blocks: int, seed: int = 42) -> Dict[str, Any]:
    docs = list(sample["documents"])
    current_len = len(docs)
    if current_len >= num_blocks:
        return {
            "question_id": sample["question_id"],
            "category": sample["category"],
            "project": sample["project"],
            "question": sample["question"],
            "documents": docs[:num_blocks]
        }
    needed = num_blocks - current_len
    rng = random.Random(seed + num_blocks + hash(sample["question_id"]))
    depts = ["HR", "Finance", "Legal", "Engineering", "Marketing", "Operations", "Sales"]
    buzz = ["integration", "synergy", "paradigm", "scalability", "leverage", "robust", "deployment"]
    for i in range(needed):
        dept = rng.choice(depts)
        bz = rng.choice(buzz)
        doc_id = f"NOISE_{i:04d}"
        text = f"{doc_id}:\nThe {dept} department is optimizing its {bz} strategies across the enterprise."
        docs.append({"doc_id": doc_id, "text": text, "contains_gold_fact": False, "contains_deprecated_fact": False})
    return {
        "question_id": sample["question_id"],
        "category": sample["category"],
        "project": sample["project"],
        "question": sample["question"],
        "documents": docs
    }

def main():
    parser = argparse.ArgumentParser(description="POC 1 Selector Run")
    parser.add_argument("--data-dir", type=str, default="experiments/kv_visibility_poc1/data")
    parser.add_argument("--model-path", type=str, default="experiments/kv_visibility_poc0/models/visibility_predictor_standard_no_position.pkl")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--samples-per-category", type=int, default=30)
    args = parser.parse_args()
    
    # Load dataset
    corpus_path = os.path.join(args.data_dir, "corpus.jsonl")
    answers_path = os.path.join(args.data_dir, "answers.jsonl")
    if not (os.path.exists(corpus_path) and os.path.exists(answers_path)):
        raise FileNotFoundError(f"Dataset files not found in {args.data_dir}")
        
    corpus_samples = []
    with open(corpus_path, "r") as f:
        for line in f:
            corpus_samples.append(json.loads(line))
            
    gold_answers = {}
    with open(answers_path, "r") as f:
        for line in f:
            item = json.loads(line)
            gold_answers[item["question_id"]] = item
            
    # Load predictor model
    with open(args.model_path, "rb") as f:
        pred_data = pickle.load(f)
    clf = pred_data["model"]
    
    # Filter by category and select samples
    cat_samples = {cat: [] for cat in ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J"]}
    for s in corpus_samples:
        if s["category"] in cat_samples:
            cat_samples[s["category"]].append(s)
            
    rng = random.Random(args.seed)
    selected_samples = []
    for cat in sorted(cat_samples.keys()):
        shuffled = list(cat_samples[cat])
        rng.shuffle(shuffled)
        selected_samples.extend(shuffled[:args.samples_per_category])
        
    print(f"Loaded {len(selected_samples)} samples ({args.samples_per_category} per category).")
    
    compiled_prompts = []
    context_sizes = [50, 200, 400]
    
    for size in context_sizes:
        print(f"Compiling prompts for context size: {size} blocks...")
        
        # Warmup sentence transformer cache for this size
        for sample in selected_samples:
            scaled = scale_sample(sample, size, seed=args.seed)
            get_block_embeddings([b["text"] for b in scaled["documents"]])
            
        for sample in selected_samples:
            q_id = sample["question_id"]
            category = sample["category"]
            project = sample["project"]
            question = sample["question"]
            gold_info = gold_answers[q_id]
            
            scaled = scale_sample(sample, size, seed=args.seed)
            documents = scaled["documents"]
            gold_ids = [idx for idx, doc in enumerate(documents) if doc.get("contains_gold_fact", False)]
            
            # --- 1. Predictor: On-the-fly (Warm cache clear to measure real CPU encoding cost) ---
            # To measure on-the-fly latency accurately, we clear the internal ST cache for these specific blocks
            clear_embedding_cache()
            t_otf = time.perf_counter()
            features_otf = extract_block_features(question, documents, project, ablation_mode="no_position")
            probs_otf = clf.predict_proba(features_otf)[:, 1]
            kept_ids_pred = [idx for idx, p in enumerate(probs_otf) if p >= 0.70]
            if len(kept_ids_pred) < 4:
                kept_ids_pred = sorted(list(np.argsort(probs_otf)[::-1][:4]))
            latency_otf = (time.perf_counter() - t_otf) * 1000.0
            
            # --- 2. Predictor: Cached (Lookup only, pre-calculate embeddings before timer) ---
            # Pre-compute block embeddings so they are cached
            block_texts = [b["text"] for b in documents]
            pre_computed_embs = get_block_embeddings(block_texts)
            
            t_cached = time.perf_counter()
            features_cached = extract_block_features(
                question, documents, project, ablation_mode="no_position",
                skip_embedding_compute=True, cached_block_embs=pre_computed_embs
            )
            probs_cached = clf.predict_proba(features_cached)[:, 1]
            kept_ids_cached = [idx for idx, p in enumerate(probs_cached) if p >= 0.70]
            if len(kept_ids_cached) < 4:
                kept_ids_cached = sorted(list(np.argsort(probs_cached)[::-1][:4]))
            latency_cached = (time.perf_counter() - t_cached) * 1000.0
            
            # --- 3. Hybrid Baseline ---
            t_hyb = time.perf_counter()
            from rank_bm25 import BM25Okapi
            corpus_tokens = [tokenize_for_bm25(b["text"]) for b in documents]
            bm25 = BM25Okapi(corpus_tokens)
            q_tokens = tokenize_for_bm25(question)
            bm25_scores = bm25.get_scores(q_tokens)
            max_bm25 = max(bm25_scores) if len(bm25_scores) > 0 and max(bm25_scores) > 0 else 1.0
            norm_bm25 = bm25_scores / max_bm25
            
            model_emb = get_embedding_model()
            q_emb = model_emb.encode(question, convert_to_tensor=False, show_progress_bar=False)
            q_norm = np.linalg.norm(q_emb)
            block_norms = np.linalg.norm(pre_computed_embs, axis=1)
            dot_products = np.dot(pre_computed_embs, q_emb)
            cos_sims = dot_products / (q_norm * block_norms + 1e-8)
            
            hybrid_scores = 0.5 * norm_bm25 + 0.5 * cos_sims
            hybrid_indices = np.argsort(hybrid_scores)[::-1]
            # Match the budget of kept blocks from the Predictor for direct fair cost scaling
            num_budget = len(kept_ids_pred)
            kept_ids_hybrid = sorted(list(hybrid_indices[:num_budget]))
            latency_hybrid = (time.perf_counter() - t_hyb) * 1000.0
            
            # --- 4. Random Baseline ---
            rng_rand = random.Random(args.seed + hash(q_id))
            kept_ids_random = sorted(rng_rand.sample(range(len(documents)), num_budget))
            
            # --- 5. Oracle / Gold ---
            kept_ids_oracle = gold_ids if len(gold_ids) > 0 else [0] # default to first if no answer
            
            # --- 6. Full Context ---
            kept_ids_full = list(range(len(documents)))
            
            # Compile prompt texts
            modes_and_ids = {
                "full_context": (kept_ids_full, 0.0),
                "oracle": (kept_ids_oracle, 0.0),
                "random": (kept_ids_random, 0.0),
                "hybrid": (kept_ids_hybrid, latency_hybrid),
                "predictor_otf": (kept_ids_pred, latency_otf),
                "predictor_cached": (kept_ids_cached, latency_cached)
            }
            
            for mode, (kept_ids, sel_latency) in modes_and_ids.items():
                prompt_text = assemble_prompt(kept_ids, question, documents)
                gold_recall = all(gid in kept_ids for gid in gold_ids) if len(gold_ids) > 0 else True
                
                compiled_prompts.append({
                    "question_id": q_id,
                    "category": category,
                    "project": project,
                    "question": question,
                    "context_size_blocks": size,
                    "mode": mode,
                    "compiled_prompt": prompt_text,
                    "selector_latency_ms": sel_latency,
                    "kept_blocks_count": len(kept_ids),
                    "token_reduction_pct": (1.0 - len(kept_ids) / len(documents)) * 100.0,
                    "gold_block_recall": gold_recall,
                    "expected_answer": gold_info["expected_answer"],
                    "is_abstention": gold_info.get("is_abstention", False),
                    "gold_block_ids": gold_ids,
                    "deprecated_block_ids": gold_info.get("deprecated_block_ids", [])
                })
                
    # Save output
    output_path = os.path.join(args.data_dir, "compiled_prompts.json")
    with open(output_path, "w") as f:
        json.dump(compiled_prompts, f, indent=2)
        
    print(f"Successfully compiled {len(compiled_prompts)} prompts.")
    print(f"Output saved to: {output_path}")

if __name__ == "__main__":
    main()


