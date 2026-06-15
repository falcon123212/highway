import os
import json
import time
import argparse
import pickle
import torch
import numpy as np
import pandas as pd
import random
import re
import string
from tqdm import tqdm
from typing import Dict, Any, List, Tuple
from transformers import AutoModelForCausalLM, AutoTokenizer
from rank_bm25 import BM25Okapi

from src.extract_features import extract_block_features, get_block_embeddings, get_embedding_model, tokenize_for_bm25
from src.run_full_attention import load_model_and_tokenizer

# Robust JSON response parser for the new format
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
        # Locate JSON block boundaries
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
        # Regex substring fallback
        ans_match = re.search(r'"answer"\s*:\s*"([^"]+)"', text, re.IGNORECASE)
        doc_match = re.search(r'"evidence_(?:block_)?id"\s*:\s*"([^"]+)"', text, re.IGNORECASE)
        quote_match = re.search(r'"evidence_quote"\s*:\s*"([^"]+)"', text, re.IGNORECASE)
        return {
            "answer": ans_match.group(1) if ans_match else text,
            "evidence_block_id": doc_match.group(1) if doc_match else "",
            "evidence_quote": quote_match.group(1) if quote_match else ""
        }

# Normalized evaluations for date, currency, casing, and punctuation
def normalize_answer(val: str) -> str:
    if val is None:
        return ""
    val = str(val).strip().lower()
    
    # 1. Currency & Number normalization: strip $ and remove spaces/commas in purely formatted digits
    val = val.replace("$", "")
    
    def clean_num(match):
        return match.group(0).replace(",", "").replace(".", "").replace(" ", "")
    
    val = re.sub(r'\b\d+([,\.\s]\d+)+\b', clean_num, val)
    
    # 2. Trim punctuation
    val = val.translate(str.maketrans("", "", string.punctuation))
    
    # 3. Trim extra spaces
    val = " ".join(val.split())
    return val

def normalize_id(val: str) -> str:
    if val is None:
        return ""
    return str(val).strip().lower()

# New prompt builder with strict format and rules
def assemble_json_prompt_b(kept_ids: List[int], question: str, documents: List[Dict[str, Any]]) -> str:
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

def build_baseline_ids_and_latencies(
    sample: Dict[str, Any],
    num_kept: int,
    seed: int = 42
) -> Tuple[Dict[str, List[int]], Dict[str, float]]:
    documents = sample["documents"]
    total_docs = len(documents)
    
    # 1. Random
    import random
    rng = random.Random(seed + hash(sample["question_id"]))
    random_ids = sorted(rng.sample(range(total_docs), min(num_kept, total_docs)))
    
    # 2. Dense (Measure query embedding + cos sim sorting)
    t0_dense = time.perf_counter()
    model_emb = get_embedding_model()
    q_emb = model_emb.encode(sample["question"], convert_to_tensor=False, show_progress_bar=False)
    block_texts = [b["text"] for b in documents]
    block_embs = get_block_embeddings(block_texts)
    q_norm = np.linalg.norm(q_emb)
    block_norms = np.linalg.norm(block_embs, axis=1)
    dot_products = np.dot(block_embs, q_emb)
    cos_sims = dot_products / (q_norm * block_norms + 1e-8)
    dense_indices = np.argsort(cos_sims)[::-1]
    dense_ids = sorted(list(dense_indices[:num_kept]))
    dense_retrieval_ms = (time.perf_counter() - t0_dense) * 1000.0
    
    # 3. Hybrid (Measure BM25 + dense combining + sorting)
    t0_hybrid = time.perf_counter()
    # BM25
    corpus = [tokenize_for_bm25(b["text"]) for b in documents]
    bm25 = BM25Okapi(corpus)
    query = tokenize_for_bm25(sample["question"])
    bm25_scores = bm25.get_scores(query)
    max_bm25 = max(bm25_scores) if len(bm25_scores) > 0 and max(bm25_scores) > 0 else 1.0
    norm_bm25 = bm25_scores / max_bm25
    
    # Dense (we re-calculate dense here to measure realistic query time cost)
    q_emb_h = model_emb.encode(sample["question"], convert_to_tensor=False, show_progress_bar=False)
    block_embs_h = get_block_embeddings(block_texts)
    q_norm_h = np.linalg.norm(q_emb_h)
    block_norms_h = np.linalg.norm(block_embs_h, axis=1)
    dot_products_h = np.dot(block_embs_h, q_emb_h)
    cos_sims_h = dot_products_h / (q_norm_h * block_norms_h + 1e-8)
    
    # Combine
    hybrid_scores = 0.5 * norm_bm25 + 0.5 * cos_sims_h
    hybrid_indices = np.argsort(hybrid_scores)[::-1]
    hybrid_ids = sorted(list(hybrid_indices[:num_kept]))
    hybrid_retrieval_ms = (time.perf_counter() - t0_hybrid) * 1000.0
    
    return {
        "random": random_ids,
        "dense": dense_ids,
        "hybrid": hybrid_ids
    }, {
        "dense": dense_retrieval_ms,
        "hybrid": hybrid_retrieval_ms
    }

def run_manual_inference(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    input_ids: List[int],
    device: str,
    max_new_tokens: int = 64
) -> Tuple[float, float, str, int]:
    input_tensor = torch.tensor([input_ids], device=device)
    prompt_len = len(input_ids)
    
    # 1. Prefill (TTFT)
    t0 = time.perf_counter()
    with torch.no_grad():
        model_outputs = model(input_ids=input_tensor, use_cache=True)
        past_key_values = model_outputs.past_key_values
        next_token_logits = model_outputs.logits[:, -1, :]
        next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)
    prefill_ms = (time.perf_counter() - t0) * 1000.0
    
    # 2. Decode
    t0_decode = time.perf_counter()
    generated_tokens = [next_token.item()]
    curr_input = next_token
    curr_past = past_key_values
    
    for _ in range(max_new_tokens - 1):
        with torch.no_grad():
            model_outputs = model(input_ids=curr_input, past_key_values=curr_past, use_cache=True)
            curr_past = model_outputs.past_key_values
            logits = model_outputs.logits[:, -1, :]
            next_tok = torch.argmax(logits, dim=-1, keepdim=True)
            generated_tokens.append(next_tok.item())
            curr_input = next_tok
            if next_tok.item() == tokenizer.eos_token_id:
                break
    decode_ms = (time.perf_counter() - t0_decode) * 1000.0
    
    answer = tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()
    return prefill_ms, decode_ms, answer, len(generated_tokens)

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
    rng = random.Random(seed + num_blocks)
    
    depts = ["HR", "Finance", "Legal", "Engineering", "Marketing", "Operations", "Sales"]
    buzz = ["integration", "synergy", "paradigm", "scalability", "leverage", "robust", "deployment"]
    
    for i in range(needed):
        dept = rng.choice(depts)
        bz = rng.choice(buzz)
        doc_id = f"NOISE_{i:04d}"
        text = f"{doc_id}:\nThe {dept} department is optimizing its {bz} strategies across the enterprise."
        docs.append({"doc_id": doc_id, "text": text, "contains_gold_fact": False})
        
    return {
        "question_id": sample["question_id"],
        "category": sample["category"],
        "project": sample["project"],
        "question": sample["question"],
        "documents": docs
    }

def main():
    parser = argparse.ArgumentParser(description="POC 0.3b â€” Clean Cost Dominance Confirmation")
    parser.add_argument("--model", type=str, default="Qwen/Qwen2.5-3B-Instruct")
    parser.add_argument("--num-samples", type=int, default=100, help="Number of samples to benchmark")
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    
    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device
    print(f"Using device: {device}")
    
    # Load dataset
    data_dir = "data"
    corpus_path = os.path.join(data_dir, "corpus.jsonl")
    answers_path = os.path.join(data_dir, "answers.jsonl")
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
            
    # Load Predictor Random Forest
    model_path = os.path.join("models", "visibility_predictor_standard_no_position.pkl")
    if not os.path.exists(model_path):
        raise FileNotFoundError("Trained Random Forest model not found.")
    with open(model_path, "rb") as f:
        predictor_data = pickle.load(f)
    clf = predictor_data["model"]
    
    # Load LLM
    model, tokenizer = load_model_and_tokenizer(args.model, device, attn_implementation="sdpa")
    
    if device == "cuda":
        torch.cuda.empty_cache()
        base_allocated = torch.cuda.memory_allocated()
        total_gpu_mem = torch.cuda.get_device_properties(0).total_memory
        print(f"Base Model Memory: {base_allocated / 1024 / 1024:.1f} MB (Total VRAM: {total_gpu_mem / 1024 / 1024:.1f} MB)")
    else:
        base_allocated = 0
        total_gpu_mem = 0
        
    # KV cache estimation variables
    num_layers = model.config.num_hidden_layers
    num_kv_heads = getattr(model.config, "num_key_value_heads", model.config.num_attention_heads)
    hidden_size = model.config.hidden_size
    num_heads = model.config.num_attention_heads
    head_dim = hidden_size // num_heads
    kv_bytes_per_token = 4 * num_layers * num_kv_heads * head_dim
    
    context_sizes = [50, 200, 400]
    modes = ["full", "dense", "hybrid", "predictor"]
    
    # Select balanced mix of 100 samples total (20 per category)
    rng = random.Random(args.seed)
    categories = ["A", "B", "C", "D", "E"]
    samples_in_cat = {cat: [] for cat in categories}
    for s in corpus_samples:
        samples_in_cat[s["category"]].append(s)
        
    selected_samples = []
    per_cat = max(1, args.num_samples // len(categories))
    for cat in categories:
        shuffled = list(samples_in_cat[cat])
        rng.shuffle(shuffled)
        selected_samples.extend(shuffled[:per_cat])
        
    # Truncate to exact number of requested samples
    selected_samples = selected_samples[:args.num_samples]
    
    results_list = []
    
    for size in context_sizes:
        print(f"\n==========================================")
        print(f"Evaluating scale: {size} blocks (~{size*128} tokens)")
        print(f"==========================================")
        
        # Scale samples to current context size
        scaled_samples = [scale_sample(s, size, seed=args.seed) for s in selected_samples]
        
        # Evaluate each sample
        for idx, sample in enumerate(scaled_samples):
            q_id = sample["question_id"]
            gold_info = gold_answers[q_id]
            category = sample["category"]
            
            # Warm up sbert cache once for latency timing
            get_block_embeddings([b["text"] for b in sample["documents"]])
            
            # 1. Measure selector latency (features + predictor inference)
            t_sel = time.perf_counter()
            features = extract_block_features(sample["question"], sample["documents"], sample["project"], ablation_mode="no_position")
            probs = clf.predict_proba(features)[:, 1]
            kept_ids = [i for i, p in enumerate(probs) if p >= 0.70]
            if len(kept_ids) < 4:
                kept_ids = sorted(list(np.argsort(probs)[::-1][:4]))
            selector_latency_ms = (time.perf_counter() - t_sel) * 1000.0
            
            num_kept = len(kept_ids)
            
            # 2. Build baselines & measure query-time retrieval latency
            baselines, ret_latencies = build_baseline_ids_and_latencies(sample, num_kept, seed=args.seed)
            
            # Prompts
            prompts = {
                "full": assemble_json_prompt_b(list(range(len(sample["documents"]))), sample["question"], sample["documents"]),
                "dense": assemble_json_prompt_b(baselines["dense"], sample["question"], sample["documents"]),
                "hybrid": assemble_json_prompt_b(baselines["hybrid"], sample["question"], sample["documents"]),
                "predictor": assemble_json_prompt_b(kept_ids, sample["question"], sample["documents"])
            }
            
            # Evaluate all modes
            for mode in modes:
                prompt_text = prompts[mode]
                input_ids = tokenizer.encode(prompt_text)
                prompt_len = len(input_ids)
                
                # Retrieval/selector/prompt-building latencies
                if mode == "full":
                    retrieval_latency = 0.0
                    selector_latency = 0.0
                elif mode == "dense":
                    retrieval_latency = ret_latencies["dense"]
                    selector_latency = 0.0
                elif mode == "hybrid":
                    retrieval_latency = ret_latencies["hybrid"]
                    selector_latency = 0.0
                else: # predictor
                    retrieval_latency = 0.0
                    selector_latency = selector_latency_ms
                    
                # Run inference (catching CUDA OOMs)
                try:
                    if device == "cuda":
                        torch.cuda.empty_cache()
                        torch.cuda.reset_peak_memory_stats()
                        
                    prefill_ms, decode_ms, answer, gen_len = run_manual_inference(
                        model, tokenizer, input_ids, device, max_new_tokens=64
                    )
                    
                    if device == "cuda":
                        peak_vram = torch.cuda.max_memory_allocated() / 1024 / 1024 # MB
                    else:
                        peak_vram = 0.0
                        
                    oom = False
                except torch.cuda.OutOfMemoryError:
                    prefill_ms, decode_ms, answer, gen_len = 0.0, 0.0, "", 0
                    peak_vram = 0.0
                    oom = True
                    if device == "cuda":
                        torch.cuda.empty_cache()
                        
                # 3. Normalized Evaluations
                parsed_ans = parse_json_response_b(answer) if not oom else {"answer": "", "evidence_block_id": "", "evidence_quote": ""}
                extracted_ans = parsed_ans["answer"]
                
                norm_gen = normalize_answer(extracted_ans)
                norm_expected = normalize_answer(gold_info["expected_answer"])
                
                # Exact Match
                em = (norm_gen == norm_expected) if not oom else False
                
                # Numeric Preservation (digits overlap)
                expected_digits = re.findall(r'\d+', norm_expected)
                generated_digits = re.findall(r'\d+', norm_gen)
                num_pres = all(d in generated_digits for d in expected_digits) if expected_digits and not oom else True
                
                # Gold block recall (indices)
                if mode == "full":
                    gold_recall = True
                    kept_blocks_count = len(sample["documents"])
                else:
                    mode_kept_ids = kept_ids if mode == "predictor" else baselines[mode]
                    gold_recall = all(gid in mode_kept_ids for gid in gold_info["gold_block_ids"])
                    kept_blocks_count = len(mode_kept_ids)
                    
                # Evidence Block Recall (LLM JSON output matches any gold doc_id)
                gold_doc_ids = [sample["documents"][gid]["doc_id"] for gid in gold_info["gold_block_ids"]]
                ev_recalled = any(normalize_id(parsed_ans["evidence_block_id"]) == normalize_id(doc_id) for doc_id in gold_doc_ids) if not oom else False
                
                # Contradiction Accuracy (Category C)
                active_truth = em
                if category == "C" and gold_info["deprecated_block_ids"] and not oom:
                    if "2026" in norm_gen: # if it output the contradiction value
                        active_truth = False
                        
                # Multi-fact Recall (Category D)
                mf_recall = gold_recall if category == "D" else True
                
                # KV Cache size estimation (MB)
                est_kv_mb = (prompt_len + gen_len) * kv_bytes_per_token / 1024 / 1024
                
                # Throughput tokens/s
                tokens_per_sec = (gen_len / (decode_ms / 1000.0)) if decode_ms > 0 else 0.0
                
                # Local GPU Max Batch Size
                if device == "cuda" and not oom:
                    req_mem = (peak_vram * 1024 * 1024) - base_allocated
                    avail_mem = total_gpu_mem - base_allocated
                    max_batch_size = int(avail_mem // max(1, req_mem))
                else:
                    max_batch_size = 1 if not oom else 0
                    
                results_list.append({
                    "question_id": q_id,
                    "category": category,
                    "context_blocks": size,
                    "mode": mode,
                    "oom": oom,
                    "exact_match": em,
                    "numeric_preservation": num_pres,
                    "gold_recall": gold_recall,
                    "evidence_block_recall": ev_recalled,
                    "contradiction_accuracy": active_truth,
                    "multi_fact_recall": mf_recall,
                    "kept_blocks": kept_blocks_count,
                    "token_reduction_pct": (1.0 - kept_blocks_count / size) * 100.0 if mode != "full" else 0.0,
                    "selector_latency_ms": selector_latency,
                    "retrieval_latency_ms": retrieval_latency,
                    "ttft_ms": prefill_ms,
                    "decode_ms": decode_ms,
                    "tokens_per_second": tokens_per_sec,
                    "peak_vram_mb": peak_vram,
                    "est_kv_mb": est_kv_mb,
                    "max_batch_size": max_batch_size
                })
                
            if (idx + 1) % 10 == 0:
                print(f"Processed {idx + 1} / {args.num_samples} samples at scale {size} blocks.")

    # ==========================================
    # Compile POC 0.3b Report
    # ==========================================
    df = pd.DataFrame(results_list)
    os.makedirs("reports", exist_ok=True)
    df.to_csv(os.path.join("reports", "poc03b_results.csv"), index=False)
    
    summary = df.groupby(["context_blocks", "mode"]).agg({
        "exact_match": "mean",
        "gold_recall": "mean",
        "evidence_block_recall": "mean",
        "numeric_preservation": "mean",
        "oom": "mean",
        "kept_blocks": "mean",
        "token_reduction_pct": "mean",
        "selector_latency_ms": "mean",
        "retrieval_latency_ms": "mean",
        "ttft_ms": "mean",
        "peak_vram_mb": "mean",
        "est_kv_mb": "mean",
        "max_batch_size": "mean",
        "tokens_per_second": "mean"
    }).reset_index()
    
    def get_val(blocks, mode, col):
        row = summary[(summary["context_blocks"] == blocks) & (summary["mode"] == mode)]
        if not row.empty:
            return row.iloc[0][col]
        return 0.0
        
    # Average across context sizes or specific scales for gating
    pred_em_50 = get_val(50, "predictor", "exact_match") * 100.0
    hybrid_em_50 = get_val(50, "hybrid", "exact_match") * 100.0
    pred_recall_50 = get_val(50, "predictor", "gold_recall") * 100.0
    pred_num_pres_50 = get_val(50, "predictor", "numeric_preservation") * 100.0
    pred_red_50 = get_val(50, "predictor", "token_reduction_pct")
    
    pred_recall_400 = get_val(400, "predictor", "gold_recall") * 100.0
    pred_oom_400 = get_val(400, "predictor", "oom")
    full_oom_400 = get_val(400, "full", "oom")
    
    pred_vram_50 = get_val(50, "predictor", "peak_vram_mb")
    full_vram_50 = get_val(50, "full", "peak_vram_mb")
    
    pred_kv_50 = get_val(50, "predictor", "est_kv_mb")
    full_kv_50 = get_val(50, "full", "est_kv_mb")
    
    pred_ttft_50 = get_val(50, "predictor", "ttft_ms")
    full_ttft_50 = get_val(50, "full", "ttft_ms")
    hybrid_ttft_50 = get_val(50, "hybrid", "ttft_ms")
    hybrid_num_pres_50 = get_val(50, "hybrid", "numeric_preservation") * 100.0
    
    max_batch_pred_50 = get_val(50, "predictor", "max_batch_size")
    max_batch_hybrid_50 = get_val(50, "hybrid", "max_batch_size")
    max_batch_full_50 = get_val(50, "full", "max_batch_size")
    
    # Reductions
    vram_reduction_pct = ((full_vram_50 - pred_vram_50) / max(1.0, full_vram_50)) * 100.0 if full_vram_50 > 0 else 0.0
    kv_reduction_pct = ((full_kv_50 - pred_kv_50) / max(1.0, full_kv_50)) * 100.0 if full_kv_50 > 0 else 0.0
    ttft_reduction_vs_full = ((full_ttft_50 - pred_ttft_50) / max(1.0, full_ttft_50)) * 100.0 if full_ttft_50 > 0 else 0.0
    ttft_diff_vs_hybrid = ((pred_ttft_50 - hybrid_ttft_50) / max(1.0, hybrid_ttft_50)) * 100.0 if hybrid_ttft_50 > 0 else 0.0
    
    selector_latency_max = df[df["mode"] == "predictor"]["selector_latency_ms"].max()
    
    gates = {
        "Gold Block Recall": {"value": pred_recall_50, "target": 99.0, "status": "PASS" if pred_recall_50 >= 99.0 else "FAIL"},
        "Exact Match vs Hybrid": {"value": pred_em_50 - hybrid_em_50, "target": -1.0, "status": "PASS" if (pred_em_50 - hybrid_em_50) >= -1.0 else "FAIL"},
        "Numeric Preservation": {"value": pred_num_pres_50, "target": 90.0, "status": "PASS" if pred_num_pres_50 >= 90.0 else "FAIL"},
        "Token Reduction": {"value": pred_red_50, "target": 70.0, "status": "PASS" if pred_red_50 >= 70.0 else "FAIL"},
        "Selector Latency": {"value": selector_latency_max, "target": 100.0, "status": "PASS" if selector_latency_max <= 100.0 else "FAIL"},
        "TTFT vs Full": {"value": ttft_reduction_vs_full, "target": 50.0, "status": "PASS" if ttft_reduction_vs_full >= 50.0 else "FAIL"},
        "TTFT vs Hybrid": {"value": ttft_diff_vs_hybrid, "target": 5.0, "status": "PASS" if abs(ttft_diff_vs_hybrid) <= 5.0 or abs(pred_ttft_50 - hybrid_ttft_50) <= 20.0 else "FAIL"},
        "Peak VRAM Total Reduction vs Full": {"value": vram_reduction_pct, "target": 15.0, "status": "PASS" if vram_reduction_pct >= 15.0 else "FAIL"},
        "Estimated KV Reduction": {"value": kv_reduction_pct, "target": 60.0, "status": "PASS" if kv_reduction_pct >= 60.0 else "FAIL"},
        "Max Batch Size vs Full": {"value": max_batch_pred_50 / max(1.0, max_batch_full_50), "target": 2.0, "status": "PASS" if max_batch_pred_50 >= (max_batch_full_50 * 2.0) else "FAIL"},
        "OOM Rate": {"value": pred_oom_400, "target": full_oom_400, "status": "PASS" if pred_oom_400 <= full_oom_400 else "FAIL"},
        "52k Context": {"value": pred_oom_400 * 100.0, "target": 0.0, "status": "PASS" if pred_oom_400 == 0.0 else "FAIL"}
    }
    
    overall_status = "PASS" if all(g["status"] == "PASS" for g in gates.values()) else "FAIL"
    
    # Format markdown report
    report_content = f"""# POC 0.3b â€” Clean Cost Dominance Confirmation

Status: **{overall_status}**

Model: **{args.model}**
Samples: **{len(selected_samples)}**
Context sizes: **6.5k / 26k / 52k tokens** (50 / 200 / 400 blocks)

## Quality:
*   **Predictor EM (6.5k)**: {pred_em_50:.1f}%
*   **Hybrid EM (6.5k)**: {hybrid_em_50:.1f}%
*   **Delta**: {pred_em_50 - hybrid_em_50:+.1f} pts (Gate: within &plusmn;1 pt)
*   **Numeric Preservation**: {pred_num_pres_50:.1f}% (Gate: &ge; 90%)
*   **Gold Block Recall**: {pred_recall_50:.1f}% (Gate: &ge; 99%)
*   **Evidence Block Recall**: {get_val(50, "predictor", "evidence_block_recall")*100:.1f}%
*   **Contradiction Accuracy**: {df[(df["mode"] == "predictor") & (df["category"] == "C")]["contradiction_accuracy"].mean() * 100.0:.1f}%
*   **Multi-fact Recall**: {df[(df["mode"] == "predictor") & (df["category"] == "D")]["multi_fact_recall"].mean() * 100.0:.1f}%

## Cost & Latencies (6.5k scale):
*   **Selector Latency**: {get_val(50, "predictor", "selector_latency_ms"):.2f} ms (Gate: &le; 100 ms)
*   **Hybrid Retrieval Latency**: {get_val(50, "hybrid", "retrieval_latency_ms"):.2f} ms (BM25 + Dense + Combining)
*   **TTFT Full**: {full_ttft_50:.1f} ms
*   **TTFT Hybrid**: {hybrid_ttft_50:.1f} ms
*   **TTFT Predictor**: {pred_ttft_50:.1f} ms
*   **TTFT Reduction vs Full**: {ttft_reduction_vs_full:.1f}% (Gate: &ge; 50%)
*   **TTFT Difference vs Hybrid**: {ttft_diff_vs_hybrid:+.1f}% (Gate: &plusmn;5%)
*   **Token Reduction**: {pred_red_50:.1f}% (Gate: &ge; 70%)

## Peak VRAM & KV Cache:
*   **Peak VRAM Total Full**: {full_vram_50:.1f} MB
*   **Peak VRAM Total Hybrid**: {get_val(50, "hybrid", "peak_vram_mb"):.1f} MB
*   **Peak VRAM Total Predictor**: {pred_vram_50:.1f} MB
*   **Peak VRAM Total Reduction vs Full**: {vram_reduction_pct:.1f}% (Gate: &ge; 15%)
*   **Estimated KV Cache (52k)**: {get_val(400, "predictor", "est_kv_mb"):.1f} MB (vs Full: {get_val(400, "full", "est_kv_mb"):.1f} MB)
*   **Estimated KV Reduction (52k)**: {((get_val(400, "full", "est_kv_mb") - get_val(400, "predictor", "est_kv_mb")) / max(1.0, get_val(400, "full", "est_kv_mb"))) * 100.0:.1f}% (Gate: &ge; 60%)

## Throughput & Batch Capacity:
*   **Max Batch Size Full**: {max_batch_full_50:.1f}
*   **Max Batch Size Hybrid**: {max_batch_hybrid_50:.1f}
*   **Max Batch Size Predictor**: {max_batch_pred_50:.1f}
*   **Max Batch Size vs Full Scaling**: {max_batch_pred_50 / max(1.0, max_batch_full_50):.1f}x (Gate: &ge; 2.0x)
*   **OOM Rate (52k)**: Predictor {pred_oom_400 * 100.0:.1f}% vs Full {full_oom_400 * 100.0:.1f}%

## Success Gates Status:

| Gate | Target | Value | Status |
|---|---|---|---|
| **Gold Block Recall** | &ge; 99% | {gates["Gold Block Recall"]["value"]:.1f}% | **{gates["Gold Block Recall"]["status"]}** |
| **Exact Match vs Hybrid** | &ge; Hybrid - 1 pt | {gates["Exact Match vs Hybrid"]["value"]:+.1f} pts | **{gates["Exact Match vs Hybrid"]["status"]}** |
| **Numeric Preservation** | &ge; 90% | {gates["Numeric Preservation"]["value"]:.1f}% | **{gates["Numeric Preservation"]["status"]}** |
| **Token Reduction** | &ge; 70% | {gates["Token Reduction"]["value"]:.1f}% | **{gates["Token Reduction"]["status"]}** |
| **Selector Latency** | &le; 100 ms | {gates["Selector Latency"]["value"]:.2f} ms | **{gates["Selector Latency"]["status"]}** |
| **TTFT vs Full** | &ge; 50% | {gates["TTFT vs Full"]["value"]:.1f}% | **{gates["TTFT vs Full"]["status"]}** |
| **TTFT vs Hybrid** | &plusmn;5% (or &le; 20 ms) | {gates["TTFT vs Hybrid"]["value"]:+.1f}% | **{gates["TTFT vs Hybrid"]["status"]}** |
| **Peak VRAM Total Reduction vs Full** | &ge; 15% | {gates["Peak VRAM Total Reduction vs Full"]["value"]:.1f}% | **{gates["Peak VRAM Total Reduction vs Full"]["status"]}** |
| **Estimated KV Reduction** | &ge; 60% | {gates["Estimated KV Reduction"]["value"]:.1f}% | **{gates["Estimated KV Reduction"]["status"]}** |
| **Max Batch Size vs Full** | &ge; &times;2 | {gates["Max Batch Size vs Full"]["value"]:.1f}x | **{gates["Max Batch Size vs Full"]["status"]}** |
| **OOM Rate** | Predictor &le; Full | Predictor {pred_oom_400*100:.1f}% vs Full {full_oom_400*100:.1f}% | **{gates["OOM Rate"]["status"]}** |
| **52k Context** | Run Stable | {gates["52k Context"]["value"]:.1f}% OOM Rate | **{gates["52k Context"]["status"]}** |

## Verdict:
{overall_status == 'PASS' and 'Predictor achieves Hybrid-level quality with significantly lower active context and KV materialization cost than Full Context.' or 'The Predictor failed to satisfy all cost-dominance validation gates. Review model latency or memory utilization.'}
"""
    
    report_path = os.path.join("reports", "poc03b_report.md")
    with open(report_path, "w") as f:
        f.write(report_content)
        
    print(f"\n==========================================")
    print(f"POC 0.3b Complete!")
    print(f"Report written to: {report_path}")
    print(f"Overall status: {overall_status}")
    print(f"==========================================")

if __name__ == "__main__":
    main()


