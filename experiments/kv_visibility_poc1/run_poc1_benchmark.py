import os
import sys
import json
import time
import re
import argparse
import random
import pickle
import string
import collections
import subprocess
import numpy as np
import pandas as pd
import requests
from typing import Dict, Any, List

# Add path for src import
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from src.extract_features import (
    extract_block_features, get_block_embeddings, clear_embedding_cache,
    get_embedding_model, tokenize_for_bm25
)
from transformers import AutoTokenizer

# Normalization & parsing functions (from collect_metrics.py)
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
    keywords = ["cannot answer", "not mentioned", "do not have", "no information", "not found", "insufficient information", "does not state", "not_found"]
    return any(kw in norm or kw.replace(" ", "") in norm.replace(" ", "") for kw in keywords)

# Strict entity helper
def block_contains_only_suffix(text: str, project: str) -> bool:
    exact_pattern = r'(?<![a-zA-Z0-9_-])' + re.escape(project) + r'(?![a-zA-Z0-9_-])'
    has_exact = bool(re.search(exact_pattern, text))
    
    suffix_pattern = re.escape(project) + r'[a-zA-Z0-9_-]'
    has_suffix = bool(re.search(suffix_pattern, text))
    
    return has_suffix and not has_exact

# Prompt assembly with new strict system instructions
def assemble_prompt(kept_ids: List[int], question: str, documents: List[Dict[str, Any]]) -> str:
    system_text = (
        "<|im_start|>system\n"
        "You are an extraction engine.\n\n"
        "Answer only from the provided context.\n"
        "Do not use outside knowledge.\n\n"
        "If the requested project/entity is not present exactly in the context, answer exactly:\n"
        "NOT_FOUND\n\n"
        "Match entity names exactly.\n"
        "Do not use suffix variants.\n\n"
        "Example:\n"
        "If asked for \"XENON-407\", do not answer using:\n"
        "- XENON-407-Legacy\n"
        "- XENON-407-A\n"
        "- XENON-407-B\n"
        "- XENON-407-Mobile\n\n"
        "Only use the exact requested entity.\n\n"
        "If multiple facts are requested, return all requested facts joined with \" and \".\n"
        "Do not add explanations.\n"
        "Do not guess.\n\n"
        "You MUST respond with a strict JSON object containing the keys 'answer' and 'evidence_quote'.\n"
        "Example format:\n"
        "{\n  \"answer\": \"15 May 2027\",\n  \"evidence_quote\": \"Project: X Active delivery date: 15 May 2027\"\n}\n"
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

def kill_vllm_server():
    print("Terminating vLLM server inside WSL2...")
    subprocess.run(["wsl", "pkill", "-f", "vllm.entrypoints.openai.api_server"])
    time.sleep(2)

def main():
    parser = argparse.ArgumentParser(description="POC 1.1 Overnight Benchmark")
    parser.add_argument("--run-name", type=str, default="poc_1_1_overnight_guarded")
    parser.add_argument("--engine", type=str, default="vllm")
    parser.add_argument("--model", type=str, default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--precision", type=str, default="fp16")
    parser.add_argument("--contexts", type=str, default="200,400")
    parser.add_argument("--samples-per-combination", type=int, default=300)
    parser.add_argument("--modes", type=str, default="hybrid,predictor_cached,predictor_cached_strict_entity,predictor_cached_guarded,oracle")
    parser.add_argument("--sanity-modes", type=str, default="full_context,random")
    parser.add_argument("--sanity-samples", type=int, default=50)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--block-size", type=int, default=128)
    parser.add_argument("--cached-embeddings", type=str, default="true")
    parser.add_argument("--save-every", type=int, default=10)
    parser.add_argument("--resume", type=str, default="true")
    parser.add_argument("--output", type=str, default="artifacts/runs/poc_1_1_overnight_guarded/results.jsonl")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--vllm-host", type=str, default="127.0.0.1", help="vLLM server host (use 127.0.0.1 to avoid DNS localhost bug)")
    args = parser.parse_args()

    # Convert args to appropriate types
    context_sizes = [int(x.strip()) for x in args.contexts.split(",") if x.strip()]
    modes = [x.strip() for x in args.modes.split(",") if x.strip()]
    sanity_modes = [x.strip() for x in args.sanity_modes.split(",") if x.strip()]
    resume_flag = args.resume.lower() == "true"
    
    print(f"=== Running POC 1.1 Overnight Benchmark: {args.run_name} ===")
    
    # 1. Dataset Check & Generation
    data_dir = "experiments/kv_visibility_poc1/data_poc1_1"
    os.makedirs(data_dir, exist_ok=True)
    corpus_path = os.path.join(data_dir, "corpus.jsonl")
    answers_path = os.path.join(data_dir, "answers.jsonl")
    
    if not (os.path.exists(corpus_path) and os.path.exists(answers_path)):
        print("Dataset files not found. Generating fresh 1000-sample dataset...")
        cmd_gen = [
            sys.executable, "experiments/kv_visibility_poc1/generate_dataset_poc1.py",
            "--num-samples", "1000",
            "--out-dir", data_dir,
            "--abstention-rate", "0.15"
        ]
        res = subprocess.run(cmd_gen, stdin=subprocess.DEVNULL)
        if res.returncode != 0:
            print("Dataset generation failed. Exiting.")
            sys.exit(1)
            
    # Load dataset
    corpus_samples = []
    with open(corpus_path, "r") as f:
        for line in f:
            corpus_samples.append(json.loads(line))
            
    gold_answers = {}
    with open(answers_path, "r") as f:
        for line in f:
            item = json.loads(line)
            gold_answers[item["question_id"]] = item
            
    # Filter and select exactly 60 samples per Category A-E
    cat_samples = {cat: [] for cat in ["A", "B", "C", "D", "E"]}
    for s in corpus_samples:
        if s["category"] in cat_samples:
            cat_samples[s["category"]].append(s)
            
    # Sort for determinism
    for cat in cat_samples:
        cat_samples[cat].sort(key=lambda x: x["question_id"])
        
    selected_samples = []
    samples_per_cat = max(1, args.samples_per_combination // 5)
    for cat in ["A", "B", "C", "D", "E"]:
        selected_samples.extend(cat_samples[cat][:samples_per_cat])
        
    print(f"Loaded {len(selected_samples)} selected benchmark samples ({samples_per_cat} per Category A-E).")
    
    # Check abstention count
    num_abstentions = sum(1 for s in selected_samples if gold_answers[s["question_id"]]["is_abstention"])
    print(f"Selected samples contain {num_abstentions} missing entity/abstention cases.")

    # Load predictor model
    model_path = "experiments/kv_visibility_poc0/models/visibility_predictor_standard_no_position.pkl"
    with open(model_path, "rb") as f:
        pred_data = pickle.load(f)
    clf = pred_data["model"]

    # 2. Serve vLLM Server
    gpu_util = 0.50
    cmd_serve = [
        sys.executable, "experiments/kv_visibility_poc1/serve_vllm.py",
        "--model", args.model,
        "--port", str(args.port),
        "--gpu-memory-utilization", str(gpu_util),
        "--timeout-seconds", "300"
    ]
    print("Starting vLLM server...")
    srv_result = subprocess.run(cmd_serve, stdin=subprocess.DEVNULL)
    if srv_result.returncode != 0:
        print("Failed to start vLLM server. Exiting.")
        kill_vllm_server()
        sys.exit(1)

    # Load tokenizer for token counting
    tokenizer = AutoTokenizer.from_pretrained(args.model)

    # 3. Setup Resume & Output File
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    completed_keys = set()
    if resume_flag and os.path.exists(args.output):
        print(f"Resuming from existing output file: {args.output}")
        with open(args.output, "r") as f:
            for line in f:
                if line.strip():
                    try:
                        record = json.loads(line)
                        completed_keys.add((record["context_blocks"], record["mode"], record["sample_id"]))
                    except Exception:
                        pass
        print(f"Skipping {len(completed_keys)} already completed runs.")

    # 4. Compile combinations
    all_combos = []
    for size in context_sizes:
        # Main modes
        for mode in modes:
            for sample in selected_samples:
                all_combos.append((size, mode, sample, 300))
        # Sanity modes (only 50 samples)
        for mode in sanity_modes:
            for sample in selected_samples[:args.sanity_samples]:
                all_combos.append((size, mode, sample, args.sanity_samples))

    # Run Loop
    url = f"http://{args.vllm_host}:{args.port}/v1/completions"
    
    # Shuffle for randomized concurrency and better execution distribution
    random.seed(42)
    random.shuffle(all_combos)
    
    # Warmup
    print("Warming up vLLM with dummy requests...")
    warmup_payload = {
        "model": args.model,
        "prompt": "Hello",
        "max_tokens": 10,
        "temperature": 0.0
    }
    try:
        requests.post(url, json=warmup_payload, timeout=30)
    except Exception:
        pass

    print(f"Starting benchmark loop. Total runs: {len(all_combos)}")
    
    for idx, (size, mode, sample, max_samples) in enumerate(all_combos):
        q_id = sample["question_id"]
        key = (size, mode, q_id)
        if key in completed_keys:
            continue
            
        # 1. Scale sample context
        scaled = scale_sample(sample, size, seed=42)
        documents = scaled["documents"]
        project_name = scaled["project"]
        question = scaled["question"]
        gold_info = gold_answers[q_id]
        gold_ids = gold_info["gold_block_ids"]
        is_abstention = gold_info["is_abstention"]
        
        # Approximate input tokens of scaled context
        original_input_tokens = len(tokenizer.encode(assemble_prompt(list(range(len(documents))), question, documents)))
        
        # 2. Extract block embeddings (cached simulation)
        block_texts = [b["text"] for b in documents]
        t_emb_start = time.perf_counter()
        pre_computed_embs = get_block_embeddings(block_texts)
        prompt_compile_ms = (time.perf_counter() - t_emb_start) * 1000.0
        
        # 3. Culling Selector Logic
        t_sel_start = time.perf_counter()
        features = extract_block_features(
            question, documents, project_name, ablation_mode="no_position",
            skip_embedding_compute=True, cached_block_embs=pre_computed_embs
        )
        probs = clf.predict_proba(features)[:, 1]
        
        kept_ids_pred = [idx for idx, p in enumerate(probs) if p >= 0.70]
        if len(kept_ids_pred) < 4:
            kept_ids_pred = sorted(list(np.argsort(probs)[::-1][:4]))
        selector_latency_ms = (time.perf_counter() - t_sel_start) * 1000.0
        
        # Determine Kept Blocks by Mode
        kept_block_ids = []
        is_guarded_abstain = False
        
        if mode == "full_context":
            kept_block_ids = list(range(len(documents)))
        elif mode == "oracle":
            kept_block_ids = gold_ids if len(gold_ids) > 0 else [0]
        elif mode == "random":
            # Match the budget of predictor_cached
            num_budget = len(kept_ids_pred)
            rng_rand = random.Random(42 + hash(q_id))
            kept_block_ids = sorted(rng_rand.sample(range(len(documents)), num_budget))
        elif mode == "hybrid":
            num_budget = len(kept_ids_pred)
            from rank_bm25 import BM25Okapi
            corpus_tokens = [tokenize_for_bm25(b["text"]) for b in documents]
            bm25 = BM25Okapi(corpus_tokens)
            q_tokens = tokenize_for_bm25(question)
            bm25_scores = bm25.get_scores(q_tokens)
            max_bm25 = max(bm25_scores) if len(bm25_scores) > 0 and max(bm25_scores) > 0 else 1.0
            norm_bm25 = bm25_scores / max_bm25
            
            q_norm = np.linalg.norm(pre_computed_embs, axis=1) # wait, query embedding norm
            model_emb = get_embedding_model()
            q_emb = model_emb.encode(question, convert_to_tensor=False, show_progress_bar=False)
            qn = np.linalg.norm(q_emb)
            dot_products = np.dot(pre_computed_embs, q_emb)
            cos_sims = dot_products / (qn * q_norm + 1e-8)
            
            hybrid_scores = 0.5 * norm_bm25 + 0.5 * cos_sims
            hybrid_indices = np.argsort(hybrid_scores)[::-1]
            kept_block_ids = sorted(list(hybrid_indices[:num_budget]))
        elif mode == "predictor_cached":
            kept_block_ids = kept_ids_pred
        elif mode == "predictor_cached_strict_entity":
            # Filter out suffix variant blocks
            for k_idx in kept_ids_pred:
                doc_text = documents[k_idx]["text"]
                if block_contains_only_suffix(doc_text, project_name):
                    continue
                kept_block_ids.append(k_idx)
        elif mode == "predictor_cached_guarded":
            # Filter out suffix variants
            for k_idx in kept_ids_pred:
                doc_text = documents[k_idx]["text"]
                if block_contains_only_suffix(doc_text, project_name):
                    continue
                kept_block_ids.append(k_idx)
                
            # Deterministic exact match check
            exact_pattern = r'(?<![a-zA-Z0-9_-])' + re.escape(project_name) + r'(?![a-zA-Z0-9_-])'
            any_exact = any(re.search(exact_pattern, documents[idx]["text"]) for idx in kept_block_ids)
            if not any_exact:
                is_guarded_abstain = True
        elif mode == "oracle_guarded":
            # Oracle blocks (gold) + deterministic guard
            base_ids = gold_ids if len(gold_ids) > 0 else [0]
            exact_pattern = r'(?<![a-zA-Z0-9_-])' + re.escape(project_name) + r'(?![a-zA-Z0-9_-])'
            for k_idx in base_ids:
                doc_text = documents[k_idx]["text"]
                if not block_contains_only_suffix(doc_text, project_name):
                    kept_block_ids.append(k_idx)
            any_exact = any(re.search(exact_pattern, documents[idx]["text"]) for idx in kept_block_ids)
            if not any_exact or is_abstention:
                # For abstention cases, oracle_guarded should also abstain deterministically
                is_guarded_abstain = True
            if not kept_block_ids:
                kept_block_ids = base_ids  # fallback
                
        # Metric: gold recall
        gold_block_recall = all(gid in kept_block_ids for gid in gold_ids) if len(gold_ids) > 0 else True
        
        # Metric: active input tokens (culled context length)
        active_input_tokens = len(tokenizer.encode(assemble_prompt(kept_block_ids, question, documents)))
        
        # Send Request to vLLM Server or Guarded Abstention
        gen_raw = ""
        ttft_ms = 0.0
        decode_ms = 0.0
        e2e_ms = 0.0
        generated_tokens = 0
        oom = False
        server_exc = ""
        
        if is_guarded_abstain:
            gen_raw = json.dumps({
                "answer": "NOT_FOUND",
                "evidence_quote": "",
                "reason": "No exact entity match found in selected context."
            })
            # Latency is zero since we didn't call LLM
            ttft_ms = 0.0
            decode_ms = 0.0
            e2e_ms = 0.0
            generated_tokens = 0
        else:
            prompt_text = assemble_prompt(kept_block_ids, question, documents)
            payload = {
                "model": args.model,
                "prompt": prompt_text,
                "max_tokens": args.max_new_tokens,
                "temperature": args.temperature,
                "top_p": args.top_p,
                "stream": True
            }
            
            # Request wrapper with Server Error Retries
            max_retries = 2
            for retry in range(max_retries):
                t_send = time.perf_counter()
                t_first = None
                t_end = None
                generated_text_chunks = []
                oom = False
                server_exc = ""
                
                try:
                    response = requests.post(url, json=payload, stream=True, timeout=200)
                    if response.status_code != 200:
                        raise RuntimeError(f"Server returned status code {response.status_code}")
                        
                    for chunk in response.iter_lines():
                        if not chunk:
                            continue
                        if t_first is None:
                            t_first = time.perf_counter()
                        line = chunk.decode("utf-8").strip()
                        if line.startswith("data: "):
                            data_str = line[6:].strip()
                            if data_str == "[DONE]":
                                break
                            try:
                                data = json.loads(data_str)
                                text = data["choices"][0]["text"]
                                generated_text_chunks.append(text)
                            except Exception:
                                pass
                    t_end = time.perf_counter()
                    
                    # Calculate timings
                    ttft_ms = (t_first - t_send) * 1000.0 if t_first is not None else 0.0
                    e2e_ms = (t_end - t_send) * 1000.0
                    decode_ms = (t_end - t_first) * 1000.0 if t_first is not None else 0.0
                    gen_raw = "".join(generated_text_chunks)
                    generated_tokens = len(tokenizer.encode(gen_raw))
                    break # Success
                    
                except Exception as e:
                    server_exc = str(e)
                    print(f"Request failed (retry {retry+1}/{max_retries}) for sample {q_id}: {e}")
                    if retry < max_retries - 1:
                        time.sleep(10)
                    else:
                        oom = True
                        
        # 4. Parse & evaluate quality metrics
        parsed = parse_json_response_b(gen_raw) if not oom else {"answer": "", "evidence_block_id": "", "evidence_quote": ""}
        extracted_answer = parsed["answer"]
        
        norm_expected = normalize_answer(gold_info["expected_answer"])
        norm_generated = normalize_answer(extracted_answer)
        
        # Exact match
        exact_match = (norm_generated == norm_expected) if not oom else False
        
        # F1
        f1 = calculate_f1(extracted_answer, gold_info["expected_answer"]) if not oom else 0.0
        
        # Numeric preservation
        expected_digits = re.findall(r'\d+', norm_expected)
        generated_digits = re.findall(r'\d+', norm_generated)
        numeric_preservation = all(d in generated_digits for d in expected_digits) if expected_digits and not oom else True
        
        # Suffix error (Category E)
        suffix_error = False
        if sample["category"] == "E" and not oom and not exact_match:
            if re.search(r'\d{1,2}\s+[a-zA-Z]+\s+\d{4}', norm_generated):
                suffix_error = True
                
        # Abstention accuracy
        abstention_correct = True
        if is_abstention:
            abstention_correct = check_abstention(extracted_answer) if not oom else False
            exact_match = abstention_correct
            
        abstention_accuracy = 1.0 if (is_abstention and abstention_correct) or (not is_abstention and not check_abstention(extracted_answer)) else 0.0
        
        # Entity Accuracy: checked against DOC block ID
        entity_accuracy = (parsed["evidence_block_id"] in [f"DOC_{gid:04d}" for gid in gold_ids]) if gold_ids and not oom else True
        
        # 5. Latency metrics
        total_latency_ms = selector_latency_ms + prompt_compile_ms + e2e_ms
        total_first_token_latency_ms = selector_latency_ms + prompt_compile_ms + ttft_ms
        decode_tokens_per_sec = (generated_tokens / (decode_ms / 1000.0)) if decode_ms > 0 else 0.0
        
        effective_context_tokens_per_sec = original_input_tokens / (total_first_token_latency_ms / 1000.0) if total_first_token_latency_ms > 0 else 0.0
        
        # 6. Error Classification
        error_type = "correct"
        if oom:
            error_type = "oom"
        elif server_exc:
            error_type = "server_error"
        elif not exact_match:
            if is_abstention and not abstention_correct:
                error_type = "missing_project_hallucination"
            elif not gold_block_recall:
                error_type = "gold_missing"
            elif sample["category"] == "E" and suffix_error:
                error_type = "suffix_confusion"
            elif expected_digits and not numeric_preservation:
                error_type = "numeric_wrong"
            elif sample["category"] == "D":
                error_type = "multi_fact_missing"
            elif sample["category"] == "C":
                error_type = "contradiction_wrong"
            else:
                # Try to parse
                try:
                    json.loads(gen_raw)
                    error_type = "model_failed_despite_gold"
                except Exception:
                    error_type = "parse_fail"

        # Convert list elements and types to standard python types to avoid JSON serialization errors
        gold_ids = [int(x) for x in gold_ids]
        kept_block_ids = [int(x) for x in kept_block_ids]
        gold_block_recall = bool(gold_block_recall)
        exact_match = bool(exact_match)
        numeric_preservation = bool(numeric_preservation)
        entity_accuracy = bool(entity_accuracy)
        suffix_error = bool(suffix_error)
        abstention_accuracy = float(abstention_accuracy)

        # Log Sample Output
        out_record = {
            "run_name": args.run_name,
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "sample_id": q_id,
            "category": sample["category"],
            "context_blocks": size,
            "mode": mode,
            "model": args.model,
            "engine": args.engine,
            "original_input_tokens": original_input_tokens,
            "original_tokens": original_input_tokens,
            "active_input_tokens": active_input_tokens,
            "active_tokens": active_input_tokens,
            "blocks_total": len(documents),
            "blocks_kept": len(kept_block_ids),
            "token_reduction": (1.0 - len(kept_block_ids) / len(documents)) * 100.0,
            "gold_block_ids": gold_ids,
            "kept_block_ids": kept_block_ids,
            "gold_block_recall": gold_block_recall,
            "selector_latency_ms": selector_latency_ms,
            "prompt_compile_ms": prompt_compile_ms,
            "guard_triggered": bool(is_guarded_abstain),
            "guard_reason": "No exact entity match found in selected context." if is_guarded_abstain else "",
            "llm_called": bool(not is_guarded_abstain),
            "ttft_ms": ttft_ms,
            "decode_ms": decode_ms,
            "decode_latency_ms": decode_ms,
            "generated_tokens": generated_tokens,
            "decode_tokens_per_sec": decode_tokens_per_sec,
            "total_latency_ms": total_latency_ms,
            "total_first_token_latency_ms": total_first_token_latency_ms,
            "first_token_latency_ms": total_first_token_latency_ms,
            "effective_context_tokens_per_sec": effective_context_tokens_per_sec,
            "exact_match": exact_match,
            "f1_score": f1,
            "numeric_preservation": numeric_preservation,
            "entity_accuracy": entity_accuracy,
            "suffix_error": suffix_error,
            "abstention_accuracy": abstention_accuracy,
            "answer": extracted_answer,
            "expected_answer": gold_info["expected_answer"],
            "evidence_quote": parsed["evidence_quote"],
            "expected_evidence": gold_info.get("evidence_quote", ""),
            "error_type": error_type,
            "oom": oom,
            "exception": server_exc,
            "cost_estimated_usd": float((active_input_tokens * 0.10 + generated_tokens * 0.20) / 1e6)
        }
        
        # Write to JSONL
        with open(args.output, "a") as out_f:
            out_f.write(json.dumps(out_f) if False else json.dumps(out_record) + "\n")
            
        if (idx+1) % 10 == 0:
            print(f"[{idx+1}/{len(all_combos)}] size={size}, mode={mode}, q_id={q_id} -> EM: {exact_match}, Err: {error_type}")

    # Always kill server
    kill_vllm_server()
    print("=== POC 1.1 overnight run complete ===")
    
    # Auto-compile report
    try:
        print("Auto-compiling final report...")
        sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__))))
        from compile_poc1_1_report import compile_report
        report_dir  = os.path.dirname(args.output)
        report_path = os.path.join(report_dir, f"{args.run_name}_report.md")
        compile_report(args.output, report_path)
    except Exception as e:
        print(f"Failed to auto-compile report: {e}")


if __name__ == "__main__":
    main()


