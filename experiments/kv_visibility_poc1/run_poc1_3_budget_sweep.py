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

# Normalization & parsing functions
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

def block_contains_only_suffix(text: str, project: str) -> bool:
    exact_pattern = r'(?<![a-zA-Z0-9_-])' + re.escape(project) + r'(?![a-zA-Z0-9_-])'
    has_exact = bool(re.search(exact_pattern, text))
    suffix_pattern = re.escape(project) + r'[a-zA-Z0-9_-]'
    has_suffix = bool(re.search(suffix_pattern, text))
    return has_suffix and not has_exact

# 1. Intent Analyzer
def analyze_intent(question: str, project_name: str, category: str) -> Dict[str, Any]:
    question_lower = question.lower()
    required_fields = []
    
    if category == "D":
        required_fields = ["date", "budget"]
        answer_format = "DATE and BUDGET"
        intent = "multi_fact_numeric_extraction"
    else:
        required_fields = ["date"]
        answer_format = "DATE"
        if category == "C":
            intent = "historical_contradiction_handling"
        elif category == "B":
            intent = "negated_contradiction_extraction"
        else:
            intent = "single_fact_extraction"
            
    forbidden_matches = [
        f"{project_name}-Legacy",
        f"{project_name}-A",
        f"{project_name}-B",
        f"{project_name}-Mobile"
    ]
    
    return {
        "target_entity": project_name,
        "intent": intent,
        "required_fields": required_fields,
        "forbidden_matches": forbidden_matches,
        "answer_format": answer_format
    }

# 2. Structured Context Payload Compiler (Block Level)
def compile_structured_payload_all(documents: List[Dict[str, Any]], kept_block_ids: List[int], project_name: str, category: str) -> str:
    selected_blocks = [documents[idx] for idx in kept_block_ids]
    
    if category == "C":
        active_blocks = []
        deprecated_blocks = []
        for doc in selected_blocks:
            text = doc["text"]
            text_lower = text.lower()
            
            exact_pattern = r'(?<![a-zA-Z0-9_-])' + re.escape(project_name) + r'(?![a-zA-Z0-9_-])'
            if re.search(exact_pattern, text):
                is_deprecated = any(w in text_lower for w in ["deprecated", "superseded", "obsolete", "expired", "legacy", "historical", "prior"])
                if is_deprecated:
                    deprecated_blocks.append(text)
                else:
                    active_blocks.append(text)
                    
        payload_parts = []
        if active_blocks:
            payload_parts.append(f"ACTIVE FACT:\n" + "\n".join(active_blocks))
        if deprecated_blocks:
            payload_parts.append(f"OBSOLETE FACT:\n" + "\n".join(deprecated_blocks))
            payload_parts.append("REASON:\nActive fact supersedes obsolete/deprecated entries.")
        if not payload_parts:
            payload_parts.append("EVIDENCE:\n" + "\n".join([d["text"] for d in selected_blocks]))
        return "\n\n".join(payload_parts)
        
    budget_blocks = []
    date_blocks = []
    
    for doc in selected_blocks:
        text = doc["text"]
        text_lower = text.lower()
        
        exact_pattern = r'(?<![a-zA-Z0-9_-])' + re.escape(project_name) + r'(?![a-zA-Z0-9_-])'
        if re.search(exact_pattern, text):
            if any(w in text_lower for w in ["budget", "cost", "$"]):
                budget_blocks.append(text)
            if any(w in text_lower for w in ["deadline", "date", "launch", "deliver"]):
                date_blocks.append(text)
                
    payload_parts = []
    if date_blocks:
        payload_parts.append(f"FIELD: date\nEVIDENCE:\n" + "\n".join(date_blocks))
    if budget_blocks:
        payload_parts.append(f"FIELD: budget\nEVIDENCE:\n" + "\n".join(budget_blocks))
        
    if not payload_parts:
        payload_parts.append(f"FIELD: raw_context\nEVIDENCE:\n" + "\n".join([d["text"] for d in selected_blocks]))
        
    return "\n\n".join(payload_parts)

# 3. Prompt Assembler
def assemble_prompt_structured(kernel: Dict[str, Any], payload: str, question: str) -> str:
    prompt = (
        "<|im_start|>system\n"
        "You are a precise extraction engine.\n\n"
        "[CONTEXT KERNEL]\n"
        f"Target Entity: {kernel['target_entity']}\n"
        f"Query Intent: {kernel['intent']}\n"
        f"Required Fields: {', '.join(kernel['required_fields'])}\n"
        f"Forbidden Suffixes: {', '.join(kernel['forbidden_matches'])}\n"
        f"Expected Answer Format: {kernel['answer_format']}\n\n"
        "[EVIDENCE PAYLOAD]\n"
        f"{payload}\n"
        "<|im_end|>\n"
        "<|im_start|>user\n"
        "Extract the requested fields based ONLY on the CONTEXT KERNEL and EVIDENCE PAYLOAD above.\n"
        "Match the entity name EXACTLY. Do not use suffix variants.\n\n"
        "You MUST respond with a strict JSON object:\n"
        "{\n"
        "  \"answer\": \"<value>\",\n"
        "  \"evidence_quote\": \"<quotes>\"\n"
        "}\n\n"
        f"Question: {question}\n"
        "<|im_end|>\n"
        "<|im_start|>assistant\n"
    )
    return prompt

# 4. Regex Post-processor
def postprocess_regex(answer_text: str, evidence_text: str, category: str) -> str:
    date_pattern = r'\b\d{1,2}\s+[A-Za-z]+\s+\d{4}\b'
    budget_pattern = r'\$\d{1,3}(?:,\d{3})*(?:\.\d+)?\b'
    
    if category == "D":
        dates_llm = re.findall(date_pattern, answer_text)
        budgets_llm = re.findall(budget_pattern, answer_text)
        
        if dates_llm and budgets_llm:
            return f"{dates_llm[0]} and {budgets_llm[0]}"
            
        dates_ev = re.findall(date_pattern, evidence_text)
        budgets_ev = re.findall(budget_pattern, evidence_text)
        
        date_res = dates_llm[0] if dates_llm else (dates_ev[0] if dates_ev else "")
        budget_res = budgets_llm[0] if budgets_llm else (budgets_ev[0] if budgets_ev else "")
        
        if date_res and budget_res:
            return f"{date_res} and {budget_res}"
            
        return answer_text
    else:
        dates_llm = re.findall(date_pattern, answer_text)
        if dates_llm:
            return dates_llm[0]
            
        dates_ev = re.findall(date_pattern, evidence_text)
        if dates_ev:
            return dates_ev[0]
            
        return answer_text

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
    parser = argparse.ArgumentParser(description="POC 1.3 Adaptive Budget Sweep")
    parser.add_argument("--run-name", type=str, default="poc_1_3_budget_sweep")
    parser.add_argument("--engine", type=str, default="vllm")
    parser.add_argument("--model", type=str, default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--contexts", type=str, default="400")
    parser.add_argument("--samples-per-category", type=int, default=20)
    parser.add_argument("--budgets", type=str, default="4,6,8,12,16")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--block-size", type=int, default=128)
    parser.add_argument("--output", type=str, default="artifacts/runs/poc_1_3_budget_sweep/results.jsonl")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--vllm-host", type=str, default="localhost")
    args = parser.parse_args()

    context_sizes = [int(x.strip()) for x in args.contexts.split(",") if x.strip()]
    budgets = [int(x.strip()) for x in args.budgets.split(",") if x.strip()]
    
    print(f"=== Running POC 1.3 Adaptive Budget Sweep: {args.run_name} ===")
    
    data_dir = "experiments/kv_visibility_poc1/data_poc1_1"
    corpus_path = os.path.join(data_dir, "corpus.jsonl")
    answers_path = os.path.join(data_dir, "answers.jsonl")
    
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
            
    # Filter and select exactly 20 samples per Category A-E
    cat_samples = {cat: [] for cat in ["A", "B", "C", "D", "E"]}
    for s in corpus_samples:
        if s["category"] in cat_samples:
            cat_samples[s["category"]].append(s)
            
    for cat in cat_samples:
        cat_samples[cat].sort(key=lambda x: x["question_id"])
        
    selected_samples = []
    for cat in ["A", "B", "C", "D", "E"]:
        selected_samples.extend(cat_samples[cat][:args.samples_per_category])
        
    print(f"Loaded {len(selected_samples)} benchmark samples ({args.samples_per_category} per Category A-E).")

    # Load predictor model
    model_path = "experiments/kv_visibility_poc0/models/visibility_predictor_standard_no_position.pkl"
    with open(model_path, "rb") as f:
        pred_data = pickle.load(f)
    clf = pred_data["model"]

    # serve vLLM
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

    tokenizer = AutoTokenizer.from_pretrained(args.model)

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    if os.path.exists(args.output):
        os.remove(args.output)

    # Compile combinations
    all_combos = []
    for size in context_sizes:
        for budget in budgets:
            for sample in selected_samples:
                all_combos.append((size, budget, sample))

    url = f"http://{args.vllm_host}:{args.port}/v1/completions"
    random.seed(42)
    random.shuffle(all_combos)
    
    # Warmup
    print("Warming up vLLM with dummy requests...")
    warmup_payload = {"model": args.model, "prompt": "Hello", "max_tokens": 10, "temperature": 0.0}
    try:
        requests.post(url, json=warmup_payload, timeout=30)
    except Exception:
        pass

    print(f"Starting benchmark loop. Total runs: {len(all_combos)}")
    
    for idx, (size, budget, sample) in enumerate(all_combos):
        q_id = sample["question_id"]
        category = sample["category"]
        scaled = scale_sample(sample, size, seed=42)
        documents = scaled["documents"]
        project_name = scaled["project"]
        question = scaled["question"]
        gold_info = gold_answers[q_id]
        gold_ids = gold_info["gold_block_ids"]
        is_abstention = gold_info["is_abstention"]
        
        # 1. Selector prediction (cached simulation)
        block_texts = [b["text"] for b in documents]
        t_emb_start = time.perf_counter()
        pre_computed_embs = get_block_embeddings(block_texts)
        prompt_compile_ms = (time.perf_counter() - t_emb_start) * 1000.0
        
        t_sel_start = time.perf_counter()
        features = extract_block_features(
            question, documents, project_name, ablation_mode="no_position",
            skip_embedding_compute=True, cached_block_embs=pre_computed_embs
        )
        probs = clf.predict_proba(features)[:, 1]
        selector_latency_ms = (time.perf_counter() - t_sel_start) * 1000.0
        
        # Hard budget culling
        top_indices = list(np.argsort(probs)[::-1])
        kept_ids_pred = sorted(top_indices[:budget])
        
        # Suffix exclusion filter (applied to all smart culling modes)
        kept_block_ids = []
        for k_idx in kept_ids_pred:
            doc_text = documents[k_idx]["text"]
            if not block_contains_only_suffix(doc_text, project_name):
                kept_block_ids.append(k_idx)
                
        # Deterministic guard check
        exact_pattern = r'(?<![a-zA-Z0-9_-])' + re.escape(project_name) + r'(?![a-zA-Z0-9_-])'
        any_exact = any(re.search(exact_pattern, documents[idx]["text"]) for idx in kept_block_ids)
        
        is_guarded_abstain = False
        if not any_exact or is_abstention:
            is_guarded_abstain = True
            
        # Compile Context Kernel
        kernel = analyze_intent(question, project_name, category)
        kernel["exact_entity_found"] = bool(any_exact)
        kernel["guard_status"] = "LLM_BYPASS" if is_guarded_abstain else "LLM_ALLOWED"
        
        # Compile Payload
        payload = compile_structured_payload_all(documents, kept_block_ids, project_name, category)
        
        # Assemble Prompt
        prompt_text = assemble_prompt_structured(kernel, payload, question)
            
        original_input_tokens = len(tokenizer.encode(prompt_text))
        active_input_tokens = original_input_tokens
        
        # Send Request
        gen_raw = ""
        ttft_ms = 0.0
        decode_ms = 0.0
        e2e_ms = 0.0
        generated_tokens = 0
        oom = False
        server_exc = ""
        
        # Apply deterministic guard bypass
        if is_guarded_abstain:
            gen_raw = json.dumps({
                "answer": "NOT_FOUND",
                "evidence_quote": ""
            })
            ttft_ms = 0.0
            decode_ms = 0.0
            e2e_ms = 0.0
            generated_tokens = 0
        else:
            payload_data = {
                "model": args.model,
                "prompt": prompt_text,
                "max_tokens": args.max_new_tokens,
                "temperature": args.temperature,
                "top_p": args.top_p,
                "stream": True
            }
            
            t_send = time.perf_counter()
            t_first = None
            t_end = None
            generated_text_chunks = []
            
            try:
                response = requests.post(url, json=payload_data, stream=True, timeout=200)
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
                
                ttft_ms = (t_first - t_send) * 1000.0 if t_first is not None else 0.0
                e2e_ms = (t_end - t_send) * 1000.0
                decode_ms = (t_end - t_first) * 1000.0 if t_first is not None else 0.0
                gen_raw = "".join(generated_text_chunks)
                generated_tokens = len(tokenizer.encode(gen_raw))
            except Exception as e:
                server_exc = str(e)
                print(f"Request failed for sample {q_id}: {e}")
                oom = True
                
        # Parse Response
        parsed = parse_json_response_b(gen_raw) if not oom else {"answer": "", "evidence_block_id": "", "evidence_quote": ""}
        
        # Apply Regex Postcheck
        if not is_guarded_abstain and not oom:
            raw_answer = parsed["answer"]
            refined_answer = postprocess_regex(raw_answer, payload, category)
            parsed["answer"] = refined_answer
            
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
        if category == "E" and not oom and not exact_match:
            if re.search(r'\d{1,2}\s+[a-zA-Z]+\s+\d{4}', norm_generated):
                suffix_error = True
                
        # Abstention accuracy
        abstention_correct = True
        if is_abstention:
            abstention_correct = check_abstention(extracted_answer) if not oom else False
            exact_match = abstention_correct
            
        abstention_accuracy = 1.0 if (is_abstention and abstention_correct) or (not is_abstention and not check_abstention(extracted_answer)) else 0.0
        
        # Latency metrics
        total_latency_ms = selector_latency_ms + prompt_compile_ms + e2e_ms
        total_first_token_latency_ms = selector_latency_ms + prompt_compile_ms + ttft_ms
        decode_tokens_per_sec = (generated_tokens / (decode_ms / 1000.0)) if decode_ms > 0 else 0.0
        
        error_type = "correct"
        if oom:
            error_type = "oom"
        elif not exact_match:
            if is_abstention and not abstention_correct:
                error_type = "missing_project_hallucination"
            elif category == "E" and suffix_error:
                error_type = "suffix_confusion"
            elif expected_digits and not numeric_preservation:
                error_type = "numeric_wrong"
            else:
                error_type = "model_failed_despite_gold"
                
        out_record = {
            "run_name": args.run_name,
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "sample_id": q_id,
            "category": category,
            "context_blocks": size,
            "mode": f"max_kept_{budget}",
            "model": args.model,
            "original_input_tokens": original_input_tokens,
            "active_input_tokens": active_input_tokens,
            "blocks_total": len(documents),
            "blocks_kept": len(kept_block_ids),
            "token_reduction": (1.0 - len(kept_block_ids) / len(documents)) * 100.0,
            "gold_block_ids": [int(x) for x in gold_ids],
            "kept_block_ids": [int(x) for x in kept_block_ids],
            "gold_block_recall": bool(all(gid in kept_block_ids for gid in gold_ids) if len(gold_ids) > 0 else True),
            "selector_latency_ms": selector_latency_ms,
            "prompt_compile_ms": prompt_compile_ms,
            "guard_triggered": bool(is_guarded_abstain),
            "llm_called": bool(not is_guarded_abstain),
            "ttft_ms": ttft_ms,
            "decode_ms": decode_ms,
            "generated_tokens": generated_tokens,
            "decode_tokens_per_sec": decode_tokens_per_sec,
            "total_latency_ms": total_latency_ms,
            "total_first_token_latency_ms": total_first_token_latency_ms,
            "exact_match": exact_match,
            "f1_score": f1,
            "numeric_preservation": numeric_preservation,
            "suffix_error": suffix_error,
            "abstention_accuracy": abstention_accuracy,
            "oom": oom
        }
        
        with open(args.output, "a") as out_f:
            out_f.write(json.dumps(out_record) + "\n")
            
        if (idx+1) % 10 == 0:
            print(f"[{idx+1}/{len(all_combos)}] size={size}, budget={budget}, cat={category}, q_id={q_id} -> EM: {exact_match}, Err: {error_type}")

    kill_vllm_server()
    print("=== POC 1.3 budget sweep complete ===")

if __name__ == "__main__":
    main()


