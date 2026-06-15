"""
Mini-run 3 â€” Numeric Bottleneck Probe
CatÃ©gorie D uniquement (multi-fact: date + budget).
Compare 4 formats de sortie LLM pour mesurer l'impact sur Numeric Preservation.

Format A â€” JSON actuel
Format B â€” Texte simple
Format C â€” StructurÃ©
Format D â€” StructurÃ© + regex postcheck
"""
import os, sys, json, time, re, random, pickle, string, collections
import subprocess
import numpy as np
import requests
from typing import Dict, Any, List

# Setup path to import run_poc1_benchmark
script_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(script_dir)
if script_dir not in sys.path:
    sys.path.insert(0, script_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from run_poc1_benchmark import (
    scale_sample, block_contains_only_suffix,
    normalize_answer, calculate_f1, check_abstention
)
from src.extract_features import (
    extract_block_features, get_block_embeddings
)
from transformers import AutoTokenizer


MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
PORT  = 8000
HOST  = "localhost"
URL   = f"http://{HOST}:{PORT}/v1/completions"
N_SAMPLES = 100
CONTEXT_BLOCKS = 200
CATEGORY = "D"

# Token pricing constants (mock for relative comparison)
PRICE_INPUT_PER_M = 0.10
PRICE_OUTPUT_PER_M = 0.20

PROMPTS = {
    "format_json_current": (
        "<|im_start|>system\n"
        "You are an extraction engine.\n"
        "Answer only from the provided context.\n"
        "If multiple facts are requested, return ALL of them joined with ' and '.\n"
        "Do not guess. Do not use outside knowledge.\n"
        "You MUST respond with a strict JSON object:\n"
        "{\"answer\": \"<value1> and <value2>\", \"evidence_quote\": \"...\"}\n"
        "<|im_end|>\n"
        "<|im_start|>user\nContext:\n"
    ),
    "format_answer_only": (
        "<|im_start|>system\n"
        "You are an extraction engine.\n"
        "Answer only from the provided context.\n"
        "If multiple facts are requested, return ALL of them joined with ' and '.\n"
        "Respond with a single line starting with ANSWER: followed by the value(s).\n"
        "Example: ANSWER: 15 May 2027 and $500,000\n"
        "Do not add any other text.\n"
        "<|im_end|>\n"
        "<|im_start|>user\nContext:\n"
    ),
    "format_fields_structured": (
        "<|im_start|>system\n"
        "You are an extraction engine.\n"
        "Answer only from the provided context.\n"
        "Respond using this exact format (only include fields that are asked):\n"
        "DATE: <date if asked>\n"
        "BUDGET: <budget if asked>\n"
        "STATUS: <status if asked>\n"
        "Do not add any other text.\n"
        "<|im_end|>\n"
        "<|im_start|>user\nContext:\n"
    ),
}

def assemble_prompt_format(kept_ids, question, documents, system_prefix):
    context_parts = []
    last_id = -2
    for idx in kept_ids:
        if last_id != -2 and idx != last_id + 1:
            context_parts.append("[...]")
        context_parts.append(documents[idx]["text"])
        last_id = idx
    context_text = "\n\n".join(context_parts)
    return system_prefix + context_text + f"\n\nQuestion: {question}<|im_end|>\n<|im_start|>assistant\n"

def extract_answer(raw: str, fmt: str) -> str:
    raw = raw.strip()
    if check_abstention(raw) or raw.upper() == "NOT_FOUND":
        return "I cannot answer this question based on the provided context."

    if fmt == "format_json_current":
        try:
            start = raw.find("{")
            end   = raw.rfind("}")
            if start != -1 and end != -1:
                data = json.loads(raw[start:end+1])
                return str(data.get("answer", raw))
        except: pass
        m = re.search(r'"answer"\s*:\s*"([^"]+)"', raw)
        return m.group(1) if m else raw
    elif fmt == "format_answer_only":
        m = re.search(r'ANSWER\s*:\s*(.+)', raw, re.IGNORECASE)
        return m.group(1).strip() if m else raw
    elif fmt == "format_fields_structured":
        parts = []
        for field in ["DATE", "BUDGET", "STATUS"]:
            m = re.search(rf'{field}\s*:\s*(.+)', raw, re.IGNORECASE)
            if m:
                val = m.group(1).strip()
                if val and not check_abstention(val) and "asked" not in val.lower() and "<" not in val and ">" not in val:
                    parts.append(val)
        return " and ".join(parts) if parts else raw
    elif fmt == "format_fields_plus_regex_postcheck":
        parts = []
        for field in ["DATE", "BUDGET", "STATUS"]:
            m = re.search(rf'{field}\s*:\s*(.+)', raw, re.IGNORECASE)
            if m:
                val = m.group(1).strip()
                if val and not check_abstention(val) and "asked" not in val.lower() and "<" not in val and ">" not in val:
                    parts.append(val)
        
        # Regex postcheck: search for standard patterns directly
        date_match = re.search(r'\b\d{1,2}\s+[A-Za-z]+\s+\d{4}\b', raw)
        budget_match = re.search(r'\$\d{1,3}(?:,\d{3})*(?:\.\d+)?\b', raw)
        
        if date_match and budget_match:
            return f"{date_match.group(0)} and {budget_match.group(0)}"
            
        return " and ".join(parts) if parts else raw
    return raw

def calculate_multi_fact_recall(generated: str, expected: str) -> float:
    if check_abstention(expected):
        return 1.0 if check_abstention(generated) else 0.0
    if check_abstention(generated):
        return 0.0
        
    norm_gen = normalize_answer(generated)
    expected_parts = [p.strip() for p in expected.lower().split(" and ") if p.strip()]
    if not expected_parts:
        return 1.0
        
    matched = 0
    for part in expected_parts:
        norm_part = normalize_answer(part)
        if norm_part in norm_gen:
            matched += 1
    return matched / len(expected_parts)

def load_data(data_dir=None):
    if data_dir is None:
        data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data_poc1_1")
    corpus, answers = [], {}
    with open(os.path.join(data_dir, "corpus.jsonl")) as f:
        for line in f:
            if line.strip(): corpus.append(json.loads(line))
    with open(os.path.join(data_dir, "answers.jsonl")) as f:
        for line in f:
            if line.strip():
                item = json.loads(line)
                answers[item["question_id"]] = item
    return corpus, answers

def call_llm(prompt_text, max_tokens=96):
    payload = {
        "model": MODEL, "prompt": prompt_text,
        "max_tokens": max_tokens, "temperature": 0, "top_p": 1, "stream": True,
    }
    t_send  = time.perf_counter()
    t_first = None
    chunks  = []
    try:
        resp = requests.post(URL, json=payload, stream=True, timeout=60)
        for chunk in resp.iter_lines():
            if not chunk: continue
            if t_first is None: t_first = time.perf_counter()
            line = chunk.decode("utf-8").strip()
            if line.startswith("data: "):
                ds = line[6:].strip()
                if ds == "[DONE]": break
                try: chunks.append(json.loads(ds)["choices"][0]["text"])
                except: pass
        t_end = time.perf_counter()
    except Exception as e:
        return "", 0.0
    return "".join(chunks), (t_first - t_send) * 1000.0 if t_first else 0.0

def kill_vllm_server():
    print("Terminating vLLM server inside WSL2...")
    subprocess.run(["wsl", "pkill", "-f", "vllm.entrypoints.openai.api_server"])
    time.sleep(2)

def main():
    print("=== MINI-RUN 3 â€” Numeric Bottleneck Format Probe (Category D) ===")
    corpus, gold_answers = load_data()

    # Select Category D only
    cat_d = [s for s in corpus if s["category"] == CATEGORY]
    cat_d.sort(key=lambda x: x["question_id"])
    cat_d = cat_d[:N_SAMPLES]
    print(f"Loaded {len(cat_d)} Category D samples")

    # 1. Terminate any stray server first
    kill_vllm_server()

    # 2. Serve vLLM Server
    gpu_util = 0.50
    serve_script = os.path.join(os.path.dirname(os.path.abspath(__file__)), "serve_vllm.py")
    cmd_serve = [
        sys.executable, serve_script,
        "--model", MODEL,
        "--port", str(PORT),
        "--gpu-memory-utilization", str(gpu_util),
        "--timeout-seconds", "300"
    ]
    print("Starting vLLM server...")
    srv_result = subprocess.run(cmd_serve, stdin=subprocess.DEVNULL)
    if srv_result.returncode != 0:
        print("Failed to start vLLM server. Exiting.")
        kill_vllm_server()
        sys.exit(1)

    model_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "experiments/kv_visibility_poc0", "models", "visibility_predictor_standard_no_position.pkl")
    with open(model_path, "rb") as f:
        clf = pickle.load(f)["model"]

    tokenizer = AutoTokenizer.from_pretrained(MODEL)

    # Warmup
    try:
        requests.post(URL, json={"model": MODEL, "prompt": "Hello", "max_tokens": 5, "temperature": 0}, timeout=15)
        print("Warmup OK")
    except:
        print("Warmup FAILED â€” is vLLM running?")

    formats = ["format_json_current", "format_answer_only", "format_fields_structured", "format_fields_plus_regex_postcheck"]
    format_results = {fmt: {"em": [], "f1": [], "num_pres": [], "parse_fail": [], "ttft": [], "gen_tokens": [], "cost": [], "failed_despite_gold": [], "recall": []}
                      for fmt in formats}

    for i, sample in enumerate(cat_d):
        scaled = scale_sample(sample, CONTEXT_BLOCKS, seed=42)
        documents = scaled["documents"]
        project_name = scaled["project"]
        question = scaled["question"]
        q_id = sample["question_id"]
        gold_info = gold_answers[q_id]
        gold_ids = gold_info["gold_block_ids"]
        expected = gold_info["expected_answer"]
        is_abstention = gold_info["is_abstention"]

        # Selector
        block_texts = [b["text"] for b in documents]
        pre_embs = get_block_embeddings(block_texts)
        features = extract_block_features(
            question, documents, project_name, ablation_mode="no_position",
            skip_embedding_compute=True, cached_block_embs=pre_embs
        )
        probs = clf.predict_proba(features)[:, 1]
        kept_ids = [j for j, p in enumerate(probs) if p >= 0.70]
        if len(kept_ids) < 4:
            kept_ids = sorted(list(np.argsort(probs)[::-1][:4]))

        # Filter suffix
        kept_block_ids = [j for j in kept_ids
                          if not block_contains_only_suffix(documents[j]["text"], project_name)]
        if not kept_block_ids:
            kept_block_ids = kept_ids

        # Guarded check
        exact_pattern = r'(?<![a-zA-Z0-9_-])' + re.escape(project_name) + r'(?![a-zA-Z0-9_-])'
        any_exact = any(re.search(exact_pattern, documents[idx]["text"]) for idx in kept_block_ids)
        is_guarded_abstain = not any_exact

        gold_block_recall = all(gid in kept_block_ids for gid in gold_ids) if len(gold_ids) > 0 else True
        norm_expected = normalize_answer(expected)
        expected_digits = re.findall(r'\d+', norm_expected)

        if is_guarded_abstain:
            # Bypassed
            for fmt in formats:
                raw = "I cannot answer this question based on the provided context."
                answer = "I cannot answer this question based on the provided context."
                ttft = 0.0
                gen_tokens = 0
                
                norm_gen = normalize_answer(answer)
                em = norm_gen == norm_expected
                if is_abstention:
                    em = True
                    
                f1 = calculate_f1(answer, expected)
                num_pres = True
                recall = 1.0 if is_abstention else 0.0
                parse_fail = 0.0
                failed_despite_gold = False
                cost = 0.0
                
                format_results[fmt]["em"].append(float(em))
                format_results[fmt]["f1"].append(f1)
                format_results[fmt]["num_pres"].append(float(num_pres))
                format_results[fmt]["parse_fail"].append(parse_fail)
                format_results[fmt]["ttft"].append(ttft)
                format_results[fmt]["gen_tokens"].append(float(gen_tokens))
                format_results[fmt]["cost"].append(cost)
                format_results[fmt]["failed_despite_gold"].append(float(failed_despite_gold))
                format_results[fmt]["recall"].append(recall)
        else:
            # Query LLM
            # 1. format_json_current
            prompt_json = assemble_prompt_format(kept_block_ids, question, documents, PROMPTS["format_json_current"])
            raw_json, ttft_json = call_llm(prompt_json, max_tokens=96)
            tokens_json = len(tokenizer.encode(raw_json))
            active_tokens_json = len(tokenizer.encode(prompt_json))
            cost_json = (active_tokens_json * PRICE_INPUT_PER_M + tokens_json * PRICE_OUTPUT_PER_M) / 1e6
            
            # 2. format_answer_only
            prompt_ans = assemble_prompt_format(kept_block_ids, question, documents, PROMPTS["format_answer_only"])
            raw_ans, ttft_ans = call_llm(prompt_ans, max_tokens=96)
            tokens_ans = len(tokenizer.encode(raw_ans))
            active_tokens_ans = len(tokenizer.encode(prompt_ans))
            cost_ans = (active_tokens_ans * PRICE_INPUT_PER_M + tokens_ans * PRICE_OUTPUT_PER_M) / 1e6

            # 3. format_fields_structured (shared with regex postcheck)
            prompt_struct = assemble_prompt_format(kept_block_ids, question, documents, PROMPTS["format_fields_structured"])
            raw_struct, ttft_struct = call_llm(prompt_struct, max_tokens=96)
            tokens_struct = len(tokenizer.encode(raw_struct))
            active_tokens_struct = len(tokenizer.encode(prompt_struct))
            cost_struct = (active_tokens_struct * PRICE_INPUT_PER_M + tokens_struct * PRICE_OUTPUT_PER_M) / 1e6

            # Evaluate each format
            for fmt in formats:
                if fmt == "format_json_current":
                    raw, ttft, gen_tokens, cost_val = raw_json, ttft_json, tokens_json, cost_json
                    answer = extract_answer(raw, fmt)
                elif fmt == "format_answer_only":
                    raw, ttft, gen_tokens, cost_val = raw_ans, ttft_ans, tokens_ans, cost_ans
                    answer = extract_answer(raw, fmt)
                elif fmt == "format_fields_structured":
                    raw, ttft, gen_tokens, cost_val = raw_struct, ttft_struct, tokens_struct, cost_struct
                    answer = extract_answer(raw, fmt)
                elif fmt == "format_fields_plus_regex_postcheck":
                    raw, ttft, gen_tokens, cost_val = raw_struct, ttft_struct, tokens_struct, cost_struct
                    answer = extract_answer(raw, "format_fields_plus_regex_postcheck")
                
                norm_gen = normalize_answer(answer)
                em = norm_gen == norm_expected
                if is_abstention:
                    em = check_abstention(answer)
                
                f1 = calculate_f1(answer, expected)
                gen_digits = re.findall(r'\d+', norm_gen)
                num_pres = all(d in gen_digits for d in expected_digits) if expected_digits else True
                
                recall = calculate_multi_fact_recall(answer, expected)
                parse_ok = (answer.strip() != raw.strip()) or check_abstention(raw)
                parse_fail = 0.0 if parse_ok else 1.0
                failed_despite_gold = (not is_abstention) and gold_block_recall and (not em)
                
                format_results[fmt]["em"].append(float(em))
                format_results[fmt]["f1"].append(f1)
                format_results[fmt]["num_pres"].append(float(num_pres))
                format_results[fmt]["parse_fail"].append(parse_fail)
                format_results[fmt]["ttft"].append(ttft)
                format_results[fmt]["gen_tokens"].append(float(gen_tokens))
                format_results[fmt]["cost"].append(cost_val)
                format_results[fmt]["failed_despite_gold"].append(float(failed_despite_gold))
                format_results[fmt]["recall"].append(recall)

        if (i + 1) % 10 == 0:
            print(f"  [{i+1}/{len(cat_d)}] sample={q_id}")

    # Always kill server
    kill_vllm_server()

    # Results table
    print("\n" + "="*85)
    print("  FORMAT PROBE RESULTS â€” Category D (multi-fact)")
    print("="*85)
    print(f"  {'Format':<35} {'EM':>6} {'F1':>6} {'NumPres':>8} {'Recall':>6} {'ParseFail':>10} {'TTFT p50':>10} {'Tokens':>6} {'Cost/Corr':>10}")
    print("-"*85)
    
    summary_data = {}
    for fmt, stats in format_results.items():
        em  = np.mean(stats["em"]) * 100
        f1  = np.mean(stats["f1"]) * 100
        num = np.mean(stats["num_pres"]) * 100
        rec = np.mean(stats["recall"]) * 100
        pf  = np.mean(stats["parse_fail"]) * 100
        tp  = np.percentile(stats["ttft"], 50) if stats["ttft"] else 0.0
        tok = np.mean(stats["gen_tokens"])
        
        total_cost = sum(stats["cost"])
        num_correct = sum(stats["em"])
        cost_per_correct = (total_cost / num_correct) * 1000 if num_correct > 0 else 0.0
        
        fdg = sum(stats["failed_despite_gold"])
        
        summary_data[fmt] = {
            "em": em, "f1": f1, "num_pres": num, "recall": rec, "parse_fail": pf, "ttft_p50": tp, "tokens": tok, "cost_per_correct": cost_per_correct, "fdg": fdg
        }
        
        print(f"  {fmt:<35} {em:>5.1f}% {f1:>5.1f}% {num:>7.1f}% {rec:>5.1f}% {pf:>9.1f}% {tp:>9.0f}ms {tok:>6.1f} {cost_per_correct:>8.4f}$")
    print("="*85)

    # Verdict
    nums = {fmt: np.mean(stats["num_pres"]) for fmt, stats in format_results.items()}
    best = max(nums, key=nums.get)
    best_val = nums[best] * 100
    base_val = nums["format_json_current"] * 100
    delta = best_val - base_val

    print(f"\n  Best format: [{best}] NumPres={best_val:.1f}% (delta vs JSON: {delta:+.1f}%)")
    if delta > 5:
        print(f"  [OK] Format alternatif ameliore Numeric â€” utiliser [{best}] cette nuit.")
    elif abs(delta) <= 5:
        print("  [WARNING] Pas d'amelioration significative -> bottleneck modele 0.5B confirme.")
    else:
        print("  [INFO] JSON reste le meilleur â€” garder format actuel.")

    # Write Markdown Report to artifacts directory
    report_path = "C:/Users/nicol/.gemini/antigravity/brain/8b6b884f-e0bd-447e-94b0-ba83143e8388/mini3_report.md"
    os.makedirs(os.path.dirname(report_path), exist_ok=True)
    
    with open(report_path, "w") as rf:
        rf.write("# Mini-run 3 Report: Numeric Format Probe\n\n")
        rf.write("## Setup\n")
        rf.write(f"- **Dataset**: Category D only (100 samples)\n")
        rf.write(f"- **Model**: `{MODEL}`\n")
        rf.write(f"- **Engine**: vLLM\n")
        rf.write(f"- **Context size**: {CONTEXT_BLOCKS} blocks\n")
        rf.write(f"- **Mode**: `predictor_cached_guarded`\n\n")
        
        rf.write("## Metrics Summary\n\n")
        rf.write("| Format | EM | F1 | Numeric Pres. | Recall | Parse Fail Rate | TTFT p50 | Avg Tokens | Cost per 1000 Correct (USD) | Model Failed Despite Gold (Count) |\n")
        rf.write("|---|---|---|---|---|---|---|---|---|---|\n")
        for fmt, s in summary_data.items():
            rf.write(f"| `{fmt}` | {s['em']:.1f}% | {s['f1']:.1f}% | {s['num_pres']:.1f}% | {s['recall']:.1f}% | {s['parse_fail']:.1f}% | {s['ttft_p50']:.0f}ms | {s['tokens']:.1f} | ${s['cost_per_correct']:.4f} | {int(s['fdg'])} |\n")
            
        rf.write("\n## Verdict\n\n")
        rf.write(f"- **Best Format**: `{best}` with **{best_val:.1f}%** Numeric Preservation.\n")
        rf.write(f"- **Delta vs JSON Baseline**: `{delta:+.1f}%`.\n\n")
        
        if delta > 5:
            rf.write(f"> [!NOTE]\n")
            rf.write(f"> **Recommendation**: Use `{best}` for the overnight run. It successfully improved Numeric Preservation by `+{delta:.1f}%`.\n")
        else:
            rf.write(f"> [!WARNING]\n")
            rf.write(f"> **Recommendation**: Keep `format_json_current`. The format changes did not yield a significant improvement (> 5%), confirming that the numeric extraction limit is indeed a model size bottleneck (Qwen 0.5B).\n")

    print(f"Report written to: {report_path}")
    print("=== Mini-run 3 format probe complete ===")

if __name__ == "__main__":
    main()


