"""
Mini-run 1 â€” Fix latence localhost vs 127.0.0.1
Compare les mÃ©triques de latence TTFT sur 40 samples, mode predictor_cached_guarded, 200 blocs.
Lance les deux URLs sÃ©quentiellement et affiche un tableau comparatif.
"""
import os, sys, json, time, re, random, pickle, string, collections
import numpy as np
import requests
from typing import Dict, Any, List

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from src.extract_features import (
    extract_block_features, get_block_embeddings, get_embedding_model
)
from run_poc1_benchmark import (
    assemble_prompt, scale_sample, block_contains_only_suffix,
    parse_json_response_b, normalize_answer, calculate_f1
)
from transformers import AutoTokenizer

MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
PORT  = 8000
N_SAMPLES = 40
CONTEXT_BLOCKS = 200
MODE = "predictor_cached_guarded"
URLS_TO_TEST = [
    ("localhost",  f"http://localhost:{PORT}/v1/completions"),
    ("127.0.0.1",  f"http://127.0.0.1:{PORT}/v1/completions"),
]

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

def run_host(label, url, samples, gold_answers, clf, tokenizer):
    print(f"\n{'='*60}")
    print(f"  Testing: {label}  ({url})")
    print(f"{'='*60}")

    # Warmup
    try:
        requests.post(url, json={"model": MODEL, "prompt": "Hello", "max_tokens": 5, "temperature": 0},
                      timeout=15)
        print("  Warmup OK")
    except Exception as e:
        print(f"  Warmup FAILED: {e}")

    ttfts, first_tok_lats, total_lats, decode_tps = [], [], [], []

    for i, sample in enumerate(samples):
        scaled = scale_sample(sample, CONTEXT_BLOCKS, seed=42)
        documents = scaled["documents"]
        project_name = scaled["project"]
        question = scaled["question"]
        q_id = sample["question_id"]
        gold_info = gold_answers[q_id]
        gold_ids = gold_info["gold_block_ids"]
        is_abstention = gold_info["is_abstention"]

        # Embeddings
        block_texts = [b["text"] for b in documents]
        pre_embs = get_block_embeddings(block_texts)
        t0 = time.perf_counter()
        features = extract_block_features(
            question, documents, project_name, ablation_mode="no_position",
            skip_embedding_compute=True, cached_block_embs=pre_embs
        )
        probs = clf.predict_proba(features)[:, 1]
        kept_ids_pred = [j for j, p in enumerate(probs) if p >= 0.70]
        if len(kept_ids_pred) < 4:
            kept_ids_pred = sorted(list(np.argsort(probs)[::-1][:4]))
        sel_ms = (time.perf_counter() - t0) * 1000.0

        # Guard
        kept_block_ids = []
        for k_idx in kept_ids_pred:
            if not block_contains_only_suffix(documents[k_idx]["text"], project_name):
                kept_block_ids.append(k_idx)
        exact_pat = r'(?<![a-zA-Z0-9_-])' + re.escape(project_name) + r'(?![a-zA-Z0-9_-])'
        any_exact = any(re.search(exact_pat, documents[j]["text"]) for j in kept_block_ids)
        is_guarded = not any_exact

        if is_guarded:
            # Instant bypass â€” record 0 latency contribution for bypass cases
            total_lats.append(sel_ms)
            continue

        prompt_text = assemble_prompt(kept_block_ids, question, documents)
        payload = {
            "model": MODEL,
            "prompt": prompt_text,
            "max_tokens": 64,
            "temperature": 0,
            "top_p": 1,
            "stream": True,
        }

        t_send = time.perf_counter()
        t_first = None
        chunks = []
        try:
            resp = requests.post(url, json=payload, stream=True, timeout=60)
            for chunk in resp.iter_lines():
                if not chunk: continue
                if t_first is None: t_first = time.perf_counter()
                line = chunk.decode("utf-8").strip()
                if line.startswith("data: "):
                    ds = line[6:].strip()
                    if ds == "[DONE]": break
                    try:
                        data = json.loads(ds)
                        chunks.append(data["choices"][0]["text"])
                    except: pass
            t_end = time.perf_counter()
        except Exception as e:
            print(f"  Sample {q_id} FAILED: {e}")
            continue

        gen_raw = "".join(chunks)
        gen_tokens = len(tokenizer.encode(gen_raw))

        ttft   = (t_first - t_send) * 1000.0 if t_first else 0.0
        decode = (t_end - t_first)   * 1000.0 if t_first else 0.0
        e2e    = (t_end  - t_send)   * 1000.0
        decode_tp = gen_tokens / (decode / 1000.0) if decode > 0 else 0.0

        first_tok_lat = sel_ms + ttft
        total_lat     = sel_ms + e2e

        ttfts.append(ttft)
        first_tok_lats.append(first_tok_lat)
        total_lats.append(total_lat)
        decode_tps.append(decode_tp)

        if (i + 1) % 10 == 0:
            print(f"  [{i+1}/{len(samples)}] TTFT={ttft:.0f}ms  FirstTok={first_tok_lat:.0f}ms  TotalLat={total_lat:.0f}ms  Decode={decode_tp:.1f}tok/s")

    def p(arr, pct): return np.percentile(arr, pct) if arr else 0.0

    print(f"\n  Results for [{label}] â€” {len(ttfts)} non-bypass samples")
    print(f"  TTFT       p50={p(ttfts,50):.1f}ms  p95={p(ttfts,95):.1f}ms")
    print(f"  FirstToken p50={p(first_tok_lats,50):.1f}ms  p95={p(first_tok_lats,95):.1f}ms")
    print(f"  TotalLat   p50={p(total_lats,50):.1f}ms  p95={p(total_lats,95):.1f}ms")
    print(f"  Decode     avg={np.mean(decode_tps):.1f} tok/s" if decode_tps else "  Decode     N/A")

    return {
        "label": label,
        "n": len(ttfts),
        "ttft_p50":   p(ttfts, 50),
        "ttft_p95":   p(ttfts, 95),
        "ft_p50":     p(first_tok_lats, 50),
        "ft_p95":     p(first_tok_lats, 95),
        "tot_p50":    p(total_lats, 50),
        "tot_p95":    p(total_lats, 95),
        "decode_avg": float(np.mean(decode_tps)) if decode_tps else 0.0,
    }


def main():
    print("=== MINI-RUN 1 â€” localhost vs 127.0.0.1 Latency Test ===")
    corpus, gold_answers = load_data()

    # Pick N_SAMPLES deterministically across categories
    cat_samples = {cat: [] for cat in ["A","B","C","D","E"]}
    for s in corpus:
        if s["category"] in cat_samples:
            cat_samples[s["category"]].append(s)
    for cat in cat_samples:
        cat_samples[cat].sort(key=lambda x: x["question_id"])
    selected = []
    per_cat = N_SAMPLES // 5
    for cat in ["A","B","C","D","E"]:
        selected.extend(cat_samples[cat][:per_cat])
    random.seed(42)
    random.shuffle(selected)
    print(f"Loaded {len(selected)} samples")

    # Load predictor
    pred_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "experiments/kv_visibility_poc0", "models", "visibility_predictor_standard_no_position.pkl")
    with open(pred_path, "rb") as f:
        pred_data = pickle.load(f)
    clf = pred_data["model"]

    tokenizer = AutoTokenizer.from_pretrained(MODEL)

    results = []
    for label, url in URLS_TO_TEST:
        res = run_host(label, url, selected, gold_answers, clf, tokenizer)
        results.append(res)

    # Final comparison table
    print("\n" + "="*70)
    print("  COMPARISON TABLE")
    print("="*70)
    print(f"{'Metric':<30} {'localhost':>15} {'127.0.0.1':>15} {'Delta':>12}")
    print("-"*70)
    def row(name, key, fmt=".1f"):
        a = results[0][key]
        b = results[1][key]
        delta = b - a
        sign = "+" if delta > 0 else ""
        print(f"  {name:<28} {a:>15{fmt}} {b:>15{fmt}} {sign+f'{delta:.1f}':>12}")

    row("TTFT p50 (ms)",           "ttft_p50")
    row("TTFT p95 (ms)",           "ttft_p95")
    row("First Token Lat p50 (ms)","ft_p50")
    row("First Token Lat p95 (ms)","ft_p95")
    row("Total Lat p50 (ms)",      "tot_p50")
    row("Total Lat p95 (ms)",      "tot_p95")
    row("Decode tok/s (avg)",      "decode_avg")
    print("="*70)

    ttft_delta = results[1]["ttft_p50"] - results[0]["ttft_p50"]
    if ttft_delta < -500:
        print(f"\n  âœ… VERDICT: 127.0.0.1 gagne {abs(ttft_delta):.0f}ms sur TTFT p50 â†’ DNS bug confirmÃ©.")
        print("     Utiliser 127.0.0.1 cette nuit.")
    elif abs(ttft_delta) < 100:
        print(f"\n  âš ï¸  VERDICT: Delta < 100ms â†’ Pas de diffÃ©rence significative.")
        print("     HypothÃ¨se DNS non confirmÃ©e sur cette machine.")
    else:
        print(f"\n  â„¹ï¸  VERDICT: Delta = {ttft_delta:.0f}ms â†’ DiffÃ©rence modÃ©rÃ©e, Ã  interprÃ©ter.")

if __name__ == "__main__":
    main()


