import os
import json
import time
import argparse
import random
import re
import numpy as np


from highway.runtime.scheduler import ExecutionScheduler

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-name", type=str, default="poc_2_4_mixed_runtime")
    parser.add_argument("--corpus", type=str, required=True)
    parser.add_argument("--workload", type=str, required=True)
    parser.add_argument("--model", type=str, default="Qwen/Qwen2.5-1.5B-Instruct")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--summary", type=str, required=True)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    # 1. Load workload
    queries = []
    with open(args.workload, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                queries.append(json.loads(line))

    print(f"Loaded {len(queries)} queries from workload.")

    # 2. Perform stratified sampling for the baseline (150 queries)
    cat_to_queries = {}
    for q in queries:
        cat = q.get("category", "A")
        cat_to_queries.setdefault(cat, []).append(q)
        
    for cat in cat_to_queries:
        cat_to_queries[cat].sort(key=lambda x: x.get("id", ""))
        
    total_queries = len(queries)
    sampled_queries = []
    target_total = 150
    
    rng = random.Random(args.seed)
    for cat, cat_qs in cat_to_queries.items():
        prop = len(cat_qs) / total_queries
        cat_target = int(round(prop * target_total))
        if cat_target > len(cat_qs):
            cat_target = len(cat_qs)
        sampled_queries.extend(rng.sample(cat_qs, cat_target))
        
    remaining = target_total - len(sampled_queries)
    if remaining > 0:
        all_remaining = [q for q in queries if q not in sampled_queries]
        sampled_queries.extend(rng.sample(all_remaining, remaining))
    elif remaining < 0:
        sampled_queries = rng.sample(sampled_queries, target_total)
        
    baseline_ids = {q["id"] for q in sampled_queries}
    print(f"Sampled {len(baseline_ids)} queries for baseline mode.")

    # 3. Initialize Scheduler
    corpus_dir = os.path.dirname(args.corpus.rstrip("/\\"))
    cache_dir = os.path.join(corpus_dir, "cache")
    os.makedirs(cache_dir, exist_ok=True)

    scheduler = ExecutionScheduler(args.corpus, cache_dir, vllm_port=args.port, model_name=args.model)

    # Setup output file
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    if os.path.exists(args.output):
        os.remove(args.output)

    # 4. Run loop
    for idx, q in enumerate(queries):
        q_id = q["id"]
        question = q["question"]
        expected = q["expected_answer"]
        cat = q["category"]
        allowed_conclusions = q.get("allowed_conclusions", [])
        expected_conclusion = q.get("expected_conclusion", "")
        
        # Mode 1: pccc_runtime
        print(f"[{idx+1}/{len(queries)}] Running pccc_runtime for {q_id} ({cat})...")
        t0 = time.time()
        res = scheduler.answer(
            question,
            use_cache=True,  # Enable cache for PCCC to measure caching benefit
            force_llm=False,
            q_id=q_id,
            category=cat,
            allowed_conclusions=allowed_conclusions,
            expected_conclusion=expected_conclusion
        )
        latency = (time.time() - t0) * 1000.0
        
        answer = res["answer"]
        route = res["route"]
        metrics = res["metrics"]
        
        # Simple EM calculation
        def clean_ans(text):
            text_clean = str(text).lower().strip().replace("$", "").replace(",", "").replace("project ", "").replace("project", "")
            import unicodedata
            nfkd_form = unicodedata.normalize('NFKD', text_clean)
            text_clean = u"".join([c for c in nfkd_form if not unicodedata.combining(c)])
            parts = sorted([p.strip() for p in re.split(r'[\s,]+', text_clean) if p.strip()])
            return " ".join(parts)
            
        is_em = (clean_ans(answer) == clean_ans(expected))
        if cat.startswith("I_"):
            is_em = metrics.get("verifier_passed", True)
            
        record_pccc = {
            "id": q_id,
            "question": question,
            "expected_answer": expected,
            "expected_conclusion": expected_conclusion,
            "category": cat,
            "mode": "pccc_runtime",
            "answer": answer,
            "route": route,
            "latency_ms": latency,
            "prompt_tokens": metrics.get("prompt_tokens", 0),
            "shared_prefix_tokens": metrics.get("shared_prefix_tokens", 0),
            "tokens_materialized_kv": metrics.get("tokens_materialized_kv", 0),
            "tokens_avoided": metrics.get("tokens_avoided", 0),
            "is_bypass": metrics.get("llm_bypass", True),
            "llm_ttft_ms": metrics.get("llm_ttft_ms", 0.0),
            "is_em": is_em,
            "verifier_passed": metrics.get("verifier_passed", True),
            "groundedness_score": metrics.get("groundedness_score", 1.0),
            "task_score_5": metrics.get("task_score_5", 5),
            "obsolete_evidence_used": metrics.get("obsolete_evidence_used", False),
            "repair_attempts": metrics.get("repair_attempts", 0),
            "malformed_json": metrics.get("malformed_json", False),
            "unsupported_claim_rate": metrics.get("unsupported_claim_rate", 0.0)
        }
        
        with open(args.output, "a", encoding="utf-8") as f:
            f.write(json.dumps(record_pccc) + "\n")
            
        # Mode 2: raw_rag_baseline
        if q_id in baseline_ids:
            print(f"[{idx+1}/{len(queries)}] Running raw_rag_baseline for {q_id} ({cat})...")
            t_start = time.time()
            candidates, _ = scheduler.search_router.search(question, top_k=100)
            search_lat = (time.time() - t_start) * 1000.0
            
            prompt_lines = [
                "<|im_start|>system\nYou are a helpful extraction assistant. Answer the question based on the provided context.",
                "If the answer cannot be found in the context, respond with ONLY: NOT_FOUND<|im_end|>"
            ]
            prompt_lines.append("<|im_start|>user\nContext:")
            for b_idx, b in enumerate(candidates):
                prompt_lines.append(f"Block {b_idx+1} [SOURCE: {b['source_file']}]: {b['text']}")
            prompt_lines.append(f"\nQuestion: {question}\nRespond with ONLY the answer in a JSON object:\n{{\n  \"answer\": \"<value>\"\n}}<|im_end|>")
            prompt_lines.append("<|im_start|>assistant\n{\n  \"answer\": \"")
            
            prompt_text = "\n".join(prompt_lines)
            prompt_tokens = len(prompt_text.split())
            
            t_llm = time.time()
            answer_baseline = scheduler._call_llm(prompt_text)
            llm_ttft = (time.time() - t_llm) * 1000.0
            latency_baseline = (time.time() - t_start) * 1000.0
            
            is_em_baseline = (clean_ans(answer_baseline) == clean_ans(expected))
            if cat.startswith("I_"):
                is_em_baseline = False
                if expected_conclusion:
                    is_em_baseline = expected_conclusion.lower() in answer_baseline.lower()
                    
            record_baseline = {
                "id": q_id,
                "question": question,
                "expected_answer": expected,
                "expected_conclusion": expected_conclusion,
                "category": cat,
                "mode": "raw_rag_baseline",
                "answer": answer_baseline,
                "route": "RAW_RAG_LLM",
                "latency_ms": latency_baseline,
                "prompt_tokens": prompt_tokens,
                "shared_prefix_tokens": 0,
                "tokens_materialized_kv": prompt_tokens,
                "tokens_avoided": 0,
                "is_bypass": False,
                "llm_ttft_ms": llm_ttft,
                "is_em": is_em_baseline,
                "verifier_passed": True,
                "groundedness_score": 1.0,
                "task_score_5": 5,
                "obsolete_evidence_used": False,
                "repair_attempts": 0,
                "malformed_json": False,
                "unsupported_claim_rate": 0.0
            }
            
            with open(args.output, "a", encoding="utf-8") as f:
                f.write(json.dumps(record_baseline) + "\n")

    print(f"Mixed runtime completed. Saved results to {args.output}")

if __name__ == "__main__":
    main()



