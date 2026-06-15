import os
import json
import time
import argparse
import random
import re
import urllib.request
import numpy as np


from highway.runtime.scheduler import ExecutionScheduler

def normalize_answer(text: str) -> str:
    text = str(text).lower().strip()
    text = text.replace("$", "").replace(",", "").replace(".", "").replace("â‚¬", "")
    text = text.replace("project ", "").replace("project", "")
    text = text.replace("department ", "").replace("department", "")
    text = re.sub(r'\band\b', "", text)
    tokens = re.split(r'[\s,]+', text)
    tokens = [t.strip() for t in tokens if t.strip()]
    tokens.sort()
    return " ".join(tokens)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-name", type=str, required=True)
    parser.add_argument("--corpus", type=str, required=True)
    parser.add_argument("--workload", type=str, required=True)
    parser.add_argument("--models", type=str, default="qwen_0_5b")
    parser.add_argument("--model-name", type=str, default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--engine", type=str, default="vllm")
    parser.add_argument("--vllm-host", type=str, default="localhost")
    parser.add_argument("--modes", type=str, required=True)
    parser.add_argument("--max-active-tokens", type=int, default=1200)
    parser.add_argument("--search-top-k", type=int, default=50)
    parser.add_argument("--fallback-search-top-k", type=int, default=150)
    parser.add_argument("--max-kept", type=int, default=6)
    parser.add_argument("--fallback-max-kept", type=int, default=12)
    parser.add_argument("--enable-answer-cache", type=str, default="true")
    parser.add_argument("--enable-proof-cache", type=str, default="true")
    parser.add_argument("--enable-evidence-cache", type=str, default="true")
    parser.add_argument("--enable-compiled-prompt-cache", type=str, default="true")
    parser.add_argument("--enable-prefix-friendly-compiler", type=str, default="true")
    parser.add_argument("--enable-long-context-fallback", type=str, default="true")
    parser.add_argument("--enable-output-verifier", type=str, default="true")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-new-tokens", type=int, default=96)
    parser.add_argument("--save-every", type=int, default=10)
    parser.add_argument("--resume", type=str, default="true")
    parser.add_argument("--max-retries", type=int, default=1)
    parser.add_argument("--timeout", type=int, default=45)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--summary", type=str, required=True)
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

    # 2. Parse modes
    modes = [m.strip() for m in args.modes.split(",") if m.strip()]
    print(f"Modes to run: {modes}")

    # 3. Initialize Scheduler
    # Mapping model name
    model_name = args.model_name
    # Port is parsed or hardcoded to 8000
    port = 8000
    
    # We write cache files to the corpus/cache directory
    corpus_dir = os.path.dirname(args.corpus.rstrip("/\\"))
    cache_dir = os.path.join(corpus_dir, "cache")
    os.makedirs(cache_dir, exist_ok=True)

    scheduler = ExecutionScheduler(args.corpus, cache_dir, vllm_port=port, model_name=model_name)
    # Update vllm host in URL if needed
    scheduler.vllm_url = f"http://{args.vllm_host}:{port}/v1/completions"

    # Setup Resume
    resume_flag = args.resume.lower() == "true"
    completed_keys = set()
    if resume_flag and os.path.exists(args.output):
        print(f"Resuming from existing output file: {args.output}")
        with open(args.output, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    try:
                        record = json.loads(line)
                        completed_keys.add((record["mode"], record["id"]))
                    except Exception:
                        pass
        print(f"Skipping {len(completed_keys)} already completed runs.")

    # Generate stratified subset of 150 queries for baseline if hybrid_topk_llm_baseline_stratified_150 is active
    baseline_selected_ids = set()
    if any("stratified_150" in m for m in modes):
        if len(queries) <= 150:
            baseline_selected_ids = {q["id"] for q in queries}
        else:
            # We perform stratified sampling: 150 queries out of len(queries)
            # Group by category
            cat_to_queries = {}
            for q in queries:
                cat = q.get("category", "A")
                cat_to_queries.setdefault(cat, []).append(q)
            
            # Sort queries in each category for determinism
            for cat in cat_to_queries:
                cat_to_queries[cat].sort(key=lambda x: x.get("id", ""))
                
            # Determine sampling count per category proportional to category count
            total_queries = len(queries)
            sampled_queries = []
            # We want to select 150 queries
            target_total = 150
            
            rng = random.Random(args.seed)
            for cat, cat_qs in cat_to_queries.items():
                prop = len(cat_qs) / total_queries
                cat_target = int(round(prop * target_total))
                if cat_target > len(cat_qs):
                    cat_target = len(cat_qs)
                # Sample without replacement
                sampled_queries.extend(rng.sample(cat_qs, cat_target))
                
            # If we are slightly off 150 due to rounding, adjust
            remaining = target_total - len(sampled_queries)
            if remaining > 0:
                all_remaining = [q for q in queries if q not in sampled_queries]
                sampled_queries.extend(rng.sample(all_remaining, remaining))
            elif remaining < 0:
                sampled_queries = rng.sample(sampled_queries, target_total)
                
            baseline_selected_ids = {q["id"] for q in sampled_queries}
        print(f"Selected {len(baseline_selected_ids)} queries for stratified baseline.")

    # Output file setup
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    out_file = open(args.output, "a", encoding="utf-8")

    # Run loop
    for idx, q in enumerate(queries):
        q_id = q["id"]
        question = q["question"]
        expected = q["expected_answer"]
        cat = q["category"]
        is_replay = q.get("is_replay", False)

        for mode in modes:
            key = (mode, q_id)
            if key in completed_keys:
                continue

            # If mode is stratified baseline and query not selected, skip running
            if "stratified_150" in mode and q_id not in baseline_selected_ids:
                # We can write a skipped record or just skip it
                continue

            t_start = time.time()
            
            # Reset cache stats at beginning if we are not testing reuse, but wait,
            # we want the cache to persist. scheduler saves automatically or at the end.
            # In Phase 8 we called save() after each run. Here we can save cache every save-every queries.

            if mode == "pccc_cache_scheduler_prefix":
                # Run with cache enabled
                res = scheduler.answer(question, use_cache=True, force_llm=False)
                answer = res["answer"]
                route = res["route"]
                metrics = res["metrics"]
                is_bypass = metrics["llm_bypass"]
                prompt_tokens = metrics["prompt_tokens"]
                kept_blocks = metrics.get("kept_blocks", len(res.get("proof_ir", {}).get("evidence", [])))
                verify_passed = metrics["verifier_passed"]
                tokens_avoided = metrics["tokens_avoided"]
                
            elif mode == "pccc_no_cache":
                # Run with cache disabled
                res = scheduler.answer(question, use_cache=False, force_llm=False)
                answer = res["answer"]
                route = res["route"]
                metrics = res["metrics"]
                is_bypass = metrics["llm_bypass"]
                prompt_tokens = metrics["prompt_tokens"]
                kept_blocks = metrics.get("kept_blocks", len(res.get("proof_ir", {}).get("evidence", [])))
                verify_passed = metrics["verifier_passed"]
                tokens_avoided = metrics["tokens_avoided"]

            elif "hybrid_topk_llm_baseline" in mode:
                # Standard QA baseline: search top-150 blocks, compile raw prompt, call LLM
                t_search = time.time()
                # We use fallback-search-top-k (150)
                candidates, query_ir = scheduler.search_router.search(question, top_k=args.fallback_search_top_k)
                search_lat_ms = (time.time() - t_search) * 1000.0
                
                # Build prompt with all candidate blocks
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
                answer = scheduler._call_llm(prompt_text)
                llm_ttft_ms = (time.time() - t_llm) * 1000.0
                
                route = "HYBRID_BASELINE"
                is_bypass = False
                kept_blocks = len(candidates)
                verify_passed = True
                tokens_avoided = 0
                
                metrics = {
                    "route": route,
                    "latency_ms": (time.time() - t_start) * 1000.0,
                    "cache_lookup_latency_ms": 0.0,
                    "search_latency_ms": search_lat_ms,
                    "ir_build_latency_ms": 0.0,
                    "llm_ttft_ms": llm_ttft_ms,
                    "prompt_tokens": prompt_tokens,
                    "tokens_materialized_kv": prompt_tokens,
                    "tokens_avoided": 0,
                    "llm_bypass": False,
                    "verifier_passed": True,
                    "shared_prefix_tokens": 0,
                    "prefix_cache_hit": False,
                    "stale_cache_error": False
                }
            
            latency = (time.time() - t_start) * 1000.0
            
            # Exact match calculation
            norm_gen = normalize_answer(answer)
            norm_exp = normalize_answer(expected)
            is_em = (norm_gen == norm_exp)

            # Record
            record = {
                "id": q_id,
                "sample_id": q_id,
                "question": question,
                "expected": expected,
                "expected_answer": expected,
                "generated": answer,
                "answer": answer,
                "category": cat,
                "mode": mode,
                "is_em": is_em,
                "exact_match": is_em,
                "is_bypass": is_bypass,
                "llm_bypass": is_bypass,
                "latency_ms": latency,
                "total_latency_ms": latency,
                "prompt_tokens": prompt_tokens,
                "prompt_tokens_approx": prompt_tokens,
                "kept_blocks": kept_blocks,
                "blocks_kept": kept_blocks,
                "is_replay": is_replay,
                "verify_passed": verify_passed,
                "tokens_avoided": tokens_avoided,
                "route": route,
                "metrics": metrics
            }

            out_file.write(json.dumps(record, ensure_ascii=False) + "\n")
            out_file.flush()

        # Save scheduler caches periodically
        if (idx + 1) % args.save_every == 0:
            scheduler.cache_manager.save()
            print(f"Processed {idx + 1}/{len(queries)} queries...")

    # Final cache save
    scheduler.cache_manager.save()
    out_file.close()
    print("Benchmark run completed successfully.")

    # Write simple summary file
    with open(args.summary, "w", encoding="utf-8") as f:
        f.write("# POC 2.3 Benchmark Summary\n\nRun completed.")

if __name__ == "__main__":
    main()



