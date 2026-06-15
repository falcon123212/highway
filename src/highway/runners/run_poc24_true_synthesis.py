import os
import json
import time
import argparse
import random
import numpy as np


from highway.runtime.scheduler import ExecutionScheduler

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-name", type=str, default="poc_2_4_true_llm_synthesis")
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

    # Load workload
    queries = []
    with open(args.workload, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                queries.append(json.loads(line))

    print(f"Loaded {len(queries)} queries from workload.")

    # Initialize Scheduler
    corpus_dir = os.path.dirname(args.corpus.rstrip("/\\"))
    cache_dir = os.path.join(corpus_dir, "cache")
    os.makedirs(cache_dir, exist_ok=True)

    scheduler = ExecutionScheduler(args.corpus, cache_dir, vllm_port=args.port, model_name=args.model)

    # Setup output file
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    
    # Clear output if exists
    if os.path.exists(args.output):
        os.remove(args.output)
    
    records = []
    
    for idx, q in enumerate(queries):
        q_id = q["id"]
        question = q["question"]
        expected = q["expected_answer"]
        cat = q["category"]
        allowed_conclusions = q.get("allowed_conclusions", [])
        expected_conclusion = q.get("expected_conclusion", "")
        
        print(f"[{idx+1}/{len(queries)}] Running query {q_id} ({cat})...")
        
        t0 = time.time()
        res = scheduler.answer(
            question,
            use_cache=False,
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
        
        verifier_audit = metrics.get("verifier_audit", {})
        
        record = {
            "id": q_id,
            "question": question,
            "expected_answer": expected,
            "expected_conclusion": expected_conclusion,
            "category": cat,
            "mode": "pccc_synthesis",
            "answer": answer,
            "route": route,
            "latency_ms": latency,
            "prompt_tokens": metrics.get("prompt_tokens", 0),
            "shared_prefix_tokens": metrics.get("shared_prefix_tokens", 0),
            "tokens_materialized_kv": metrics.get("tokens_materialized_kv", 0),
            "llm_ttft_ms": metrics.get("llm_ttft_ms", 0.0),
            "groundedness_score": metrics.get("groundedness_score", 1.0),
            "task_score_5": metrics.get("task_score_5", 5),
            "obsolete_evidence_used": metrics.get("obsolete_evidence_used", False),
            "repair_attempts": metrics.get("repair_attempts", 0),
            "malformed_json": metrics.get("malformed_json", False),
            "unsupported_claim_rate": metrics.get("unsupported_claim_rate", 0.0),
            "verifier_passed": metrics.get("verifier_passed", True),
            "errors": verifier_audit.get("errors", [])
        }
        
        records.append(record)
        
        # Write to JSONL
        with open(args.output, "a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")
            
    print(f"Run completed. Saved results to {args.output}")

if __name__ == "__main__":
    main()



