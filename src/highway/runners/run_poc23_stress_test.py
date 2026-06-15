import os
import json
import time
from highway.runtime.scheduler import ExecutionScheduler
from run_pccc_benchmark import normalize_answer

def run_stress_test():
    print("=== Running POC 2.3 Stress Test Benchmark ===")
    
    corpus_dir = "corpus_poc2_stress"
    index_dir = os.path.join(corpus_dir, "index")
    cache_dir = os.path.join(corpus_dir, "cache")
    os.makedirs(cache_dir, exist_ok=True)
    
    gold_qa_path = os.path.join(corpus_dir, "questions", "qa_gold.json")
    with open(gold_qa_path, "r", encoding="utf-8") as f:
        queries = json.load(f)
        
    # Initialize scheduler
    model_name = "Qwen/Qwen2.5-0.5B-Instruct"
    scheduler = ExecutionScheduler(index_dir, cache_dir, vllm_port=8000, model_name=model_name)
    scheduler.vllm_url = "http://localhost:8000/v1/completions"
    
    print(f"Loaded {len(queries)} stress test queries.")
    
    modes = ["PCCC (No Cache)", "PCCC (With Cache Replay)", "Standard LLM Baseline"]
    
    results = {}
    
    # 1. Mode: PCCC (No Cache)
    print("\n--- Mode 1: PCCC (No Cache) ---")
    results["PCCC (No Cache)"] = []
    for q in queries:
        question = q["question"]
        expected = q["expected_answer"]
        q_id = q["id"]
        
        t_start = time.time()
        res = scheduler.answer(question, use_cache=False)
        lat = (time.time() - t_start) * 1000.0
        
        ans = res["answer"]
        route = res["route"]
        
        is_em = (normalize_answer(ans) == normalize_answer(expected))
        results["PCCC (No Cache)"].append({
            "id": q_id,
            "question": question,
            "expected": expected,
            "generated": ans,
            "route": route,
            "latency_ms": lat,
            "is_em": is_em
        })
        print(f"[{q_id}] EM={is_em} | Route={route} | Lat={lat:.1f}ms | Q: '{question}' -> Ans: '{ans}'")
        
    # 2. Mode: PCCC (With Cache Replay) - run again with cache enabled
    print("\n--- Mode 2: PCCC (With Cache Replay) ---")
    results["PCCC (With Cache Replay)"] = []
    # Force saving the previous runs to cache first
    scheduler.cache_manager.save()
    
    for q in queries:
        question = q["question"]
        expected = q["expected_answer"]
        q_id = q["id"]
        
        t_start = time.time()
        res = scheduler.answer(question, use_cache=True)
        lat = (time.time() - t_start) * 1000.0
        
        ans = res["answer"]
        route = res["route"]
        
        is_em = (normalize_answer(ans) == normalize_answer(expected))
        results["PCCC (With Cache Replay)"].append({
            "id": q_id,
            "question": question,
            "expected": expected,
            "generated": ans,
            "route": route,
            "latency_ms": lat,
            "is_em": is_em
        })
        print(f"[{q_id}] EM={is_em} | Route={route} | Lat={lat:.1f}ms | Q: '{question}' -> Ans: '{ans}'")

    # 3. Mode: Standard LLM Baseline
    print("\n--- Mode 3: Standard LLM Baseline ---")
    results["Standard LLM Baseline"] = []
    for q in queries:
        question = q["question"]
        expected = q["expected_answer"]
        q_id = q["id"]
        
        t_start = time.time()
        # Retrieve top-50 blocks for baseline
        candidates, _ = scheduler.search_router.search(question, top_k=50)
        
        # Build raw prompt
        prompt_lines = [
            "<|im_start|>system\nYou are a helpful extraction assistant. Answer the question based on the provided context.",
            "If the answer cannot be found in the context, respond with ONLY: NOT_FOUND<|im_end|>"
        ]
        prompt_lines.append("<|im_start|>user\nContext:")
        for idx, b in enumerate(candidates):
            prompt_lines.append(f"Block {idx+1} [SOURCE: {b['source_file']}]: {b['text']}")
        prompt_lines.append(f"\nQuestion: {question}\nRespond with ONLY the answer in a JSON object:\n{{\n  \"answer\": \"<value>\"\n}}<|im_end|>")
        prompt_lines.append("<|im_start|>assistant\n{\n  \"answer\": \"")
        
        prompt_text = "\n".join(prompt_lines)
        ans = scheduler._call_llm(prompt_text)
        lat = (time.time() - t_start) * 1000.0
        
        is_em = (normalize_answer(ans) == normalize_answer(expected))
        results["Standard LLM Baseline"].append({
            "id": q_id,
            "question": question,
            "expected": expected,
            "generated": ans,
            "route": "HYBRID_BASELINE",
            "latency_ms": lat,
            "is_em": is_em
        })
        print(f"[{q_id}] EM={is_em} | Route=HYBRID_BASELINE | Lat={lat:.1f}ms | Q: '{question}' -> Ans: '{ans}'")

    # Compile a quick markdown report of stress test results
    report_lines = [
        "# POC 2.3 Stress Test Results Report",
        f"Generated at: {time.strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        "This report evaluates the **PCCC (Proof-Carrying Context Compiler)** scheduling, culling, and routing logic against a highly challenging stress-test dataset consisting of deep suffix distractors, temporal cascades, complete project obsolescence, multi-hop comparisons, and low Signal-to-Noise Ratio (SNR) context.",
        "",
        "## Summary table",
        "",
        "| Query ID | Difficulty / Objective | Expected Answer | PCCC (No Cache) Route / EM | PCCC (Cache Replay) Route / EM | Standard LLM Baseline EM |",
        "|---|---|---|---|---|---|",
    ]
    
    for idx, q in enumerate(queries):
        q_id = q["id"]
        desc = q["reasoning"]
        expected = q["expected_answer"]
        
        # PCCC No Cache
        r_nc = results["PCCC (No Cache)"][idx]
        nc_status = f"`{r_nc['route']}` / **{'PASS' if r_nc['is_em'] else 'FAIL'}**"
        
        # PCCC Cache Replay
        r_c = results["PCCC (With Cache Replay)"][idx]
        c_status = f"`{r_c['route']}` / **{'PASS' if r_c['is_em'] else 'FAIL'}**"
        
        # Baseline
        r_b = results["Standard LLM Baseline"][idx]
        b_status = f"**{'PASS' if r_b['is_em'] else 'FAIL'}**"
        
        report_lines.append(f"| `{q_id}` | {desc} | `{expected}` | {nc_status} | {c_status} | {b_status} |")
        
    report_lines.append("\n## Detailed Analysis by Query")
    for idx, q in enumerate(queries):
        q_id = q["id"]
        q_text = q["question"]
        expected = q["expected_answer"]
        desc = q["reasoning"]
        
        report_lines.append(f"\n### {q_id}: {q_text}")
        report_lines.append(f"* **Objective**: {desc}")
        report_lines.append(f"* **Expected Answer**: `{expected}`")
        report_lines.append("")
        
        for mode in modes:
            r = results[mode][idx]
            report_lines.append(f"  * **{mode}**: Generated = `{r['generated']}` | Route = `{r['route']}` | Latency = `{r['latency_ms']:.1f} ms` | EM = **{'PASS' if r['is_em'] else 'FAIL'}**")
            
    report_md = "\n".join(report_lines)
    
    # Save report inside the Highway project directory
    with open("artifacts/runs/poc_2_3_night_safe_mixed_execution/stress_test_report.md", "w", encoding="utf-8") as f:
        f.write(report_md)
        
    print(f"\nSaved stress test report to artifacts/runs/poc_2_3_night_safe_mixed_execution/stress_test_report.md")

if __name__ == "__main__":
    run_stress_test()


