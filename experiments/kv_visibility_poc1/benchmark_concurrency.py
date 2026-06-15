import os
import json
import time
import asyncio
import aiohttp
import argparse
import numpy as np
from typing import Dict, Any, List
from transformers import AutoTokenizer

async def send_streaming_request(
    session: aiohttp.ClientSession,
    url: str,
    model_name: str,
    prompt_text: str,
    max_new_tokens: int = 64
) -> Dict[str, Any]:
    payload = {
        "model": model_name,
        "prompt": prompt_text,
        "max_tokens": max_new_tokens,
        "temperature": 0.0,
        "stream": True
    }
    
    t_send = time.perf_counter()
    t_first = None
    t_end = None
    generated_text_chunks = []
    oom = False
    status_code = 200
    
    try:
        async with session.post(url, json=payload, timeout=aiohttp.ClientTimeout(total=200)) as response:
            status_code = response.status
            if status_code != 200:
                oom = True
                t_end = time.perf_counter()
            else:
                # Read the response stream chunk by chunk
                async for chunk in response.content:
                    if not chunk:
                        continue
                    if t_first is None:
                        t_first = time.perf_counter()
                    
                    # Parse Server-Sent Events (SSE) format: "data: {...}"
                    lines = chunk.decode("utf-8").split("\n")
                    for line in lines:
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
    except Exception as e:
        print(f"Connection error or timeout during request: {e}")
        oom = True
        t_end = time.perf_counter()
        
    # Calculate timings in milliseconds
    ttft_ms = (t_first - t_send) * 1000.0 if t_first is not None else 0.0
    e2e_ms = (t_end - t_send) * 1000.0
    decode_ms = (t_end - t_first) * 1000.0 if t_first is not None else 0.0
    
    response_text = "".join(generated_text_chunks).strip()
    
    return {
        "oom": oom,
        "status_code": status_code,
        "ttft_ms": ttft_ms,
        "decode_ms": decode_ms,
        "e2e_ms": e2e_ms,
        "response_text": response_text
    }

async def run_concurrency_batch(
    url: str,
    model_name: str,
    prompts: List[Dict[str, Any]],
    concurrency: int,
    tokenizer: AutoTokenizer,
    max_new_tokens: int = 64
) -> List[Dict[str, Any]]:
    # Split the prompts into batches of size 'concurrency'
    results = []
    sem = asyncio.Semaphore(concurrency)
    
    async def worker(session: aiohttp.ClientSession, item: Dict[str, Any]):
        async with sem:
            prompt_text = item["compiled_prompt"]
            prompt_len = len(tokenizer.encode(prompt_text))
            
            # Record starting timestamp
            t0 = time.perf_counter()
            res = await send_streaming_request(session, url, model_name, prompt_text, max_new_tokens)
            
            # Post-process response to get generated token count
            gen_len = 0
            if not res["oom"]:
                gen_len = len(tokenizer.encode(res["response_text"]))
                
            # Compute throughput metrics
            tokens_per_sec_in = (prompt_len / (res["ttft_ms"] / 1000.0)) if res["ttft_ms"] > 0 else 0.0
            tokens_per_sec_out = (gen_len / (res["decode_ms"] / 1000.0)) if res["decode_ms"] > 0 else 0.0
            
            # Effective throughput: original context length / total time
            # original_input_tokens = context_size_blocks * 128 (approx)
            original_input_tokens = item["context_size_blocks"] * 128
            total_time_sec = (res["e2e_ms"] + item["selector_latency_ms"]) / 1000.0
            effective_throughput = (original_input_tokens / total_time_sec) if total_time_sec > 0 else 0.0
            
            results.append({
                "question_id": item["question_id"],
                "category": item["category"],
                "project": item["project"],
                "question": item["question"],
                "context_size_blocks": item["context_size_blocks"],
                "mode": item["mode"],
                "selector_latency_ms": item["selector_latency_ms"],
                "kept_blocks_count": item["kept_blocks_count"],
                "token_reduction_pct": item["token_reduction_pct"],
                "gold_block_recall": item["gold_block_recall"],
                "expected_answer": item["expected_answer"],
                "is_abstention": item["is_abstention"],
                "gold_block_ids": item["gold_block_ids"],
                "deprecated_block_ids": item["deprecated_block_ids"],
                "oom": res["oom"],
                "status_code": res["status_code"],
                "ttft_ms": res["ttft_ms"],
                "decode_ms": res["decode_ms"],
                "e2e_ms": res["e2e_ms"],
                "generated_tokens": gen_len,
                "input_tokens": prompt_len,
                "tokens_per_sec_in": tokens_per_sec_in,
                "tokens_per_sec_out": tokens_per_sec_out,
                "effective_context_tokens_per_sec": effective_throughput,
                "generated_text": res["response_text"]
            })
            
    async with aiohttp.ClientSession() as session:
        tasks = [worker(session, item) for item in prompts]
        await asyncio.gather(*tasks)
        
    return results

def main():
    parser = argparse.ArgumentParser(description="vLLM Concurrency Benchmark Client")
    parser.add_argument("--prompts-file", type=str, default="experiments/kv_visibility_poc1/data/compiled_prompts.json")
    parser.add_argument("--tokenizer", type=str, default="Qwen/Qwen2.5-3B-Instruct")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--concurrency", type=int, default=1)
    parser.add_argument("--context-size", type=int, default=50)
    parser.add_argument("--mode", type=str, default="predictor_cached")
    parser.add_argument("--output", type=str, default="experiments/kv_visibility_poc1/data/benchmark_results.json")
    parser.add_argument("--max-new-tokens", type=int, default=64)
    args = parser.parse_args()
    
    # Load prompts
    with open(args.prompts_file, "r") as f:
        prompts_all = json.load(f)
        
    # Filter prompts to benchmark subset
    filtered = [
        p for p in prompts_all 
        if p["context_size_blocks"] == args.context_size and p["mode"] == args.mode
    ]
    
    if not filtered:
        print(f"No prompts found for context size {args.context_size} and mode {args.mode}")
        sys.exit(1)
        
    print(f"Running benchmark: Mode={args.mode}, ContextSize={args.context_size} blocks, Concurrency={args.concurrency}, Prompts={len(filtered)}")
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    
    url = f"http://localhost:{args.port}/v1/completions"
    
    # Run event loop
    loop = asyncio.get_event_loop()
    results = loop.run_until_complete(
        run_concurrency_batch(url, args.tokenizer, filtered, args.concurrency, tokenizer, args.max_new_tokens)
    )
    
    # Save results
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(results, f, indent=2)
        
    print(f"Benchmark complete. Results saved to: {args.output}")

if __name__ == "__main__":
    main()


