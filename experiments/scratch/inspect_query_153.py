import sys
sys.path.append("src")

from highway.runtime.scheduler import ExecutionScheduler
import json

scheduler = ExecutionScheduler("data/corpus_poc2/index", "data/corpus_poc2/cache", vllm_port=8000, model_name="Qwen/Qwen2.5-1.5B-Instruct")

with open("data/workloads/gh_only_116.jsonl", "r") as f:
    for line in f:
        data = json.loads(line)
        if data["id"] == "q_153":
            question = data["question"]
            break

candidates, query_ir = scheduler.search_router.search(question, top_k=50)
print(f"Number of retrieved candidates: {len(candidates)}")

active, suppressed, forbidden = scheduler.evidence_resolver.resolve(candidates, query_ir)
print(f"Number of active evidence blocks: {len(active)}")
for idx, ev in enumerate(active):
    print(f"Block {idx+1} [SOURCE: {ev['source_file']}]: {ev['text']}")



