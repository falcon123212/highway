import sys
sys.path.append("src")

from highway.runtime.scheduler import ExecutionScheduler
import json

scheduler = ExecutionScheduler("data/corpus_poc2/index", "data/corpus_poc2/cache", vllm_port=8000, model_name="Qwen/Qwen2.5-1.5B-Instruct")

with open("data/workloads/gh_only_116.jsonl", "r") as f:
    for line in f:
        data = json.loads(line)
        if data["id"] == "q_137":
            question = data["question"]
            break

# Search router
candidates, query_ir = scheduler.search_router.search(question, top_k=50)
active, suppressed, forbidden = scheduler.evidence_resolver.resolve(candidates, query_ir)
proof_ir = scheduler.ir_builder.build_ir(query_ir, active, suppressed, forbidden)
prompt = scheduler.compiler.compile(proof_ir)

# We can start vLLM or call the LLM if the server is running. But server is killed.
# Let's just print the compiled prompt to inspect it.
print("--- PROMPT ---")
print(prompt)



