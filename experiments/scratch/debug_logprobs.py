import sys
import json
import urllib.request

sys.path.append("src")

from highway.retrieval.search import SearchRouter
from highway.retrieval.evidence_resolver import EvidenceResolver
from highway.retrieval.ir_builder import IRBuilder
from highway.runtime.compiler import ContextCompiler

qa_path = "data/corpus_poc2/questions/qa_gold.json"
index_dir = "data/corpus_poc2/index"

with open(qa_path, "r", encoding="utf-8") as f:
    qa_pairs = json.load(f)

router = SearchRouter(index_dir)
resolver = EvidenceResolver()
ir_builder = IRBuilder()
compiler = ContextCompiler()

# Get q_002
q = next(item for item in qa_pairs if item["id"] == "q_060")

candidates, query_ir = router.search(q["question"], top_k=50)
active, suppressed, forbidden = resolver.resolve(candidates, query_ir)
ir = ir_builder.build_ir(query_ir, active, suppressed, forbidden)
prompt = compiler.compile(ir)

# We call the model API with logprobs enabled!
# First we need to make sure the server is running.
# Wait, let's write the code to send request to localhost:8000
vllm_url = "http://localhost:8000/v1/completions"
data = {
    "model": "Qwen/Qwen2.5-0.5B-Instruct",
    "prompt": prompt,
    "max_tokens": 10,
    "temperature": 0.0,
    "logprobs": 5,
    "repetition_penalty": 1.0
}
headers = {"Content-Type": "application/json"}
try:
    req = urllib.request.Request(vllm_url, data=json.dumps(data).encode("utf-8"), headers=headers)
    with urllib.request.urlopen(req, timeout=10) as response:
        res = json.loads(response.read().decode("utf-8"))
        print("\n--- MODEL RESPONSE ---")
        print(repr(res["choices"][0]["text"]))
        print("\n--- LOGPROBS FOR FIRST 5 TOKENS ---")
        logprobs_list = res["choices"][0].get("logprobs", {}).get("top_logprobs", [])
        for idx, token_dict in enumerate(logprobs_list[:5]):
            print(f"Token {idx}: {token_dict}")
except Exception as e:
    print(f"Failed to query server: {e}")



