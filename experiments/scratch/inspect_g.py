import sys
sys.path.append("src")
import json
from highway.retrieval.search import SearchRouter
from highway.retrieval.evidence_resolver import EvidenceResolver
from highway.retrieval.ir_builder import IRBuilder
from highway.runtime.compiler import ContextCompiler

router = SearchRouter("data/corpus_poc2/index")
resolver = EvidenceResolver()
ir_builder = IRBuilder()
compiler = ContextCompiler()

with open("data/corpus_poc2/questions/qa_gold.json", "r", encoding="utf-8") as f:
    qa_pairs = json.load(f)

# Find q_002
q = next(item for item in qa_pairs if item["id"] == "q_002")
print(f"Question: {q['question']}")

candidates, query_ir = router.search(q["question"], top_k=50)
active, suppressed, forbidden = resolver.resolve(candidates, query_ir)
ir = ir_builder.build_ir(query_ir, active, suppressed, forbidden)
prompt_text = compiler.compile(ir)

print("=== COMPILED PROMPT ===")
print(prompt_text)



