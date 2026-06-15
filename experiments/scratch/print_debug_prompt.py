import sys
import json

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

# Pick specific questions
target_ids = ["q_050"]

for q in qa_pairs:
    if q["id"] in target_ids:
        print(f"\n=========================================")
        print(f"QUESTION ID: {q['id']}")
        print(f"QUESTION: {q['question']}")
        print(f"EXPECTED: {q['expected_answer']}")
        print(f"=========================================")
        
        candidates, query_ir = router.search(q["question"], top_k=50)
        active, suppressed, forbidden = resolver.resolve(candidates, query_ir)
        ir = ir_builder.build_ir(query_ir, active, suppressed, forbidden)
        prompt = compiler.compile(ir)
        print(prompt)



