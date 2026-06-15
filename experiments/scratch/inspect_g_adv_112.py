import sys
sys.path.append("src")
sys.path.append("src")
from highway.runtime.scheduler import ExecutionScheduler

scheduler = ExecutionScheduler("data/corpus_poc2/index", "data/corpus_poc2/cache")
question = "Which project has a higher budget: Project VEGA or Project NEPTUNE?"
query_ir = scheduler.search_router.query_parser.parse(question)
print("Query IR:", query_ir)

# Search
evidence_pool, scores = scheduler.search_router.search(question, top_k=50)
print(f"Retrieved {len(evidence_pool)} blocks.")

for idx, b in enumerate(evidence_pool):
    if "adv_doc_0112.txt" in b["source_file"]:
        print(f"FOUND adv_doc_0112.txt in pool at rank {idx+1}: {b}")
    if "adv_doc_0172.txt" in b["source_file"]:
        print(f"FOUND adv_doc_0172.txt in pool at rank {idx+1}: {b}")

active, suppressed, forbidden = scheduler.evidence_resolver.resolve(evidence_pool, query_ir)
print(f"Active count: {len(active)}")
for b in active:
    if "adv_doc_0112.txt" in b["source_file"]:
        print("FOUND adv_doc_0112.txt in active!")
    if "adv_doc_0172.txt" in b["source_file"]:
        print("FOUND adv_doc_0172.txt in active!")



