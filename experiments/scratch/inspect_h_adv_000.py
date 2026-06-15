import sys
sys.path.append("src")
sys.path.append("src")
from highway.runtime.scheduler import ExecutionScheduler
import json

scheduler = ExecutionScheduler("data/corpus_poc2/index", "data/corpus_poc2/cache")
question = "List all project names managed by Jean Dupont."
query_ir = scheduler.search_router.query_parser.parse(question)
print("Query IR:", query_ir)

# Search
evidence_pool, scores = scheduler.search_router.search(question, top_k=50)
print(f"Retrieved {len(evidence_pool)} blocks.")

# Check if adv_doc_0200.txt is in the pool
found_pool = False
for idx, b in enumerate(evidence_pool):
    if "adv_doc_0200.txt" in b["source_file"]:
        print(f"FOUND adv_doc_0200.txt in evidence pool at rank {idx+1}: {b}")
        found_pool = True
if not found_pool:
    print("adv_doc_0200.txt NOT found in evidence pool!")

# Resolve
active, suppressed, forbidden = scheduler.evidence_resolver.resolve(evidence_pool, query_ir)
print(f"Active evidence count: {len(active)}")
found_active = False
for b in active:
    if "adv_doc_0200.txt" in b["source_file"]:
        print(f"FOUND adv_doc_0200.txt in active evidence: {b}")
        found_active = True
if not found_active:
    print("adv_doc_0200.txt NOT found in active evidence!")
    
# Let's see if it was suppressed
for b in suppressed:
    if "adv_doc_0200.txt" in b["source_file"]:
        print(f"FOUND adv_doc_0200.txt in suppressed evidence: {b}")



