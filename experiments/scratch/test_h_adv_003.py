import sys
sys.path.append("src")
sys.path.append("src")
from highway.runtime.scheduler import ExecutionScheduler
from highway.kernels.compute_kernels import AggregationKernel

scheduler = ExecutionScheduler("data/corpus_poc2/index", "data/corpus_poc2/cache")
question = "List all project names managed by Alice Martin."
query_ir = scheduler.search_router.query_parser.parse(question)

evidence_pool, _ = scheduler.search_router.search(question, top_k=50)
active, suppressed, forbidden = scheduler.evidence_resolver.resolve(evidence_pool, query_ir)

# Find adv_doc_0203 block
block = None
for b in active:
    if "adv_doc_0203.txt" in b["source_file"]:
        block = b
        break

print("Block text:")
print(block["text"])

kernel = AggregationKernel()
text_folded = kernel._fold_accents(block["text"])
canonical_manager = "Alice Martin"

print("Confirms management in block:", kernel._confirms_management(text_folded, canonical_manager))

for proj in scheduler.ir_builder.project_names:
    if proj in ["HELIOS", "IRIS", "KRONOS", "LUNA", "METEOR"]:
        print(f"\nChecking project: {proj}")
        lines = block["text"].split("\n")
        for line in lines:
            val = kernel._confirms_management_in_line(line, canonical_manager, proj)
            print(f"  Line: '{line}' -> confirms: {val}")



