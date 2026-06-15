import sys
sys.path.append("src")

from highway.retrieval.search import SearchRouter
from highway.retrieval.evidence_resolver import EvidenceResolver
from highway.retrieval.ir_builder import IRBuilder

router = SearchRouter("data/corpus_poc2/index")
resolver = EvidenceResolver()
ir_builder = IRBuilder()

question = "What is the approved budget of Project HELIOS?"
candidates, query_ir = router.search(question, top_k=50)
print("Query IR:", query_ir)
print(f"\nRetrieved {len(candidates)} candidates.")

active, suppressed, forbidden = resolver.resolve(candidates, query_ir)
print(f"\nActive evidence ({len(active)}):")
for b in active:
    print(f" - {b['block_id']} in {b['source_file']}: {repr(b['text'])}")

print(f"\nSuppressed evidence ({len(suppressed)}):")
for b in suppressed:
    print(f" - {b['block_id']} in {b['source_file']} ({b.get('suppression_reason')}): {repr(b['text'])}")

print("\nForbidden matches:", forbidden)

ir = ir_builder.build_ir(query_ir, active, suppressed, forbidden)
print("\nIR guard decision:", ir["guard_decision"])



