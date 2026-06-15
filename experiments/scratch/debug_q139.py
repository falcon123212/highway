import sys
sys.path.append("src")

from highway.retrieval.search import SearchRouter
from highway.retrieval.evidence_resolver import EvidenceResolver
from highway.retrieval.ir_builder import IRBuilder

router = SearchRouter("data/corpus_poc2/index")
resolver = EvidenceResolver()
ir_builder = IRBuilder()

question = "Which project has a higher budget: Project NEXUS or Project QUASAR?"
candidates, query_ir = router.search(question, top_k=50)

print("Target entities:", query_ir["target_entities"])

active, suppressed, forbidden = resolver.resolve(candidates, query_ir)
print(f"\nActive evidence blocks ({len(active)}):")
for b in active:
    print(f" - {b['block_id']} | File: {b['source_file']}")
    print(f"   is_amendment: {b.get('is_amendment')} | is_base: {b.get('is_base')} | Date: {b.get('parsed_date')}")
    print(f"   Text: {repr(b['text'])}")

ir = ir_builder.build_ir(query_ir, active, suppressed, forbidden)
print("\nGuard decision:", ir["guard_decision"])



