import json
import collections

failures_by_sub = collections.defaultdict(list)

with open("artifacts/runs/poc_2_3_4_kernel_hardening/results.jsonl", "r", encoding="utf-8") as f:
    for line in f:
        if line.strip():
            r = json.loads(line)
            if not r["is_em"]:
                cat = r["category"]
                meta = r.get("metadata", {})
                sub_type = meta.get("sub_type", "unknown")
                failures_by_sub[(cat, sub_type)].append({
                    "id": r["id"],
                    "question": r["question"],
                    "expected": r["expected"],
                    "generated": r["generated"],
                    "route": r["route"],
                    "audit": r["metrics"].get("kernel_audit", {})
                })

print("FAILURES SUMMARY:")
for key, list_f in sorted(failures_by_sub.items()):
    print(f"\nCategory {key[0]} Sub-type {key[1]} (Count={len(list_f)}):")
    # print first 3 examples
    for f in list_f[:3]:
        print(f"  ID: {f['id']}")
        print(f"  Question: {f['question']}")
        print(f"  Expected: {f['expected']}")
        print(f"  Generated: {f['generated']}")
        print(f"  Route: {f['route']}")
        print(f"  Audit: {f['audit']}")
        print("-" * 40)



