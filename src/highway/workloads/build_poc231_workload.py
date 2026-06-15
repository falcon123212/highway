import os
import json
import argparse
import random

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)

    corpus_dir = os.path.dirname(args.corpus.rstrip("/\\"))
    qa_path = os.path.join(corpus_dir, "questions", "qa_gold.json")

    if not os.path.exists(qa_path):
        print(f"Error: {qa_path} not found.")
        return

    with open(qa_path, "r", encoding="utf-8") as f:
        qa_data = json.load(f)

    # Group by category
    by_cat = {}
    for q in qa_data:
        cat = q["category"]
        by_cat.setdefault(cat, []).append(q)

    print("Available queries by category:")
    for cat in sorted(by_cat.keys()):
        print(f"  Category {cat}: {len(by_cat[cat])}")

    # Select all G and H queries
    selected_g = by_cat.get("G", [])
    selected_h = by_cat.get("H", [])
    
    # Sample from others
    # Total G + H = 71 + 45 = 116 queries
    # We need 84 more queries to make exactly 200
    # Let's allocate:
    # A: 10, B: 10, C: 20, D: 15, E: 15, F: 14 -> 84 queries
    n_allocations = {
        "A": 10,
        "B": 10,
        "C": 20,
        "D": 15,
        "E": 15,
        "F": 14
    }

    other_selected = []
    for cat, count in n_allocations.items():
        pool = by_cat.get(cat, [])
        if len(pool) < count:
            sampled = random.choices(pool, k=count)
        else:
            sampled = random.sample(pool, k=count)
        other_selected.extend(sampled)

    final_workload = selected_g + selected_h + other_selected
    random.shuffle(final_workload)

    # Write to JSONL
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        for q in final_workload:
            f.write(json.dumps(q, ensure_ascii=False) + "\n")

    print(f"Generated focused workload with {len(final_workload)} queries under {args.output}:")
    print(f"  - Category G: {len(selected_g)}")
    print(f"  - Category H: {len(selected_h)}")
    for cat, count in n_allocations.items():
        print(f"  - Category {cat}: {count}")

if __name__ == "__main__":
    main()


