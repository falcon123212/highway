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

    # Filter G and H queries
    gh_queries = [q for q in qa_data if q["category"] in ["G", "H"]]
    
    # Shuffle for benchmark randomness
    random.shuffle(gh_queries)

    # Write to JSONL
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        for q in gh_queries:
            f.write(json.dumps(q, ensure_ascii=False) + "\n")

    print(f"Generated G/H-only workload with {len(gh_queries)} queries under {args.output}:")
    print(f"  - Category G: {len([q for q in gh_queries if q['category'] == 'G'])}")
    print(f"  - Category H: {len([q for q in gh_queries if q['category'] == 'H'])}")

if __name__ == "__main__":
    main()


