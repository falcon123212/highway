import os
import json
import argparse
import random

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--n-extractive", type=int, default=125)
    parser.add_argument("--n-not-found", type=int, default=75)
    parser.add_argument("--n-suffix-conflict", type=int, default=75)
    parser.add_argument("--n-llm-synthesis", type=int, default=150)
    parser.add_argument("--n-long-context", type=int, default=50)
    parser.add_argument("--n-cache-replay", type=int, default=25)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)

    # Load qa_gold.json
    # Note that qa_gold.json is inside data/corpus_poc2/questions/qa_gold.json
    # args.corpus is "data/corpus_poc2/index", so qa_gold.json is at "data/corpus_poc2/questions/qa_gold.json"
    corpus_dir = os.path.dirname(args.corpus.rstrip("/\\"))
    qa_path = os.path.join(corpus_dir, "questions", "qa_gold.json")

    if not os.path.exists(qa_path):
        print(f"Error: {qa_path} not found.")
        return

    with open(qa_path, "r", encoding="utf-8") as f:
        qa_data = json.load(f)

    # Group by categories
    extractive_pool = [q for q in qa_data if q["category"] in ["A", "B"]]
    not_found_pool = [q for q in qa_data if q["category"] == "E"]
    suffix_conflict_pool = [q for q in qa_data if q["category"] == "F"]
    llm_synthesis_pool = [q for q in qa_data if q["category"] in ["C", "G", "H"]]
    long_context_pool = [q for q in qa_data if q["category"] == "D"]

    # Sampling function
    def sample_pool(pool, n):
        if not pool:
            return []
        # If we need more than available, we do sampling with replacement (choices)
        if len(pool) < n:
            return random.choices(pool, k=n)
        else:
            return random.sample(pool, k=n)

    selected_extractive = sample_pool(extractive_pool, args.n_extractive)
    selected_not_found = sample_pool(not_found_pool, args.n_not_found)
    selected_suffix_conflict = sample_pool(suffix_conflict_pool, args.n_suffix_conflict)
    selected_llm_synthesis = sample_pool(llm_synthesis_pool, args.n_llm_synthesis)
    selected_long_context = sample_pool(long_context_pool, args.n_long_context)

    # Combine the main workload (total = 475 queries)
    main_workload = (
        selected_extractive +
        selected_not_found +
        selected_suffix_conflict +
        selected_llm_synthesis +
        selected_long_context
    )
    
    # Shuffle main workload
    random.shuffle(main_workload)

    # Generate cache replays: repeat 25 queries from the main workload at the end of the workload
    # To test cache replay effectively, we pick 25 queries from the selected ones and append them.
    # We choose 25 queries from the main workload (or we could select 25 from selected_extractive/selected_llm_synthesis)
    cache_replay_base = random.sample(main_workload, k=args.n_cache_replay)
    
    # We mark the repeated ones as cache_replay for evaluation/tracking, but keep their question
    cache_replay = []
    for q in cache_replay_base:
        q_copy = q.copy()
        # Tag it so that the runner knows it's a replay and we can track it
        q_copy["is_replay"] = True
        cache_replay.append(q_copy)

    final_workload = main_workload + cache_replay

    # Write to JSONL
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        for q in final_workload:
            f.write(json.dumps(q, ensure_ascii=False) + "\n")

    print(f"Generated {len(final_workload)} queries under {args.output}")
    print(f"  - Extractive: {len(selected_extractive)}")
    print(f"  - NotFound: {len(selected_not_found)}")
    print(f"  - SuffixConflict: {len(selected_suffix_conflict)}")
    print(f"  - LLMSynthesis: {len(selected_llm_synthesis)}")
    print(f"  - LongContext: {len(selected_long_context)}")
    print(f"  - CacheReplay: {len(cache_replay)}")

if __name__ == "__main__":
    main()


