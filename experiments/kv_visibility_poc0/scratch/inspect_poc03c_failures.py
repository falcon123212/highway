import pandas as pd

df = pd.read_csv("reports/poc03c_mini_results.csv")
print("Total rows:", len(df))

# Filter to where mode is new_predictor
df_new = df[df["mode"] == "new_predictor"]
print("\nNew Predictor stats:")
print(df_new.groupby("category").agg({
    "exact_match": "mean",
    "numeric_preservation": "mean",
    "gold_recall": "mean"
}))

# Show failures (where exact_match is False)
print("\nFailures:")
# We need to load answers.jsonl or answers key to show details.
import json
gold_answers = {}
with open("data/answers.jsonl", "r") as f:
    for line in f:
        item = json.loads(line)
        gold_answers[item["question_id"]] = item

for idx, row in df_new[df_new["exact_match"] == False].iterrows():
    q_id = row["question_id"]
    gold = gold_answers[q_id]
    print(f"Q_ID: {q_id} | Cat: {row['category']}")
    print(f"  Expected: {gold['expected_answer']}")
    print(f"  Gold doc IDs: {gold['gold_block_ids']}")
    print(f"  Kept blocks: {row['kept_blocks']}")
    # Let's print generated answer if possible, but results.csv doesn't save generated answer text?
    # Let's check what columns are in results.csv
    # The columns are: question_id, category, mode, oom, exact_match, numeric_preservation, gold_recall, suffix_error, contradiction_accuracy, multi_fact_recall, kept_blocks, token_reduction_pct, selector_latency_ms, ttft_ms
    # Oh, results.csv doesn't save the generated answer text. But we can print the question text and expected answer.
    print(f"  Question: {gold.get('question', '')}")


