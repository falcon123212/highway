import pandas as pd

df = pd.read_csv("reports/poc03b_results.csv")
df50 = df[df["context_blocks"] == 50]
print("POC 0.3b Category Stats (Context Blocks = 50):")
print(df50.groupby(["mode", "category"]).agg({
    "exact_match": "mean",
    "numeric_preservation": "mean",
    "gold_recall": "mean"
}))


