import json
with open('artifacts/runs/poc_2_3_2_model_sweep/results_qwen_1_5b.jsonl') as f:
    for line in f:
        r = json.loads(line)
        if r['category'] == 'G':
            print(f"{r['id']}: {r['question']}\nExpected: {r['expected']}\nGot: {r['generated']}\nIsEM: {r['is_em']}\n")



