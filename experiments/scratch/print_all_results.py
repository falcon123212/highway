import json
with open('artifacts/runs/poc_2_3_2_model_sweep/results_qwen_1_5b.jsonl') as f:
    for line in f:
        r = json.loads(line)
        print(f"ID: {r['id']} ({r['category']})")
        print(f"Question: {r['question']}")
        print(f"Expected: {r['expected']}")
        print(f"Generated: {r['generated']}")
        # print metrics/reasoning if available
        # wait, the generated JSON is parsed in scheduler.py, does it store the full response including reasoning?
        # let's check
        if 'metrics' in r:
            pass
        print("-" * 50)



