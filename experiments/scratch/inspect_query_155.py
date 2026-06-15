import json

with open("data/workloads/gh_only_116.jsonl", "r") as f:
    for line in f:
        data = json.loads(line)
        if data["id"] == "q_155":
            print(json.dumps(data, indent=2))
            break



