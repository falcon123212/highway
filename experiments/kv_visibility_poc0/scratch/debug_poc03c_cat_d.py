import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import json
import torch
import random
from transformers import AutoTokenizer, AutoModelForCausalLM
from typing import List, Dict, Any
from run_poc03b import scale_sample, run_manual_inference, parse_json_response_b

def assemble_prompt_new(kept_ids: List[int], question: str, documents: List[Dict[str, Any]]) -> str:
    system_text = (
        "<|im_start|>system\n"
        "You are a helpful assistant. Answer the question based on the provided context.\n"
        "You MUST respond with a strict JSON object containing the keys 'answer', 'evidence_block_id', and 'evidence_quote'.\n"
        "Rules:\n"
        "- Copy numbers, dates, IDs exactly from the evidence.\n"
        "- Do not paraphrase numeric values.\n"
        "- Do not explain.\n"
        "- Match the requested project name EXACTLY. Do not answer using a project that has an extra suffix. "
        "For example, if asked for 'Project X', do NOT match it to 'Project X-Legacy', 'Project X-A', or 'Project X-B'. Only match 'Project X' exactly.\n"
        "- If the answer is a date, return only the date string from the evidence.\n"
        "- If the answer is a budget, return only the exact budget value from the evidence.\n"
        "- If the question asks for both a date and a budget, return both formatted exactly as 'DATE and BUDGET' (for example: '15 May 2027 and $150,000').\n"
        "Output ONLY the raw JSON block. Do not include markdown code block formatting or explanation.\n"
        "Example format:\n"
        '{\n  "answer": "15 May 2027",\n  "evidence_block_id": "DOC_0012",\n  "evidence_quote": "Project: X Active delivery date: 15 May 2027"\n}\n'
        "<|im_end|>\n"
        "<|im_start|>user\nContext:\n"
    )
    context_parts = []
    last_id = -2
    for idx in kept_ids:
        if last_id != -2 and idx != last_id + 1:
            context_parts.append("[...]")
        context_parts.append(documents[idx]["text"])
        last_id = idx
    context_text = "\n\n".join(context_parts)
    question_text = f"\n\nQuestion: {question}<|im_end|>\n<|im_start|>assistant\n"
    return system_text + context_text + question_text

device = "cuda" if torch.cuda.is_available() else "cpu"
model_name = "Qwen/Qwen2.5-3B-Instruct"

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16, device_map=device)
model.eval()

# Load corpus and answers
with open("data/corpus.jsonl", "r") as f:
    corpus = [json.loads(line) for line in f]
    
with open("data/answers.jsonl", "r") as f:
    answers = {json.loads(line)["question_id"]: json.loads(line) for line in f}

# Find a category D sample
sample = next(s for s in corpus if s["category"] == "D")
q_id = sample["question_id"]
gold = answers[q_id]

print(f"Sample Q_ID: {q_id}")
print(f"Question: {sample['question']}")
print(f"Expected: {gold['expected_answer']}")
print(f"Gold block IDs: {gold['gold_block_ids']}")

# Scale to 50 blocks
scaled = scale_sample(sample, 50, seed=42)
documents = scaled["documents"]

# In POC 0.3c-mini, new predictor kept the gold blocks because Gold Recall was 100%.
gold_block_indices = sorted(list(set(gold["gold_block_ids"])))
print(f"Gold block indices: {gold_block_indices}")

prompt = assemble_prompt_new(gold_block_indices, scaled["question"], documents)
print("\n--- Prompt ---")
print(prompt)
print("--------------\n")

# Run inference
input_ids = tokenizer.encode(prompt)
prefill_ms, decode_ms, answer, gen_len = run_manual_inference(
    model, tokenizer, input_ids, device, max_new_tokens=64
)

print("\n--- Raw Answer ---")
print(answer)
print("------------------\n")

parsed = parse_json_response_b(answer)
print("\nParsed:", parsed)


