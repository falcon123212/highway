import torch
import time
from transformers import AutoModelForCausalLM, AutoTokenizer
from src.run_full_attention import load_model_and_tokenizer, AttentionHook
from src.tokenize_blocks import tokenize_and_map_blocks
import json

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, tokenizer = load_model_and_tokenizer("Qwen/Qwen2.5-0.5B-Instruct", device)
    
    # Load one sample
    sample = json.loads(open("data/corpus.jsonl").readline())
    gold_info = json.loads(open("data/answers.jsonl").readline())
    
    tokenized_sample = tokenize_and_map_blocks(tokenizer, sample, gold_info, block_size=128)
    input_ids = tokenized_sample["full_input_ids"]
    input_tensor = torch.tensor([input_ids], device=device)
    prompt_len = len(input_ids)
    num_layers = getattr(model.config, "num_hidden_layers", 24)
    
    print("\n--- Testing model.generate(max_new_tokens=1) ---")
    hook_obj1 = AttentionHook(prompt_len, num_layers)
    handles1 = []
    for name, module in model.named_modules():
        if module.__class__.__name__.endswith("Attention"):
            handles1.append(module.register_forward_hook(hook_obj1))
            
    t0 = time.perf_counter()
    with torch.no_grad():
        model.generate(
            input_tensor,
            max_new_tokens=1,
            output_attentions=True,
            return_dict_in_generate=True,
            use_cache=True,
            pad_token_id=tokenizer.eos_token_id
        )
    print(f"Time: {(time.perf_counter() - t0)*1000.0:.2f} ms")
    for h in handles1: h.remove()
    
    print("\n--- Testing model(input_tensor, output_attentions=True) ---")
    hook_obj2 = AttentionHook(prompt_len, num_layers)
    handles2 = []
    for name, module in model.named_modules():
        if module.__class__.__name__.endswith("Attention"):
            handles2.append(module.register_forward_hook(hook_obj2))
            
    t0 = time.perf_counter()
    with torch.no_grad():
        model(input_tensor, output_attentions=True)
    print(f"Time: {(time.perf_counter() - t0)*1000.0:.2f} ms")
    for h in handles2: h.remove()

if __name__ == "__main__":
    main()


