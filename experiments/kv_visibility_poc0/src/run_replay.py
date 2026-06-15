import torch
import time
from typing import Dict, Any, Tuple
from transformers import AutoModelForCausalLM, AutoTokenizer

def run_replay_inference(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    replay_prompts: Dict[str, Dict[str, Any]],
    device: str,
    max_new_tokens: int = 64
) -> Dict[str, Dict[str, Any]]:
    """
    Runs inference on all three replay prompts (visibility, random, bm25).
    """
    results = {}
    
    for mode in ["visibility", "random", "bm25"]:
        prompt_data = replay_prompts[mode]
        input_ids = prompt_data["token_ids"]
        prompt_len = len(input_ids)
        
        input_tensor = torch.tensor([input_ids], device=device)
        
        t0 = time.perf_counter()
        with torch.no_grad():
            outputs = model.generate(
                input_tensor,
                max_new_tokens=max_new_tokens,
                output_attentions=False,
                return_dict_in_generate=True,
                use_cache=True,
                pad_token_id=tokenizer.eos_token_id
            )
        duration_ms = (time.perf_counter() - t0) * 1000.0
        
        generated_ids = outputs.sequences[0][prompt_len:]
        answer = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
        
        results[mode] = {
            "answer": answer,
            "ttft_ms": duration_ms,  # For short generations without intermediate measurements, this serves as total time / TTFT proxy
            "input_tokens": prompt_len,
            "output_tokens": len(generated_ids),
            "kept_ids": prompt_data["ids"]
        }
        
    return results


