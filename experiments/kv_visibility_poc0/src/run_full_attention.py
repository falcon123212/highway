import torch
import time
from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import Dict, Any, List, Tuple

class AttentionHook:
    """
    Forward hook that intercepts attention weights, aggregates them on the fly
    across layers and heads for the last query token (which predicts the next token),
    and then replaces the raw weights with None to free memory immediately.
    """
    def __init__(self, prompt_len: int, num_layers: int):
        self.prompt_len = prompt_len
        self.num_layers = num_layers
        self.accumulated_attention = torch.zeros(prompt_len, dtype=torch.float32)
        self.call_count = 0
        
    def __call__(self, module, input, output):
        # output is a tuple: (attn_output, attn_weights)
        if isinstance(output, tuple) and len(output) >= 2:
            attn_output, attn_weights = output[0], output[1]
            
            if attn_weights is not None:
                # attn_weights shape: (batch_size, num_heads, query_len, key_len)
                k_len = attn_weights.shape[-1]
                slice_end = min(self.prompt_len, k_len)
                
                # Extract attention of the last query token to all prompt key tokens
                # Shape: (num_heads, slice_end)
                attn_slice = attn_weights[0, :, -1, :slice_end].detach().cpu().float()
                
                # Average over heads
                attn_mean_heads = attn_slice.mean(dim=0) # (slice_end,)
                
                # Accumulate
                self.accumulated_attention[:slice_end] += attn_mean_heads
                self.call_count += 1
                
            # Replace attn_weights with None to prevent memory leaks/accumulation
            return attn_output, None
            
        return output

def load_model_and_tokenizer(model_name: str, device: str, attn_implementation: str = "sdpa") -> Tuple[AutoModelForCausalLM, AutoTokenizer]:
    """Loads the causal LM model and tokenizer."""
    print(f"Loading tokenizer for {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    
    print(f"Loading model {model_name} on {device} using {attn_implementation}...")
    
    # Configure precision and implementation
    if "cuda" in device:
        torch_dtype = torch.float16
        device_map = device
    else:
        torch_dtype = torch.float32
        device_map = "cpu"
        
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch_dtype,
        attn_implementation=attn_implementation,
        device_map=device_map,
        trust_remote_code=True
    )
    model.eval()
    
    return model, tokenizer

def run_full_attention_inference(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    tokenized_sample: Dict[str, Any],
    device: str,
    max_new_tokens: int = 64
) -> Dict[str, Any]:
    """
    Runs inference on the full context and returns the generated answer,
    the block attention scores, the TTFT, and other metrics.
    """
    input_ids = tokenized_sample["full_input_ids"]
    input_tensor = torch.tensor([input_ids], device=device)
    prompt_len = len(input_ids)
    
    # Identify number of layers
    num_layers = getattr(model.config, "num_hidden_layers", 24)
    
    # Setup the hook to intercept attentions
    hook_obj = AttentionHook(prompt_len=prompt_len, num_layers=num_layers)
    
    # Register the hook on all Attention modules
    hook_handles = []
    for name, module in model.named_modules():
        if module.__class__.__name__.endswith("Attention"):
            hook_handles.append(module.register_forward_hook(hook_obj))
            
    # Measure TTFT (Time to First Token)
    t0 = time.perf_counter()
    with torch.no_grad():
        outputs = model.generate(
            input_tensor,
            max_new_tokens=max_new_tokens,
            output_attentions=True,
            return_dict_in_generate=True,
            use_cache=True,
            pad_token_id=tokenizer.eos_token_id
        )
    ttft = (time.perf_counter() - t0) * 1000.0 # ms
    
    # Remove the hooks immediately
    for handle in hook_handles:
        handle.remove()
        
    generated_ids = outputs.sequences[0][prompt_len:]
    answer = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
    
    # Normalize accumulated attention scores
    # Number of steps generated is hook_obj.call_count / num_layers
    num_steps = hook_obj.call_count / num_layers if hook_obj.call_count > 0 else 1
    final_attention = hook_obj.accumulated_attention / num_steps
    
    # Aggregate attention per block
    block_scores = {}
    context_total_mass = 0.0
    
    for block in tokenized_sample["blocks"]:
        block_id = block["block_id"]
        start = block["token_start"]
        end = block["token_end"]
        
        if start < prompt_len:
            actual_end = min(end, prompt_len)
            mass = final_attention[start:actual_end].sum().item()
        else:
            mass = 0.0
            
        block_scores[block_id] = mass
        context_total_mass += mass
        
    # Normalize over context blocks
    if context_total_mass > 0:
        for block_id in block_scores:
            block_scores[block_id] /= context_total_mass
            
    return {
        "question_id": tokenized_sample["question_id"],
        "category": tokenized_sample["category"],
        "answer": answer,
        "block_scores": block_scores,
        "ttft_ms": ttft,
        "input_tokens": prompt_len,
        "output_tokens": len(generated_ids)
    }


