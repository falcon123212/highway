import torch
from typing import List, Dict, Any, Tuple

def extract_block_attentions(
    attentions: Tuple[Tuple[torch.Tensor]],
    blocks: List[Dict[str, Any]],
    prompt_len: int
) -> Dict[int, float]:
    """
    Extracts attention weights, slices them to only cover the prompt tokens,
    averages over heads, layers, and generation steps, and aggregates per block.
    
    Args:
        attentions: tuple of tuples of tensors.
                   Outer tuple: generation steps.
                   Inner tuple: layers.
                   Tensor shape: (batch_size, num_heads, 1, current_seq_len)
        blocks: list of block dicts containing token_start and token_end.
        prompt_len: length of the input prompt.
        
    Returns:
        Dict[block_id -> normalized attention mass]
    """
    # Number of steps and layers
    num_steps = len(attentions)
    num_layers = len(attentions[0])
    
    # Initialize accumulated attention for prompt tokens
    accumulated_attention = torch.zeros(prompt_len, dtype=torch.float32)
    
    for step_idx in range(num_steps):
        # Tuple of layers for this step
        step_attentions = attentions[step_idx]
        
        step_acc = torch.zeros(prompt_len, dtype=torch.float32)
        for layer_idx in range(num_layers):
            # Shape: (batch_size, num_heads, 1, current_seq_len)
            layer_attn = step_attentions[layer_idx]
            
            # Slice to only look at the prompt tokens
            # If current_seq_len is smaller than prompt_len (should not happen, but safety check)
            curr_len = layer_attn.shape[-1]
            slice_end = min(prompt_len, curr_len)
            
            # Extract and squeeze batch and query dimensions
            # layer_attn[0] shape: (num_heads, 1, current_seq_len)
            # We select query index 0 (the only query token) and slice prompt tokens
            attn_prompt = layer_attn[0, :, 0, :slice_end].float() # (num_heads, slice_end)
            
            # Average over heads
            attn_mean_heads = attn_prompt.mean(dim=0) # (slice_end,)
            
            # Accumulate across layers
            if slice_end < prompt_len:
                step_acc[:slice_end] += attn_mean_heads
            else:
                step_acc += attn_mean_heads
                
        # Average step_acc over layers
        step_acc /= num_layers
        
        # Accumulate over generation steps
        accumulated_attention += step_acc
        
    # Average over generation steps
    accumulated_attention /= num_steps
    
    # Now aggregate by block
    block_attention_mass = {}
    context_total_mass = 0.0
    
    for block in blocks:
        block_id = block["block_id"]
        start = block["token_start"]
        end = block["token_end"]
        
        # Sum attention weights in this block's token range
        if start < prompt_len:
            actual_end = min(end, prompt_len)
            mass = accumulated_attention[start:actual_end].sum().item()
        else:
            mass = 0.0
            
        block_attention_mass[block_id] = mass
        context_total_mass += mass
        
    # Normalize attention mass over context blocks so they sum to 1.0 (or keep raw fraction)
    # The spec shows: "Block 17: 38.4% ... Block 63: 9.8%" - which are normalized / raw percentages.
    # We will normalize over the context blocks to see the relative distribution.
    if context_total_mass > 0:
        for block_id in block_attention_mass:
            block_attention_mass[block_id] /= context_total_mass
            
    return block_attention_mass


