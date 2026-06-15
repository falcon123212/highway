import torch
from transformers import AutoTokenizer
from typing import List, Dict, Any

def tokenize_and_map_blocks(
    tokenizer: AutoTokenizer,
    sample: Dict[str, Any],
    gold_info: Dict[str, Any],
    block_size: int = 128
) -> Dict[str, Any]:
    """
    Tokenizes the sample's documents and structures them into blocks of exactly `block_size` tokens.
    Each document is tokenized separately and padded/truncated to `block_size` tokens.
    This ensures that Block i corresponds exactly to Document i.
    """
    # System instruction template for Qwen
    # We use the official Qwen chat template style but construct it manually for precise token indexing.
    # Qwen format: <|im_start|>system\n{system_content}<|im_end|>\n<|im_start|>user\nContext:\n
    system_text = "<|im_start|>system\nYou are a helpful assistant. Answer the question based on the provided context. Be concise and precise.<|im_end|>\n<|im_start|>user\nContext:\n"
    system_ids = tokenizer.encode(system_text)
    
    # We will pad with newline token
    newline_token_id = tokenizer.encode("\n")[0]
    
    blocks = []
    context_token_ids = []
    
    current_token_offset = len(system_ids)
    
    for i, doc in enumerate(sample["documents"]):
        doc_text = doc["text"]
        doc_tokens = tokenizer.encode(doc_text)
        
        # Pad or truncate to block_size
        if len(doc_tokens) < block_size:
            padded_tokens = doc_tokens + [newline_token_id] * (block_size - len(doc_tokens))
        else:
            padded_tokens = doc_tokens[:block_size]
            
        context_token_ids.extend(padded_tokens)
        
        # Determine if this block is gold or deprecated
        contains_gold = i in gold_info["gold_block_ids"]
        contains_dep = i in gold_info["deprecated_block_ids"]
        
        blocks.append({
            "block_id": i,
            "token_start": current_token_offset,
            "token_end": current_token_offset + block_size,
            "contains_gold_fact": contains_gold,
            "contains_deprecated_fact": contains_dep,
            "text": doc_text
        })
        
        current_token_offset += block_size
        
    # Question text template:
    # \n\nQuestion: {question}<|im_end|>\n<|im_start|>assistant\n
    question_text = f"\n\nQuestion: {sample['question']}<|im_end|>\n<|im_start|>assistant\n"
    question_ids = tokenizer.encode(question_text)
    
    full_input_ids = system_ids + context_token_ids + question_ids
    
    return {
        "question_id": sample["question_id"],
        "category": sample["category"],
        "full_input_ids": full_input_ids,
        "blocks": blocks,
        "system_len": len(system_ids),
        "context_len": len(context_token_ids),
        "question_len": len(question_ids),
        "question_text": sample["question_id"]
    }


