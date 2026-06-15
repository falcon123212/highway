import re
import string
from typing import Dict, Any, List, Set

def normalize_text(text: str) -> str:
    """Normalize text by lowercasing, removing punctuation, and stripping whitespace."""
    if not text:
        return ""
    text = text.lower()
    text = "".join(c for c in text if c not in string.punctuation)
    # Remove common articles
    words = text.split()
    filtered_words = [w for w in words if w not in ["a", "an", "the"]]
    return " ".join(filtered_words)

def extract_numbers(text: str) -> List[str]:
    """Extract sequences of digits from text."""
    return re.findall(r'\d+', text)

def compute_exact_match(expected: str, generated: str) -> bool:
    """Check if the expected answer matches the generated answer (allowing substring match)."""
    norm_expected = normalize_text(expected)
    norm_generated = normalize_text(generated)
    
    if not norm_expected:
        return False
        
    # If they are exactly equal after normalization
    if norm_expected == norm_generated:
        return True
        
    # If expected is contained in generated and the generated isn't excessively long
    # (to prevent a model that dumps context from getting 100% match)
    if norm_expected in norm_generated and len(norm_generated) < len(norm_expected) * 4 + 30:
        return True
        
    return False

def compute_numeric_preservation(expected: str, generated: str) -> bool:
    """Check if all digit sequences in the expected answer are present in the generated answer."""
    expected_nums = extract_numbers(expected)
    generated_nums = extract_numbers(generated)
    
    if not expected_nums:
        return True # No numbers to preserve
        
    # Check if every expected number is present in the generated numbers list
    for num in expected_nums:
        if num not in generated_nums:
            return False
    return True

def estimate_kv_bytes(
    tokens: int,
    num_layers: int,
    num_kv_heads: int,
    head_dim: int,
    bytes_per_element: int = 2
) -> int:
    """
    KV bytes = tokens * layers * kv_heads * head_dim * 2 * bytes_per_element
    """
    return tokens * num_layers * num_kv_heads * head_dim * 2 * bytes_per_element

def evaluate_sample(
    sample_id: str,
    category: str,
    expected_answer: str,
    gold_block_ids: List[int],
    deprecated_block_ids: List[int],
    full_result: Dict[str, Any],
    replay_results: Dict[str, Dict[str, Any]],
    model_config: Any
) -> Dict[str, Any]:
    """
    Evaluates one sample across all modes and calculates metrics.
    """
    # Extract model config details
    num_layers = getattr(model_config, "num_hidden_layers", 24)
    num_kv_heads = getattr(model_config, "num_key_value_heads", 2)
    hidden_size = getattr(model_config, "hidden_size", 896)
    num_heads = getattr(model_config, "num_attention_heads", 14)
    head_dim = getattr(model_config, "head_dim", hidden_size // num_heads)
    
    # Assume 2 bytes per element (float16/bfloat16) unless model uses float32
    torch_dtype = getattr(model_config, "torch_dtype", None)
    bytes_per_element = 4 if torch_dtype == "float32" else 2
    
    eval_data = {}
    
    modes = ["full", "visibility", "random", "bm25"]
    
    for mode in modes:
        if mode == "full":
            ans = full_result["answer"]
            tokens = full_result["input_tokens"]
            ttft = full_result["ttft_ms"]
            kept_ids = list(range(50)) # all context blocks
        else:
            mode_res = replay_results[mode]
            ans = mode_res["answer"]
            tokens = mode_res["input_tokens"]
            ttft = mode_res["ttft_ms"]
            kept_ids = mode_res["kept_ids"]
            
        em = compute_exact_match(expected_answer, ans)
        num_pres = compute_numeric_preservation(expected_answer, ans)
        
        # Gold Block Recall
        gold_recall = all(gid in kept_ids for gid in gold_block_ids) if gold_block_ids else True
        
        # Wrong Block Retention (Deprecated block in context for Contradiction tests)
        wrong_retention = any(did in kept_ids for did in deprecated_block_ids) if deprecated_block_ids else False
        
        # Active Truth Accuracy:
        # In contradiction category, did the model output active date?
        active_truth = em
        if category == "C" and deprecated_block_ids:
            # If model outputs the deprecated answer instead of active one, active_truth is False
            # We check if any deprecated dates are outputted
            # (In build_dataset.py, deprecated dates are 2026, active dates are 2027)
            # If the generated answer contains "2026" or matches the old date, it's wrong.
            if "2026" in ans or "2026" in normalize_text(ans):
                active_truth = False
                
        kv_bytes = estimate_kv_bytes(
            tokens=tokens,
            num_layers=num_layers,
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
            bytes_per_element=bytes_per_element
        )
        
        eval_data[mode] = {
            "answer": ans,
            "exact_match": em,
            "numeric_preservation": num_pres,
            "gold_recall": gold_recall,
            "wrong_retention": wrong_retention,
            "active_truth": active_truth,
            "input_tokens": tokens,
            "ttft_ms": ttft,
            "kv_bytes": kv_bytes,
            "kept_blocks": len(kept_ids)
        }
        
    return eval_data


