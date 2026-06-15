import random
from rank_bm25 import BM25Okapi
from typing import List, Dict, Any, Tuple

def tokenize_for_bm25(text: str) -> List[str]:
    """Simple word tokenization for BM25."""
    # Convert to lowercase and split by spaces, removing basic punctuation
    clean_text = "".join(c if c.isalnum() or c.isspace() else " " for c in text.lower())
    return clean_text.split()

def build_replay_prompts(
    sample: Dict[str, Any],
    kept_visibility_ids: List[int],
    tokenizer: Any,
    seed: int = 42
) -> Dict[str, Dict[str, Any]]:
    """
    Builds the 3 replay prompts: visibility, random, and BM25.
    Returns prompt text and token list for each.
    """
    documents = sample["documents"]
    question = sample["question"]
    num_kept = len(kept_visibility_ids)
    total_docs = len(documents)
    
    # 1. Random Selection (same number of blocks, deterministic seed based on question_id)
    rng = random.Random(seed + hash(sample["question_id"]))
    random_ids = sorted(rng.sample(range(total_docs), min(num_kept, total_docs)))
    
    # 2. BM25 Selection (same number of blocks)
    corpus = [tokenize_for_bm25(doc["text"]) for doc in documents]
    bm25 = BM25Okapi(corpus)
    query = tokenize_for_bm25(question)
    doc_scores = bm25.get_scores(query)
    
    # Rank by BM25 score
    ranked_indices = sorted(range(total_docs), key=lambda idx: doc_scores[idx], reverse=True)
    bm25_ids = sorted(ranked_indices[:num_kept])
    
    # Common function to reconstruct the full prompt for a set of kept doc indices
    def assemble_prompt(kept_ids: List[int]) -> str:
        system_text = "<|im_start|>system\nYou are a helpful assistant. Answer the question based on the provided context. Be concise and precise.<|im_end|>\n<|im_start|>user\nContext:\n"
        
        context_parts = []
        last_id = -2
        for idx in kept_ids:
            # If there is a gap, optionally add elision indicator
            if last_id != -2 and idx != last_id + 1:
                context_parts.append("[...]")
            context_parts.append(documents[idx]["text"])
            last_id = idx
            
        context_text = "\n\n".join(context_parts)
        question_text = f"\n\nQuestion: {question}<|im_end|>\n<|im_start|>assistant\n"
        
        return system_text + context_text + question_text

    # Assemble all 3 prompts
    visibility_prompt = assemble_prompt(kept_visibility_ids)
    random_prompt = assemble_prompt(random_ids)
    bm25_prompt = assemble_prompt(bm25_ids)
    
    return {
        "visibility": {
            "text": visibility_prompt,
            "ids": kept_visibility_ids,
            "token_ids": tokenizer.encode(visibility_prompt)
        },
        "random": {
            "text": random_prompt,
            "ids": random_ids,
            "token_ids": tokenizer.encode(random_prompt)
        },
        "bm25": {
            "text": bm25_prompt,
            "ids": bm25_ids,
            "token_ids": tokenizer.encode(bm25_prompt)
        }
    }


