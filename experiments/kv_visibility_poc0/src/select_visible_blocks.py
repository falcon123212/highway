from typing import List, Dict, Any, Set, Tuple

def select_blocks_policy(
    blocks: List[Dict[str, Any]],
    block_scores: Dict[int, float],
    project_entity: str,
    top_k: int = 24,
    recent_n: int = 2
) -> Tuple[List[Dict[str, Any]], List[int]]:
    """
    Applies the hot/warm/cold classification and selection policy.
    
    Args:
        blocks: list of block dicts.
        block_scores: dict mapping block_id -> attention mass.
        project_entity: the project name (entity) from the question (e.g., "NOVA-0").
        top_k: keep top K attention blocks.
        recent_n: keep the last N blocks of the context.
        
    Returns:
        List of blocks updated with policy info and kept status.
        List of kept block IDs.
    """
    # Rank blocks by attention mass
    ranked_blocks = sorted(block_scores.items(), key=lambda x: x[1], reverse=True)
    top_k_ids = {bid for bid, _ in ranked_blocks[:top_k]}
    
    # Recency window
    num_blocks = len(blocks)
    recent_ids = {i for i in range(max(0, num_blocks - recent_n), num_blocks)}
    
    kept_block_ids = []
    updated_blocks = []
    
    for block in blocks:
        block_id = block["block_id"]
        score = block_scores.get(block_id, 0.0)
        
        # Categorize
        if score >= 0.05:
            policy = "HOT"
        elif score >= 0.01:
            policy = "WARM"
        else:
            policy = "COLD"
            
        # Entity matching
        # Simple case-insensitive match for the project entity in the block text
        entity_matched = project_entity.lower() in block["text"].lower()
        
        # Selection rule: top_k OR recent OR entity_matched
        is_top_k = block_id in top_k_ids
        is_recent = block_id in recent_ids
        
        keep = is_top_k or is_recent or entity_matched
        
        if keep:
            kept_block_ids.append(block_id)
            
        updated_block = dict(block)
        updated_block.update({
            "attention_mass": score,
            "policy": policy,
            "is_top_k": is_top_k,
            "is_recent": is_recent,
            "entity_matched": entity_matched,
            "keep": keep
        })
        updated_blocks.append(updated_block)
        
    return updated_blocks, sorted(kept_block_ids)


