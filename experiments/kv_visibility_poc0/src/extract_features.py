import re
import string
import numpy as np
from typing import List, Dict, Any
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer, util

# Lazy loader for sentence transformer model
_embedding_model = None
_block_embedding_cache = {}

def get_embedding_model() -> SentenceTransformer:
    global _embedding_model
    if _embedding_model is None:
        # Load small, fast model
        _embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
    return _embedding_model

def get_block_embeddings(texts: List[str]) -> np.ndarray:
    model = get_embedding_model()
    embs = []
    uncached_texts = []
    uncached_indices = []
    
    for idx, txt in enumerate(texts):
        if txt in _block_embedding_cache:
            embs.append((idx, _block_embedding_cache[txt]))
        else:
            uncached_texts.append(txt)
            uncached_indices.append(idx)
            
    if uncached_texts:
        # Encode uncached texts as a batch
        uncached_embs = model.encode(uncached_texts, convert_to_tensor=False, show_progress_bar=False)
        for idx, emb in zip(uncached_indices, uncached_embs):
            _block_embedding_cache[texts[idx]] = emb
            embs.append((idx, emb))
            
    # Sort embeddings back to original order
    embs.sort(key=lambda x: x[0])
    return np.array([e[1] for e in embs], dtype=np.float32)

def tokenize_for_bm25(text: str) -> List[str]:
    """Simple word tokenization for BM25."""
    clean_text = "".join(c if c.isalnum() or c.isspace() else " " for c in text.lower())
    return clean_text.split()

def is_exact_project_match(project: str, text: str) -> float:
    project_lower = project.lower()
    text_lower = text.lower()
    
    start = 0
    while True:
        pos = text_lower.find(project_lower, start)
        if pos == -1:
            break
        # Check if the character right after the match is a suffix character (like -A or -Legacy)
        end_pos = pos + len(project_lower)
        if end_pos < len(text_lower):
            next_char = text_lower[end_pos]
            if next_char == '-' and end_pos + 1 < len(text_lower) and text_lower[end_pos+1].isalnum():
                # Skip distractor projects (e.g., APEX-400-A, APEX-400-Legacy)
                start = end_pos
                continue
        return 1.0
    return 0.0

def extract_block_features(
    question: str,
    blocks: List[Dict[str, Any]],
    project_entity: str,
    ablation_mode: str = "full"
) -> np.ndarray:
    """
    Extracts a feature matrix of shape (num_blocks, 7) for a single sample.
    
    Features:
    1. bm25_score: Lexical match score.
    2. semantic_score: Cosine similarity of MiniLM embeddings.
    3. relative_position: Index of the block divided by total blocks.
    4. recency: Distance from the end of the context.
    5. entity_match: Binary match of project entity.
    6. number_match: Overlap of digits between question and block text.
    7. status_active: Binary flag if status is ACTIVE.
    """
    num_blocks = len(blocks)
    
    # 1. Lexical Scores (BM25)
    corpus = [tokenize_for_bm25(b["text"]) for b in blocks]
    bm25 = BM25Okapi(corpus)
    query = tokenize_for_bm25(question)
    bm25_scores = bm25.get_scores(query)
    # Normalize BM25 scores
    max_bm25 = max(bm25_scores) if len(bm25_scores) > 0 and max(bm25_scores) > 0 else 1.0
    bm25_scores = [s / max_bm25 for s in bm25_scores]
    
    # 2. Semantic Scores (Sentence Transformers with Caching & NumPy Cosine Similarity)
    model = get_embedding_model()
    q_emb = model.encode(question, convert_to_tensor=False, show_progress_bar=False)
    block_texts = [b["text"] for b in blocks]
    block_embs = get_block_embeddings(block_texts)
    
    q_norm = np.linalg.norm(q_emb)
    block_norms = np.linalg.norm(block_embs, axis=1)
    dot_products = np.dot(block_embs, q_emb)
    cos_sims = dot_products / (q_norm * block_norms + 1e-8)
    
    # 3. Numeric Overlap Extraction
    question_nums = set(re.findall(r'\d+', question))
    
    feature_matrix = []
    
    for i, block in enumerate(blocks):
        block_text = block["text"]
        
        # Position / Recency
        rel_pos = i / num_blocks
        recency = (num_blocks - 1 - i) / num_blocks
        
        # Entity Match (using precise match)
        entity_match = is_exact_project_match(project_entity, block_text)
        
        # Number Match
        block_nums = set(re.findall(r'\d+', block_text))
        common_nums = len(question_nums.intersection(block_nums))
        number_match = float(common_nums) / max(1, len(question_nums))
        
        # Status checks
        status_active = 1.0 if "active" in block_text.lower() else 0.0
        
        features = [
            bm25_scores[i],       # Lexical
            float(cos_sims[i]),   # Semantic
            rel_pos,              # Position
            recency,              # Recency
            entity_match,         # Entity Match
            number_match,         # Numeric Match
            status_active         # Status
        ]
        if ablation_mode == "no_position":
            features = [features[0], features[1], features[4], features[5], features[6]]
        elif ablation_mode == "semantic_only":
            features = [features[1], features[4]]
        feature_matrix.append(features)
        
    return np.array(feature_matrix, dtype=np.float32)


