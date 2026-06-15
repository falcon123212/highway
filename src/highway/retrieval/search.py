import os
import json
import re
import numpy as np
import pickle
from typing import List, Dict, Any, Tuple
from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi
from highway.retrieval.query_parser import QueryParser
from highway.runtime.hardware_budget import HardwareBudget
from highway.storage.out_of_core_index import OutOfCoreIndex

class SearchRouter:
    def __init__(self, index_dir: str, storage_mode: str = "auto", hardware_budget: HardwareBudget | None = None):
        self.index_dir = index_dir
        self.storage_mode = "legacy"
        self.last_storage_metrics = {}

        if storage_mode not in {"auto", "legacy", "out_of_core"}:
            raise ValueError(f"Unsupported storage mode: {storage_mode}")

        if storage_mode in {"auto", "out_of_core"} and OutOfCoreIndex.is_out_of_core_index(index_dir):
            self.storage_mode = "out_of_core"
            self.out_of_core_index = OutOfCoreIndex(index_dir, hardware_budget=hardware_budget)
            self.query_parser = self.out_of_core_index.query_parser
            return

        if storage_mode == "out_of_core":
            raise FileNotFoundError(f"Out-of-core index manifest not found in {index_dir}")
        
        # Load blocks
        self.blocks = []
        blocks_path = os.path.join(index_dir, "blocks.jsonl")
        if not os.path.exists(blocks_path):
            raise FileNotFoundError(f"Blocks file not found at {blocks_path}")
        with open(blocks_path, "r", encoding="utf-8") as f:
            for line in f:
                self.blocks.append(json.loads(line))
        
        # Load embeddings
        embeddings_path = os.path.join(index_dir, "embeddings.npy")
        if not os.path.exists(embeddings_path):
            raise FileNotFoundError(f"Embeddings file not found at {embeddings_path}")
        self.embeddings = np.load(embeddings_path)
        
        # Load BM25
        bm25_path = os.path.join(index_dir, "bm25.pkl")
        if not os.path.exists(bm25_path):
            raise FileNotFoundError(f"BM25 index not found at {bm25_path}")
        with open(bm25_path, "rb") as f:
            self.bm25 = pickle.load(f)
            
        # Load entities
        entity_path = os.path.join(index_dir, "entity_list.json")
        if not os.path.exists(entity_path):
            raise FileNotFoundError(f"Entities list not found at {entity_path}")
        with open(entity_path, "r", encoding="utf-8") as f:
            self.entities = json.load(f)
            
        self.query_parser = QueryParser(self.entities)
        self.embed_model = SentenceTransformer('all-MiniLM-L6-v2')
        
    def search(self, question: str, top_k: int = 50) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        if self.storage_mode == "out_of_core":
            results, query_ir, telemetry = self.out_of_core_index.search(question, top_k=top_k)
            self.last_storage_metrics = telemetry
            return results, query_ir

        # Parse query
        query_ir = self.query_parser.parse(question)
        target_entities = query_ir["target_entities"]
        
        # 1. BM25 Search
        q_tokens = question.lower().split()
        bm25_scores = self.bm25.get_scores(q_tokens)
        
        # 2. Dense Search
        q_emb = self.embed_model.encode(question, convert_to_numpy=True)
        q_norm = np.linalg.norm(q_emb)
        block_norms = np.linalg.norm(self.embeddings, axis=1)
        dot_products = np.dot(self.embeddings, q_emb)
        cos_sims = dot_products / (q_norm * block_norms + 1e-8)
        
        # 3. Reciprocal Rank Fusion (RRF)
        # Argsort gives indices from smallest to largest, so we invert to get largest to smallest
        bm25_order = np.argsort(-bm25_scores)
        dense_order = np.argsort(-cos_sims)
        
        # Build rank lookups (0-indexed rank)
        bm25_ranks = np.zeros(len(self.blocks), dtype=int)
        dense_ranks = np.zeros(len(self.blocks), dtype=int)
        
        for rank, idx in enumerate(bm25_order):
            bm25_ranks[idx] = rank
        for rank, idx in enumerate(dense_order):
            dense_ranks[idx] = rank
            
        rrf_scores = np.zeros(len(self.blocks))
        for idx in range(len(self.blocks)):
            # standard RRF constant is 60
            rrf_scores[idx] = (1.0 / (60.0 + bm25_ranks[idx])) + (1.0 / (60.0 + dense_ranks[idx]))
            
        # 4. Entity Boost
        if target_entities:
            for idx, block in enumerate(self.blocks):
                block_text_lower = block["text"].lower()
                for entity in target_entities:
                    entity_lower = entity.lower()
                    pattern_full = r'(?:\b|(?<=_))' + re.escape(entity_lower) + r'(?:\b|(?=_))'
                    base_name = entity_lower.replace("project ", "").strip()
                    pattern_base = r'(?:\b|(?<=_))' + re.escape(base_name) + r'(?:\b|(?=_))'
                    if re.search(pattern_full, block_text_lower) or re.search(pattern_base, block_text_lower):
                        rrf_scores[idx] += 0.3  # Add calibrated entity boost
                        
        # Sort and return top_k
        top_k_indices = np.argsort(-rrf_scores)[:top_k]
        
        results = []
        for rank, idx in enumerate(top_k_indices):
            block_info = dict(self.blocks[idx])
            block_info["retrieval_score"] = float(rrf_scores[idx])
            block_info["retrieval_rank"] = rank + 1
            block_info["bm25_score"] = float(bm25_scores[idx])
            block_info["cosine_similarity"] = float(cos_sims[idx])
            results.append(block_info)

        self.last_storage_metrics = {
            "storage_mode": "legacy",
            "blocks_materialized": len(self.blocks),
            "embedding_rows_scanned": len(self.blocks),
            "bytes_read": 0,
            "index_bytes": 0,
        }
        return results, query_ir


