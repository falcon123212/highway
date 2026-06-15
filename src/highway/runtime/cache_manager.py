import os
import json
import hashlib
from typing import Dict, Any, List, Optional

from highway.paths import DEFAULT_CACHE_DIR

class CacheManager:
    def __init__(self, cache_dir: Optional[str] = None):
        self.cache_dir = cache_dir or str(DEFAULT_CACHE_DIR)
        os.makedirs(self.cache_dir, exist_ok=True)
        
        # Versions path
        self.version_path = os.path.join(self.cache_dir, "version.json")
        self.corpus_version = self._load_corpus_version()
        
        # Cache file paths
        self.l0_path = os.path.join(self.cache_dir, "l0_answer_cache.json")
        self.l1_path = os.path.join(self.cache_dir, "l1_proof_cache.json")
        self.l2_path = os.path.join(self.cache_dir, "l2_evidence_cache.json")
        self.l3_path = os.path.join(self.cache_dir, "l3_prompt_cache.json")
        
        # Load caches
        self.l0_cache = self._load_cache_file(self.l0_path)
        self.l1_cache = self._load_cache_file(self.l1_path)
        self.l2_cache = self._load_cache_file(self.l2_path)
        self.l3_cache = self._load_cache_file(self.l3_path)
        
        # Statistics trackers
        self.stats = {
            "l0_hits": 0, "l0_misses": 0,
            "l1_hits": 0, "l1_misses": 0,
            "l2_hits": 0, "l2_misses": 0,
            "l3_hits": 0, "l3_misses": 0,
            "stale_cache_errors": 0
        }

    def _load_corpus_version(self) -> int:
        if os.path.exists(self.version_path):
            try:
                with open(self.version_path, "r") as f:
                    data = json.load(f)
                    return data.get("corpus_version", 1)
            except Exception:
                pass
        return 1

    def _save_corpus_version(self):
        with open(self.version_path, "w") as f:
            json.dump({"corpus_version": self.corpus_version}, f)

    def increment_corpus_version(self):
        self.corpus_version += 1
        self._save_corpus_version()
        # Clear/invalidate all caches on disk and in memory
        self.l0_cache = {}
        self.l1_cache = {}
        self.l2_cache = {}
        self.l3_cache = {}
        self._save_all_caches()
        print(f"[CacheManager] Corpus version incremented to {self.corpus_version}. Caches invalidated.")

    def _load_cache_file(self, path: str) -> Dict[str, Any]:
        if os.path.exists(path):
            try:
                with open(path, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception:
                pass
        return {}

    def _save_cache_file(self, path: str, cache_dict: Dict[str, Any]):
        try:
            with open(path, "w", encoding="utf-8") as f:
                json.dump(cache_dict, f, indent=2, ensure_ascii=False)
        except Exception as e:
            print(f"[CacheManager] Error saving cache to {path}: {e}")

    def _save_all_caches(self):
        self._save_cache_file(self.l0_path, self.l0_cache)
        self._save_cache_file(self.l1_path, self.l1_cache)
        self._save_cache_file(self.l2_path, self.l2_cache)
        self._save_cache_file(self.l3_path, self.l3_cache)

    def save(self):
        self._save_all_caches()

    def reset_stats(self):
        for k in self.stats:
            self.stats[k] = 0

    # --- L0 Answer Cache ---
    # Key = hash(proof_ir) + hash(output_schema) + corpus_version
    def get_answer(self, proof_ir: Dict[str, Any], output_schema: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        # Hash elements to make a secure key
        proof_str = json.dumps(proof_ir, sort_keys=True)
        schema_str = json.dumps(output_schema, sort_keys=True)
        proof_hash = hashlib.sha256(proof_str.encode("utf-8")).hexdigest()
        schema_hash = hashlib.sha256(schema_str.encode("utf-8")).hexdigest()
        
        key = f"{proof_hash}_{schema_hash}_v{self.corpus_version}"
        if key in self.l0_cache:
            self.stats["l0_hits"] += 1
            return self.l0_cache[key]
        self.stats["l0_misses"] += 1
        return None

    def set_answer(self, proof_ir: Dict[str, Any], output_schema: Dict[str, Any], answer: str, evidence_ids: List[str], verifier_status: str):
        proof_str = json.dumps(proof_ir, sort_keys=True)
        schema_str = json.dumps(output_schema, sort_keys=True)
        proof_hash = hashlib.sha256(proof_str.encode("utf-8")).hexdigest()
        schema_hash = hashlib.sha256(schema_str.encode("utf-8")).hexdigest()
        
        key = f"{proof_hash}_{schema_hash}_v{self.corpus_version}"
        self.l0_cache[key] = {
            "key": key,
            "answer": answer,
            "evidence_ids": evidence_ids,
            "verifier_status": verifier_status
        }

    # --- L1 Proof IR Cache ---
    # Key = query_ir_hash + corpus_version
    def get_proof_ir(self, query_ir_hash: str) -> Optional[Dict[str, Any]]:
        key = f"{query_ir_hash}_v{self.corpus_version}"
        if key in self.l1_cache:
            self.stats["l1_hits"] += 1
            return self.l1_cache[key]
        self.stats["l1_misses"] += 1
        return None

    def set_proof_ir(self, query_ir_hash: str, proof_ir: Dict[str, Any]):
        key = f"{query_ir_hash}_v{self.corpus_version}"
        self.l1_cache[key] = proof_ir

    # --- L2 Evidence Pool Cache ---
    # Key = query_ir_hash + search_config_hash + corpus_version
    def get_evidence_pool(self, query_ir_hash: str, search_config_hash: str) -> Optional[List[Dict[str, Any]]]:
        key = f"{query_ir_hash}_{search_config_hash}_v{self.corpus_version}"
        if key in self.l2_cache:
            self.stats["l2_hits"] += 1
            return self.l2_cache[key]
        self.stats["l2_misses"] += 1
        return None

    def set_evidence_pool(self, query_ir_hash: str, search_config_hash: str, evidence_pool: List[Dict[str, Any]]):
        key = f"{query_ir_hash}_{search_config_hash}_v{self.corpus_version}"
        self.l2_cache[key] = evidence_pool

    # --- L3 Compiled Prompt Cache ---
    # Key = hash(proof_ir) + compiler_version + model_id
    def get_compiled_prompt(self, proof_ir: Dict[str, Any], compiler_version: str, model_id: str) -> Optional[str]:
        proof_str = json.dumps(proof_ir, sort_keys=True)
        proof_hash = hashlib.sha256(proof_str.encode("utf-8")).hexdigest()
        
        key = f"{proof_hash}_{compiler_version}_{model_id}_v{self.corpus_version}"
        if key in self.l3_cache:
            self.stats["l3_hits"] += 1
            return self.l3_cache[key]
        self.stats["l3_misses"] += 1
        return None

    def set_compiled_prompt(self, proof_ir: Dict[str, Any], compiler_version: str, model_id: str, compiled_prompt: str):
        proof_str = json.dumps(proof_ir, sort_keys=True)
        proof_hash = hashlib.sha256(proof_str.encode("utf-8")).hexdigest()
        
        key = f"{proof_hash}_{compiler_version}_{model_id}_v{self.corpus_version}"
        self.l3_cache[key] = compiled_prompt


