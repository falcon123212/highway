from dataclasses import dataclass


@dataclass(frozen=True)
class HardwareBudget:
    max_ram_mb: int = 512
    max_index_window_mb: int = 64
    max_candidates: int = 200
    max_context_tokens: int = 4096
    semantic_ann_k: int = 2000
    semantic_rerank_k: int = 2000
    semantic_lexical_k: int = 5000
    semantic_ef_search: int = 128
    semantic_rescue_enabled: bool = True
    semantic_full_scan_fallback_max_blocks: int = 10000
    semantic_reranker_backend: str = "cross_encoder"
    semantic_reranker_input_k: int = 1000
    semantic_reranker_output_k: int = 500
    semantic_reranker_batch_size: int = 32
    semantic_reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    semantic_reranker_local_files_only: bool = False

    @property
    def max_ram_bytes(self) -> int:
        return int(self.max_ram_mb * 1024 * 1024)

    @property
    def max_index_window_bytes(self) -> int:
        return int(self.max_index_window_mb * 1024 * 1024)
