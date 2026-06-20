from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Mapping

from highway.errors import ConfigurationError

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - Python 3.10 fallback
    try:
        import tomli as tomllib  # type: ignore[no-redef]
    except ModuleNotFoundError:  # pragma: no cover
        tomllib = None  # type: ignore[assignment]


# Experimental configurations and global constants for the Highway RAG system.

# Ingestion & Chunking
CHUNK_SIZE = 128
CHUNK_OVERLAP = 32

# Retrieval Models
DEFAULT_EMBEDDING_MODEL = "BAAI/bge-base-en-v1.5"
DEFAULT_FALLBACK_EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
DEFAULT_RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"

# Retrieval Fusion & Rescue
DEFAULT_RRF_K = 60.0
DEFAULT_WEIGHT_BM25 = 0.6
DEFAULT_WEIGHT_DENSE = 0.4
DEFAULT_RESCUE_LIMIT = 5

# Cascade & Adaptive Budget Governor
DEFAULT_CASCADE_BUDGETS = (256, 321, 384, 512)
DEFAULT_CPU_EXTRACTIVE_CONFIDENCE_THRESHOLD = 0.85


@dataclass(frozen=True)
class IngestionConfig:
    chunk_size: int = CHUNK_SIZE
    chunk_overlap: int = CHUNK_OVERLAP


@dataclass(frozen=True)
class RetrievalConfig:
    default_embedding_model: str = DEFAULT_EMBEDDING_MODEL
    default_fallback_embedding_model: str = DEFAULT_FALLBACK_EMBEDDING_MODEL
    default_reranker_model: str = DEFAULT_RERANKER_MODEL
    default_rrf_k: float = DEFAULT_RRF_K
    default_weight_bm25: float = DEFAULT_WEIGHT_BM25
    default_weight_dense: float = DEFAULT_WEIGHT_DENSE
    default_rescue_limit: int = DEFAULT_RESCUE_LIMIT


@dataclass(frozen=True)
class GovernorConfig:
    cascade_budgets: tuple[int, ...] = DEFAULT_CASCADE_BUDGETS
    cpu_extractive_confidence_threshold: float = DEFAULT_CPU_EXTRACTIVE_CONFIDENCE_THRESHOLD


@dataclass(frozen=True)
class HighwayConfig:
    ingestion: IngestionConfig = IngestionConfig()
    retrieval: RetrievalConfig = RetrievalConfig()
    governor: GovernorConfig = GovernorConfig()


_SECTION_TYPES = {
    "ingestion": IngestionConfig,
    "retrieval": RetrievalConfig,
    "governor": GovernorConfig,
}


def _section_update(section_name: str, section: Any, values: Mapping[str, Any]) -> Any:
    allowed = set(section.__dataclass_fields__.keys())
    unknown = set(values) - allowed
    if unknown:
        raise ConfigurationError(
            message=f"Unknown config keys in [{section_name}]: {', '.join(sorted(unknown))}",
            details={"section": section_name, "unknown_keys": sorted(unknown)},
        )
    normalized = dict(values)
    if section_name == "governor" and "cascade_budgets" in normalized:
        normalized["cascade_budgets"] = tuple(int(v) for v in normalized["cascade_budgets"])
    return replace(section, **normalized)


def load_config(path: str | Path | None = None) -> HighwayConfig:
    cfg = HighwayConfig()
    if path is None:
        default_path = Path("config.toml")
        if not default_path.exists():
            return cfg
        path = default_path

    if tomllib is None:
        raise ConfigurationError(message="TOML config loading requires Python 3.11+ or tomli.")

    config_path = Path(path)
    with config_path.open("rb") as fh:
        data = tomllib.load(fh)

    unknown_sections = set(data) - set(_SECTION_TYPES)
    if unknown_sections:
        raise ConfigurationError(
            message=f"Unknown config sections: {', '.join(sorted(unknown_sections))}",
            details={"unknown_sections": sorted(unknown_sections)},
        )

    return HighwayConfig(
        ingestion=_section_update("ingestion", cfg.ingestion, data.get("ingestion", {})),
        retrieval=_section_update("retrieval", cfg.retrieval, data.get("retrieval", {})),
        governor=_section_update("governor", cfg.governor, data.get("governor", {})),
    )
