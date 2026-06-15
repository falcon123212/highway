import json
import re
import sqlite3
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence

import numpy as np

from highway.storage.vector_index import build_vector_index


TERM_RE = re.compile(r"[a-z0-9_]+")


def tokenize_for_postings(text: str) -> List[str]:
    return TERM_RE.findall(text.lower())


def _embedding_dtype(name: str):
    normalized = name.lower()
    if normalized == "fp16":
        return np.float16
    if normalized == "fp32":
        return np.float32
    raise ValueError(f"Unsupported embedding dtype: {name}")


def write_out_of_core_index(
    index_dir: str | Path,
    blocks: Sequence[Dict[str, Any]],
    embeddings: np.ndarray,
    entities: Iterable[str],
    embedding_dtype: str = "fp32",
    vector_backend: str = "none",
    ann_params: Dict[str, Any] | None = None,
    embedding_metadata: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    index_path = Path(index_dir)
    index_path.mkdir(parents=True, exist_ok=True)

    embeddings_array = np.asarray(embeddings, dtype=_embedding_dtype(embedding_dtype))
    if len(blocks) != int(embeddings_array.shape[0]):
        raise ValueError("blocks and embeddings must have the same first dimension")

    blocks_file = index_path / "blocks.jsonl"
    offsets_file = index_path / "block_offsets.json"
    embeddings_file = index_path / "embeddings.npy"
    postings_file = index_path / "postings.sqlite"
    entities_file = index_path / "entity_list.json"
    manifest_file = index_path / "manifest.json"

    offsets = []
    with blocks_file.open("wb") as f:
        for block_idx, block in enumerate(blocks):
            offset = f.tell()
            payload = (json.dumps(block, ensure_ascii=False) + "\n").encode("utf-8")
            f.write(payload)
            offsets.append({
                "block_idx": block_idx,
                "block_id": block["block_id"],
                "offset": offset,
                "byte_length": len(payload),
                "source_file": block.get("source_file", ""),
                "category": block.get("category", ""),
                "token_count": int(block.get("token_count", 0)),
                "chunk_index": int(block.get("chunk_index", 0)),
            })

    offsets_file.write_text(json.dumps(offsets, indent=2), encoding="utf-8")
    np.save(embeddings_file, embeddings_array)
    entities_file.write_text(json.dumps(sorted(set(entities)), indent=2), encoding="utf-8")

    with sqlite3.connect(postings_file) as conn:
        conn.execute("DROP TABLE IF EXISTS term_postings")
        conn.execute("DROP TABLE IF EXISTS block_meta")
        conn.execute("CREATE TABLE term_postings(term TEXT NOT NULL, block_idx INTEGER NOT NULL, tf INTEGER NOT NULL)")
        conn.execute(
            "CREATE TABLE block_meta("
            "block_idx INTEGER PRIMARY KEY, block_id TEXT NOT NULL, source_file TEXT, category TEXT, "
            "token_count INTEGER, offset INTEGER NOT NULL, byte_length INTEGER NOT NULL)"
        )

        for meta, block in zip(offsets, blocks):
            conn.execute(
                "INSERT INTO block_meta(block_idx, block_id, source_file, category, token_count, offset, byte_length) "
                "VALUES (?, ?, ?, ?, ?, ?, ?)",
                (
                    meta["block_idx"],
                    meta["block_id"],
                    meta["source_file"],
                    meta["category"],
                    meta["token_count"],
                    meta["offset"],
                    meta["byte_length"],
                ),
            )
            tokens = tokenize_for_postings(f"{block.get('text', '')} {block.get('source_file', '')}")
            for term, tf in Counter(tokens).items():
                conn.execute(
                    "INSERT INTO term_postings(term, block_idx, tf) VALUES (?, ?, ?)",
                    (term, meta["block_idx"], int(tf)),
                )

        conn.execute("CREATE INDEX IF NOT EXISTS idx_term_postings_term ON term_postings(term)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_term_postings_block ON term_postings(block_idx)")

    ann_metadata = build_vector_index(
        embeddings_path=embeddings_file,
        output_path=index_path,
        backend=vector_backend,
        params=ann_params or {},
    )

    manifest = {
        "layout": "highway_out_of_core_v1",
        "storage_mode": "out_of_core",
        "blocks_file": blocks_file.name,
        "offsets_file": offsets_file.name,
        "embeddings_file": embeddings_file.name,
        "postings_file": postings_file.name,
        "entity_file": entities_file.name,
        "num_blocks": len(blocks),
        "embedding_shape": list(embeddings_array.shape),
        "embedding_dtype": str(embeddings_array.dtype),
        "vector_backend": vector_backend,
        "ann_file": ann_metadata.get("ann_file"),
        "ann_metric": ann_metadata.get("ann_metric", "inner_product"),
        "ann_params": ann_metadata.get("ann_params", ann_params or {}),
        "ann_available": bool(ann_metadata.get("ann_available", False)),
        "ann_fallback_reason": ann_metadata.get("ann_fallback_reason", ""),
    }
    manifest.update(dict(embedding_metadata or {}))
    manifest_file.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest
