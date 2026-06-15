import os
import json
import re
import numpy as np
import pickle
import glob
from typing import List, Dict, Any
from sentence_transformers import SentenceTransformer
from highway.paths import DEFAULT_CORPUS_DIR, DEFAULT_INDEX_DIR
from highway.storage.index_writer import write_out_of_core_index

# Fallback token counter in case transformers tokenizer is slow/fails
class SimpleWordTokenizer:
    def tokenize(self, text: str) -> List[str]:
        return text.split()

def get_tokenizer():
    try:
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct", local_files_only=True)
        print("Using Qwen2.5-0.5B-Instruct tokenizer.")
        return tokenizer
    except Exception as e:
        print(f"Fallback to simple word splitter: {e}")
        return SimpleWordTokenizer()

def clean_text(text: str) -> str:
    # Basic cleaning, remove multiple consecutive newlines and extra spaces
    lines = [line.strip() for line in text.splitlines()]
    non_empty_lines = [line for line in lines if line]
    return "\n".join(non_empty_lines)

def chunk_text(text: str, tokenizer, block_size: int = 128) -> List[Dict[str, Any]]:
    # Tokenize the text
    if hasattr(tokenizer, "encode"):
        tokens = tokenizer.encode(text)
        is_hf = True
    else:
        tokens = tokenizer.tokenize(text)
        is_hf = False
        
    chunks = []
    num_tokens = len(tokens)
    
    # We do non-overlapping blocks of size 128
    for i in range(0, num_tokens, block_size):
        chunk_tokens = tokens[i : i + block_size]
        if is_hf:
            chunk_text = tokenizer.decode(chunk_tokens)
            token_count = len(chunk_tokens)
        else:
            chunk_text = " ".join(chunk_tokens)
            token_count = len(chunk_tokens)
            
        chunks.append({
            "text": chunk_text,
            "token_count": token_count
        })
    return chunks

BLACKLIST_ENTITIES = {
    "project", "amendment", "contract", "division", "status", "update", "technical", 
    "report", "service", "summary", "specifications", "internal", "only", "to", "no", "kv",
    "approved budget", "completion date", "delivery date", "detailed analysis", 
    "executive summary", "financial overview", "financial request", "key takeaways", 
    "lead unit", "legacy", "mobile", "operations analyst", "project owner", 
    "security classification", "target entity", "timeline", "finance committee", 
    "archive", "division", "operational", "optimization", "analysis", "takeaways",
    "author", "department", "location", "division", "division", "officially",
    "january", "february", "march", "april", "may", "june", "july", "august",
    "september", "october", "november", "december"
}

def extract_entities_from_text(text: str) -> List[str]:
    # We match words like KRONOS, NEPTUNE, KRONOS-Legacy, PROJECT-A etc.
    words = re.findall(r'\b[A-Z][A-Z0-9_\-]+', text)
    # Also find names of people
    people_matches = re.findall(r'\b[A-Z][a-z]+\s+[A-Z][a-z]+\b', text)
    
    candidates = list(set(words + people_matches))
    filtered = []
    for c in candidates:
        c_clean = c.strip().lower()
        # Filter out if in blacklist, too short, or matches generic headers
        if c_clean in BLACKLIST_ENTITIES:
            continue
        if len(c_clean) <= 2:
            continue
        if any(bl in c_clean for bl in ["location", "approved", "timeline", "noise"]):
            continue
        filtered.append(c)
    return filtered

def ingest_corpus(
    corpus_dir: str,
    output_dir: str,
    layout: str = "legacy",
    embedding_dtype: str = "fp32",
    vector_backend: str = "none",
):
    if layout not in {"legacy", "out_of_core", "both"}:
        raise ValueError(f"Unsupported ingestion layout: {layout}")

    print(f"=== Starting Ingestion of {corpus_dir} ===")
    os.makedirs(output_dir, exist_ok=True)
    
    tokenizer = get_tokenizer()
    model = SentenceTransformer('all-MiniLM-L6-v2')
    
    blocks = []
    entity_set = set()
    
    # Find all txt files recursively
    search_path = os.path.join(corpus_dir, "documents", "**", "*.txt")
    txt_files = glob.glob(search_path, recursive=True)
    print(f"Found {len(txt_files)} source files.")
    
    global_block_id = 0
    for file_path in sorted(txt_files):
        # Determine relative category path, e.g. "reports/neptune_status_report.txt"
        parts = os.path.normpath(file_path).split(os.sep)
        # Category is the folder under 'documents'
        try:
            doc_idx = parts.index("documents")
            category = parts[doc_idx + 1]
            rel_file_path = "/".join(parts[doc_idx + 1:])
        except ValueError:
            category = "unknown"
            rel_file_path = os.path.basename(file_path)
            
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()
            
        cleaned_content = clean_text(content)
        file_chunks = chunk_text(cleaned_content, tokenizer, block_size=128)
        
        # Scan for entities
        file_entities = extract_entities_from_text(cleaned_content)
        entity_set.update(file_entities)
        
        for chunk_idx, chunk in enumerate(file_chunks):
            block_id = f"block_{global_block_id:04d}"
            blocks.append({
                "block_id": block_id,
                "text": chunk["text"],
                "source_file": rel_file_path,
                "category": category,
                "token_count": chunk["token_count"],
                "chunk_index": chunk_idx
            })
            global_block_id += 1
            
    print(f"Total blocks chunked: {len(blocks)}")
    
    # Save blocks.jsonl
    blocks_file = os.path.join(output_dir, "blocks.jsonl")
    with open(blocks_file, "w", encoding="utf-8") as f:
        for block in blocks:
            f.write(json.dumps(block) + "\n")
            
    # Compute dense embeddings
    print("Computing block embeddings...")
    block_texts = [b["text"] for b in blocks]
    embeddings = model.encode(block_texts, convert_to_numpy=True, show_progress_bar=True)

    # Build Entity list
    # Let's save the set of entities as a sorted list
    entities_list = sorted(list(entity_set))

    if layout in {"legacy", "both"}:
        embeddings_file = os.path.join(output_dir, "embeddings.npy")
        np.save(embeddings_file, embeddings)

        # Build BM25
        print("Building BM25 index...")
        from rank_bm25 import BM25Okapi

        # Simple word tokenization for BM25
        corpus_tokens = [text.lower().split() for text in block_texts]
        bm25 = BM25Okapi(corpus_tokens)
        bm25_file = os.path.join(output_dir, "bm25.pkl")
        with open(bm25_file, "wb") as f:
            pickle.dump(bm25, f)

        entity_file = os.path.join(output_dir, "entity_list.json")
        with open(entity_file, "w", encoding="utf-8") as f:
            json.dump(entities_list, f, indent=2)

    if layout in {"out_of_core", "both"}:
        print("Writing out-of-core index layout...")
        write_out_of_core_index(
            output_dir,
            blocks=blocks,
            embeddings=embeddings,
            entities=entities_list,
            embedding_dtype=embedding_dtype,
            vector_backend=vector_backend,
        )

    print(f"=== Ingestion Complete! ===")
    print(f"Saved blocks to {blocks_file}")
    if layout in {"legacy", "both"}:
        print(f"Saved embeddings to {embeddings_file}")
        print(f"Saved BM25 to {bm25_file}")
        print(f"Saved entities to {entity_file}")
    if layout in {"out_of_core", "both"}:
        print(f"Saved out-of-core manifest to {os.path.join(output_dir, 'manifest.json')}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus-dir", type=str, default=str(DEFAULT_CORPUS_DIR))
    parser.add_argument("--output-dir", type=str, default=str(DEFAULT_INDEX_DIR))
    parser.add_argument("--layout", choices=["legacy", "out_of_core", "both"], default="legacy")
    parser.add_argument("--embedding-dtype", choices=["fp32", "fp16"], default="fp32")
    parser.add_argument(
        "--vector-backend",
        choices=["none", "numpy_flat", "faiss_flat", "faiss_hnsw", "faiss_ivf_flat"],
        default="none",
    )
    args = parser.parse_args()
    ingest_corpus(
        args.corpus_dir,
        args.output_dir,
        layout=args.layout,
        embedding_dtype=args.embedding_dtype,
        vector_backend=args.vector_backend,
    )


