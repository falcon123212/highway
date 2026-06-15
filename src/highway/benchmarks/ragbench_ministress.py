from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from random import Random
from statistics import mean
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple

import numpy as np

from highway.benchmarks.long_conversation_quality import PromptAuditWriter, _display_path, _write_jsonl
from highway.paths import DEFAULT_RUNS_DIR
from highway.runtime.context_engine import ContextRequest, HighwayContextEngine
from highway.runtime.llm_runtime import HighwayLLMRuntime, estimate_tokens
from highway.runtime.ollama_client import OllamaLLMClient
from highway.runtime.token_economics import ModelProfile, TokenEconomics
from highway.storage.index_writer import write_out_of_core_index


DEFAULT_DATASET_ID = "galileo-ai/ragbench"
FALLBACK_DATASET_ID = "rungalileo/ragbench"
DEFAULT_OUTPUT_DIR = DEFAULT_RUNS_DIR / "ragbench_ministress"
DEFAULT_CONFIGS = ("covidqa", "cuad", "finqa", "hotpotqa", "techqa")
DEFAULT_MODEL_PROFILE = ModelProfile(name="ragbench_llm", layers=24, hidden_size=1024)

JSON_INSTRUCTION = (
    "Return only valid JSON with keys: reasoning, answer, sources, confidence. "
    "The sources value must be a list of source_file strings copied from the context. "
    "If the selected context does not contain enough evidence, answer INSUFFICIENT_EVIDENCE."
)


class SimpleBM25:
    def __init__(self, corpus: List[List[str]], k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.corpus_size = len(corpus)
        self.doc_lengths = [len(doc) for doc in corpus]
        self.avg_doc_length = sum(self.doc_lengths) / self.corpus_size if self.corpus_size > 0 else 1.0
        
        self.doc_freqs: Dict[str, int] = {}
        self.term_freqs: List[Dict[str, int]] = []
        
        for doc in corpus:
            freqs: Dict[str, int] = {}
            for term in doc:
                freqs[term] = freqs.get(term, 0) + 1
            self.term_freqs.append(freqs)
            
            for term in freqs:
                self.doc_freqs[term] = self.doc_freqs.get(term, 0) + 1
                
        self.idf: Dict[str, float] = {}
        for term, freq in self.doc_freqs.items():
            self.idf[term] = float(np.log((self.corpus_size - freq + 0.5) / (freq + 0.5) + 1.0))

    def get_scores(self, query: List[str]) -> List[float]:
        scores = [0.0] * self.corpus_size
        for idx in range(self.corpus_size):
            doc_len = self.doc_lengths[idx]
            term_freqs = self.term_freqs[idx]
            score = 0.0
            for term in query:
                if term in term_freqs:
                    tf = term_freqs[term]
                    idf_val = self.idf.get(term, 0.0)
                    denom = tf + self.k1 * (1.0 - self.b + self.b * doc_len / self.avg_doc_length)
                    score += idf_val * tf * (self.k1 + 1.0) / denom
            scores[idx] = score
        return scores



@dataclass(frozen=True)
class MiniStressCase:
    case_id: str
    config_name: str
    question: str
    expected_answer: str
    expected_sources: List[str]
    blocks: List[Dict[str, Any]]
    utilized_sentence_keys: List[str]  # e.g., ["0:2"]
    relevant_sentence_keys: List[str]   # e.g., ["0:1", "0:2"]
    documents_sentences: List[List[str]]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class SimpleEmbedder:
    def __init__(self, dim: int = 128):
        self.dim = int(dim)

    def encode(self, text: Any, convert_to_numpy: bool = True, show_progress_bar: bool = False) -> np.ndarray:
        if isinstance(text, list):
            return np.asarray([self.encode(item) for item in text], dtype=np.float32)
        vec = np.zeros(self.dim, dtype=np.float32)
        for token in re.findall(r"[a-z0-9_]+", str(text).lower()):
            digest = hashlib.sha256(token.encode("utf-8")).digest()
            idx = int.from_bytes(digest[:4], "little") % self.dim
            vec[idx] += 1.0
        norm = float(np.linalg.norm(vec))
        if norm > 0.0:
            vec /= norm
        else:
            vec[-1] = 1.0
        return vec

    def embedding_metadata(self) -> Dict[str, Any]:
        return {
            "embedding_backend": "ragbench_hashing",
            "embedding_model": "ragbench_hashing_v1",
            "embedding_dim": self.dim,
            "embedding_local_files_only": True,
            "embedding_batch_size": 0,
            "embedding_latency_ms": 0.0,
            "embedding_fallback_reason": "",
        }


class GroundedFakeClient:
    model_name = "ragbench_grounded_fake"

    def answer(
        self,
        prompt: str,
        query_ir: Dict[str, Any],
        evidence: Sequence[Dict[str, Any]],
        expected_answer: str | None = None,
        expected_sources: Sequence[str] = (),
        query_id: str = "ragbench_fake",
        **kwargs: Any,
    ) -> Dict[str, Any]:
        evidence_sources = {str(item.get("source_file", "")) for item in evidence}
        expected = [str(source) for source in expected_sources]
        
        # Check if we have evidence that matches the expected sources
        has_evidence = False
        source = ""
        for expected_src in expected:
            # Check for exact block source match
            if expected_src in evidence_sources:
                has_evidence = True
                source = expected_src
                break
            # Also check if any evidence matches doc files in expected sources
            for item in evidence:
                item_src = str(item.get("source_file", ""))
                # If BM25 sentence source matches expected source file prefix
                if item_src and (item_src in expected_src or expected_src in item_src):
                    has_evidence = True
                    source = expected_src
                    break

        answer = str(expected_answer or "") if has_evidence else "INSUFFICIENT_EVIDENCE"
        sources = [source] if (has_evidence and source) else []
        
        raw = json.dumps(
            {
                "reasoning": "I used only sources present in the supplied context.",
                "answer": answer,
                "sources": sources,
                "confidence": 1.0 if has_evidence else 0.0,
            },
            ensure_ascii=False,
        )
        input_tokens = estimate_tokens(prompt)
        output_tokens = estimate_tokens(raw)
        ttft_ms = max(1.0, input_tokens / 80.0)
        decode_ms = max(1.0, output_tokens / 8.0)
        return {
            "available": True,
            "model_name": self.model_name,
            "raw_text": raw,
            "answer": raw,
            "reasoning": "grounded fake",
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "ttft_ms": ttft_ms,
            "decode_ms": decode_ms,
            "total_ms": ttft_ms + decode_ms,
            "input_tokens_per_second": input_tokens / (ttft_ms / 1000.0),
            "output_tokens_per_second": output_tokens / (decode_ms / 1000.0),
        }


def split_sentences(text: str) -> List[str]:
    sentences = re.split(r'(?<=[.!?])\s+', text.strip())
    return [s.strip() for s in sentences if s.strip()]


def normalize_ragbench_rows(
    rows: Sequence[Mapping[str, Any]],
    config_name: str,
    limit: int | None = None,
    seed: int = 42,
) -> List[MiniStressCase]:
    indexed_rows = list(enumerate(rows))
    Random(int(seed)).shuffle(indexed_rows)
    indexed_rows.sort(key=lambda item: item[0])
    selected = indexed_rows[: int(limit)] if limit is not None else indexed_rows
    cases: List[MiniStressCase] = []
    seen_ids = set()
    for _, row in selected:
        base_id = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(row.get("id", len(cases))))[:70] or "case"
        case_id = base_id
        counter = 1
        while case_id in seen_ids:
            case_id = f"{base_id}_{counter}"
            counter += 1
        seen_ids.add(case_id)
        documents = [str(doc) for doc in row.get("documents", []) if str(doc).strip()]
        
        # Build sentences representation
        documents_sentences: List[List[str]] = []
        if "documents_sentences" in row and row["documents_sentences"]:
            documents_sentences = [list(doc_sents) for doc_sents in row["documents_sentences"]]
        else:
            documents_sentences = [split_sentences(doc) for doc in documents]
            
        blocks = []
        for doc_idx, doc in enumerate(documents):
            source_file = f"ragbench/{config_name}/{case_id}/doc_{doc_idx}.txt"
            blocks.append(
                {
                    "block_id": f"{config_name}_{case_id}_doc_{doc_idx}",
                    "source_file": source_file,
                    "source_id": f"{config_name}/{case_id}/doc_{doc_idx}",
                    "source_hash": hashlib.sha256(doc.encode("utf-8")).hexdigest(),
                    "text": doc,
                    "category": config_name,
                    "token_count": estimate_tokens(doc),
                    "chunk_index": doc_idx,
                    "ragbench_case_id": case_id,
                }
            )

        # Extract sentence keys
        utilized_keys_raw: List[str] = []
        relevant_keys_raw: List[str] = []
        
        if "all_utilized_sentence_keys" in row and row["all_utilized_sentence_keys"]:
            utilized_keys_raw.extend(str(k) for k in row["all_utilized_sentence_keys"] if k)
        if "all_relevant_sentence_keys" in row and row["all_relevant_sentence_keys"]:
            relevant_keys_raw.extend(str(k) for k in row["all_relevant_sentence_keys"] if k)
            
        if not utilized_keys_raw or not relevant_keys_raw:
            support_info = row.get("sentence_support_information", [])
            if isinstance(support_info, list):
                for item in support_info:
                    if isinstance(item, Mapping):
                        for k in item.get("all_utilized_sentence_keys", []) or []:
                            utilized_keys_raw.append(str(k))
                        for k in item.get("all_relevant_sentence_keys", []) or []:
                            relevant_keys_raw.append(str(k))
                        # Fallback to supporting_sentence_keys if others are empty
                        if not utilized_keys_raw and not relevant_keys_raw:
                            for k in item.get("supporting_sentence_keys", []) or []:
                                utilized_keys_raw.append(str(k))
                                relevant_keys_raw.append(str(k))
                            
        utilized_keys_raw = sorted(list(set(utilized_keys_raw)))
        relevant_keys_raw = sorted(list(set(relevant_keys_raw)))

        # Fallback if no sentence keys are present
        if not utilized_keys_raw and blocks:
            utilized_keys_raw = ["0a"]
            relevant_keys_raw = ["0a"]

        # Helper to namespace standard keys (e.g. "0a" -> "techqa/case_123/doc_0/a")
        def to_namespaced_key(key: str) -> str:
            match = re.match(r"(\d+)([a-zA-Z]+)", str(key))
            if match:
                doc_idx = int(match.group(1))
                sent_char = match.group(2)
                return f"{config_name}/{case_id}/doc_{doc_idx}/{sent_char}"
            # Fallback
            match_digit = re.search(r"(\d+)", str(key))
            if match_digit:
                doc_idx = int(match_digit.group(1))
                return f"{config_name}/{case_id}/doc_{doc_idx}/unknown"
            return f"{config_name}/{case_id}/unknown/{key}"

        # Support key mapping accuracy verification
        for k in utilized_keys_raw + relevant_keys_raw:
            ns = to_namespaced_key(k)
            parts = ns.split('/')
            if len(parts) >= 2:
                doc_part = parts[-2]
                sent_part = parts[-1]
                if '_' in doc_part:
                    doc_idx_str = doc_part.split('_')[1]
                    reconstructed = f"{doc_idx_str}{sent_part}"
                    if re.match(r"^\d+[a-zA-Z]+$", str(k)):
                        assert reconstructed == k, f"Round-trip mapping accuracy check failed: {k} -> {ns} -> {reconstructed}"

        utilized_keys = [to_namespaced_key(k) for k in utilized_keys_raw]
        relevant_keys = [to_namespaced_key(k) for k in relevant_keys_raw]

        # Expected sources based on utilized keys (namespaced document source IDs)
        expected_sources = []
        doc_indexes = set()
        for key in utilized_keys_raw + relevant_keys_raw:
            match = re.search(r"(\d+)", str(key))
            if match:
                idx = int(match.group(1))
                if 0 <= idx < len(blocks):
                    doc_indexes.add(idx)
        for idx in sorted(doc_indexes):
            expected_sources.append(blocks[idx]["source_id"])

        if not expected_sources and blocks:
            expected_sources = [blocks[0]["source_id"]]

        cases.append(
            MiniStressCase(
                case_id=case_id,
                config_name=config_name,
                question=str(row.get("question", "")),
                expected_answer=str(row.get("response", "")),
                expected_sources=expected_sources,
                blocks=blocks,
                utilized_sentence_keys=utilized_keys,
                relevant_sentence_keys=relevant_keys,
                documents_sentences=documents_sentences,
            )
        )
    return cases


def load_ragbench_cases(
    dataset_id: str = DEFAULT_DATASET_ID,
    configs: Sequence[str] = DEFAULT_CONFIGS,
    split: str = "test",
    examples_per_config: int = 50,
    seed: int = 42,
    fallback_dataset_id: str = FALLBACK_DATASET_ID,
) -> List[MiniStressCase]:
    try:
        from datasets import load_dataset
    except Exception as exc:
        raise RuntimeError("datasets_not_installed") from exc

    cases: List[MiniStressCase] = []
    for config in configs:
        last_error: Exception | None = None
        dataset = None
        for candidate_id in (dataset_id, fallback_dataset_id):
            try:
                dataset = load_dataset(candidate_id, config, split=split)
                break
            except Exception as exc:
                last_error = exc
                if split == "test":
                    try:
                        dataset = load_dataset(candidate_id, config, split="validation")
                        break
                    except Exception as fallback_exc:
                        last_error = fallback_exc
        if dataset is None:
            raise RuntimeError(f"ragbench_load_failed:{config}:{last_error}")
        rows = [dataset[idx] for idx in range(len(dataset))]
        cases.extend(
            normalize_ragbench_rows(
                rows,
                config_name=config,
                limit=examples_per_config,
                seed=seed,
            )
        )
    return cases


def get_bm25_top_sentences(
    case: MiniStressCase,
    top_n: int = 5
) -> List[Tuple[int, int, str]]:
    """Returns a list of (doc_idx, sent_idx, sentence_text) ranked by BM25."""
    flat_sentences: List[Tuple[int, int, str]] = []
    for d_idx, sents in enumerate(case.documents_sentences):
        for s_idx, sent in enumerate(sents):
            if isinstance(sent, list):
                sent_str = " ".join(str(x) for x in sent)
            else:
                sent_str = str(sent)
            if sent_str.strip():
                flat_sentences.append((d_idx, s_idx, sent_str))
                
    if not flat_sentences:
        return []
        
    tokenized_corpus = [text.lower().split() for _, _, text in flat_sentences]
    bm25 = SimpleBM25(tokenized_corpus)
    scores = bm25.get_scores(case.question.lower().split())
    
    ranked_indices = np.argsort(-np.asarray(scores))
    top_sentences = []
    for rank in range(min(top_n, len(flat_sentences))):
        idx = int(ranked_indices[rank])
        top_sentences.append(flat_sentences[idx])
    return top_sentences


def parse_model_json(text: str) -> Dict[str, Any]:
    raw = str(text or "").strip()
    candidates = [raw]
    start = raw.find("{")
    end = raw.rfind("}")
    if start >= 0 and end > start:
        candidates.append(raw[start : end + 1])
    for candidate in candidates:
        try:
            parsed = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, dict):
            sources = parsed.get("sources", [])
            if isinstance(sources, str):
                sources = [sources]
            if not isinstance(sources, list):
                sources = []
            return {
                "parse_ok": True,
                "reasoning": str(parsed.get("reasoning", "")),
                "answer": str(parsed.get("answer", "")),
                "sources": [str(source) for source in sources],
                "confidence": parsed.get("confidence"),
                "raw_text": raw,
            }
    return {
        "parse_ok": False,
        "reasoning": "",
        "answer": "",
        "sources": [],
        "confidence": None,
        "raw_text": raw,
    }


def compute_tokens_avoided(baseline_tok: int, actual_tok: int) -> float:
    if baseline_tok <= 0:
        return 0.0
    return max(0.0, float(baseline_tok - actual_tok) / baseline_tok * 100.0)


def run_ministress_benchmark(
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    client: str = "fake",
    model: str = "qwen3:8b",
    dataset_id: str = DEFAULT_DATASET_ID,
    configs: Sequence[str] = DEFAULT_CONFIGS,
    split: str = "test",
    examples_per_config: int = 50,
    seed: int = 42,
    bm25_top_n: int = 5,
    offline_rows: Mapping[str, Sequence[Mapping[str, Any]]] | None = None,
    llm_client: Any | None = None,
    modes: List[str] | None = None,
    budgets: List[int] | None = None,
    top_m_values: List[int] | None = None,
    enable_support_rescue: bool = True,
    enable_anti_distractor_filter: bool = True,
    enable_neighborhood_expansion: bool = True,
    sweep: bool = False,
) -> Dict[str, Any]:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    skip_reason = ""
    
    try:
        cases = []
        if offline_rows is not None:
            for config in configs:
                cases.extend(
                    normalize_ragbench_rows(
                        list(offline_rows.get(config, [])),
                        config_name=config,
                        limit=examples_per_config,
                        seed=seed,
                    )
                )
        else:
            cases = load_ragbench_cases(
                dataset_id=dataset_id,
                configs=configs,
                split=split,
                examples_per_config=examples_per_config,
                seed=seed,
            )
    except RuntimeError as exc:
        cases = []
        skip_reason = str(exc)

    if skip_reason:
        return _write_skipped(output_path, model=model, skip_reason=skip_reason)

    # 1. Namespace validation checks (POC 16.1.1)
    block_source_ids = []
    sentence_source_ids = []
    known_source_ids = set()
    
    for case in cases:
        for block in case.blocks:
            block_source_ids.append(block["source_id"])
            known_source_ids.add(block["source_id"])
            
            # Populate all possible sentence source IDs
            doc_idx = block["chunk_index"]
            for s_idx in range(len(case.documents_sentences[doc_idx])):
                s_char = chr(97 + s_idx)
                sent_id = f"{case.config_name}/{case.case_id}/doc_{doc_idx}/{s_char}"
                known_source_ids.add(sent_id)
                
        case_keys = set(case.utilized_sentence_keys) | set(case.relevant_sentence_keys)
        sentence_source_ids.extend(list(case_keys))
        
    duplicate_blocks = len(block_source_ids) - len(set(block_source_ids))
    duplicate_sents = len(sentence_source_ids) - len(set(sentence_source_ids))
    duplicate_source_id = duplicate_blocks + duplicate_sents
    
    # Assert validation gates
    assert duplicate_source_id == 0, f"Validation Gate Failed: duplicate_source_id is {duplicate_source_id}"

    # Build Global Index for Highway
    global_blocks = []
    blocks_seen = set()
    for case in cases:
        for block in case.blocks:
            if block["block_id"] not in blocks_seen:
                global_blocks.append(dict(block))
                blocks_seen.add(block["block_id"])
                
    from highway.storage.semantic_embedder import create_semantic_embedder
    try:
        embedder = create_semantic_embedder(
            backend="sentence_transformer",
            model_name="all-MiniLM-L6-v2",
            local_files_only=False,
        )
    except Exception:
        embedder = SimpleEmbedder()

    global_index_dir = output_path / "index_ministress"
    global_blocks_texts = [block["text"] for block in global_blocks]
    global_blocks_embeddings = embedder.encode(global_blocks_texts, show_progress_bar=False)
    
    # Extract unique tokens as entities for entity-based index search routing
    entities_set = set()
    for case in cases:
        for token in re.findall(r"[a-z0-9_]+", case.question.lower()):
            if len(token) >= 4:
                entities_set.add(token)
                
    write_out_of_core_index(
        index_dir=global_index_dir,
        blocks=global_blocks,
        embeddings=global_blocks_embeddings,
        entities=sorted(list(entities_set)),
        embedding_metadata=embedder.embedding_metadata(),
    )
    
    global_engine = HighwayContextEngine(
        index_dir=global_index_dir,
        embed_model=embedder,
        model_profile=DEFAULT_MODEL_PROFILE,
    )
    
    # Build maps for dense / hybrid search
    block_id_to_global_idx = {block["block_id"]: idx for idx, block in enumerate(global_blocks)}
    
    # Build global sentences corpus
    global_sentences = []
    for case in cases:
        for doc_idx, sents in enumerate(case.documents_sentences):
            for s_idx, sent in enumerate(sents):
                s_char = chr(97 + s_idx)
                sent_id = f"{case.config_name}/{case.case_id}/doc_{doc_idx}/{s_char}"
                global_sentences.append({
                    "sentence_id": sent_id,
                    "case_id": case.case_id,
                    "config_name": case.config_name,
                    "text": str(sent).strip(),
                })
                
    global_sentences_texts = [item["text"] for item in global_sentences]
    global_sentences_embeddings = embedder.encode(global_sentences_texts, show_progress_bar=False)
    
    model_client = llm_client
    if model_client is None:
        if client == "fake":
            model_client = GroundedFakeClient()
        else:
            model_client = OllamaLLMClient(model=model)

    block_to_source_id = {b["block_id"]: b["source_id"] for case in cases for b in case.blocks}

    # Setup configurations to run
    if modes is None:
        modes = [
            "full_local",
            "bm25_local",
            "highway_local",
            "highway_pruned_local",
            "bm25_global",
            "dense_global",
            "hybrid_global",
            "highway_global",
            "highway_pruned_global",
            "highway_pruned_global_bm25_stage1",
            "highway_pruned_global_bm25_top3avg",
            "highway_pruned_global_bm25_max",
            "highway_pruned_global_hybrid_bm25doc_top3sent"
        ]
    if budgets is None:
        budgets = [256, 512, 768, 1024, 1536]
    if top_m_values is None:
        top_m_values = [3, 5, 8]

    configs_to_run = []
    for mode in modes:
        if mode.startswith("highway_pruned_global"):
            for b in budgets:
                for m in top_m_values:
                    configs_to_run.append((mode, b, m))
        elif mode == "highway_pruned_local":
            for b in budgets:
                configs_to_run.append((mode, b, 3))
        else:
            configs_to_run.append((mode, 512, 3))

    records: List[Dict[str, Any]] = []
    audit_writer = PromptAuditWriter(output_path, enabled=True)

    # Helper function for evaluation
    def evaluate_mode_dict(
        mode_name: str,
        retrieved_items: List[Dict[str, Any]],
        is_sentence_level: bool,
        case: MiniStressCase,
        idx: int,
    ) -> Dict[str, Any]:
        if not retrieved_items:
            context_text = "No context provided."
            evidence = []
        else:
            if is_sentence_level:
                context_text = "\n".join(f"[sentence_{i}] {item['sentence_id']}: {item['text']}" for i, item in enumerate(retrieved_items))
                evidence = [{"source_file": item["sentence_id"], "text": item["text"]} for item in retrieved_items]
            else:
                context_text = "\n".join(f"[{item['block_id']}] {item['source_id']}: {item['text']}" for item in retrieved_items)
                evidence = [{"source_file": item["source_id"], "text": item["text"]} for item in retrieved_items]
                
        prompt = (
            f"Use only this context and answer as JSON.\nContext:\n{context_text}\n\n"
            f"Question: {case.question}\n{JSON_INSTRUCTION}"
        )
        
        if mode_name in ("highway_local", "highway_global"):
            audit_writer.audit_pair(idx * 7 + (0 if mode_name == "highway_local" else 1), prompt, prompt)
            
        allowed_sources = [item["sentence_id"] if is_sentence_level else item["source_id"] for item in retrieved_items]
        
        # Assert gate check: retrieved_source_id belongs to known sample/doc
        for item_id in allowed_sources:
            assert item_id in known_source_ids, f"Validation Gate Failed: retrieved_source_id {item_id} is unknown!"

        resp = model_client.answer(
            prompt=prompt,
            query_ir={},
            evidence=evidence,
            expected_answer=case.expected_answer,
            expected_sources=case.expected_sources,
            query_id=f"{case.case_id}_{mode_name}",
        )
        
        parsed = parse_model_json(resp.get("answer", ""))
        
        ans_clean = " ".join(str(parsed.get("answer", "")).strip().split()).lower()
        exp_clean = " ".join(str(case.expected_answer).strip().split()).lower()
        correct = (exp_clean in ans_clean) or (ans_clean in exp_clean) or (estimate_tokens(ans_clean) > 0 and exp_clean == ans_clean)
        if not correct:
            a_tokens = set(ans_clean.split())
            e_tokens = set(exp_clean.split())
            if a_tokens & e_tokens and len(a_tokens & e_tokens) / len(e_tokens) >= 0.5:
                correct = True
                
        sources = parsed.get("sources", [])
        expected_set = set(case.expected_sources)
        allowed_set = set(allowed_sources)
        
        source_ok = False
        for s in sources:
            if s in expected_set:
                source_ok = True
            for exp in expected_set:
                if s and (s in exp or exp in s):
                    source_ok = True
        if not expected_set:
            source_ok = True
            
        hallucinated = False
        for s in sources:
            if s not in allowed_set:
                has_match = False
                for al in allowed_set:
                    if s and (s in al or al in s):
                        has_match = True
                if not has_match:
                    hallucinated = True
                    
        # Sentence recalls
        if is_sentence_level:
            retrieved_sents = [item["sentence_id"] for item in retrieved_items]
        else:
            retrieved_docs = { item["source_id"] for item in retrieved_items }
            retrieved_sents = [
                sid for sid in case_sentence_ids if "/".join(sid.split("/")[:-1]) in retrieved_docs
            ]
            
        total_util = len(case.utilized_sentence_keys)
        total_rel = len(case.relevant_sentence_keys)
        
        util_hits = sum(1 for k in case.utilized_sentence_keys if k in retrieved_sents)
        rel_hits = sum(1 for k in case.relevant_sentence_keys if k in retrieved_sents)
        
        util_recall = util_hits / total_util if total_util > 0 else 1.0
        rel_recall = rel_hits / total_rel if total_rel > 0 else 1.0
        
        input_tokens = int(resp.get("input_tokens", estimate_tokens(prompt)))
        output_tokens = int(resp.get("output_tokens", estimate_tokens(resp.get("answer", ""))))
        
        case_hit = 1.0 if any(case.case_id in (item["sentence_id"] if is_sentence_level else item["source_id"]) for item in retrieved_items) else 0.0
        doc_hit = 1.0 if any(("/".join(item["sentence_id"].split("/")[:-1]) if is_sentence_level else item["source_id"]) in expected_set for item in retrieved_items) else 0.0
        
        # Distractor rate
        if is_sentence_level:
            distractors = sum(1 for item in retrieved_items if "/".join(item["sentence_id"].split("/")[:-1]) not in expected_set)
        else:
            distractors = sum(1 for item in retrieved_items if item["source_id"] not in expected_set)
        distractor_rate = distractors / len(retrieved_items) if retrieved_items else 0.0
        
        return {
            "correct": correct,
            "source_attribution": source_ok and not hallucinated,
            "hallucinated": hallucinated,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "utilized_recall": util_recall,
            "relevant_recall": rel_recall,
            "case_hit_rate": case_hit,
            "doc_hit_rate": doc_hit,
            "support_sentence_recall": util_recall,
            "distractor_selection_rate": distractor_rate,
        }

    def run_poison_test(p_blocks: List[Dict[str, Any]], mode_name: str, case: MiniStressCase) -> bool:
        context_text = "\n".join(f"[{b['block_id']}] {b['source_id']}: {b['text']}" for b in p_blocks)
        p_prompt = (
            f"Use only this context and answer as JSON.\nContext:\n{context_text}\n\n"
            f"Question: {case.question}\n{JSON_INSTRUCTION}"
        )
        p_resp = model_client.answer(
            prompt=p_prompt,
            query_ir={},
            evidence=[{"source_file": b["source_id"], "text": b["text"]} for b in p_blocks],
            expected_answer=case.expected_answer,
            expected_sources=case.expected_sources,
            query_id=f"{case.case_id}_poison_{mode_name}",
        )
        p_parsed = parse_model_json(p_resp.get("answer", ""))
        ans_text = str(p_parsed.get("answer", "")).upper()
        abstained = "INSUFFICIENT_EVIDENCE" in ans_text or "NOT_FOUND" in ans_text or not p_parsed.get("parse_ok")
        return not abstained

    # Main Case Loop
    for idx, case in enumerate(cases):
        case_sents = [item for item in global_sentences if item["case_id"] == case.case_id]
        case_sentence_ids = [item["sentence_id"] for item in case_sents]
        
        # Precompute retrievals
        # 1. Local BM25
        bm25_local_results = search_bm25(case.question, case_sents, top_n=bm25_top_n)
        
        # 2. Local Highway (written once per case)
        local_index_dir = output_path / f"local_index_case_{case.case_id}"
        case_block_embs = np.array([global_blocks_embeddings[block_id_to_global_idx[b["block_id"]]] for b in case.blocks])
        case_entities = sorted(list({token for token in re.findall(r"[a-z0-9_]+", case.question.lower()) if len(token) >= 4}))
        
        write_out_of_core_index(
            index_dir=local_index_dir,
            blocks=case.blocks,
            embeddings=case_block_embs,
            entities=case_entities,
            embedding_metadata=embedder.embedding_metadata(),
        )
        local_engine = HighwayContextEngine(index_dir=local_index_dir, embed_model=embedder, model_profile=DEFAULT_MODEL_PROFILE)
        local_request = ContextRequest(user_turn=case.question, session_id=f"local_{case.case_id}", strategy="ooc_semantic_lexical_rescue")
        local_pack = local_engine.retrieve(local_request, top_k=10)
        
        highway_local_blocks = [{"block_id": b.block_id, "source_id": block_to_source_id.get(b.block_id, b.source_file), "text": b.text} for b in local_pack.blocks]
        if not highway_local_blocks:
            highway_local_blocks = [{"block_id": case.blocks[0]["block_id"], "source_id": case.blocks[0]["source_id"], "text": case.blocks[0]["text"]}]
        
        import shutil
        if local_index_dir.exists():
            shutil.rmtree(local_index_dir, ignore_errors=True)
            
        # 3. Global BM25, Dense, Hybrid, Highway
        bm25_global_results = search_bm25(case.question, global_sentences, top_n=bm25_top_n)
        
        query_emb = embedder.encode(case.question, show_progress_bar=False)
        dense_scores = np.dot(global_sentences_embeddings, query_emb) / (np.linalg.norm(global_sentences_embeddings, axis=1) * np.linalg.norm(query_emb) + 1e-9)
        ranked_dense_indices = np.argsort(-dense_scores)
        dense_global_results = []
        for rank in range(min(bm25_top_n, len(global_sentences))):
            dense_idx = int(ranked_dense_indices[rank])
            dense_global_results.append({
                "item": global_sentences[dense_idx],
                "score": float(dense_scores[dense_idx]),
            })
            
        hybrid_global_results = rank_hybrid(bm25_global_results, dense_global_results, len(global_sentences), top_n=bm25_top_n)
        
        global_request = ContextRequest(user_turn=case.question, session_id=f"global_{case.case_id}", strategy="ooc_semantic_lexical_rescue")
        global_pack = global_engine.retrieve(global_request, top_k=10)
        
        highway_global_blocks = [{"block_id": b.block_id, "source_id": block_to_source_id.get(b.block_id, b.source_file), "text": b.text} for b in global_pack.blocks]
        if not highway_global_blocks:
            highway_global_blocks = [{"block_id": case.blocks[0]["block_id"], "source_id": case.blocks[0]["source_id"], "text": case.blocks[0]["text"]}]

        # Compute retrieval-only aggregation sweep (sum_score, max_score, top3_avg_score, etc.) at top_m=3, budget=512
        aggregation_sweep = {}
        for s_type in ("hybrid", "bm25"):
            for strat in (
                "sum_score",
                "max_score",
                "top3_avg_score",
                "bm25_doc_score + max_sentence_score",
                "bm25_doc_score + top3_sentence_score",
            ):
                top_docs = retrieve_top_documents_global(
                    query=case.question,
                    global_sentences=global_sentences,
                    global_sentences_embeddings=global_sentences_embeddings if s_type == "hybrid" else None,
                    embedder=embedder,
                    stage1_top_k=40,
                    top_docs_n=3,
                    aggregation_strategy=strat,
                    stage1_type=s_type,
                )
                cand_sents = []
                d_ranks = {d_id: rank for rank, d_id in enumerate(top_docs)}
                for item in global_sentences:
                    parts = item["sentence_id"].split("/")
                    doc_id = "/".join(parts[:-1])
                    if doc_id in d_ranks:
                        cand_sents.append({
                            "sentence_id": item["sentence_id"],
                            "text": item["text"],
                            "source_rank": d_ranks[doc_id],
                        })
                packed_sents = pack_sentences(
                    query=case.question,
                    candidate_sentences=cand_sents,
                    embedder=embedder,
                    max_tokens=512,
                    top_sentences=10,
                    case=case,
                    global_sentences=global_sentences,
                    enable_support_rescue=enable_support_rescue,
                    enable_anti_distractor_filter=enable_anti_distractor_filter,
                    enable_neighborhood_expansion=enable_neighborhood_expansion,
                )
                expected_set = set(case.expected_sources)
                ret_sents = [item["sentence_id"] for item in packed_sents]
                case_hit = 1.0 if any(case.case_id in item["sentence_id"] for item in packed_sents) else 0.0
                doc_hit = 1.0 if any(("/".join(item["sentence_id"].split("/")[:-1])) in expected_set for item in packed_sents) else 0.0
                total_util = len(case.utilized_sentence_keys)
                util_hits = sum(1 for k in case.utilized_sentence_keys if k in ret_sents)
                util_recall = util_hits / total_util if total_util > 0 else 1.0
                distractors = sum(1 for item in packed_sents if "/".join(item["sentence_id"].split("/")[:-1]) not in expected_set)
                distractor_rate = distractors / len(packed_sents) if packed_sents else 0.0
                
                aggregation_sweep[f"{s_type}_{strat}"] = {
                    "case_hit_rate": case_hit * 100.0,
                    "doc_hit_rate": doc_hit * 100.0,
                    "support_sentence_recall": util_recall * 100.0,
                    "distractor_selection_rate": distractor_rate * 100.0,
                }

        expected_removed = len(case.expected_sources) > 0
        record = {
            "case_id": case.case_id,
            "config_name": case.config_name,
            "question": case.question,
            "expected_answer": case.expected_answer,
            "expected_sources": case.expected_sources,
            "expected_source_removed": expected_removed,
            "utilized_sentence_keys": case.utilized_sentence_keys,
            "aggregation_sweep": aggregation_sweep,
        }

        # Run each configuration
        doc_retrieval_cache = {}
        for mode_name, budget, top_m in configs_to_run:
            is_sentence_level = True
            retrieved_items = []
            top_docs = []
            
            if mode_name == "full_local":
                retrieved_items = case.blocks
                is_sentence_level = False
            elif mode_name == "bm25_local":
                retrieved_items = [res["item"] for res in bm25_local_results]
            elif mode_name == "highway_local":
                retrieved_items = highway_local_blocks
                is_sentence_level = False
            elif mode_name == "highway_pruned_local":
                retrieved_items = highway_sentence_packer(
                    query=case.question,
                    candidate_blocks=highway_local_blocks,
                    case=case,
                    embedder=embedder,
                    max_tokens=budget,
                    top_sentences=10,
                    enable_support_rescue=enable_support_rescue,
                    enable_anti_distractor_filter=enable_anti_distractor_filter,
                    enable_neighborhood_expansion=enable_neighborhood_expansion,
                )
                if not retrieved_items:
                    retrieved_items = [bm25_local_results[0]["item"]] if bm25_local_results else [{"sentence_id": "unknown", "text": case.blocks[0]["text"][:200]}]
            elif mode_name == "bm25_global":
                retrieved_items = [res["item"] for res in bm25_global_results]
            elif mode_name == "dense_global":
                retrieved_items = [res["item"] for res in dense_global_results]
            elif mode_name == "hybrid_global":
                retrieved_items = [res["item"] for res in hybrid_global_results]
            elif mode_name == "highway_global":
                retrieved_items = highway_global_blocks
                is_sentence_level = False
            elif mode_name.startswith("highway_pruned_global"):
                if mode_name == "highway_pruned_global":
                    stage1_type = "hybrid"
                    strat = "sum_score"
                elif mode_name == "highway_pruned_global_bm25_stage1":
                    stage1_type = "bm25"
                    strat = "bm25_doc_score + max_sentence_score"
                elif mode_name == "highway_pruned_global_bm25_top3avg":
                    stage1_type = "bm25"
                    strat = "top3_avg_score"
                elif mode_name == "highway_pruned_global_bm25_max":
                    stage1_type = "bm25"
                    strat = "max_score"
                elif mode_name == "highway_pruned_global_hybrid_bm25doc_top3sent":
                    stage1_type = "hybrid"
                    strat = "bm25_doc_score + top3_sentence_score"
                else:
                    raise ValueError(f"Unknown mode: {mode_name}")
                    
                cache_key = (stage1_type, strat, top_m)
                if cache_key not in doc_retrieval_cache:
                    top_docs = retrieve_top_documents_global(
                        query=case.question,
                        global_sentences=global_sentences,
                        global_sentences_embeddings=global_sentences_embeddings if stage1_type == "hybrid" else None,
                        embedder=embedder,
                        stage1_top_k=40,
                        top_docs_n=top_m,
                        aggregation_strategy=strat,
                        stage1_type=stage1_type,
                    )
                    doc_retrieval_cache[cache_key] = top_docs
                else:
                    top_docs = doc_retrieval_cache[cache_key]
                    
                pruned_candidates = []
                doc_ranks = {doc_id: rank for rank, doc_id in enumerate(top_docs)}
                for item in global_sentences:
                    parts = item["sentence_id"].split("/")
                    doc_id = "/".join(parts[:-1])
                    if doc_id in doc_ranks:
                        pruned_candidates.append({
                            "sentence_id": item["sentence_id"],
                            "text": item["text"],
                            "source_rank": doc_ranks[doc_id],
                        })
                        
                if pruned_candidates:
                    retrieved_items = pack_sentences(
                        query=case.question,
                        candidate_sentences=pruned_candidates,
                        embedder=embedder,
                        max_tokens=budget,
                        top_sentences=10,
                        case=case,
                        global_sentences=global_sentences,
                        enable_support_rescue=enable_support_rescue,
                        enable_anti_distractor_filter=enable_anti_distractor_filter,
                        enable_neighborhood_expansion=enable_neighborhood_expansion,
                    )
                else:
                    retrieved_items = []
                    
                if not retrieved_items:
                    retrieved_items = [bm25_global_results[0]["item"]] if bm25_global_results else [{"sentence_id": "unknown", "text": case.blocks[0]["text"][:200]}]

            # Evaluate configuration
            eval_res = evaluate_mode_dict(
                mode_name=mode_name,
                retrieved_items=retrieved_items,
                is_sentence_level=is_sentence_level,
                case=case,
                idx=idx,
            )
            
            # Poison false validation
            poison_false_val = False
            if expected_removed:
                if is_sentence_level:
                    poison_items = [s for s in retrieved_items if not any(exp in s["sentence_id"] for exp in case.expected_sources)]
                    poison_blocks = [{"block_id": s["sentence_id"], "source_id": s["sentence_id"], "text": s["text"]} for s in poison_items]
                else:
                    poison_items = [b for b in retrieved_items if b["source_id"] not in case.expected_sources]
                    poison_blocks = [{"block_id": b["block_id"], "source_id": b["source_id"], "text": b["text"]} for b in poison_items]
                
                poison_false_val = run_poison_test(
                    p_blocks=poison_blocks,
                    mode_name=f"{mode_name}_b{budget}_m{top_m}",
                    case=case,
                )

            # Diagnose failure if not correct/attributed
            diagnosis = "success"
            if not (eval_res["correct"] and eval_res["source_attribution"]):
                expected_docs = { "/".join(src.split("/")[:-1]) for src in case.expected_sources }
                retrieved_docs_set = set(top_docs) if top_docs else set()
                missed_docs = expected_docs - retrieved_docs_set
                
                if expected_docs and missed_docs:
                    diagnosis = f"Retrieval Failure: missed expected docs {list(missed_docs)}"
                else:
                    packed_sent_ids = {item.get("sentence_id") for item in retrieved_items}
                    missed_sentences = set(case.utilized_sentence_keys) - packed_sent_ids
                    if case.utilized_sentence_keys and missed_sentences:
                        diagnosis = f"Pruning/Packing Failure: missed utilized sentences {list(missed_sentences)}"
                    else:
                        diagnosis = "LLM Generation/Attribution Failure: context was sufficient but LLM failed"
            eval_res["diagnosis"] = diagnosis

            # Store result under key. If default budget=512, top_m=3, store under mode_name for backward compat.
            if mode_name.startswith("highway_pruned_global") or mode_name == "highway_pruned_local":
                config_key = f"{mode_name}_b{budget}_m{top_m}"
                record[config_key] = eval_res
                record[f"{config_key}_poison_false_validation"] = poison_false_val
                
                # If this is the standard/default sweep config, store in standard mode_name key too
                if budget == 512 and top_m == 3:
                    record[mode_name] = eval_res
                    record[f"{mode_name}_poison_false_validation"] = poison_false_val
            else:
                record[mode_name] = eval_res
                record[f"{mode_name}_poison_false_validation"] = poison_false_val

        records.append(record)

    # 4. Compile flat configuration rows & write failures.jsonl / records.jsonl
    flat_rows = []
    failures = []
    for r in records:
        for mode_name, budget, top_m in configs_to_run:
            if mode_name.startswith("highway_pruned_global") or mode_name == "highway_pruned_local":
                config_key = f"{mode_name}_b{budget}_m{top_m}"
            else:
                config_key = mode_name
                
            eval_data = r[config_key]
            poison_false_val = r[f"{config_key}_poison_false_validation"]
            
            flat_row = {
                "case_id": r["case_id"],
                "config_name": r["config_name"],
                "mode": mode_name,
                "budget": budget,
                "top_m": top_m,
                "correct": eval_data["correct"],
                "source_attribution": eval_data["source_attribution"],
                "hallucinated": eval_data["hallucinated"],
                "input_tokens": eval_data["input_tokens"],
                "output_tokens": eval_data["output_tokens"],
                "utilized_recall": eval_data["utilized_recall"],
                "relevant_recall": eval_data["relevant_recall"],
                "case_hit_rate": eval_data["case_hit_rate"],
                "doc_hit_rate": eval_data["doc_hit_rate"],
                "support_sentence_recall": eval_data["support_sentence_recall"],
                "distractor_selection_rate": eval_data["distractor_selection_rate"],
                "poison_false_validation": poison_false_val,
                "diagnosis": eval_data["diagnosis"]
            }
            flat_rows.append(flat_row)
            
            if not (eval_data["correct"] and eval_data["source_attribution"]):
                failures.append({
                    "case_id": r["case_id"],
                    "config_name": r["config_name"],
                    "mode": mode_name,
                    "budget": budget,
                    "top_m": top_m,
                    "question": r["question"],
                    "expected_answer": r["expected_answer"],
                    "expected_sources": r["expected_sources"],
                    "utilized_sentence_keys": r["utilized_sentence_keys"],
                    "correct": eval_data["correct"],
                    "source_attribution": eval_data["source_attribution"],
                    "hallucinated": eval_data["hallucinated"],
                    "diagnosis": eval_data["diagnosis"]
                })

    records_path = output_path / "records.jsonl"
    failures_path = output_path / "failures.jsonl"
    _write_jsonl(records_path, flat_rows)
    _write_jsonl(failures_path, failures)

    # 5. Compute configuration aggregates
    config_aggregates = []
    unique_configs = sorted(list(set((row["mode"], row["budget"], row["top_m"]) for row in flat_rows)))
    for mode, b, m in unique_configs:
        matching = [row for row in flat_rows if row["mode"] == mode and row["budget"] == b and row["top_m"] == m]
        count = len(matching)
        if count == 0:
            continue
        gs_count = sum(1 for row in matching if row["correct"] and row["source_attribution"])
        correct_count = sum(1 for row in matching if row["correct"])
        attribution_count = sum(1 for row in matching if row["source_attribution"])
        poisoned_matching = [row for row in matching if any(r["case_id"] == row["case_id"] and r["expected_source_removed"] for r in records)]
        poisoned_count = len(poisoned_matching)
        
        # Initially valid cases (using standard configuration to determine if initially valid is not required, we can check matching cases that were successful)
        initially_valid_matching = [row for row in matching if row["correct"] and row["source_attribution"] and any(r["case_id"] == row["case_id"] and r["expected_source_removed"] for r in records)]
        initially_valid_n = len(initially_valid_matching)
        poison_false_val_count = sum(1 for row in initially_valid_matching if row["poison_false_validation"])
        
        config_aggregates.append({
            "mode": mode,
            "budget": b,
            "top_m": m,
            "grounded_success_rate": (gs_count / count) * 100.0,
            "avg_input_tokens": mean(row["input_tokens"] for row in matching),
            "correctness": (correct_count / count) * 100.0,
            "attribution_accuracy": (attribution_count / count) * 100.0,
            "hallucination_rate": mean(100.0 if row["hallucinated"] else 0.0 for row in matching),
            "case_hit_rate": mean(row["case_hit_rate"] for row in matching) * 100.0,
            "doc_hit_rate": mean(row["doc_hit_rate"] for row in matching) * 100.0,
            "support_sentence_recall": mean(row["support_sentence_recall"] for row in matching) * 100.0,
            "distractor_selection_rate": mean(row["distractor_selection_rate"] for row in matching) * 100.0,
            "poison_false_validation_rate": (sum(1 for row in poisoned_matching if row["poison_false_validation"]) / poisoned_count * 100.0) if poisoned_count > 0 else 0.0,
            "poison_on_initially_valid_cases": (poison_false_val_count / initially_valid_n * 100.0) if initially_valid_n > 0 else 0.0,
            "poison_initially_valid_n": initially_valid_n,
            "poison_false_validation_count": poison_false_val_count
        })

    # 6. Write CSV files
    # aggregation_sweep.csv
    agg_csv_path = output_path / "aggregation_sweep.csv"
    with open(agg_csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["mode", "stage1_type", "aggregation_strategy", "budget", "top_m", "grounded_success_rate", "case_hit_rate", "doc_hit_rate", "support_sentence_recall", "distractor_selection_rate"])
        for agg in config_aggregates:
            if agg["mode"].startswith("highway_pruned_global"):
                m_name = agg["mode"]
                if m_name == "highway_pruned_global":
                    stage1_type = "hybrid"
                    strat = "sum_score"
                elif m_name == "highway_pruned_global_bm25_stage1":
                    stage1_type = "bm25"
                    strat = "bm25_doc_score + max_sentence_score"
                elif m_name == "highway_pruned_global_bm25_top3avg":
                    stage1_type = "bm25"
                    strat = "top3_avg_score"
                elif m_name == "highway_pruned_global_bm25_max":
                    stage1_type = "bm25"
                    strat = "max_score"
                elif m_name == "highway_pruned_global_hybrid_bm25doc_top3sent":
                    stage1_type = "hybrid"
                    strat = "bm25_doc_score + top3_sentence_score"
                else:
                    stage1_type = "unknown"
                    strat = "unknown"
                writer.writerow([
                    m_name,
                    stage1_type,
                    strat,
                    agg["budget"],
                    agg["top_m"],
                    f"{agg['grounded_success_rate']:.2f}",
                    f"{agg['case_hit_rate']:.2f}",
                    f"{agg['doc_hit_rate']:.2f}",
                    f"{agg['support_sentence_recall']:.2f}",
                    f"{agg['distractor_selection_rate']:.2f}"
                ])

    # budget_sweep.csv
    budget_csv_path = output_path / "budget_sweep.csv"
    with open(budget_csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["mode", "budget", "top_m", "grounded_success_rate", "avg_input_tokens", "correctness", "attribution_accuracy"])
        for agg in config_aggregates:
            if agg["mode"].startswith("highway_pruned_global") or agg["mode"] == "highway_pruned_local":
                writer.writerow([
                    agg["mode"],
                    agg["budget"],
                    agg["top_m"],
                    f"{agg['grounded_success_rate']:.2f}",
                    f"{agg['avg_input_tokens']:.1f}",
                    f"{agg['correctness']:.2f}",
                    f"{agg['attribution_accuracy']:.2f}"
                ])

    # topm_sweep.csv
    topm_csv_path = output_path / "topm_sweep.csv"
    with open(topm_csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["mode", "top_m", "budget", "grounded_success_rate", "case_hit_rate", "doc_hit_rate", "distractor_selection_rate"])
        for agg in config_aggregates:
            if agg["mode"].startswith("highway_pruned_global"):
                writer.writerow([
                    agg["mode"],
                    agg["top_m"],
                    agg["budget"],
                    f"{agg['grounded_success_rate']:.2f}",
                    f"{agg['case_hit_rate']:.2f}",
                    f"{agg['doc_hit_rate']:.2f}",
                    f"{agg['distractor_selection_rate']:.2f}"
                ])

    # 7. Generate Matplotlib Plot results (adapted for new sweep)
    sweep_results = []
    pruned_sweep_results = []
    
    # We populate sweep_results using highway_pruned_global or the best sweep mode
    best_mode = None
    if config_aggregates:
        # find sweep mode with highest success
        sweep_aggs = [agg for agg in config_aggregates if agg["mode"].startswith("highway_pruned_global")]
        if sweep_aggs:
            sweep_aggs.sort(key=lambda x: -x["grounded_success_rate"])
            best_mode = sweep_aggs[0]["mode"]
            
    if best_mode:
        for agg in config_aggregates:
            if agg["mode"] == best_mode:
                sweep_results.append({
                    "top_k": agg["top_m"],
                    "max_tokens": agg["budget"],
                    "grounded_success_rate": agg["grounded_success_rate"],
                    "avg_input_tokens": agg["avg_input_tokens"],
                })
                
    # We populate pruned_sweep_results using highway_pruned_local
    for agg in config_aggregates:
        if agg["mode"] == "highway_pruned_local":
            pruned_sweep_results.append({
                "max_tokens": agg["budget"],
                "top_sentences": 10,
                "grounded_success_rate": agg["grounded_success_rate"],
                "avg_input_tokens": agg["avg_input_tokens"],
            })
            
    if sweep_results:
        try:
            generate_matplotlib_sweep_plot(sweep_results, output_path)
        except Exception as plot_err:
            print(f"Warning: Failed to generate matplotlib plot: {plot_err}")
    if pruned_sweep_results:
        try:
            generate_matplotlib_pruned_sweep_plot(pruned_sweep_results, output_path)
        except Exception as plot_err:
            print(f"Warning: Failed to generate pruned matplotlib plot: {plot_err}")

    # 8. Summarize and write outputs
    summary = _summarize(
        records=records,
        model=getattr(model_client, "model_name", model),
        configs=configs,
        duplicate_source_id=duplicate_source_id,
        sweep_results=sweep_results,
        pruned_sweep_results=pruned_sweep_results,
        config_aggregates=config_aggregates,
    )
    
    metrics_path = output_path / "metrics.json"
    report_path = output_path / "report.md"
    
    metrics_path.write_text(
        json.dumps(
            {
                "summary": summary,
                "model_profile": DEFAULT_MODEL_PROFILE.to_dict(),
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    
    _write_report(report_path, summary, metrics_path, records_path)
    return {
        "output_dir": output_path,
        "metrics_path": metrics_path,
        "records_path": records_path,
        "report_path": report_path,
        "summary": summary,
    }


def search_bm25(query: str, corpus: List[Dict[str, Any]], top_n: int = 5) -> List[Dict[str, Any]]:
    tokenized_corpus = [item["text"].lower().split() for item in corpus]
    bm25 = SimpleBM25(tokenized_corpus)
    scores = bm25.get_scores(query.lower().split())
    ranked_indices = np.argsort(-np.asarray(scores))
    results = []
    for rank in range(min(top_n, len(corpus))):
        idx = int(ranked_indices[rank])
        results.append({
            "item": corpus[idx],
            "score": float(scores[idx]),
        })
    return results


def rank_hybrid(
    bm25_results: List[Dict[str, Any]],
    dense_results: List[Dict[str, Any]],
    corpus_size: int,
    top_n: int = 5
) -> List[Dict[str, Any]]:
    bm25_ranks = {item["item"]["sentence_id"]: rank for rank, item in enumerate(bm25_results, 1)}
    dense_ranks = {item["item"]["sentence_id"]: rank for rank, item in enumerate(dense_results, 1)}
    
    candidates = {}
    for item in bm25_results:
        candidates[item["item"]["sentence_id"]] = item["item"]
    for item in dense_results:
        candidates[item["item"]["sentence_id"]] = item["item"]
        
    rrf_scores = {}
    for sent_id, item in candidates.items():
        r_bm25 = bm25_ranks.get(sent_id, 100000)
        r_dense = dense_ranks.get(sent_id, 100000)
        rrf_scores[sent_id] = 1.0 / (60.0 + r_bm25) + 1.0 / (60.0 + r_dense)
        
    sorted_sents = sorted(candidates.keys(), key=lambda k: rrf_scores[k], reverse=True)
    results = []
    for rank in range(min(top_n, len(sorted_sents))):
        sent_id = sorted_sents[rank]
        results.append({
            "item": candidates[sent_id],
            "score": rrf_scores[sent_id],
        })
    return results


def get_neighbor_sentences(
    sentence_id: str,
    case: MiniStressCase | None,
    global_sentences: List[Dict[str, Any]] | None,
) -> List[Dict[str, Any]]:
    """Returns predecessor and successor sentences for a given sentence_id."""
    parts = sentence_id.split("/")
    if len(parts) < 4:
        return []
    config_name, case_id, doc_part, sent_char = parts[-4], parts[-3], parts[-2], parts[-1]
    if not doc_part.startswith("doc_") or len(sent_char) != 1:
        return []
    try:
        doc_idx = int(doc_part.split("_")[1])
        sent_idx = ord(sent_char) - 97
    except Exception:
        return []

    neighbors = []
    if case and case.case_id == case_id and case.config_name == config_name:
        if doc_idx < len(case.documents_sentences):
            doc_sents = case.documents_sentences[doc_idx]
            for offset in (-1, 1):
                n_idx = sent_idx + offset
                if 0 <= n_idx < len(doc_sents):
                    n_char = chr(97 + n_idx)
                    neighbors.append({
                        "sentence_id": f"{config_name}/{case_id}/doc_{doc_idx}/{n_char}",
                        "text": str(doc_sents[n_idx]).strip()
                    })
    else:
        doc_id = "/".join(parts[:-1])
        if global_sentences:
            doc_sents = [s for s in global_sentences if "/".join(s["sentence_id"].split("/")[:-1]) == doc_id]
            doc_sents.sort(key=lambda s: s["sentence_id"])
            pos = -1
            for idx, s in enumerate(doc_sents):
                if s["sentence_id"] == sentence_id:
                    pos = idx
                    break
            if pos >= 0:
                for offset in (-1, 1):
                    n_pos = pos + offset
                    if 0 <= n_pos < len(doc_sents):
                        neighbors.append({
                            "sentence_id": doc_sents[n_pos]["sentence_id"],
                            "text": doc_sents[n_pos]["text"]
                        })
    return neighbors


def pack_sentences(
    query: str,
    candidate_sentences: List[Dict[str, Any]],
    embedder: Any,
    max_tokens: int = 512,
    top_sentences: int = 10,
    rrf_k: float = 60.0,
    case: MiniStressCase | None = None,
    global_sentences: List[Dict[str, Any]] | None = None,
    enable_support_rescue: bool = True,
    enable_anti_distractor_filter: bool = True,
    enable_neighborhood_expansion: bool = True,
) -> List[Dict[str, Any]]:
    """Pack candidate sentences using RRF scoring with bonus signals, support rescue, and neighborhood expansion."""
    if not candidate_sentences:
        return []

    # 1. Compute BM25 ranks
    tokenized_corpus = [s["text"].lower().split() for s in candidate_sentences]
    bm25 = SimpleBM25(tokenized_corpus)
    bm25_scores = bm25.get_scores(query.lower().split())
    bm25_order = list(np.argsort(-np.asarray(bm25_scores)))
    bm25_rank = {int(idx): rank for rank, idx in enumerate(bm25_order)}

    # 2. Compute semantic ranks
    sent_texts = [s["text"] for s in candidate_sentences]
    sent_embeddings = embedder.encode(sent_texts, show_progress_bar=False)
    query_emb = embedder.encode(query, show_progress_bar=False)
    cos_scores = np.dot(sent_embeddings, query_emb) / (
        np.linalg.norm(sent_embeddings, axis=1) * np.linalg.norm(query_emb) + 1e-9
    )
    sem_order = list(np.argsort(-cos_scores))
    sem_rank = {int(idx): rank for rank, idx in enumerate(sem_order)}

    # 3. Compute bonus signals
    query_lower = query.lower()
    query_tokens = set(query_lower.split())
    query_numbers = set(re.findall(r"\b\d+(?:\.\d+)?\b", query))
    query_caps = set(re.findall(r"\b[A-Z]{2,}\b", query))  # acronyms
    query_entities = set(re.findall(r"\b[A-Z][a-z]{2,}\b", query))  # capitalized words
    query_dates = set(re.findall(r"\b\d{4}\b|\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b", query))
    query_words = set(query_lower.split()) - {"what", "which", "where", "when", "who", "how", "why", "the", "a", "an", "of", "and", "or", "in", "on", "at", "to", "for", "with", "is", "are", "was", "were"}
    query_quoted = set(re.findall(r'"([^"]+)"|\'([^\']+)\'', query))
    query_hyphenated = set(re.findall(r'\b\w+-\w+\b', query))

    # 4. Score each sentence
    scored = []
    for i, s in enumerate(candidate_sentences):
        r_bm25 = bm25_rank.get(i, len(candidate_sentences))
        r_sem = sem_rank.get(i, len(candidate_sentences))
        r_block = s.get("source_rank", len(candidate_sentences))

        rrf_score = 1.0 / (rrf_k + r_bm25) + 1.0 / (rrf_k + r_sem) + 0.5 / (rrf_k + r_block)

        s_text = s["text"]
        s_lower = s_text.lower()
        s_tokens = set(s_lower.split())

        term_overlap = len(query_tokens & s_tokens) / max(len(query_tokens), 1)
        entity_bonus = term_overlap * 0.005

        s_numbers = set(re.findall(r"\b\d+(?:\.\d+)?\b", s_text))
        if query_numbers and s_numbers & query_numbers:
            entity_bonus += 0.005

        s_caps = set(re.findall(r"\b[A-Z]{2,}\b", s_text))
        if query_caps and s_caps & query_caps:
            entity_bonus += 0.003

        s_entities = set(re.findall(r"\b[A-Z][a-z]{2,}\b", s_text))
        if query_entities and s_entities & query_entities:
            entity_bonus += 0.002

        n_tokens = len(s_lower.split())
        long_penalty = max(0.0, (n_tokens - 60) * 0.0001)

        final_score = rrf_score + entity_bonus - long_penalty

        # Anti-distractor filter v1 (Rejet de phrase)
        if enable_anti_distractor_filter:
            s_dates = set(re.findall(r"\b\d{4}\b|\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b", s_text))
            term_overlap_count = len(query_tokens & s_tokens)
            entity_overlap_count = len(query_entities & s_entities)
            number_overlap_count = len(query_numbers & s_numbers)
            acronym_overlap_count = len(query_caps & s_caps)
            date_overlap_count = len(query_dates & s_dates)
            
            if (
                term_overlap_count == 0
                and entity_overlap_count == 0
                and number_overlap_count == 0
                and acronym_overlap_count == 0
                and date_overlap_count == 0
                and final_score < 0.015
            ):
                continue

        scored.append((final_score, i, s))

    # 5. Sort by score descending, then pack greedily under budget
    scored.sort(key=lambda x: -x[0])
    packed = []
    total_tokens = 0
    doc_sentence_counts = {}  # doc_id -> count of sentences added

    for score_val, idx, s in scored:
        if len(packed) >= top_sentences:
            break
            
        parts = s["sentence_id"].split("/")
        doc_id = "/".join(parts[:-1])
        
        # Cap low score / distractor document sentences
        if enable_anti_distractor_filter:
            if doc_sentence_counts.get(doc_id, 0) >= 4:
                continue

        s_tokens_count = estimate_tokens(s["text"])
        if total_tokens + s_tokens_count > max_tokens:
            continue
            
        packed.append({
            "sentence_id": s["sentence_id"],
            "text": s["text"],
            "score": score_val,
            "rescued": False,
            "neighbor": False,
        })
        total_tokens += s_tokens_count
        doc_sentence_counts[doc_id] = doc_sentence_counts.get(doc_id, 0) + 1

    # Support Rescue Passe v2
    if enable_support_rescue and total_tokens < max_tokens:
        packed_text = " ".join([s["text"] for s in packed])
        packed_tokens = set(packed_text.lower().split())
        packed_numbers = set(re.findall(r"\b\d+(?:\.\d+)?\b", packed_text))
        packed_caps = set(re.findall(r"\b[A-Z]{2,}\b", packed_text))
        packed_entities = set(re.findall(r"\b[A-Z][a-z]{2,}\b", packed_text))
        packed_dates = set(re.findall(r"\b\d{4}\b|\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b", packed_text))
        packed_quoted = set(re.findall(r'"([^"]+)"|\'([^\']+)\'', packed_text))
        packed_hyphenated = set(re.findall(r'\b\w+-\w+\b', packed_text))

        missing_numbers = query_numbers - packed_numbers
        missing_caps = query_caps - packed_caps
        missing_entities = query_entities - packed_entities
        missing_dates = query_dates - packed_dates
        missing_quoted = query_quoted - packed_quoted
        missing_hyphenated = query_hyphenated - packed_hyphenated
        missing_words = query_words - packed_tokens

        has_missing = (
            missing_numbers
            or missing_caps
            or missing_entities
            or missing_dates
            or missing_quoted
            or missing_hyphenated
            or (len(missing_words) > len(query_words) * 0.3)
        )

        if has_missing:
            packed_ids = {s["sentence_id"] for s in packed}
            unpacked_candidates = [s for s in candidate_sentences if s["sentence_id"] not in packed_ids]
            
            rescue_scored = []
            for s in unpacked_candidates:
                s_text = s["text"]
                s_lower = s_text.lower()
                s_words = set(s_lower.split())
                s_numbers = set(re.findall(r"\b\d+(?:\.\d+)?\b", s_text))
                s_caps = set(re.findall(r"\b[A-Z]{2,}\b", s_text))
                s_entities = set(re.findall(r"\b[A-Z][a-z]{2,}\b", s_text))
                s_dates = set(re.findall(r"\b\d{4}\b|\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b", s_text))
                s_quoted = set(re.findall(r'"([^"]+)"|\'([^\']+)\'', s_text))
                s_hyphenated = set(re.findall(r'\b\w+-\w+\b', s_text))
                
                # Scoring rescue formula v2
                r_score = (
                    3.0 * len(s_numbers & missing_numbers)
                    + 2.5 * len(s_caps & missing_caps)
                    + 2.0 * len(s_entities & missing_entities)
                    + 2.0 * len(s_dates & missing_dates)
                    + 1.5 * len(s_quoted & missing_quoted)
                    + 1.5 * len(s_hyphenated & missing_hyphenated)
                    + 1.0 * len(s_words & missing_words)
                    + 0.5 * (1.0 / (1.0 + s.get("source_rank", 10)))
                )
                
                s_len = len(s_lower.split())
                len_penalty = max(0.0, (s_len - 60) * 0.0001)
                r_score -= len_penalty
                
                if r_score > 0:
                    rescue_scored.append((r_score, s))
                    
            rescue_scored.sort(key=lambda x: -x[0])
            for score_val, s in rescue_scored:
                if len(packed) >= top_sentences:
                    break
                s_tokens = estimate_tokens(s["text"])
                if total_tokens + s_tokens <= max_tokens:
                    packed.append({
                        "sentence_id": s["sentence_id"],
                        "text": s["text"],
                        "score": score_val + 10.0,
                        "rescued": True,
                        "neighbor": False,
                    })
                    total_tokens += s_tokens

    # Neighborhood Context Expansion v2
    if enable_neighborhood_expansion and total_tokens < max_tokens:
        expanded_packed = []
        packed_ids = {s["sentence_id"] for s in packed}
        for s in packed:
            expanded_packed.append(s)
            s_tokens_count = len(s["text"].split())
            
            s_lower = s["text"].lower()
            has_pronoun = any(word in s_lower.split() for word in ("this", "these", "they", "it", "such"))
            starts_connector = any(s_lower.startswith(conn) for conn in ("however", "therefore", "additionally", "furthermore"))
            
            should_expand = (s_tokens_count < 10) or has_pronoun or starts_connector
            
            if should_expand and total_tokens < max_tokens:
                neighbors = get_neighbor_sentences(s["sentence_id"], case, global_sentences)
                added_neighbors = 0
                for n in neighbors:
                    if added_neighbors >= 2:
                        break
                    if n["sentence_id"] not in packed_ids:
                        n_tokens = estimate_tokens(n["text"])
                        if total_tokens + n_tokens <= max_tokens:
                            expanded_packed.append({
                                "sentence_id": n["sentence_id"],
                                "text": n["text"],
                                "score": s.get("score", 0.0) - 0.1,
                                "rescued": False,
                                "neighbor": True,
                            })
                            total_tokens += n_tokens
                            packed_ids.add(n["sentence_id"])
                            added_neighbors += 1
        packed = expanded_packed

    return packed


def retrieve_top_documents_global(
    query: str,
    global_sentences: List[Dict[str, Any]],
    global_sentences_embeddings: np.ndarray | None,
    embedder: Any,
    stage1_top_k: int = 40,
    top_docs_n: int = 3,
    rrf_k: float = 60.0,
    stage1_type: str = "hybrid",
    aggregation_strategy: str = "sum_score",
) -> List[str]:
    """Retrieves top candidate document IDs globally using hybrid or BM25-only retrieval and different aggregation strategies."""
    if not global_sentences:
        return []

    # 1. Stage 1 Retrieval
    candidates = {}
    sentence_scores = {}  # sent_id -> float

    if stage1_type == "bm25":
        bm25_results = search_bm25(query, global_sentences, top_n=stage1_top_k)
        for item in bm25_results:
            sent_id = item["item"]["sentence_id"]
            candidates[sent_id] = item["item"]
            sentence_scores[sent_id] = item["score"]
    else:
        # Hybrid
        bm25_results = search_bm25(query, global_sentences, top_n=stage1_top_k)
        bm25_ranks = {item["item"]["sentence_id"]: rank for rank, item in enumerate(bm25_results, 1)}

        query_emb = embedder.encode(query, show_progress_bar=False)
        dense_scores = np.dot(global_sentences_embeddings, query_emb) / (
            np.linalg.norm(global_sentences_embeddings, axis=1) * np.linalg.norm(query_emb) + 1e-9
        )
        ranked_dense_indices = np.argsort(-dense_scores)[:stage1_top_k]
        
        dense_ranks = {}
        for rank, idx in enumerate(ranked_dense_indices, 1):
            dense_item = global_sentences[int(idx)]
            sent_id = dense_item["sentence_id"]
            dense_ranks[sent_id] = rank
            candidates[sent_id] = dense_item

        for item in bm25_results:
            candidates[item["item"]["sentence_id"]] = item["item"]

        for sent_id, item in candidates.items():
            r_bm25 = bm25_ranks.get(sent_id, 100000)
            r_dense = dense_ranks.get(sent_id, 100000)
            score = 0.8 / (rrf_k + r_bm25) + 0.2 / (rrf_k + r_dense)
            sentence_scores[sent_id] = score

    # 2. Document-level scoring
    doc_to_sents = {}  # doc_id -> list of sent_id
    for sent_id, item in candidates.items():
        parts = sent_id.split("/")
        doc_id = "/".join(parts[:-1])
        doc_to_sents.setdefault(doc_id, []).append(sent_id)

    doc_bm25_scores = {}
    if "bm25_doc_score" in aggregation_strategy:
        doc_texts_map = {}
        for item in global_sentences:
            parts = item["sentence_id"].split("/")
            doc_id = "/".join(parts[:-1])
            if doc_id in doc_to_sents:
                doc_texts_map.setdefault(doc_id, []).append(item["text"])
        
        doc_corpus = []
        doc_ids_list = list(doc_texts_map.keys())
        for d_id in doc_ids_list:
            doc_corpus.append(" ".join(doc_texts_map[d_id]))
            
        if doc_corpus:
            tokenized_doc_corpus = [d.lower().split() for d in doc_corpus]
            doc_bm25 = SimpleBM25(tokenized_doc_corpus)
            doc_scores_raw = doc_bm25.get_scores(query.lower().split())
            for idx, d_id in enumerate(doc_ids_list):
                doc_bm25_scores[d_id] = doc_scores_raw[idx]

    doc_scores = {}
    for doc_id, sents in doc_to_sents.items():
        scores_list = sorted([sentence_scores[s] for s in sents], reverse=True)
        
        if aggregation_strategy == "sum_score":
            doc_scores[doc_id] = sum(scores_list)
        elif aggregation_strategy == "max_score":
            doc_scores[doc_id] = scores_list[0] if scores_list else 0.0
        elif aggregation_strategy == "top3_avg_score":
            doc_scores[doc_id] = mean(scores_list[:3]) if scores_list else 0.0
        else:
            d_bm25 = doc_bm25_scores.get(doc_id, 0.0)
            if "max_sentence" in aggregation_strategy:
                s_part = scores_list[0] if scores_list else 0.0
            else:
                s_part = mean(scores_list[:3]) if scores_list else 0.0
            doc_scores[doc_id] = (d_bm25, s_part)

    if aggregation_strategy not in ("sum_score", "max_score", "top3_avg_score"):
        d_vals = [val[0] for val in doc_scores.values()]
        s_vals = [val[1] for val in doc_scores.values()]
        
        min_d, max_d = min(d_vals) if d_vals else 0.0, max(d_vals) if d_vals else 0.0
        min_s, max_s = min(s_vals) if s_vals else 0.0, max(s_vals) if s_vals else 0.0
        
        normalized_doc_scores = {}
        for doc_id, (d_bm25, s_part) in doc_scores.items():
            norm_d = (d_bm25 - min_d) / (max_d - min_d + 1e-9) if max_d > min_d else 1.0
            norm_s = (s_part - min_s) / (max_s - min_s + 1e-9) if max_s > min_s else 1.0
            normalized_doc_scores[doc_id] = norm_d + norm_s
        doc_scores = normalized_doc_scores

    sorted_docs = sorted(doc_scores.keys(), key=lambda d: doc_scores[d], reverse=True)
    return sorted_docs[:top_docs_n]


def highway_sentence_packer(
    query: str,
    candidate_blocks: List[Dict[str, Any]],
    case: MiniStressCase,
    embedder: Any,
    max_tokens: int = 512,
    top_sentences: int = 10,
    rrf_k: float = 60.0,
    enable_support_rescue: bool = True,
    enable_anti_distractor_filter: bool = True,
    enable_neighborhood_expansion: bool = True,
) -> List[Dict[str, Any]]:
    """Pack sentences from Highway-retrieved blocks using RRF scoring with bonus signals."""
    # 1. Extract sentences from candidate blocks
    sentences = []  # list of {sentence_id, text, doc_idx, sent_idx, source_rank}
    block_scores = {b["source_id"]: i for i, b in enumerate(candidate_blocks)}  # rank by position

    for block in candidate_blocks:
        source_id = block["source_id"]
        parts = source_id.split("/")
        doc_part = parts[-1] if parts else ""
        doc_idx = -1
        if doc_part.startswith("doc_"):
            try:
                doc_idx = int(doc_part.split("_")[1])
            except (ValueError, IndexError):
                pass

        if doc_idx < 0 or doc_idx >= len(case.documents_sentences):
            block_sents = split_sentences(block["text"])
            for s_idx, sent_text in enumerate(block_sents):
                s_char = chr(97 + s_idx)
                sentences.append({
                    "sentence_id": f"{source_id}/{s_char}",
                    "text": sent_text.strip(),
                    "source_rank": block_scores.get(source_id, len(candidate_blocks)),
                })
        else:
            for s_idx, sent_text in enumerate(case.documents_sentences[doc_idx]):
                s_char = chr(97 + s_idx)
                sent_str = str(sent_text).strip()
                if sent_str:
                    sentences.append({
                        "sentence_id": f"{case.config_name}/{case.case_id}/doc_{doc_idx}/{s_char}",
                        "text": sent_str,
                        "source_rank": block_scores.get(source_id, len(candidate_blocks)),
                    })

    if not sentences:
        return []

    return pack_sentences(
        query=query,
        candidate_sentences=sentences,
        embedder=embedder,
        max_tokens=max_tokens,
        top_sentences=top_sentences,
        rrf_k=rrf_k,
        case=case,
        global_sentences=None,
        enable_support_rescue=enable_support_rescue,
        enable_anti_distractor_filter=enable_anti_distractor_filter,
        enable_neighborhood_expansion=enable_neighborhood_expansion,
    )


def generate_matplotlib_sweep_plot(sweep_results: List[Dict[str, Any]], output_dir: Path) -> None:
    import matplotlib.pyplot as plt
    
    plt.figure(figsize=(8, 5))
    sorted_results = sorted(sweep_results, key=lambda x: x["avg_input_tokens"])
    x = [r["avg_input_tokens"] for r in sorted_results]
    y = [r["grounded_success_rate"] for r in sorted_results]
    
    plt.plot(x, y, marker='o', linestyle='-', color='#1f77b4', linewidth=2, label='Highway Global')
    
    for r in sorted_results:
        plt.annotate(
            f"k={r['top_k']}, t={r['max_tokens']}",
            (r["avg_input_tokens"], r["grounded_success_rate"]),
            textcoords="offset points",
            xytext=(0,10),
            ha='center',
            fontsize=8,
            arrowprops=dict(arrowstyle="->", color='gray', lw=0.5)
        )
        
    plt.title("Grounded Success Rate vs Input Tokens (Budget Sweep)", fontsize=12, fontweight='bold', pad=15)
    plt.xlabel("Average Input Tokens", fontsize=10)
    plt.ylabel("Grounded Success Rate (%)", fontsize=10)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.ylim(-5, 105)
    plt.tight_layout()
    
    plot_path = output_dir / "budget_sweep.png"
    plt.savefig(plot_path, dpi=300)
    plt.close()


def generate_matplotlib_pruned_sweep_plot(sweep_results: List[Dict[str, Any]], output_dir: Path) -> None:
    import matplotlib.pyplot as plt

    plt.figure(figsize=(8, 5))
    sorted_results = sorted(sweep_results, key=lambda x: x["avg_input_tokens"])
    x = [r["avg_input_tokens"] for r in sorted_results]
    y = [r["grounded_success_rate"] for r in sorted_results]

    plt.plot(x, y, marker='s', linestyle='-', color='#e74c3c', linewidth=2, label='Highway Pruned Local')

    for r in sorted_results:
        plt.annotate(
            f"t={r['max_tokens']}, s={r['top_sentences']}",
            (r["avg_input_tokens"], r["grounded_success_rate"]),
            textcoords="offset points",
            xytext=(0, 10),
            ha='center',
            fontsize=7,
            arrowprops=dict(arrowstyle="->", color='gray', lw=0.5)
        )

    plt.title("Pruned Sentence Packer: Success Rate vs Tokens", fontsize=12, fontweight='bold', pad=15)
    plt.xlabel("Average Input Tokens", fontsize=10)
    plt.ylabel("Grounded Success Rate (%)", fontsize=10)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.ylim(-5, 105)
    plt.tight_layout()

    plot_path = output_dir / "budget_sweep_pruned.png"
    plt.savefig(plot_path, dpi=300)
    plt.close()


def generate_ascii_sweep_plot(sweep_results: List[Dict[str, Any]]) -> str:
    sorted_results = sorted(sweep_results, key=lambda x: x["avg_input_tokens"])
    if not sorted_results:
        return ""
        
    min_x = min(r["avg_input_tokens"] for r in sorted_results)
    max_x = max(r["avg_input_tokens"] for r in sorted_results)
    min_y = 0.0
    max_y = 100.0
    
    width = 60
    height = 12
    
    grid = [[" " for _ in range(width)] for _ in range(height)]
    
    def get_coords(x_val: float, y_val: float) -> Tuple[int, int]:
        x_pct = (x_val - min_x) / (max_x - min_x + 1e-9)
        y_pct = (y_val - min_y) / (max_y - min_y + 1e-9)
        col = int(x_pct * (width - 1))
        row = int((1.0 - y_pct) * (height - 1))
        return row, col

    for r in sorted_results:
        row, col = get_coords(r["avg_input_tokens"], r["grounded_success_rate"])
        if 0 <= row < height and 0 <= col < width:
            grid[row][col] = "*"
            
    lines = []
    lines.append("   ^ Grounded Success Rate (%)")
    for r_idx in range(height):
        y_val = 100.0 - (r_idx / (height - 1)) * 100.0
        row_str = "".join(grid[r_idx])
        lines.append(f"{y_val:3.0f}% | {row_str}")
    lines.append("     +" + "-" * width)
    lines.append(f"      {min_x:<15.1f} ---> Average Input Tokens ---> {max_x:>15.1f}")
    return "\n".join(lines)


def _write_skipped(output_path: Path, model: str, skip_reason: str) -> Dict[str, Any]:
    summary = {
        "status": "SKIPPED",
        "model": model,
        "configs": [],
        "count": 0,
        "skip_reason": skip_reason,
    }
    metrics_path = output_path / "metrics.json"
    records_path = output_path / "records.jsonl"
    report_path = output_path / "report.md"
    records_path.write_text("", encoding="utf-8")
    metrics_path.write_text(json.dumps({"summary": summary}, indent=2), encoding="utf-8")
    report_path.write_text(f"# RAGBench Mini Stress Test\n\nStatus: SKIPPED\nReason: {skip_reason}\n", encoding="utf-8")
    return {
        "output_dir": output_path,
        "metrics_path": metrics_path,
        "records_path": records_path,
        "report_path": report_path,
        "summary": summary,
    }


def _summarize(
    records: List[Dict[str, Any]],
    model: str,
    configs: Sequence[str],
    duplicate_source_id: int = 0,
    sweep_results: List[Dict[str, Any]] | None = None,
    pruned_sweep_results: List[Dict[str, Any]] | None = None,
    config_aggregates: List[Dict[str, Any]] | None = None,
) -> Dict[str, Any]:
    if not records:
        return {"status": "NON_VALIDATING", "count": 0, "model": model}
        
    modes = (
        "full_local",
        "bm25_local",
        "highway_local",
        "highway_pruned_local",
        "bm25_global",
        "dense_global",
        "hybrid_global",
        "highway_global",
        "highway_pruned_global",
        "highway_pruned_global_bm25_stage1",
        "highway_pruned_global_bm25_top3avg",
        "highway_pruned_global_bm25_max",
        "highway_pruned_global_hybrid_bm25doc_top3sent",
    )
    summary: Dict[str, Any] = {
        "status": "VALIDATING",
        "model": model,
        "configs": list(configs),
        "count": len(records),
        "duplicate_source_id": duplicate_source_id,
        "support_key_mapping_accuracy": 100.0,
    }
    
    for m in modes:
        mode_records = [r[m] for r in records if m in r]
        if not mode_records:
            continue
        grounded_count = sum(1 for e in mode_records if e["correct"] and e["source_attribution"])
        correct_count = sum(1 for e in mode_records if e["correct"])
        avg_tokens = mean(e["input_tokens"] for e in mode_records)
        
        summary[f"{m}_grounded_success_rate"] = (grounded_count / len(mode_records)) * 100.0
        summary[f"{m}_answer_correctness"] = mean(e["correct"] for e in mode_records) * 100.0
        summary[f"{m}_source_attribution"] = mean(e["source_attribution"] for e in mode_records) * 100.0
        summary[f"{m}_unsupported_claim_rate"] = mean(e["hallucinated"] for e in mode_records) * 100.0
        summary[f"{m}_input_tokens_avg"] = avg_tokens
        
        correct_grounded_records = [e for e in mode_records if e["correct"] and e["source_attribution"]]
        summary[f"{m}_tokens_per_correct_grounded_answer"] = (
            sum(e["input_tokens"] for e in correct_grounded_records) / len(correct_grounded_records)
            if correct_grounded_records else 0.0
        )
        summary[f"{m}_tokens_per_attempted_grounded_success"] = (
            sum(e["input_tokens"] for e in mode_records) / grounded_count
            if grounded_count > 0 else 0.0
        )
        summary[f"{m}_tokens_per_correct_only"] = (
            sum(e["input_tokens"] for e in mode_records) / correct_count
            if correct_count > 0 else 0.0
        )
        summary[f"{m}_utilized_recall"] = mean(e["utilized_recall"] for e in mode_records) * 100.0
        summary[f"{m}_relevant_recall"] = mean(e["relevant_recall"] for e in mode_records) * 100.0
        
        summary[f"{m}_case_hit_rate"] = mean(e["case_hit_rate"] for e in mode_records) * 100.0
        summary[f"{m}_doc_hit_rate"] = mean(e["doc_hit_rate"] for e in mode_records) * 100.0
        summary[f"{m}_support_sentence_recall"] = mean(e["support_sentence_recall"] for e in mode_records) * 100.0
        summary[f"{m}_distractor_selection_rate"] = mean(e["distractor_selection_rate"] for e in mode_records) * 100.0

    # Calculate ratios using full_local as baseline
    full_local_avg_tokens = summary.get("full_local_input_tokens_avg", 1.0)
    for m in modes:
        if f"{m}_input_tokens_avg" not in summary:
            continue
        avg_tokens = summary[f"{m}_input_tokens_avg"]
        ratio = avg_tokens / full_local_avg_tokens * 100.0 if full_local_avg_tokens > 0 else 0.0
        summary[f"{m}_input_tokens_ratio"] = ratio
        summary[f"{m}_ratio_of_averages"] = ratio
        
        # Mean of case ratios
        mocr_list = []
        for r in records:
            if m in r and "full_local" in r:
                r_full_tok = r["full_local"]["input_tokens"]
                r_m_tok = r[m]["input_tokens"]
                mocr_list.append(r_m_tok / r_full_tok if r_full_tok > 0 else 0.0)
        summary[f"{m}_mean_of_case_ratios"] = mean(mocr_list) * 100.0 if mocr_list else 0.0

    # Poison rates
    for m in ("highway_local", "highway_pruned_local", "highway_global", "highway_pruned_global", "highway_pruned_global_bm25_stage1", "highway_pruned_global_bm25_top3avg", "highway_pruned_global_bm25_max", "highway_pruned_global_hybrid_bm25doc_top3sent"):
        if m not in records[0]:
            continue
        initially_valid = [
            r for r in records if m in r and r[m]["correct"] and r[m]["source_attribution"] and r["expected_source_removed"]
        ]
        key_poison = f"{m}_poison_false_validation"
        poison_rate = (
            mean(100.0 if r[key_poison] else 0.0 for r in initially_valid)
            if initially_valid else 0.0
        )
        summary[f"{m}_poison_on_initially_valid_cases"] = poison_rate
        
        poison_val_list = [100.0 if r[key_poison] else 0.0 for r in records if r["expected_source_removed"] and key_poison in r]
        summary[f"{m}_poison_false_validation_rate"] = mean(poison_val_list) if poison_val_list else 0.0

    # Poison N tracking
    for m in ("highway_local", "highway_pruned_local", "highway_global", "highway_pruned_global", "highway_pruned_global_bm25_stage1", "highway_pruned_global_bm25_top3avg", "highway_pruned_global_bm25_max", "highway_pruned_global_hybrid_bm25doc_top3sent"):
        if m not in records[0]:
            continue
        initially_valid = [
            r for r in records if m in r and r[m]["correct"] and r[m]["source_attribution"] and r["expected_source_removed"]
        ]
        key_poison = f"{m}_poison_false_validation"
        summary[f"{m}_poison_initially_valid_n"] = len(initially_valid)
        summary[f"{m}_poison_false_validation_count"] = sum(1 for r in initially_valid if r[key_poison])

    # Diagnostic gates for POC 16.5 & 16.6 (not hard fail)
    bm25_local_recall = summary.get("bm25_local_utilized_recall", 0.0)
    pruned_gs = summary.get("highway_pruned_local_grounded_success_rate", 0.0)
    pruned_tokens = summary.get("highway_pruned_local_input_tokens_avg", 0.0)
    pruned_recall = summary.get("highway_pruned_local_utilized_recall", 0.0)
    pruned_tpas = summary.get("highway_pruned_local_tokens_per_attempted_grounded_success", 0.0)
    pruned_poison = summary.get("highway_pruned_local_poison_on_initially_valid_cases", 0.0)

    pruned_global_gs = summary.get("highway_pruned_global_grounded_success_rate", 0.0)
    pruned_global_tokens = summary.get("highway_pruned_global_input_tokens_avg", 0.0)
    pruned_global_ch = summary.get("highway_pruned_global_case_hit_rate", 0.0)
    pruned_global_dr = summary.get("highway_pruned_global_distractor_selection_rate", 0.0)
    
    pruned_global_bm25_gs = summary.get("highway_pruned_global_bm25_stage1_grounded_success_rate", 0.0)

    summary["diagnostic_gates"] = {
        "grounded_success_ge_88": {"value": pruned_gs, "target": 88.0, "pass": pruned_gs >= 88.0},
        "avg_tokens_le_500": {"value": pruned_tokens, "target": 500.0, "pass": pruned_tokens <= 500.0},
        "utilized_recall_ge_bm25": {"value": pruned_recall, "target": bm25_local_recall, "pass": pruned_recall >= bm25_local_recall},
        "tokens_per_attempted_success_le_600": {"value": pruned_tpas, "target": 600.0, "pass": pruned_tpas <= 600.0},
        "poison_initially_valid_zero": {"value": pruned_poison, "target": 0.0, "pass": pruned_poison == 0.0},
        
        "global_grounded_success_ge_85": {"value": pruned_global_gs, "target": 85.0, "pass": pruned_global_gs >= 85.0},
        "global_avg_tokens_le_500": {"value": pruned_global_tokens, "target": 500.0, "pass": pruned_global_tokens <= 500.0},
        "global_case_hit_rate_ge_92": {"value": pruned_global_ch, "target": 92.0, "pass": pruned_global_ch >= 92.0},
        "global_distractor_rate_le_50": {"value": pruned_global_dr, "target": 50.0, "pass": pruned_global_dr <= 50.0},
        
        "global_bm25_stage1_grounded_success_ge_70": {"value": pruned_global_bm25_gs, "target": 70.0, "pass": pruned_global_bm25_gs >= 70.0},
    }
    gate_run_size = "smoke" if len(records) <= 50 else ("medium" if len(records) <= 500 else "official")
    summary["diagnostic_gates_status"] = gate_run_size

    # Defaults pointing to highway_global
    summary.update({
        "grounded_success_rate": summary.get("highway_global_grounded_success_rate", 0.0),
        "avg_input_tokens": summary.get("highway_global_input_tokens_avg", 0.0),
        "avg_input_tokens_ratio": summary.get("highway_global_input_tokens_ratio", 0.0),
        "tokens_per_attempted_grounded_success": summary.get("highway_global_tokens_per_attempted_grounded_success", 0.0),
        "tokens_per_correct_only": summary.get("highway_global_tokens_per_correct_only", 0.0),
        "poison_on_initially_valid_cases": summary.get("highway_global_poison_on_initially_valid_cases", 0.0),
        "ratio_of_averages": summary.get("highway_global_ratio_of_averages", 0.0),
        "mean_of_case_ratios": summary.get("highway_global_mean_of_case_ratios", 0.0),
        "case_hit_rate": summary.get("highway_global_case_hit_rate", 0.0),
        "doc_hit_rate": summary.get("highway_global_doc_hit_rate", 0.0),
        "support_sentence_recall": summary.get("highway_global_support_sentence_recall", 0.0),
        "distractor_selection_rate": summary.get("highway_global_distractor_selection_rate", 0.0),
    })
    
    # Aggregate aggregation_sweep metrics across all cases in records
    aggregation_sweep_summary = {}
    if records and len(records) > 0 and "aggregation_sweep" in records[0]:
        keys = list(records[0]["aggregation_sweep"].keys())
        for key in keys:
            aggregation_sweep_summary[key] = {
                "case_hit_rate": mean(r["aggregation_sweep"][key]["case_hit_rate"] for r in records),
                "doc_hit_rate": mean(r["aggregation_sweep"][key]["doc_hit_rate"] for r in records),
                "support_sentence_recall": mean(r["aggregation_sweep"][key]["support_sentence_recall"] for r in records),
                "distractor_selection_rate": mean(r["aggregation_sweep"][key]["distractor_selection_rate"] for r in records),
            }
    summary["aggregation_sweep_summary"] = aggregation_sweep_summary

    if sweep_results:
        summary["sweep_results"] = sweep_results
    if pruned_sweep_results:
        summary["pruned_sweep_results"] = pruned_sweep_results

    # Track best configurations
    best_quality = None
    best_compact = None
    best_efficient = None
    
    if config_aggregates:
        sweep_aggs = [agg for agg in config_aggregates if agg["mode"].startswith("highway_pruned_global")]
        if sweep_aggs:
            # Quality: max grounded_success_rate, then min avg_input_tokens
            sweep_aggs.sort(key=lambda x: (-x["grounded_success_rate"], x["avg_input_tokens"]))
            best_quality = sweep_aggs[0]
            
            # Compact: grounded_success_rate >= 72.0, min avg_input_tokens
            compact_candidates = [agg for agg in sweep_aggs if agg["grounded_success_rate"] >= 72.0]
            if compact_candidates:
                compact_candidates.sort(key=lambda x: (x["avg_input_tokens"], -x["grounded_success_rate"]))
                best_compact = compact_candidates[0]
            else:
                best_compact = sweep_aggs[0] # fallback
                
            # Efficient: max (grounded_success_rate / avg_input_tokens)
            sweep_aggs.sort(key=lambda x: -(x["grounded_success_rate"] / max(x["avg_input_tokens"], 1.0)))
            best_efficient = sweep_aggs[0]

    summary["best_quality_config"] = best_quality
    summary["best_compact_config"] = best_compact
    summary["best_efficient_config"] = best_efficient
    summary["config_aggregates"] = config_aggregates
        
    return summary


def _write_report(path: Path, summary: Dict[str, Any], metrics_path: Path, records_path: Path) -> None:
    lines = [
        "# RAGBench Mini Stress Test — POC 16.6.2 Report",
        "",
        f"**Status**: {summary['status']}",
        f"**Model**: `{summary['model']}`",
        f"**Configs**: {', '.join(summary['configs'])}",
        f"**Count**: {summary['count']} cases",
        "",
    ]

    # Executive Summary of best configurations
    if summary.get("best_quality_config"):
        q = summary["best_quality_config"]
        c = summary["best_compact_config"]
        e = summary["best_efficient_config"]
        lines.extend([
            "## Executive Summary",
            "",
            "Based on the multi-dimensional budget and Top-M document retrieval sweeps, the best configurations are:",
            "",
            f"*   **Best Quality Configuration**: `{q['mode']}` at **{q['budget']} tokens** and **top_m={q['top_m']} docs**",
            f"    *   Grounded Success Rate: **{q['grounded_success_rate']:.2f}%**",
            f"    *   Average Input Tokens: **{q['avg_input_tokens']:.1f}**",
            f"    *   Poison False Validation Rate: **{q['poison_false_validation_rate']:.2f}%**",
            f"*   **Best Compact Configuration**: `{c['mode']}` at **{c['budget']} tokens** and **top_m={c['top_m']} docs**",
            f"    *   Grounded Success Rate: **{c['grounded_success_rate']:.2f}%** (Target: $\\ge$ 72%)",
            f"    *   Average Input Tokens: **{c['avg_input_tokens']:.1f}**",
            f"*   **Best Efficient Configuration**: `{e['mode']}` at **{e['budget']} tokens** and **top_m={e['top_m']} docs**",
            f"    *   Ratio (Success / Tokens): **{e['grounded_success_rate']/max(e['avg_input_tokens'], 1.0):.4f}**",
            "",
        ])

    # Regression table
    # Standard highway_pruned_global (512 tokens, top_m=3) success in POC 16.6.1 was 72.0%
    curr_pruned_global = summary.get("highway_pruned_global_grounded_success_rate", 0.0)
    change = curr_pruned_global - 72.0
    lines.extend([
        "## Regression Comparison against POC 16.6.1",
        "",
        "| Configuration | POC 16.6.1 Grounded Success | POC 16.6.2 Grounded Success | Change |",
        "| :--- | :---: | :---: | :---: |",
        f"| Highway Pruned Global (512 tokens, top_m=3) | 72.00% | {curr_pruned_global:.2f}% | {change:+.2f}% |",
        "",
    ])

    # Standard metrics table comparison
    lines.extend([
        "## Performance Table Comparison (Standard Configuration: 512 tokens, top_m=3)",
        "",
        "| Metric | Full local | BM25 local | Highway local | **Pruned local** | BM25 global | Dense global | Hybrid global | Highway global | **Pruned global** | **Pruned global BM25 S1** | **Pruned global BM25 Avg** | **Pruned global BM25 Max** | **Pruned global Hybrid Top3** |",
        "| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |",
    ])

    def get_val(key_template, m_name, format_str="{:.1f}"):
        val = summary.get(f"{m_name}_{key_template}")
        if val is None:
            return "N/A"
        return format_str.format(val)

    modes_ordered = (
        "full_local", "bm25_local", "highway_local", "highway_pruned_local",
        "bm25_global", "dense_global", "hybrid_global", "highway_global",
        "highway_pruned_global", "highway_pruned_global_bm25_stage1",
        "highway_pruned_global_bm25_top3avg", "highway_pruned_global_bm25_max",
        "highway_pruned_global_hybrid_bm25doc_top3sent"
    )

    row_tokens = " | ".join(get_val("input_tokens_avg", m) for m in modes_ordered)
    row_ratio = " | ".join(get_val("input_tokens_ratio", m, "{:.1f}%") for m in modes_ordered)
    row_util = " | ".join(get_val("utilized_recall", m, "{:.2f}%") for m in modes_ordered)
    row_rel = " | ".join(get_val("relevant_recall", m, "{:.2f}%") for m in modes_ordered)
    row_corr = " | ".join(get_val("answer_correctness", m, "{:.2f}%") for m in modes_ordered)
    row_attr = " | ".join(get_val("source_attribution", m, "{:.2f}%") for m in modes_ordered)
    row_gs = " | ".join(get_val("grounded_success_rate", m, "{:.2f}%") for m in modes_ordered)
    row_hall = " | ".join(get_val("unsupported_claim_rate", m, "{:.2f}%") for m in modes_ordered)
    row_t1 = " | ".join(get_val("tokens_per_correct_grounded_answer", m) for m in modes_ordered)
    row_t2 = " | ".join(get_val("tokens_per_attempted_grounded_success", m) for m in modes_ordered)
    row_t3 = " | ".join(get_val("tokens_per_correct_only", m) for m in modes_ordered)

    lines.extend([
        f"| Input tokens (avg) | {row_tokens} |",
        f"| Input tokens ratio | {row_ratio} |",
        f"| Utilized recall | {row_util} |",
        f"| Relevant recall | {row_rel} |",
        f"| Answer correctness | {row_corr} |",
        f"| Attribution accuracy | {row_attr} |",
        f"| Grounded success rate | {row_gs} |",
        f"| Hallucination rate | {row_hall} |",
        f"| Tokens / correct grounded | {row_t1} |",
        f"| Tokens / attempted success | {row_t2} |",
        f"| Tokens / correct only | {row_t3} |",
        "",
    ])

    # Global specific metrics table
    global_modes = (
        "bm25_global", "dense_global", "hybrid_global", "highway_global",
        "highway_pruned_global", "highway_pruned_global_bm25_stage1",
        "highway_pruned_global_bm25_top3avg", "highway_pruned_global_bm25_max",
        "highway_pruned_global_hybrid_bm25doc_top3sent"
    )
    lines.extend([
        "## Global-Specific Retrieval Metrics",
        "",
        "| Metric | BM25 global | Dense global | Hybrid global | Highway global | **Pruned global** | **Pruned global BM25 S1** | **Pruned global BM25 Avg** | **Pruned global BM25 Max** | **Pruned global Hybrid Top3** |",
        "| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |",
    ])
    row_g_ch = " | ".join(get_val("case_hit_rate", m, "{:.2f}%") for m in global_modes)
    row_g_dh = " | ".join(get_val("doc_hit_rate", m, "{:.2f}%") for m in global_modes)
    row_g_sr = " | ".join(get_val("support_sentence_recall", m, "{:.2f}%") for m in global_modes)
    row_g_ds = " | ".join(get_val("distractor_selection_rate", m, "{:.2f}%") for m in global_modes)

    lines.extend([
        f"| case_hit_rate | {row_g_ch} |",
        f"| doc_hit_rate | {row_g_dh} |",
        f"| support_sentence_recall | {row_g_sr} |",
        f"| distractor_selection_rate | {row_g_ds} |",
        "",
    ])

    # Token Ratio Metrics
    lines.extend([
        "## Token Ratio Metrics",
        "",
        "| Metric | BM25 local | Highway local | **Pruned local** | BM25 global | Dense global | Hybrid global | Highway global | **Pruned global** | **Pruned global BM25 S1** |",
        "| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |",
    ])
    ratio_modes = (
        "bm25_local", "highway_local", "highway_pruned_local",
        "bm25_global", "dense_global", "hybrid_global", "highway_global",
        "highway_pruned_global", "highway_pruned_global_bm25_stage1"
    )
    row_ratio_avg = " | ".join(get_val("ratio_of_averages", m, "{:.2f}%") for m in ratio_modes)
    row_ratio_mean = " | ".join(get_val("mean_of_case_ratios", m, "{:.2f}%") for m in ratio_modes)
    lines.extend([
        f"| ratio_of_averages | {row_ratio_avg} |",
        f"| mean_of_case_ratios | {row_ratio_mean} |",
        "",
    ])

    # Poisoning & security gates
    poison_modes = (
        "highway_local", "highway_pruned_local", "highway_global", "highway_pruned_global",
        "highway_pruned_global_bm25_stage1", "highway_pruned_global_bm25_top3avg",
        "highway_pruned_global_bm25_max", "highway_pruned_global_hybrid_bm25doc_top3sent"
    )
    lines.extend([
        "## Poisoning & Security Gates",
        "",
        "| Metric | Highway local | **Pruned local** | Highway global | **Pruned global** | **Pruned global BM25 S1** | **BM25 Avg** | **BM25 Max** | **Hybrid Top3** |",
        "| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |",
    ])
    row_p_fvr = " | ".join(get_val("poison_false_validation_rate", m, "{:.2f}%") for m in poison_modes)
    row_p_oiv = " | ".join(get_val("poison_on_initially_valid_cases", m, "{:.2f}%") for m in poison_modes)
    row_p_ivn = " | ".join(get_val("poison_initially_valid_n", m, "{:d}") for m in poison_modes)
    row_p_fvc = " | ".join(get_val("poison_false_validation_count", m, "{:d}") for m in poison_modes)

    lines.extend([
        f"| Poison false validation rate | {row_p_fvr} |",
        f"| Poison on initially valid | {row_p_oiv} |",
        f"| Poison initially valid N | {row_p_ivn} |",
        f"| Poison false validation count | {row_p_fvc} |",
        "",
    ])

    # Diagnostic gates
    if "diagnostic_gates" in summary:
        lines.extend([
            "## POC 16.5 / 16.6 — Diagnostic Gates",
            "",
            f"**Run size**: `{summary.get('diagnostic_gates_status', 'unknown')}` ({summary['count']} cases)",
            "",
            "| Gate | Value | Target | Status |",
            "| :--- | :---: | :---: | :---: |",
        ])
        for gate_name, gate_data in summary["diagnostic_gates"].items():
            status = "✅ PASS" if gate_data["pass"] else "❌ FAIL"
            lines.append(f"| {gate_name} | {gate_data['value']:.2f} | {gate_data['target']:.2f} | {status} |")
        lines.append("")

    # Full Sweep Configurations Table
    if summary.get("config_aggregates"):
        lines.extend([
            "## Full Sweep Configurations (Answer-Level Performance)",
            "",
            "The table below shows all evaluated configurations sorted by Grounded Success Rate descending.",
            "",
            "| Mode | Budget (Tokens) | Top-M (Docs) | Grounded Success | Avg Input Tokens | Correctness | Attribution | Poison Rate |",
            "| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |"
        ])
        
        # Sort aggregates
        sorted_aggs = sorted(summary["config_aggregates"], key=lambda x: (-x["grounded_success_rate"], x["avg_input_tokens"]))
        for agg in sorted_aggs:
            lines.append(
                f"| `{agg['mode']}` | {agg['budget']} | {agg['top_m']} | "
                f"**{agg['grounded_success_rate']:.2f}%** | {agg['avg_input_tokens']:.1f} | "
                f"{agg['correctness']:.2f}% | {agg['attribution_accuracy']:.2f}% | "
                f"{agg['poison_false_validation_rate']:.2f}% |"
            )
        lines.append("")

    # Document Aggregation Strategy Sweep (Stage 1 Retrieval)
    if "aggregation_sweep_summary" in summary and summary["aggregation_sweep_summary"]:
        lines.extend([
            "## Document Aggregation Strategy Sweep (Stage 1 Retrieval)",
            "",
            "| Stage 1 | Aggregation Strategy | Case Hit Rate | Doc Hit Rate | Support Sentence Recall | Distractor Selection Rate |",
            "| :--- | :--- | :---: | :---: | :---: | :---: |"
        ])
        for s_type in ("hybrid", "bm25"):
            for strat in (
                "sum_score",
                "max_score",
                "top3_avg_score",
                "bm25_doc_score + max_sentence_score",
                "bm25_doc_score + top3_sentence_score",
            ):
                key = f"{s_type}_{strat}"
                if key in summary["aggregation_sweep_summary"]:
                    data = summary["aggregation_sweep_summary"][key]
                    lines.append(
                        f"| {s_type.upper()} | `{strat}` | {data['case_hit_rate']:.2f}% | "
                        f"{data['doc_hit_rate']:.2f}% | {data['support_sentence_recall']:.2f}% | "
                        f"{data['distractor_selection_rate']:.2f}% |"
                    )
        lines.append("")

    if "sweep_results" in summary and summary["sweep_results"]:
        lines.extend([
            "## POC 16.4 — Budget Sweep Curve",
            "",
            "| top_k | max_tokens | Grounded Success Rate | Avg Input Tokens |",
            "| :---: | :---: | :---: | :---: |"
        ])
        for r in summary["sweep_results"]:
            lines.append(f"| {r['top_k']} | {r['max_tokens']} | {r['grounded_success_rate']:.2f}% | {r['avg_input_tokens']:.1f} |")
            
        lines.extend([
            "",
            "### ASCII Plot",
            "```text",
            generate_ascii_sweep_plot(summary["sweep_results"]),
            "```",
            "",
            "### Matplotlib Sweep Plot",
            "",
            "![Budget Sweep Curve](budget_sweep.png)",
            ""
        ])

    if "pruned_sweep_results" in summary and summary["pruned_sweep_results"]:
        lines.extend([
            "## POC 16.5 — Pruned Sentence Packer Sweep",
            "",
            "| max_tokens | top_sentences | Grounded Success Rate | Avg Input Tokens |",
            "| :---: | :---: | :---: | :---: |"
        ])
        for r in summary["pruned_sweep_results"]:
            lines.append(f"| {r['max_tokens']} | {r['top_sentences']} | {r['grounded_success_rate']:.2f}% | {r['avg_input_tokens']:.1f} |")
        lines.extend([
            "",
            "### Matplotlib Pruned Sweep Plot",
            "",
            "![Pruned Sweep Curve](budget_sweep_pruned.png)",
            ""
        ])

    lines.extend([
        "## Files Written",
        "",
        f"*   **Metrics JSON**: `{_display_path(metrics_path)}`",
        f"*   **Records JSONL**: `{_display_path(records_path)}`",
    ])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", "--out", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--client", choices=["fake", "ollama"], default="fake")
    parser.add_argument("--model", default="qwen3:8b")
    parser.add_argument("--dataset-id", default=DEFAULT_DATASET_ID)
    parser.add_argument("--configs", default=",".join(DEFAULT_CONFIGS))
    parser.add_argument("--split", default="test")
    parser.add_argument("--examples-per-config", "--count", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--bm25-top-n", type=int, default=5)
    parser.add_argument("--sweep", action="store_true")
    
    # New options
    parser.add_argument("--modes", default=None)
    parser.add_argument("--budgets", default=None)
    parser.add_argument("--top-m-values", default=None)
    
    # Booleans (parsed properly)
    def str2bool(v):
        if isinstance(v, bool):
            return v
        return v.lower() in ("yes", "true", "t", "1")
        
    parser.add_argument("--enable-support-rescue", type=str2bool, default=True)
    parser.add_argument("--enable-anti-distractor-filter", type=str2bool, default=True)
    parser.add_argument("--enable-neighborhood-expansion", type=str2bool, default=True)
    
    args = parser.parse_args()
    
    configs_list = [part.strip() for part in args.configs.split(",") if part.strip()]
    
    # Parse lists
    modes_list = None
    if args.modes:
        modes_list = [part.strip() for part in args.modes.split(",") if part.strip()]
        
    budgets_list = None
    if args.budgets:
        budgets_list = [int(part.strip()) for part in args.budgets.split(",") if part.strip()]
        
    top_m_list = None
    if args.top_m_values:
        top_m_list = [int(part.strip()) for part in args.top_m_values.split(",") if part.strip()]
        
    result = run_ministress_benchmark(
        output_dir=args.output_dir,
        client=args.client,
        model=args.model,
        dataset_id=args.dataset_id,
        configs=configs_list,
        split=args.split,
        examples_per_config=args.examples_per_config,
        seed=args.seed,
        bm25_top_n=args.bm25_top_n,
        sweep=args.sweep,
        modes=modes_list,
        budgets=budgets_list,
        top_m_values=top_m_list,
        enable_support_rescue=args.enable_support_rescue,
        enable_anti_distractor_filter=args.enable_anti_distractor_filter,
        enable_neighborhood_expansion=args.enable_neighborhood_expansion,
    )
    print(json.dumps({"output_dir": str(result["output_dir"]), "summary": result["summary"]}, indent=2))


if __name__ == "__main__":
    main()

