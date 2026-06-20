import os
import json
import time
import urllib.request
import re
import hashlib
import logging
from typing import Dict, Any, List, Tuple, Optional
from highway.runtime.cache_manager import CacheManager
from highway.retrieval.search import SearchRouter
from highway.retrieval.evidence_resolver import EvidenceResolver
from highway.retrieval.ir_builder import IRBuilder
from highway.runtime.compiler import ContextCompiler
from highway.runtime.output_verifier import OutputVerifier
from highway.kernels.compute_kernels import ComparisonKernel, AggregationKernel, CanonicalFactStore, FieldLookupKernel, MultiFactKernel, EvidencePackBuilder, ClaimLevelVerifier
from highway.errors import ContextOverflowError, HighwayError, LLMUnavailableError, MalformedJSONError

logger = logging.getLogger("highway.scheduler")


class ExecutionScheduler:
    def __init__(self, index_dir: str, cache_dir: str, vllm_port: int = 8000, model_name: str = "Qwen/Qwen2.5-0.5B-Instruct"):
        self.search_router = SearchRouter(index_dir)
        self.evidence_resolver = EvidenceResolver()
        self.ir_builder = IRBuilder()
        self.compiler = ContextCompiler()
        self.verifier = OutputVerifier()
        self.cache_manager = CacheManager(cache_dir)
        
        self.vllm_port = vllm_port
        self.model_name = model_name
        self.vllm_url = f"http://localhost:{self.vllm_port}/v1/completions"
        self.search_config_hash = "default_k50"
        
        # Logging dictionary to track execution paths and metrics per query
        self.last_query_metrics = {}
        
        # Compute Kernels for deterministic G/H resolution
        self.comparison_kernel = ComparisonKernel()
        self.aggregation_kernel = AggregationKernel()
        self.claim_verifier = ClaimLevelVerifier()
        
        self.MODEL_CONTEXT_LIMITS = {
            "Qwen/Qwen2.5-0.5B-Instruct": 1200,
            "Qwen/Qwen2.5-1.5B-Instruct": 4096,
            "Qwen/Qwen2.5-3B-Instruct": 4096,
            "Qwen/Qwen2.5-7B-Instruct-GPTQ-Int4": 2048,
            "mistralai/Mistral-7B-Instruct-v0.3-GPTQ": 8192,
        }

    def _record_storage_metrics(self):
        storage_metrics = getattr(self.search_router, "last_storage_metrics", {})
        if storage_metrics:
            self.last_query_metrics.update(storage_metrics)

    def _record_execution_error(self, error: HighwayError) -> str:
        self.last_query_metrics["execution_error"] = error.code
        self.last_query_metrics["execution_error_structured"] = error.to_dict()
        return error.to_legacy_answer()

    def _call_llm(self, prompt: str) -> str:
        try:
            data = {
                "model": self.model_name,
                "prompt": prompt,
                "max_tokens": 96,
                "temperature": 0.0,
                "repetition_penalty": 1.0,
                "stop": ["<|im_end|>", "Question:"]
            }
            headers = {"Content-Type": "application/json"}
            req = urllib.request.Request(self.vllm_url, data=json.dumps(data).encode("utf-8"), headers=headers)
            with urllib.request.urlopen(req, timeout=15) as response:
                res_data = json.loads(response.read().decode("utf-8"))
                raw_answer = res_data["choices"][0]["text"].strip()
                
                # Parse JSON answer envelope if needed
                text = raw_answer.strip()
                if text.startswith("```"):
                    lines = text.split("\n")
                    if lines[0].startswith("```"):
                        lines = lines[1:]
                    if lines[-1].strip() == "```":
                        lines = lines[:-1]
                    text = "\n".join(lines).strip()

                # If the text has a trailing JSON structure like "\n}", reconstruct and parse
                if not text.startswith("{") and not text.startswith("["):
                    candidates_to_try = [
                        '{"answer": "' + text,
                        '{"answer": "' + text + '"}'
                    ]
                    for cand in candidates_to_try:
                        try:
                            parsed = json.loads(cand)
                            if isinstance(parsed, dict) and "answer" in parsed:
                                return str(parsed["answer"]).strip()
                        except Exception:
                            pass

                try:
                    start_obj = text.find("{")
                    start_arr = text.find("[")
                    start, end = -1, -1
                    if start_obj != -1 and (start_arr == -1 or start_obj < start_arr):
                        start = start_obj
                        end = text.rfind("}")
                    elif start_arr != -1:
                        start = start_arr
                        end = text.rfind("]")
                    if start != -1 and end != -1:
                        json_str = text[start:end+1]
                        parsed = json.loads(json_str)
                        if isinstance(parsed, list):
                            items = []
                            for item in parsed:
                                if isinstance(item, dict):
                                    items.append(str(next(iter(item.values()))).strip())
                                else:
                                    items.append(str(item).strip())
                            return ", ".join(items)
                        if isinstance(parsed, dict):
                            if "answer" in parsed:
                                return str(parsed["answer"]).strip()
                            return str(next(iter(parsed.values()))).strip()
                except Exception:
                    pass

                # Regex fallback
                match = re.search(r'"answer"\s*:\s*"([^"]+)"', text, re.IGNORECASE)
                if match:
                    return match.group(1).strip()

                # Manual cleaning fallback
                cleaned_text = re.sub(r'"\s*}\s*$', '', text)
                cleaned_text = re.sub(r'^\s*"', '', cleaned_text)
                return cleaned_text.strip()
        except Exception as e:
            logger.error("vLLM serving request failed", extra={"error": str(e)})
            error_str = str(e)
            if "400" in error_str and ("maximum context length" in error_str or "context length" in error_str):
                return self._record_execution_error(ContextOverflowError(message=error_str))
            else:
                return self._record_execution_error(LLMUnavailableError(message=error_str))

    def route_execution(self, proof_ir: Dict[str, Any]) -> str:
        intent = proof_ir["query"].get("intent", "single_fact_lookup")
        status = proof_ir["proof"].get("status", "INSUFFICIENT")
        guard_dec = proof_ir.get("guard_decision", {})
        llm_required_by_intent = proof_ir.get("llm_required_by_intent", False)
        category = proof_ir.get("category", "")
        
        # Check if Category I (synthesis)
        if category.startswith("I_") or intent == "LLM_SYNTHESIS" or proof_ir["query"].get("category", "").startswith("I_"):
            return "LLM_SYNTHESIS"
        
        # L0/Guard: NOT_FOUND Guard (highest priority)
        if guard_dec.get("action") == "BYPASS_LLM" and guard_dec.get("answer") == "NOT_FOUND":
            return "NOT_FOUND"
        
        # Compute Kernel routes for G/H (before LLM paths)
        if proof_ir.get("compute_kernel_eligible", False):
            if intent == "comparison":
                return "COMPUTE_COMPARISON"
            elif intent == "aggregation":
                return "COMPUTE_AGGREGATION"
        
        # L5: Long-Context Fallback conditions
        if status in ["PARTIAL", "INSUFFICIENT"] and guard_dec.get("action") != "BYPASS_LLM":
            return "LONG_CONTEXT_FALLBACK"
            
        # Safety net: intent is LLM-required â€” override any bypass decision
        if llm_required_by_intent and guard_dec.get("action") == "BYPASS_LLM":
            if guard_dec.get("answer") != "NOT_FOUND":
                return "LLM_COMPILED"  # force LLM path
            
        # Guard: Deterministic bypass
        if guard_dec.get("action") == "BYPASS_LLM" and guard_dec.get("answer") != "NOT_FOUND":
            return "DETERMINISTIC"
            
        # Default LLM serving path
        return "LLM_COMPILED"

    def long_context_executor(self, query_ir: Dict[str, Any], proof_ir: Dict[str, Any]) -> str:
        # L5 Fallback Pipeline:
        # 1. Expanded Search: Retrieve top-100 blocks
        question = query_ir["question"]
        candidates, _ = self.search_router.search(question, top_k=100)
        self._record_storage_metrics()
        
        # 2. Expanded Resolution: Keep all blocks matching entities, skip strict temporal pruning to let LLM see history
        target_entities = query_ir.get("target_entities", [])
        expanded_active = []
        for b in candidates:
            text_lower = b["text"].lower()
            has_entity = False
            for ent in target_entities:
                pattern = r'(?<![a-zA-Z0-9_\-])' + re.escape(ent.lower()) + r'(?![a-zA-Z0-9_\-])'
                if re.search(pattern, text_lower):
                    has_entity = True
                    break
            if has_entity:
                expanded_active.append(b)
                
        if not expanded_active:
            expanded_active = candidates[:10] # Fallback to top-10 chunks
            
        # 3. Create long-context fallback prompt
        prompt_lines = []
        prompt_lines.append("<|im_start|>system\nYou are a precise helper. Extract the requested answer from the following long context.")
        prompt_lines.append("If not found, respond with ONLY: NOT_FOUND<|im_end|>")
        prompt_lines.append("<|im_start|>user\nContext:")
        for idx, b in enumerate(expanded_active):
            prompt_lines.append(f"Block {idx+1} [SOURCE: {b['source_file']}]: {b['text']}")
        prompt_lines.append(f"\nQuestion: {question}\nRespond with ONLY the exact requested value(s) in a JSON object:\n{{\n  \"answer\": \"<value>\"\n}}<|im_end|>")
        prompt_lines.append("<|im_start|>assistant\n{\n  \"answer\": \"")
        
        prompt_text = "\n".join(prompt_lines)
        
        # Log metadata
        self.last_query_metrics["long_context_blocks"] = len(expanded_active)
        self.last_query_metrics["long_context_tokens"] = len(prompt_text.split())
        
        return self._call_llm(prompt_text)

    def answer(self, question: str, use_cache: bool = True, force_llm: bool = False, q_id: Optional[str] = None, disable_llm_for_computable: bool = False, category: Optional[str] = None, allowed_conclusions: Optional[List[str]] = None, expected_conclusion: Optional[str] = None) -> Dict[str, Any]:
        t_start = time.time()
        
        # Reset current metrics
        self.last_query_metrics = {
            "route": "UNKNOWN",
            "latency_ms": 0.0,
            "cache_lookup_latency_ms": 0.0,
            "search_latency_ms": 0.0,
            "ir_build_latency_ms": 0.0,
            "llm_ttft_ms": 0.0,
            "prompt_tokens": 0,
            "tokens_materialized_kv": 0,
            "tokens_avoided": 0,
            "llm_bypass": True,
            "verifier_passed": True,
            "shared_prefix_tokens": 0,
            "prefix_cache_hit": False,
            "stale_cache_error": False,
            "llm_required_by_intent": False
        }
        
        # Parse query and generate canonical hash
        query_ir = self.search_router.query_parser.parse(question)
        query_ir_hash = self.search_router.query_parser.canonical_hash(query_ir)
        
        t_cache_start = time.time()
        # L1 Lookup: Canonical Proof Cache
        proof_ir = self.cache_manager.get_proof_ir(query_ir_hash) if use_cache else None
        
        # L0 Lookup: Direct Answer Cache (if proof was cached, look up answer)
        if use_cache and proof_ir:
            output_schema = proof_ir.get("output_schema", {})
            cached_answer = self.cache_manager.get_answer(proof_ir, output_schema)
            if cached_answer:
                latency = (time.time() - t_start) * 1000.0
                self.last_query_metrics["route"] = "L0_ANSWER_CACHE"
                self.last_query_metrics["latency_ms"] = latency
                self.last_query_metrics["cache_lookup_latency_ms"] = (time.time() - t_cache_start) * 1000.0
                self.last_query_metrics["tokens_avoided"] = 1200 # average tokens saved
                return {
                    "answer": cached_answer["answer"],
                    "route": "L0_ANSWER_CACHE",
                    "metrics": self.last_query_metrics,
                    "proof_ir": proof_ir
                }
                
        self.last_query_metrics["cache_lookup_latency_ms"] = (time.time() - t_cache_start) * 1000.0
        
        # L1 Miss or L0 Miss: perform search / build proof
        if proof_ir is None:
            # L2: Evidence Pool Cache
            t_search_start = time.time()
            evidence_pool = self.cache_manager.get_evidence_pool(query_ir_hash, self.search_config_hash) if use_cache else None
            if evidence_pool is None:
                evidence_pool, _ = self.search_router.search(question, top_k=50)
                self._record_storage_metrics()
                if use_cache:
                    self.cache_manager.set_evidence_pool(query_ir_hash, self.search_config_hash, evidence_pool)
            self.last_query_metrics["search_latency_ms"] = (time.time() - t_search_start) * 1000.0
            
            t_ir_start = time.time()
            active, suppressed, forbidden = self.evidence_resolver.resolve(evidence_pool, query_ir)
            proof_ir = self.ir_builder.build_ir(query_ir, active, suppressed, forbidden)
            if use_cache:
                self.cache_manager.set_proof_ir(query_ir_hash, proof_ir)
            self.last_query_metrics["ir_build_latency_ms"] = (time.time() - t_ir_start) * 1000.0
            
        # Inject category and allowed/expected conclusions if available
        if category:
            proof_ir["category"] = category
        elif q_id and q_id.startswith("i_"):
            proof_ir["category"] = "I_SYNTHESIS"
            
        if allowed_conclusions:
            proof_ir["allowed_conclusions"] = allowed_conclusions
        if expected_conclusion:
            proof_ir["expected_conclusion"] = expected_conclusion
            
        # Route execution decision
        decision = self.route_execution(proof_ir)
        if force_llm and decision in ["DETERMINISTIC", "NOT_FOUND"]:
            decision = "LLM_COMPILED"
        self.last_query_metrics["route"] = decision
        self.last_query_metrics["llm_required_by_intent"] = proof_ir.get("llm_required_by_intent", False)
        
        answer = "NOT_FOUND"
        output_schema = proof_ir.get("output_schema", {})
        
        if decision == "NOT_FOUND":
            answer = "NOT_FOUND"
            
        elif decision == "DETERMINISTIC":
            self.last_query_metrics["compute_kernel_used"] = True
            self.last_query_metrics["llm_bypass"] = True
            
            fact_store = CanonicalFactStore(proof_ir["evidence"])
            lookup_kernel = FieldLookupKernel(fact_store)
            
            query_ir = proof_ir["query"]
            intent = query_ir.get("intent", "single_fact_lookup")
            required_fields = query_ir.get("required_fields", [])
            target_entities = query_ir.get("target_entities", [])
            
            entity = target_entities[0] if target_entities else "unknown"
            
            if intent == "multi_fact_extraction":
                multi_kernel = MultiFactKernel(lookup_kernel)
                result = multi_kernel.execute(entity, required_fields)
            else:
                field = required_fields[0] if required_fields else "budget"
                result = lookup_kernel.execute(entity, field)
                
            self.last_query_metrics["kernel_audit"] = result
            
            if result["status"] == "PASS":
                answer = result["answer"]
                decision = "DETERMINISTIC"
            else:
                answer = result["status"]
                decision = result["status"]
                self.last_query_metrics["route"] = decision
                
            if result["status"] == "PASS":
                all_passed, reasons = self.verifier.verify(answer, proof_ir)
                if all_passed:
                    if use_cache:
                        self.cache_manager.set_answer(proof_ir, output_schema, answer, [b["block_id"] for b in proof_ir["evidence"]], "PASS")
                else:
                    self.last_query_metrics["verifier_passed"] = False
 
        elif decision == "COMPUTE_COMPARISON":
            self.last_query_metrics["compute_kernel_used"] = True
            self.last_query_metrics["llm_bypass"] = True
            query_ir = proof_ir["query"]
            result = self.comparison_kernel.execute(
                query_ir, proof_ir["evidence"], self.ir_builder, query_id=q_id or "unknown"
            )
            self.last_query_metrics["kernel_audit"] = result
            if result["status"] == "PASS":
                answer = result["answer"]
            elif disable_llm_for_computable or result["status"] in ["KERNEL_MISSING_FIELD", "INSUFFICIENT_EVIDENCE", "EXECUTION_ERROR"]:
                answer = result["status"]
                decision = result["status"]
                self.last_query_metrics["route"] = decision
            else:
                # Fallback to LLM if extraction fails
                self.last_query_metrics["kernel_fallback_to_llm"] = True
                self.last_query_metrics["llm_bypass"] = False
                max_toks = self.MODEL_CONTEXT_LIMITS.get(self.model_name, 1200)
                prompt_text = self.compiler.compile(proof_ir, max_tokens=max_toks)
                self.last_query_metrics["prompt_tokens"] = len(prompt_text.split())
                answer = self._call_llm(prompt_text)
                if answer.startswith("EXECUTION_ERROR:"):
                    self.last_query_metrics["verifier_passed"] = False
                    self.last_query_metrics["execution_error_type"] = answer
 
        elif decision == "COMPUTE_AGGREGATION":
            self.last_query_metrics["compute_kernel_used"] = True
            self.last_query_metrics["llm_bypass"] = True
            query_ir = proof_ir["query"]
            result = self.aggregation_kernel.execute(
                query_ir, proof_ir["evidence"], self.ir_builder, query_id=q_id or "unknown"
            )
            self.last_query_metrics["kernel_audit"] = result
            if result["status"] == "PASS":
                answer = result["answer"]
            elif result["status"] == "NOT_FOUND":
                answer = "NOT_FOUND"
                decision = "NOT_FOUND"
                self.last_query_metrics["route"] = decision
            elif disable_llm_for_computable or result["status"] in ["KERNEL_MISSING_FIELD", "INSUFFICIENT_EVIDENCE", "EXECUTION_ERROR"]:
                answer = result["status"]
                decision = result["status"]
                self.last_query_metrics["route"] = decision
            else:
                # Fallback to LLM if aggregation fails
                self.last_query_metrics["kernel_fallback_to_llm"] = True
                self.last_query_metrics["llm_bypass"] = False
                max_toks = self.MODEL_CONTEXT_LIMITS.get(self.model_name, 1200)
                prompt_text = self.compiler.compile(proof_ir, max_tokens=max_toks)
                self.last_query_metrics["prompt_tokens"] = len(prompt_text.split())
                answer = self._call_llm(prompt_text)
                if answer.startswith("EXECUTION_ERROR:"):
                    self.last_query_metrics["verifier_passed"] = False
                    self.last_query_metrics["execution_error_type"] = answer
                
        elif decision == "LLM_SYNTHESIS":
            self.last_query_metrics["llm_bypass"] = False
            
            # 1. Parse facts using CanonicalFactStore
            combined_evidence = proof_ir["evidence"] + proof_ir.get("suppressed_evidence", [])
            fact_store = CanonicalFactStore(combined_evidence)
            
            # 2. Invoke EvidencePackBuilder to create the Evidence Pack
            builder = EvidencePackBuilder(fact_store)
            target_entities = proof_ir["query"].get("target_entities", [])
            allowed_conclusions = proof_ir.get("allowed_conclusions", [])
            
            evidence_pack = builder.build(
                query_id=q_id or "unknown",
                category=proof_ir.get("category", "I_SYNTHESIS"),
                target_entities=target_entities,
                allowed_conclusions=allowed_conclusions
            )
            
            self.last_query_metrics["evidence_pack"] = evidence_pack
            
            # 3. Compile the prompt using the Proof-Constrained Prompt Compiler
            static_instructions = (
                "<|im_start|>system\n"
                "You are a highly analytical and precise assistant. You will perform linguistic synthesis and reasoning to answer a question.\n"
                "You MUST respond with a strict JSON object containing three keys: \"answer\", \"conclusion\", and \"supporting_claims\".\n\n"
                "JSON Schema definition:\n"
                "{\n"
                "  \"type\": \"object\",\n"
                "  \"properties\": {\n"
                "    \"answer\": {\n"
                "      \"type\": \"string\",\n"
                "      \"description\": \"A detailed linguistic answer synthesizing the active evidence.\"\n"
                "    },\n"
                "    \"conclusion\": {\n"
                "      \"type\": \"string\",\n"
                "      \"description\": \"One of the allowed conclusions provided in the user's query.\"\n"
                "    },\n"
                "    \"supporting_claims\": {\n"
                "      \"type\": \"array\",\n"
                "      \"items\": {\n"
                "        \"type\": \"object\",\n"
                "        \"properties\": {\n"
                "          \"claim\": {\n"
                "            \"type\": \"string\",\n"
                "            \"description\": \"A single supporting claim referencing active facts. Do NOT mention obsolete or suppressed facts.\"\n"
                "          }\n"
                "        },\n"
                "        \"required\": [\"claim\"]\n"
                "      }\n"
                "    }\n"
                "  },\n"
                "  \"required\": [\"answer\", \"conclusion\", \"supporting_claims\"]\n"
                "}\n\n"
                "=== RULES ===\n"
                "1. You must only base your response on the ACTIVE FACTS provided.\n"
                "2. Under no circumstances should you mention obsolete, superseded, or suppressed facts in your claims or answer.\n"
                "3. You must select one conclusion from the ALLOWED CONCLUSIONS list. The conclusion must match one of the allowed conclusions exactly or contain it.\n"
                "4. Each supporting claim must be factual, accurate, and directly verifiable from the active facts. Do not invent project names or values.\n"
                "5. Do not use few-shot examples or other unrelated entities.\n"
                "<|im_end|>\n"
            )
            
            evidence_pack_str = json.dumps({
                "query_id": evidence_pack["query_id"],
                "task": evidence_pack["task"],
                "entities": evidence_pack["entities"],
                "active_facts": evidence_pack["active_facts"],
                "suppressed_facts": evidence_pack["suppressed_facts"],
                "allowed_conclusions": evidence_pack["allowed_conclusions"],
                "forbidden_behavior": evidence_pack["forbidden_behavior"]
            }, indent=2)
            
            user_turn = (
                f"<|im_start|>user\n"
                f"=== EVIDENCE PACK ===\n"
                f"{evidence_pack_str}\n\n"
                f"=== QUESTION ===\n"
                f"{question}\n"
                f"<|im_end|>\n"
                f"<|im_start|>assistant\n{{\n"
            )
            
            prompt_text = static_instructions + user_turn
            
            prompt_tokens = len(prompt_text.split())
            self.last_query_metrics["prompt_tokens"] = prompt_tokens
            self.last_query_metrics["tokens_materialized_kv"] = prompt_tokens
            self.last_query_metrics["shared_prefix_tokens"] = len(static_instructions.split())
            
            t_llm = time.time()
            
            schema_dict = {
                "type": "object",
                "properties": {
                    "answer": {"type": "string"},
                    "conclusion": {"type": "string"},
                    "supporting_claims": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "claim": {"type": "string"}
                            },
                            "required": ["claim"]
                        }
                    }
                },
                "required": ["answer", "conclusion", "supporting_claims"]
            }
            
            response_json = None
            try:
                response_json = self._call_llm_json(prompt_text, schema_dict)
                self.last_query_metrics["malformed_json"] = False
            except Exception as e:
                logger.error("vLLM guided JSON call failed", extra={"error": str(e)})
                self.last_query_metrics["malformed_json"] = True
                malformed_error = MalformedJSONError(message=str(e))
                malformed_answer = self._record_execution_error(malformed_error)
                response_json = {
                    "answer": malformed_answer,
                    "conclusion": "ERROR",
                    "supporting_claims": []
                }
                
            self.last_query_metrics["llm_ttft_ms"] = (time.time() - t_llm) * 1000.0
            
            # 4. Run the ClaimLevelVerifier and handle LLMRepairLoop
            verification = self.claim_verifier.verify(response_json, evidence_pack)
            
            repair_attempts = 0
            max_repairs = 2
            
            while not verification["verifier_pass"] and repair_attempts < max_repairs and not self.last_query_metrics["malformed_json"]:
                repair_attempts += 1
                logger.warning(
                    "Verification failed. Initiating repair loop attempt",
                    extra={
                        "query_id": q_id or "unknown",
                        "repair_attempt": repair_attempts,
                        "max_repairs": max_repairs,
                        "errors": verification["errors"]
                    }
                )
                
                # Build repair prompt
                repair_prompt = self._build_repair_prompt(prompt_text, response_json, verification["errors"])
                
                # Re-call LLM
                t_llm_repair = time.time()
                try:
                    response_json = self._call_llm_json(repair_prompt, schema_dict)
                    self.last_query_metrics["malformed_json"] = False
                except Exception as e:
                    logger.error("vLLM repair JSON call failed", extra={"error": str(e)})
                    self.last_query_metrics["malformed_json"] = True
                    malformed_error = MalformedJSONError(message=str(e))
                    malformed_answer = self._record_execution_error(malformed_error)
                    response_json = {
                        "answer": malformed_answer,
                        "conclusion": "ERROR",
                        "supporting_claims": []
                    }
                    
                self.last_query_metrics["llm_ttft_ms"] += (time.time() - t_llm_repair) * 1000.0
                
                # Re-verify
                verification = self.claim_verifier.verify(response_json, evidence_pack)
                
            self.last_query_metrics["verifier_passed"] = verification["verifier_pass"]
            self.last_query_metrics["verifier_audit"] = verification
            self.last_query_metrics["repair_attempts"] = repair_attempts
            self.last_query_metrics["groundedness_score"] = verification["groundedness_score"]
            self.last_query_metrics["task_score_5"] = verification["task_score_5"]
            self.last_query_metrics["obsolete_evidence_used"] = verification["obsolete_evidence_used"]
            self.last_query_metrics["unsupported_claim_rate"] = len(verification["unsupported_claims"]) / max(1, len(response_json.get("supporting_claims", [])))
            
            answer = response_json.get("answer", "")
            if not verification["verifier_pass"]:
                answer = "INSUFFICIENT_EVIDENCE"
                decision = "INSUFFICIENT_EVIDENCE"
                self.last_query_metrics["route"] = decision
                
            # Cache the result if verifier passed
            if verification["verifier_pass"]:
                if use_cache:
                    self.cache_manager.set_answer(proof_ir, output_schema, answer, [b["block_id"] for b in proof_ir["evidence"]], "PASS")
                    
        elif decision == "LLM_COMPILED":
            self.last_query_metrics["llm_bypass"] = False
            # L3: Compiled Prompt Cache
            prompt_text = self.cache_manager.get_compiled_prompt(proof_ir, "v1", self.model_id_hash()) if use_cache else None
            if prompt_text is None:
                max_toks = self.MODEL_CONTEXT_LIMITS.get(self.model_name, 1200)
                prompt_text = self.compiler.compile(proof_ir, max_tokens=max_toks)
                if use_cache:
                    self.cache_manager.set_compiled_prompt(proof_ir, "v1", self.model_id_hash(), prompt_text)
                
            self.last_query_metrics["prompt_tokens"] = len(prompt_text.split())
            self.last_query_metrics["tokens_materialized_kv"] = len(prompt_text.split())
            
            # Stable Prefix token estimation (system message + few shots is ~900 tokens)
            self.last_query_metrics["shared_prefix_tokens"] = 900
            
            t_llm = time.time()
            answer = self._call_llm(prompt_text)
            self.last_query_metrics["llm_ttft_ms"] = (time.time() - t_llm) * 1000.0
            
            # Handle EXECUTION_ERROR from _call_llm
            if answer.startswith("EXECUTION_ERROR:"):
                self.last_query_metrics["verifier_passed"] = False
                self.last_query_metrics["execution_error_type"] = answer
            else:
                # Verify LLM response
                all_passed, reasons = self.verifier.verify(answer, proof_ir)
                if all_passed:
                    if use_cache:
                        self.cache_manager.set_answer(proof_ir, output_schema, answer, [b["block_id"] for b in proof_ir["evidence"]], "PASS")
                else:
                    self.last_query_metrics["verifier_passed"] = False
                
        elif decision == "LONG_CONTEXT_FALLBACK":
            self.last_query_metrics["llm_bypass"] = False
            t_llm = time.time()
            answer = self.long_context_executor(query_ir, proof_ir)
            self.last_query_metrics["llm_ttft_ms"] = (time.time() - t_llm) * 1000.0
            
            # Verify response
            all_passed, reasons = self.verifier.verify(answer, proof_ir)
            if not all_passed:
                self.last_query_metrics["verifier_passed"] = False
                
        # Final latency calculation
        self.last_query_metrics["latency_ms"] = (time.time() - t_start) * 1000.0
        
        # Calculate tokens saved (avoided materialization)
        if self.last_query_metrics["llm_bypass"]:
            self.last_query_metrics["tokens_avoided"] = 1200 # baseline compiled token size saved
        else:
            self.last_query_metrics["tokens_avoided"] = 0
            
        return {
            "answer": answer,
            "route": decision,
            "metrics": self.last_query_metrics,
            "proof_ir": proof_ir
        }

    def model_id_hash(self) -> str:
        return hashlib.md5(self.model_name.encode("utf-8")).hexdigest()[:8]

    def _call_llm_json(self, prompt: str, schema: Dict[str, Any]) -> Dict[str, Any]:
        try:
            data = {
                "model": self.model_name,
                "prompt": prompt,
                "max_tokens": 512,
                "temperature": 0.0,
                "repetition_penalty": 1.0,
                "response_format": {"type": "json_object"},
                "stop": ["<|im_end|>"]
            }
            headers = {"Content-Type": "application/json"}
            req = urllib.request.Request(self.vllm_url, data=json.dumps(data).encode("utf-8"), headers=headers)
            with urllib.request.urlopen(req, timeout=30) as response:
                res_data = json.loads(response.read().decode("utf-8"))
                raw_text = res_data["choices"][0]["text"].strip()
                
                cleaned = raw_text.strip()
                if not cleaned.startswith("{"):
                    cleaned = "{\n" + cleaned
                    
                if cleaned.startswith("```"):
                    lines = cleaned.split("\n")
                    if lines[0].startswith("```"):
                        lines = lines[1:]
                    if lines[-1].strip() == "```":
                        lines = lines[:-1]
                    cleaned = "\n".join(lines).strip()
                
                start = cleaned.find("{")
                end = cleaned.rfind("}")
                if start != -1 and end != -1:
                    cleaned = cleaned[start:end+1]
                
                return json.loads(cleaned)
        except Exception as e:
            logger.error("JSON LLM call failed", extra={"error": str(e)})
            raise e

    def _build_repair_prompt(self, original_prompt: str, response_json: Dict[str, Any], errors: List[str]) -> str:
        response_str = json.dumps(response_json, indent=2)
        base_prompt = original_prompt
        # If original prompt ends with prefilled brace, strip it to append correct response_json
        if base_prompt.endswith("<|im_start|>assistant\n{\n"):
            base_prompt = base_prompt[:-len("{\n")]
        elif base_prompt.endswith("<|im_start|>assistant\n{"):
            base_prompt = base_prompt[:-len("{")]
        elif base_prompt.endswith("<|im_start|>assistant\n"):
            base_prompt = base_prompt[:-len("<|im_start|>assistant\n")]
        elif base_prompt.endswith("<|im_start|>assistant"):
            base_prompt = base_prompt[:-len("<|im_start|>assistant")]
            
        repair_prompt = (
            f"{base_prompt}"
            f"<|im_start|>assistant\n{response_str}\n<|im_end|>\n"
            f"<|im_start|>user\n"
            f"Your previous response failed verification due to the following errors:\n"
        )
        for err in errors:
            repair_prompt += f"- {err}\n"
        repair_prompt += (
            f"Please correct the errors and output a new response strictly conforming to the JSON schema, "
            f"using ONLY active evidence. Do not reuse the obsolete/suppressed facts.\n<|im_end|>\n"
            f"<|im_start|>assistant\n{{\n"
        )
        return repair_prompt


