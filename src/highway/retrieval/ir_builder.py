import re
from typing import List, Dict, Any

class IRBuilder:
    def __init__(self):
        self.LLM_REQUIRED_INTENTS = set()  # G/H now handled by compute kernels
        self.project_names = [
            "NEPTUNE", "KRONOS", "ECLIPSE", "FALCON", "IRIS", "JUPITER", "METEOR", "NEXUS",
            "ORION", "PHOENIX", "QUASAR", "SIRIUS", "TITAN", "VEGA", "ZENITH", "AURORA",
            "BEACON", "CHRONOS", "DAWN", "GENESIS", "HELIOS", "LUNA", "PEGASE", "SOLARIS"
        ]
        self.people = [
            "Jean Dupont", "Alice Martin", "Pierre Leroy", "Marie Dubois", "Thomas Petit",
            "Sophie Richard", "Michel Bernard", "Julie Thomas", "Nicolas Durand", "Emma Michel"
        ]
        self.departments = ["Engineering", "Finance", "HR", "Legal", "Marketing", "Operations", "Sales"]
        self.locations = ["Paris", "Lyon", "Marseille", "Toulouse", "Nice", "Nantes", "Strasbourg", "Bordeaux"]

    def _extract_field(self, text: str, field: str) -> str:
        text_clean = text.replace('\n', ' ')
        if field == "budget":
            patterns = [
                r'updated budget is increased to\s+(\$\d{1,3}(?:,\d{3})*)',
                r'updated budget to\s+(\$\d{1,3}(?:,\d{3})*)',
                r'official budget for Project\s+[A-Za-z0-9_\-]+\s+is\s+(\$\d{1,3}(?:,\d{3})*)',
                r'Approved Budget:\s+(\$\d{1,3}(?:,\d{3})*)',
                r'financial allocation request of\s+(\$\d{1,3}(?:,\d{3})*)\s+was approved',
                r'budget is\s+(\$\d{1,3}(?:,\d{3})*)',
                r'budget of\s+[A-Za-z0-9_\-]+\s+was\s+(\$\d{1,3}(?:,\d{3})*)',
                r'budget of\s+(\$\d{1,3}(?:,\d{3})*)',
                r'(\$\d{1,3}(?:,\d{3})*)'
            ]
            for p in patterns:
                m = re.search(p, text_clean, re.IGNORECASE)
                if m:
                    return m.group(1)
                    
        elif field == "deadline":
            patterns = [
                r'officially extended to\s+(\d{1,2}\s+[A-Za-z]+\s+\d{4})',
                r'final deadline of Project\s+[A-Za-z0-9_\-]+\s+is set to\s+(\d{1,2}\s+[A-Za-z]+\s+\d{4})',
                r'Completion Date:\s+(\d{1,2}\s+[A-Za-z]+\s+\d{4})',
                r'milestone deadline is set to\s+(\d{1,2}\s+[A-Za-z]+\s+\d{4})',
                r'Delivery Date:\s+(\d{1,2}\s+[A-Za-z]+\s+\d{4})',
                r'completion is set to\s+(\d{1,2}\s+[A-Za-z]+\s+\d{4})',
                r'(\d{1,2}\s+[A-Za-z]+\s+\d{4})'
            ]
            for p in patterns:
                m = re.search(p, text_clean, re.IGNORECASE)
                if m:
                    return m.group(1)
                    
        elif field == "owner":
            for person in self.people:
                if person.lower() in text.lower():
                    return person
                    
        elif field == "department":
            patterns = [
                r'Department:\s*([A-Za-z]+)',
                r'Lead Unit:\s*([A-Za-z]+)',
                r'overseen by the\s*([A-Za-z]+)\s+unit'
            ]
            for p in patterns:
                m = re.search(p, text_clean, re.IGNORECASE)
                if m and m.group(1) in self.departments:
                    return m.group(1)
            for dept in self.departments:
                if re.search(r'\b' + re.escape(dept) + r'\b', text, re.IGNORECASE):
                    if dept.lower() == "finance" and "finance committee" in text.lower() and "department: finance" not in text.lower() and "lead unit: finance" not in text.lower():
                        continue
                    return dept
                    
        elif field == "location":
            patterns = [
                r'Location:\s*([A-Za-z]+)',
                r'based in\s*([A-Za-z]+)'
            ]
            for p in patterns:
                m = re.search(p, text_clean, re.IGNORECASE)
                if m and m.group(1) in self.locations:
                    return m.group(1)
            for loc in self.locations:
                if re.search(r'\b' + re.escape(loc) + r'\b', text, re.IGNORECASE):
                    return loc
                    
        return None

    def build_ir(self, query_ir: Dict[str, Any], active_evidence: List[Dict[str, Any]], suppressed_evidence: List[Dict[str, Any]], forbidden_matches: List[str]) -> Dict[str, Any]:
        # 1. Determine Proof Status and Field Coverage
        required_fields = query_ir.get("required_fields", [])
        target_entities = query_ir.get("target_entities", [])
        intent = query_ir.get("intent", "single_fact_lookup")
        
        # Check field coverage in active evidence
        field_coverage = {}
        for field in required_fields:
            field_coverage[field] = False
            # Check if any active block mentions the field keyword or seems to contain the evidence
            for ev in active_evidence:
                text_lower = ev["text"].lower()
                if field == "budget" and ("budget" in text_lower or "$" in text_lower):
                    field_coverage[field] = True
                elif field == "deadline" and ("deadline" in text_lower or "date" in text_lower):
                    field_coverage[field] = True
                elif field == "owner" and ("owner" in text_lower or "author" in text_lower or "manager" in text_lower or any(p.lower() in text_lower for p in self.people)):
                    field_coverage[field] = True
                elif field == "department" and ("department" in text_lower or "unit" in text_lower or any(d.lower() in text_lower for d in self.departments)):
                    field_coverage[field] = True
                elif field == "location" and ("location" in text_lower or "based in" in text_lower or any(l.lower() in text_lower for l in self.locations)):
                    field_coverage[field] = True
                    
        # Calculate completeness score
        if required_fields:
            completeness_score = sum(1 for f in required_fields if field_coverage.get(f, False)) / len(required_fields)
        else:
            completeness_score = 1.0 if active_evidence else 0.0
            
        # Determine proof status
        if not active_evidence and not suppressed_evidence:
            proof_status = "ABSENT"
        elif not active_evidence and suppressed_evidence:
            proof_status = "INSUFFICIENT" # Only distractors
        elif completeness_score == 1.0:
            proof_status = "COMPLETE"
        elif completeness_score > 0.0:
            proof_status = "PARTIAL"
        else:
            proof_status = "INSUFFICIENT"
            
        # 2. Guard Chain Evaluation (evaluated in order)
        guard_decision = {
            "llm_required": True,
            "action": "PROCEED",
            "answer": None,
            "reason": None
        }
        
        # Guard 1: Entity Absence or Empty Active Evidence
        if proof_status == "ABSENT" or (intent != "aggregation" and (not target_entities or len(active_evidence) == 0)):
            guard_decision = {
                "llm_required": False,
                "action": "BYPASS_LLM",
                "answer": "NOT_FOUND",
                "reason": "Target entity not found in any evidence block or no active evidence remains"
            }
        # Guard 2: Suffix Ambiguity
        elif not active_evidence and any(se.get("suppression_reason") == "suffix_distractor" for se in suppressed_evidence):
            guard_decision = {
                "llm_required": False,
                "action": "BYPASS_LLM",
                "answer": "NOT_FOUND",
                "reason": "Only suffix-distractor matches found, no exact entity match"
            }
        # Guard 3: Deterministic Bypass for COMPLETE proof
        elif proof_status == "COMPLETE":
            if intent in self.LLM_REQUIRED_INTENTS:
                guard_decision = {
                    "llm_required": True,
                    "action": "PROCEED",
                    "answer": None,
                    "reason": f"Intent '{intent}' is LLM-required â€” deterministic bypass forbidden"
                }
            elif intent in {"comparison", "aggregation"}:
                # Compute kernel eligible â€” let the scheduler route to COMPUTE_* kernels
                guard_decision = {
                    "llm_required": False,
                    "action": "PROCEED",
                    "answer": None,
                    "reason": f"Intent '{intent}' is compute-kernel-eligible â€” routed to kernel"
                }
            else:
                # Attempt to extract answers deterministically
                bypass_answer = None
                
                if intent == "multi_fact_extraction":
                    # Extract multiple fields
                    extracted_values = []
                    success = True
                    for field in required_fields:
                        val = None
                        for b in active_evidence:
                            val = self._extract_field(b["text"], field)
                            if val:
                                break
                        if val:
                            extracted_values.append(val)
                        else:
                            success = False
                            break
                    if success and len(extracted_values) == len(required_fields):
                        bypass_answer = " and ".join(extracted_values)
                        
                else:
                    # single_fact_lookup / status_resolution
                    field = required_fields[0] if required_fields else "budget" # default fallback
                    for b in active_evidence:
                        val = self._extract_field(b["text"], field)
                        if val:
                            bypass_answer = val
                            break
                
                if bypass_answer is not None:
                    guard_decision = {
                        "llm_required": False,
                        "action": "BYPASS_LLM",
                        "answer": bypass_answer,
                        "reason": f"Deterministic bypass succeeded for complete proof of intent {intent}"
                    }
                    
        # Guard 4: Partial proof
        elif proof_status == "PARTIAL":
            if intent in self.LLM_REQUIRED_INTENTS:
                guard_decision = {
                    "llm_required": True,
                    "action": "PROCEED",
                    "answer": None,
                    "reason": f"Intent '{intent}' is LLM-required â€” proceeding without warning"
                }
            else:
                missing = [f for f in required_fields if not field_coverage[f]]
                guard_decision = {
                    "llm_required": True,
                    "action": "PROCEED_WITH_WARNING",
                    "answer": None,
                    "reason": f"Missing evidence for fields: {missing}"
                }
            
        # 3. Output Schema Specification
        example = ""
        if intent == "comparison":
            example = "Project NAME (budget of BUDGET)"
        elif intent == "aggregation":
            example = "PROJECT_1, PROJECT_2"
        elif intent == "multi_fact_extraction":
            parts = []
            for f in required_fields:
                if f == "deadline":
                    parts.append("DATE")
                else:
                    parts.append(f.upper())
            example = " and ".join(parts)
        else:
            if required_fields:
                f = required_fields[0]
                if f == "deadline":
                    example = "DATE"
                else:
                    example = f.upper()
            else:
                example = "VALUE"
                
        output_schema = {
            "example": example
        }
        
        return {
            "query": query_ir,
            "proof": {
                "status": proof_status,
                "completeness_score": completeness_score,
                "field_coverage": field_coverage
            },
            "evidence": active_evidence,
            "suppressed_evidence": suppressed_evidence,
            "forbidden_matches": forbidden_matches,
            "guard_decision": guard_decision,
            "output_schema": output_schema,
            "compute_kernel_eligible": intent in {"comparison", "aggregation"},
            "llm_required_by_intent": intent in self.LLM_REQUIRED_INTENTS
        }


