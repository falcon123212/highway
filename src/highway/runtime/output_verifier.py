import re
from typing import Dict, Any, List, Tuple

class OutputVerifier:
    def __init__(self):
        pass

    def extract_terms(self, text: str) -> List[str]:
        # Extract numbers (e.g. $450,000, 2027) and capitalized names (e.g. Alice Martin)
        # to check if they are grounded in the evidence.
        numbers = re.findall(r'\b\$?\d+(?:,\d+)*(?:\.\d+)?(?:Mâ‚¬|k|M)?\b', text)
        names = re.findall(r'\b[A-Z][a-z]+\s+[A-Z][a-z]+\b', text)
        return list(set(numbers + names))

    def verify(self, answer: str, ir: Dict[str, Any]) -> Tuple[bool, List[str]]:
        reasons = []
        answer_clean = answer.strip()
        
        # If the IR states ABSENT or we bypassed because of an empty/absent proof, the answer must be NOT_FOUND
        if ir["proof"]["status"] == "ABSENT" or (ir["guard_decision"]["action"] == "BYPASS_LLM" and ir["guard_decision"].get("answer") == "NOT_FOUND"):
            if "NOT_FOUND" in answer_clean:
                return True, []
            else:
                reasons.append("Answer should have been NOT_FOUND due to absent evidence / LLM bypass.")
                return False, reasons

        # If LLM answered NOT_FOUND, check if that was reasonable (e.g. if proof is INSUFFICIENT or PARTIAL)
        if "NOT_FOUND" in answer_clean:
            # If the proof is complete, answering NOT_FOUND is a false negative
            if ir["proof"]["status"] == "COMPLETE":
                reasons.append("Model answered NOT_FOUND but evidence was COMPLETE.")
                return False, reasons
            return True, []

        # 1. Check for Forbidden Entities in the answer
        forbidden = ir.get("forbidden_matches", [])
        for f in forbidden:
            f_lower = f.lower()
            # Boundary check
            pattern = r'(?<![a-zA-Z0-9_\-])' + re.escape(f_lower) + r'(?![a-zA-Z0-9_\-])'
            if re.search(pattern, answer_clean.lower()):
                reasons.append(f"Forbidden entity '{f}' was mentioned in the answer.")

        # 2. Check Citation Grounding
        # Combine all active evidence texts
        evidence_text = " ".join([ev["text"] for ev in ir["evidence"]]).lower()
        extracted = self.extract_terms(answer_clean)
        
        for term in extracted:
            term_clean = term.lower().replace("$", "").replace(",", "")
            # We check if this term is grounded in the evidence text
            if term_clean not in evidence_text.replace(",", ""):
                # Allow minor formatting differences
                reasons.append(f"Answer term '{term}' is not grounded in the active evidence.")

        # 3. Check format example compliance
        example_format = ir["output_schema"].get("example", "")
        if "and" in example_format.lower() and "and" not in answer_clean.lower():
            reasons.append("Answer does not comply with the requested format (missing 'and' between facts).")

        all_passed = len(reasons) == 0
        return all_passed, reasons


