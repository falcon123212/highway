import re
from typing import List, Dict, Any

class QueryParser:
    def __init__(self, entity_list: List[str]):
        # Sort entities by length descending to match longer entities first (e.g. NEPTUNE-Legacy before NEPTUNE)
        self.entity_list = sorted(entity_list, key=len, reverse=True)
        self.entity_token_index = self._build_entity_token_index(self.entity_list)
        
        self.field_map = {
            "budget": ["budget", "coÃ»t", "montant", "prix", "cost", "amount", "financial"],
            "deadline": ["deadline", "date", "quand", "when", "Ã©chÃ©ance", "dÃ©lai", "delivery date"],
            "owner": ["manager", "owner", "responsable", "directeur", "director", "who", "author", "technician"],
            "department": ["department", "unit", "division", "service"],
            "location": ["location", "oÃ¹", "where", "site", "city", "pays", "ville"]
        }

    def _build_entity_token_index(self, entities: List[str]) -> Dict[str, List[str]]:
        token_index: Dict[str, List[str]] = {}
        ignored_tokens = {"project", "projects"}
        for entity in entities:
            tokens = re.findall(r"[a-z0-9_\-]+", entity.lower())
            for token in tokens:
                if token in ignored_tokens:
                    continue
                token_index.setdefault(token, []).append(entity)
        return token_index

    def _candidate_entities(self, question: str) -> List[str]:
        tokens = re.findall(r"[a-z0-9_\-]+", question.lower())
        seen = set()
        candidates = []
        for token in tokens:
            for entity in self.entity_token_index.get(token, []):
                if entity not in seen:
                    candidates.append(entity)
                    seen.add(entity)
        return sorted(candidates, key=len, reverse=True)

    def parse(self, question: str) -> Dict[str, Any]:
        question_clean = question.lower()
        
        # 1. Entity Extraction
        matched_entities = []
        # We search for entities in the question
        for entity in self._candidate_entities(question):
            entity_lower = entity.lower()
            # Use custom boundary to prevent matching substrings followed by hyphen or underscore
            pattern = r'(?<![a-zA-Z0-9_\-])' + re.escape(entity_lower) + r'(?![a-zA-Z0-9_\-])'
            if re.search(pattern, question_clean):
                # Check if this entity is a substring of an already matched longer entity
                # to prevent matching "NEPTUNE" when "NEPTUNE-Legacy" is already matched.
                if not any(entity_lower in existing.lower() for existing in matched_entities):
                    matched_entities.append(entity)
        
        # 2. Field Detection (ordered by appearance in the question)
        field_positions = []
        for field, keywords in self.field_map.items():
            first_pos = float('inf')
            for kw in keywords:
                pos = question_clean.find(kw)
                if pos != -1 and pos < first_pos:
                    first_pos = pos
            if first_pos != float('inf'):
                field_positions.append((field, first_pos))
        
        field_positions.sort(key=lambda x: x[1])
        detected_fields = [f[0] for f in field_positions]
        
        # 3. Intent Classification
        intent = "single_fact_lookup" # default
        if len(detected_fields) >= 2:
            intent = "multi_fact_extraction"
        elif any(kw in question_clean for kw in ["compare", "vs", "diffÃ©rence", "higher", "lower", "larger", "smaller"]):
            intent = "comparison"
        elif any(kw in question_clean for kw in ["list all", "summarize all", "all project", "projects managed by"]):
            intent = "aggregation"
        elif any(kw in question_clean for kw in ["obsolete", "active", "updated", "supersede", "revised", "contract", "amendment"]):
            intent = "status_resolution"
            
        # If no entities matched and we are looking for a project-like pattern,
        # it might be an absent entity
        is_absent_candidate = len(matched_entities) == 0
        if is_absent_candidate:
            intent = "abstention_candidate"
            # Try to extract a potential unseen entity (e.g. project name in caps or capitalized nouns)
            # This helps the Search Router know what to search for or what entity is absent.
            potential = re.findall(r'\b[A-Z][A-Z0-9_\-]+\b', question)
            if potential:
                matched_entities = potential
            else:
                # Fallback: extract capitalized word
                potential_cap = re.findall(r'\b[A-Z][a-z]+\b', question)
                if potential_cap:
                    matched_entities = potential_cap
        
        # Constraints
        constraints = {
            "strict_entity": True,
            "status_preference": "active" if intent != "status_resolution" else "all",
            "temporal_preference": "latest"
        }
        reference_match = re.search(r'\bref_[0-9a-f]{10}\b', question_clean)
        if reference_match:
            constraints["reference_marker"] = reference_match.group(0)
        
        return {
            "question": question,
            "target_entities": matched_entities,
            "required_fields": detected_fields,
            "intent": intent,
            "constraints": constraints
        }

    def canonical_hash(self, parsed_ir: Dict[str, Any]) -> str:
        import hashlib
        entities = sorted([e.lower() for e in parsed_ir.get("target_entities", [])])
        fields = sorted([f.lower() for f in parsed_ir.get("required_fields", [])])
        intent = parsed_ir.get("intent", "single_fact_lookup").lower()
        question = re.sub(r'\s+', ' ', parsed_ir.get("question", "").lower()).strip()
        constraints = parsed_ir.get("constraints", {})
        constraints_repr = ",".join(
            f"{str(k).lower()}={str(v).lower()}"
            for k, v in sorted(constraints.items())
        )
        
        # Build canonical representation string
        repr_str = (
            f"question:{question}|entities:{','.join(entities)}|fields:{','.join(fields)}|"
            f"intent:{intent}|constraints:{constraints_repr}"
        )
        return hashlib.sha256(repr_str.encode("utf-8")).hexdigest()


