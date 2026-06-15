import re
import os
from typing import List, Dict, Any, Tuple

class EvidenceResolver:
    def __init__(self):
        # Revision keywords that indicate a block contains an update
        self.revision_keywords = [
            "amendment", "amend", "supersedes", "superseded", "obsolete", 
            "updated", "revised", "revision", "new budget", "extended to", "retired"
        ]
        
        self.months = [
            "january", "february", "march", "april", "may", "june",
            "july", "august", "september", "october", "november", "december"
        ]

    def extract_date(self, text: str) -> Tuple[int, int, int]:
        # Tries to extract a date from text. Returns (year, month_idx, day)
        # 1. Look for DD Month YYYY
        pattern = r'\b(\d{1,2})\s+([a-zA-Z]+)\s+(\d{4})\b'
        match = re.search(pattern, text)
        if match:
            day = int(match.group(1))
            month_str = match.group(2).lower()
            year = int(match.group(3))
            month_idx = -1
            if month_str in self.months:
                month_idx = self.months.index(month_str)
            return (year, month_idx, day)
            
        # 2. Look for YYYY
        pattern_yr = r'\b(\d{4})\b'
        match_yr = re.findall(pattern_yr, text)
        if match_yr:
            # Return the latest year found in the text
            years = [int(y) for y in match_yr if 1990 <= int(y) <= 2100]
            if years:
                return (max(years), -1, -1)
                
        return (-1, -1, -1)

    def is_later_date(self, date1: Tuple[int, int, int], date2: Tuple[int, int, int]) -> bool:
        # Returns True if date1 is strictly later than date2
        if date1[0] != date2[0]:
            return date1[0] > date2[0]
        if date1[1] != date2[1]:
            return date1[1] > date2[1]
        if date1[2] != date2[2]:
            return date1[2] > date2[2]
        return False

    def resolve(self, candidates: List[Dict[str, Any]], query_ir: Dict[str, Any]) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[str]]:
        target_entities = query_ir["target_entities"]
        reference_marker = query_ir.get("constraints", {}).get("reference_marker")
        if reference_marker:
            marker = reference_marker.lower()
            candidates = [
                block for block in candidates
                if marker in block.get("text", "").lower()
            ]
        
        # Expand target entities to include their base names (without "Project ")
        expanded_target_entities = []
        for ent in target_entities:
            expanded_target_entities.append(ent)
            base_ent = ent.replace("Project ", "").replace("project ", "").strip()
            if base_ent != ent:
                expanded_target_entities.append(base_ent)
        target_entities = list(set(expanded_target_entities))
        
        active_evidence = []
        suppressed_evidence = []
        forbidden_matches = []
        
        # --- STAGE 1: SUFFIX DISTRACTOR RESOLUTION (Category F) ---
        stage1_candidates = []
        
        for block in candidates:
            text_lower = block["text"].lower()
            is_distractor = False
            matched_distractors = []
            
            for entity in target_entities:
                entity_lower = entity.lower()
                # Check for suffix distractor: target_entity followed immediately by alphanumeric / hyphen
                # e.g. "NEPTUNE-Legacy" or "NEPTUNE-Mobile"
                distractor_pattern = r'(?<![a-zA-Z0-9_\-])' + re.escape(entity_lower) + r'([a-zA-Z0-9_\-]+)'
                distractor_matches = re.findall(distractor_pattern, text_lower)
                
                # Check for exact match
                exact_pattern = r'(?<![a-zA-Z0-9_\-])' + re.escape(entity_lower) + r'(?![a-zA-Z0-9_\-])'
                has_exact = bool(re.search(exact_pattern, text_lower))
                
                if distractor_matches:
                    for dm in distractor_matches:
                        matched_distractors.append(entity + dm)
                    # If it has distractor occurrences and NO exact occurrences, it is a distractor block
                    if not has_exact:
                        is_distractor = True
                        
            if is_distractor:
                block_copy = dict(block)
                block_copy["suppression_reason"] = "suffix_distractor"
                block_copy["suppression_detail"] = f"Contains distractor entities: {matched_distractors}"
                suppressed_evidence.append(block_copy)
                forbidden_matches.extend(matched_distractors)
            else:
                stage1_candidates.append(block)
                
        forbidden_matches = list(set(forbidden_matches))
        
        # --- STAGE 2: TEMPORAL SUPERSESSION RESOLUTION (Category C) ---
        # Group stage1 candidates by entity if there are multiple candidates
        # For simplicity, if query intent is status_resolution or we have temporal indicators,
        # we check the files for supersession.
        
        if query_ir["intent"] == "aggregation":
            stage2_candidates = list(stage1_candidates)
        else:
            # Let's extract metadata for each block
            for block in stage1_candidates:
                block["parsed_date"] = self.extract_date(block["text"])
                filename = os.path.basename(block["source_file"].replace("\\", "/")).lower()
                block["is_amendment"] = "amendment" in filename or "update" in filename
                block["is_base"] = "base" in filename or ("contract" in filename and "amendment" not in filename)
                
            stage2_candidates = list(stage1_candidates)
            
            # We perform temporal pruning if we have blocks from the same project/entity
            # that contradict each other.
            for entity in target_entities:
                entity_lower = entity.lower()
                entity_blocks = []
                
                for b in stage2_candidates:
                    # If block mentions this entity
                    exact_pattern = r'(?<![a-zA-Z0-9_\-])' + re.escape(entity_lower) + r'(?![a-zA-Z0-9_\-])'
                    if re.search(exact_pattern, b["text"].lower()):
                        entity_blocks.append(b)
                        
                if len(entity_blocks) > 1:
                    # We have multiple blocks for the same entity! Resolve conflicts.
                    amendments = [b for b in entity_blocks if b["is_amendment"]]
                    bases = [b for b in entity_blocks if b["is_base"]]
                    
                    # Check dates
                    dates = [b["parsed_date"] for b in entity_blocks if b["parsed_date"] != (-1, -1, -1)]
                    amendment_dates = [b["parsed_date"] for b in entity_blocks if b["is_amendment"] and b["parsed_date"] != (-1, -1, -1)]
                    
                    latest_date = (-1, -1, -1)
                    if dates:
                        # Find the latest date
                        for d in dates:
                            if self.is_later_date(d, latest_date):
                                latest_date = d
                                
                    latest_amendment_date = (-1, -1, -1)
                    if amendment_dates:
                        # Find the latest amendment date
                        for d in amendment_dates:
                            if self.is_later_date(d, latest_amendment_date):
                                latest_amendment_date = d
                                
                    for b in entity_blocks:
                        is_superseded = False
                        reason = ""
                        
                        # Heuristic 1: Base contract is superseded by amendment
                        if b["is_base"] and amendments:
                            is_superseded = True
                            reason = "obsolete (superseded by amendment document)"
                            
                        # Heuristic 2: Older date is superseded by later date
                        elif b["parsed_date"] != (-1, -1, -1):
                            if b["is_amendment"] and latest_amendment_date != (-1, -1, -1):
                                if self.is_later_date(latest_amendment_date, b["parsed_date"]):
                                    is_superseded = True
                                    reason = f"obsolete (superseded by later amendment date: {latest_amendment_date})"
                            elif not b["is_amendment"] and latest_date != (-1, -1, -1):
                                if self.is_later_date(latest_date, b["parsed_date"]):
                                    is_superseded = True
                                    reason = f"obsolete (superseded by later date: {latest_date})"
                                
                        if is_superseded:
                            block_copy = dict(b)
                            block_copy["suppression_reason"] = "obsolete"
                            block_copy["suppression_detail"] = reason
                            suppressed_evidence.append(block_copy)
                            # Remove from stage2_candidates so it cannot be resurrected by other entities
                            stage2_candidates = [x for x in stage2_candidates if x["block_id"] != b["block_id"]]
                
        # Remove duplicates from stage2_candidates while maintaining order
        seen_ids = set()
        unique_stage2 = []
        for b in stage2_candidates:
            if b["block_id"] not in seen_ids:
                unique_stage2.append(b)
                seen_ids.add(b["block_id"])
                
        # Stage 3: Entity Presence Filter
        # Only keep blocks that contain at least one target entity (if target_entities is non-empty)
        if target_entities and reference_marker and query_ir.get("intent") == "aggregation":
            active_evidence = unique_stage2
        elif target_entities:
            filtered_active = []
            for b in unique_stage2:
                text_lower = b["text"].lower()
                has_entity = False
                for ent in target_entities:
                    ent_lower = ent.lower()
                    pattern = r'(?<![a-zA-Z0-9_\-])' + re.escape(ent_lower) + r'(?![a-zA-Z0-9_\-])'
                    if re.search(pattern, text_lower):
                        has_entity = True
                        break
                if has_entity:
                    filtered_active.append(b)
                else:
                    block_copy = dict(b)
                    block_copy["suppression_reason"] = "unrelated_entity"
                    block_copy["suppression_detail"] = f"Does not contain target entity: {target_entities}"
                    suppressed_evidence.append(block_copy)
            active_evidence = filtered_active
        else:
            active_evidence = unique_stage2
            
        # Stage 4: Comparison-specific Filter
        if query_ir.get("intent") == "comparison":
            q_lower = query_ir.get("question", "").lower()
            filtered_comp = []
            for b in active_evidence:
                text_lower = b["text"].lower()
                if "budget" in q_lower:
                    if "budget" in text_lower or "$" in text_lower:
                        filtered_comp.append(b)
                elif "deadline" in q_lower or "date" in q_lower:
                    if "deadline" in text_lower or "date" in text_lower or "completion" in text_lower:
                        filtered_comp.append(b)
                else:
                    filtered_comp.append(b)
            if filtered_comp:
                active_evidence = filtered_comp
            
        return active_evidence, suppressed_evidence, forbidden_matches


