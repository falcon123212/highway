from typing import Dict, Any, List

class ContextCompiler:
    def __init__(self):
        pass

    def _format_user_turn(self, question: str, target_entity: str, intent: str, required_fields: str, expected_format: str, evidence_lines: List[str], forbidden_matches: List[str] = None) -> List[str]:
        lines = []
        lines.append(f"Question: {question}")
        if target_entity:
            lines.append(f"TARGET ENTITY: {target_entity}")
        if intent:
            lines.append(f"QUERY INTENT: {intent}")
        if required_fields:
            lines.append(f"REQUIRED FIELDS: {required_fields}")
        if forbidden_matches:
            lines.append(f"FORBIDDEN ENTITIES (DO NOT MATCH): {', '.join(forbidden_matches)}")
        if expected_format:
            lines.append(f"EXPECTED ANSWER FORMAT: {expected_format}")
            
        lines.append("\n=== ACTIVE EVIDENCE ===")
        lines.extend(evidence_lines)
        return lines

    def compile(self, ir: Dict[str, Any], max_tokens: int = 1200) -> str:
        query = ir["query"]
        evidence = ir["evidence"]
        suppressed_evidence = ir["suppressed_evidence"]
        forbidden_matches = ir["forbidden_matches"]
        output_schema = ir["output_schema"]
        
        prompt = []
        
        # Section 1: Context Kernel (~100 tokens, Stable Prefix)
        prompt.append("<|im_start|>system")
        prompt.append("You are a precise extraction engine. Answer the question using ONLY the provided active evidence.")
        prompt.append("Respond with ONLY the exact requested value(s) and nothing else. Do NOT write full sentences, do NOT repeat the question, and do NOT add conversational filler.")
        prompt.append("If a value is mentioned in both ACTIVE and OBSOLETE evidence, or if the active evidence explicitly says a value is superseded/old/obsolete, output ONLY the new updated/active value.")
        prompt.append("\n=== INSTRUCTIONS ===")
        prompt.append("- Base your answer ONLY on the ACTIVE EVIDENCE provided in the user turn.")
        prompt.append("- Match entity names EXACTLY. Do not use forbidden suffix variants.")
        prompt.append("- If the query asks for multiple fields (e.g., both deadline and owner of a project), you MUST join the values with the word ' and ' (e.g., Value1 and Value2). Do NOT use commas or semicolons to separate them. Do NOT use this rule for comparison or aggregation queries.")
        prompt.append("- If there is no active evidence, or if the active evidence does not contain the answer, output: NOT_FOUND")
        prompt.append("- For management queries, 'Contract Manager', 'Project Manager', 'Owner', 'Director', 'Author', or leadership/management roles indicate the person manages that project. An author of a project report (e.g. 'Author: Name' at the top of a 'PROJECT NAME' report) is the manager of that project. You MUST include all such projects in your answer.")
        prompt.append("- For comparison queries (e.g., comparing budgets of two projects), you MUST compare the values digit-by-digit in your reasoning. Note that single digits follow the strict order: 9 > 8 > 7 > 6 > 5 > 4 > 3 > 2 > 1 > 0 (for example, 9 is greater than 7). For example, comparing $987,000 and $967,000: both are 6 digits; in the ten-thousands place, 8 is greater than 6, so $987,000 is higher. Output ONLY the single project name (with budget, e.g., 'Project Name (budget of $Value)') that satisfies the query condition.")
        prompt.append("- For aggregation/listing queries (e.g. 'List all project names...'), you MUST list ALL matching projects/entities found in the ACTIVE EVIDENCE, not just two. Do not be limited by the example format 'PROJECT_1, PROJECT_2'; if there are three, four, or more matching projects, you MUST list all of them, separated by commas.")
        prompt.append("- You MUST respond with a strict JSON object containing a 'reasoning' key for step-by-step thinking/calculations, and an 'answer' key for the final extracted value:")
        prompt.append("{\n  \"reasoning\": \"<step-by-step thinking>\",\n  \"answer\": \"<value>\"\n}")
        prompt.append("<|im_end|>")
        
        # Section 1.5: Few-shot Examples (as separate turns, Stable Prefix)
        # Few-shot 1: deadline and owner (multi_fact_extraction)
        ex1_evidence = ["ACTIVE EVIDENCE 1 [SOURCE: project_x_report.txt]: Project X is managed by John Smith. The approved budget for Project X is $250,000, and the delivery deadline is set to 15 May 2027."]
        ex1_turn = self._format_user_turn(
            question="What is the deadline and manager of Project X?",
            target_entity="Project X",
            intent="multi_fact_extraction",
            required_fields="deadline, owner",
            expected_format="DATE and OWNER",
            evidence_lines=ex1_evidence
        )
        prompt.append("<|im_start|>user\n" + "\n".join(ex1_turn) + "\n<|im_end|>")
        prompt.append("<|im_start|>assistant\n{\n  \"reasoning\": \"Project X manager is John Smith; deadline is 15 May 2027.\",\n  \"answer\": \"15 May 2027 and John Smith\"\n}\n<|im_end|>")
        
        # Few-shot 4: budget comparison (comparison - second is higher)
        ex4_evidence = [
            "ACTIVE EVIDENCE 1 [SOURCE: project_x_report.txt]: Project X is managed by John Smith. The approved budget for Project X is $250,000.",
            "ACTIVE EVIDENCE 2 [SOURCE: project_y_report.txt]: Project Y is managed by Sarah Jenkins. The approved budget for Project Y is $450,000."
        ]
        ex4_turn = self._format_user_turn(
            question="Which project has a higher budget: Project X or Project Y?",
            target_entity="Project X, Project Y",
            intent="comparison",
            required_fields=None,
            expected_format="Project NAME (budget of BUDGET)",
            evidence_lines=ex4_evidence
        )
        prompt.append("<|im_start|>user\n" + "\n".join(ex4_turn) + "\n<|im_end|>")
        prompt.append("<|im_start|>assistant\n{\n  \"reasoning\": \"Project Y budget ($450,000) vs Project X budget ($250,000). Both are 6 digits. In the hundred-thousands place, 4 is greater than 2, so $450,000 is higher.\",\n  \"answer\": \"Project Y (budget of $450,000)\"\n}\n<|im_end|>")
        
        # Few-shot 4b: budget comparison (comparison - first is higher)
        ex4b_evidence = [
            "ACTIVE EVIDENCE 1 [SOURCE: project_a_report.txt]: The budget for Project A is $800,000.",
            "ACTIVE EVIDENCE 2 [SOURCE: project_b_report.txt]: The budget for Project B is $300,000."
        ]
        ex4b_turn = self._format_user_turn(
            question="Which project has a higher budget: Project A or Project B?",
            target_entity="Project A, Project B",
            intent="comparison",
            required_fields=None,
            expected_format="Project NAME (budget of BUDGET)",
            evidence_lines=ex4b_evidence
        )
        prompt.append("<|im_start|>user\n" + "\n".join(ex4b_turn) + "\n<|im_end|>")
        prompt.append("<|im_start|>assistant\n{\n  \"reasoning\": \"Project A budget ($800,000) vs Project B budget ($300,000). Both are 6 digits. In the hundred-thousands place, 8 is greater than 3, so $800,000 is higher.\",\n  \"answer\": \"Project A (budget of $800,000)\"\n}\n<|im_end|>")
        
        # Few-shot 5: list projects (aggregation)
        ex5_evidence = [
            "ACTIVE EVIDENCE 1 [SOURCE: specs/project_x.txt]: Project X is under the leadership of John Smith.",
            "ACTIVE EVIDENCE 2 [SOURCE: reports/project_z_status_report.txt]:\n=========================================\nPROJECT PROGRESS REPORT: PROJECT Z\n=========================================\nAuthor: John Smith\nDepartment: Operations\nProject Z status is normal.",
            "ACTIVE EVIDENCE 3 [SOURCE: contracts/project_w.txt]: Contract Manager: John Smith. Project W budget is $120,000."
        ]
        ex5_turn = self._format_user_turn(
            question="List all project names managed by John Smith.",
            target_entity="John Smith",
            intent="aggregation",
            required_fields=None,
            expected_format="PROJECT_1, PROJECT_2, ...",
            evidence_lines=ex5_evidence
        )
        prompt.append("<|im_start|>user\n" + "\n".join(ex5_turn) + "\n<|im_end|>")
        prompt.append("<|im_start|>assistant\n{\n  \"reasoning\": \"John Smith is the manager of Project X, Project Z, and Project W.\",\n  \"answer\": \"Project X, Project Z, Project W\"\n}\n<|im_end|>")
        
        # Section 2: Real Query Turn (Dynamic Suffix) with Pruning Loop to fit max_tokens
        target_entities = query.get("target_entities", [])
        required_fields = query.get("required_fields", [])
        
        prompt_prefix = list(prompt)
        
        while True:
            evidence_lines = []
            if query["intent"] == "status_resolution":
                evidence_lines.append("ACTIVE EVIDENCE (VALID):")
                for ev in evidence:
                    evidence_lines.append(f"  - [SOURCE: {ev['source_file']}]: {ev['text']}")
                    
                obs_ev = [ev for ev in suppressed_evidence if ev.get("suppression_reason") == "obsolete"]
                if obs_ev:
                    evidence_lines.append("\nOBSOLETE/SUPERSEDED EVIDENCE (DO NOT USE):")
                    for ev in obs_ev:
                        detail = ev.get("suppression_detail", "superseded by newer document")
                        evidence_lines.append(f"  - [SOURCE: {ev['source_file']}] [STATUS: obsolete - {detail}]: {ev['text']}")
            else:
                for idx, ev in enumerate(evidence):
                    evidence_lines.append(f"ACTIVE EVIDENCE {idx + 1} [SOURCE: {ev['source_file']}]: {ev['text']}")
                    
            real_turn = self._format_user_turn(
                question=query['question'],
                target_entity=', '.join(target_entities) if target_entities else None,
                intent=query.get('intent', 'single_fact_lookup'),
                required_fields=', '.join(required_fields) if required_fields else None,
                expected_format=output_schema.get("example"),
                evidence_lines=evidence_lines,
                forbidden_matches=forbidden_matches
            )
            
            candidate_prompt = list(prompt_prefix)
            candidate_prompt.append("<|im_start|>user\n" + "\n".join(real_turn) + "\n<|im_end|>")
            candidate_prompt.append("<|im_start|>assistant\n{\n  \"reasoning\": \"")
            
            prompt_text = "\n".join(candidate_prompt)
            
            approx_tokens = int(len(prompt_text) / 4.15) + 15
            if len(evidence) <= 1 or approx_tokens <= max_tokens - 96:
                return prompt_text
            else:
                # Remove the last evidence block (least relevant based on ranking)
                evidence = evidence[:-1]


