import os
import sys
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

device = "cuda" if torch.cuda.is_available() else "cpu"
model_id = "Qwen/Qwen2.5-1.5B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.float16).to(device)

def run_query(prompt):
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=96,
            temperature=0.0,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id
        )
    generated_ids = outputs[0][inputs.input_ids.shape[1]:]
    response = tokenizer.decode(generated_ids, skip_special_tokens=True)
    return response

# System prompt with management instruction
sys_mgmt = """You are a precise extraction engine. Answer the question using ONLY the provided active evidence.
Respond with ONLY the exact requested value(s) and nothing else. Do NOT write full sentences, do NOT repeat the question, and do NOT add conversational filler.
If a value is mentioned in both ACTIVE and OBSOLETE evidence, or if the active evidence explicitly says a value is superseded/old/obsolete, output ONLY the new updated/active value.

=== INSTRUCTIONS ===
- Base your answer ONLY on the ACTIVE EVIDENCE provided in the user turn.
- Match entity names EXACTLY. Do not use forbidden suffix variants.
- If the query asks for multiple fields, you MUST join the values with the word ' and ' (e.g., Value1 and Value2). Do NOT use commas or semicolons to separate them.
- If there is no active evidence, or if the active evidence does not contain the answer, output: NOT_FOUND
- For management queries, 'Contract Manager', 'Project Manager', 'Owner', 'Director', 'Author', or leadership/management roles indicate the person manages that project.
- You MUST respond with a strict JSON object containing a 'reasoning' key for step-by-step thinking/calculations, and an 'answer' key for the final extracted value:
{
  "reasoning": "<step-by-step thinking>",
  "answer": "<value>"
}"""

# Option 5A: New Few-shot 5 with "Project X, Project Z" in answer
few_shots_5A = """<|im_start|>user
Question: Who manages Project X?
TARGET ENTITY: Project X
QUERY INTENT: single_fact_lookup
REQUIRED FIELDS: owner
EXPECTED ANSWER FORMAT: OWNER

=== ACTIVE EVIDENCE ===
ACTIVE EVIDENCE 1 [SOURCE: project_x_report.txt]: Project X is managed by John Smith in the Engineering department. The approved budget for Project X is $250,000, and the delivery deadline is set to 15 May 2027.
<|im_end|>
<|im_start|>assistant
{
  "reasoning": "Project X manager is John Smith.",
  "answer": "John Smith"
}
<|im_end|>
<|im_start|>user
Question: What is the budget of Project X?
TARGET ENTITY: Project X
QUERY INTENT: single_fact_lookup
REQUIRED FIELDS: budget
EXPECTED ANSWER FORMAT: BUDGET

=== ACTIVE EVIDENCE ===
ACTIVE EVIDENCE 1 [SOURCE: project_x_report.txt]: Project X is managed by John Smith in the Engineering department. The approved budget for Project X is $250,000, and the delivery deadline is set to 15 May 2027.
<|im_end|>
<|im_start|>assistant
{
  "reasoning": "Project X budget is $250,000.",
  "answer": "$250,000"
}
<|im_end|>
<|im_start|>user
Question: What is the deadline and manager of Project X?
TARGET ENTITY: Project X
QUERY INTENT: multi_fact_extraction
REQUIRED FIELDS: deadline, owner
EXPECTED ANSWER FORMAT: DATE and OWNER

=== ACTIVE EVIDENCE ===
ACTIVE EVIDENCE 1 [SOURCE: project_x_report.txt]: Project X is managed by John Smith in the Engineering department. The approved budget for Project X is $250,000, and the delivery deadline is set to 15 May 2027.
<|im_end|>
<|im_start|>assistant
{
  "reasoning": "Project X manager is John Smith; deadline is 15 May 2027.",
  "answer": "15 May 2027 and John Smith"
}
<|im_end|>
<|im_start|>user
Question: Which project has a higher budget: Project X or Project Y?
TARGET ENTITY: Project X, Project Y
QUERY INTENT: comparison
EXPECTED ANSWER FORMAT: Project NAME (budget of BUDGET)

=== ACTIVE EVIDENCE ===
ACTIVE EVIDENCE 1 [SOURCE: project_x_report.txt]: Project X is managed by John Smith. The approved budget for Project X is $250,000.
ACTIVE EVIDENCE 2 [SOURCE: project_y_report.txt]: Project Y is managed by Sarah Jenkins. The approved budget for Project Y is $450,000.
<|im_end|>
<|im_start|>assistant
{
  "reasoning": "Project Y budget ($450,000) is higher than Project X budget ($250,000).",
  "answer": "Project Y (budget of $450,000)"
}
<|im_end|>
<|im_start|>user
Question: List all project names managed by John Smith.
TARGET ENTITY: John Smith
QUERY INTENT: aggregation
EXPECTED ANSWER FORMAT: PROJECT_1, PROJECT_2

=== ACTIVE EVIDENCE ===
ACTIVE EVIDENCE 1 [SOURCE: specs/project_x.txt]: Project X is under the leadership of John Smith.
ACTIVE EVIDENCE 2 [SOURCE: reports/project_z.txt]: Author: John Smith. Project Z status is normal.
<|im_end|>
<|im_start|>assistant
{
  "reasoning": "John Smith is the manager of Project X and Project Z.",
  "answer": "Project X, Project Z"
}
<|im_end|>"""

# Test query (q_137) input
q_137_input = """<|im_start|>user
Question: Which project has a higher budget: Project JUPITER or Project ORION?
TARGET ENTITY: JUPITER, ORION
QUERY INTENT: comparison
REQUIRED FIELDS: budget
EXPECTED ANSWER FORMAT: Project NAME (budget of BUDGET)

=== ACTIVE EVIDENCE ===
ACTIVE EVIDENCE 1 [SOURCE: reports/jupiter_status_report.txt]: 2. Financial Overview
The financial allocation is fully approved. The official budget for Project JUPITER is $967,000.
No supplementary cost overrun is expected at this stage.
ACTIVE EVIDENCE 2 [SOURCE: reports/orion_status_report.txt]: 2. Financial Overview
The financial allocation is fully approved. The official budget for Project ORION is $987,000.
No supplementary cost overrun is expected at this stage.
<|im_end|>
<|im_start|>assistant
{
  "reasoning": """

# Test query (q_155) input
q_155_input = """<|im_start|>user
Question: List all project names managed by Emma Michel.
TARGET ENTITY: Emma Michel
QUERY INTENT: aggregation
EXPECTED ANSWER FORMAT: PROJECT_1, PROJECT_2

=== ACTIVE EVIDENCE ===
ACTIVE EVIDENCE 1 [SOURCE: specs/aurora_specifications.txt]: =========================================
TECHNICAL SPECIFICATIONS: PROJECT AURORA
=========================================
Project Owner: Emma Michel
Lead Unit: Finance
This document describes the technical architecture for project AURORA.
The project is overseen by the Finance unit, with Emma Michel as director.
ACTIVE EVIDENCE 2 [SOURCE: contracts/vega_base_contract.txt]: =========================================
SERVICE CONTRACT: PROJECT VEGA
=========================================
Date: 21 August 2026
Approved Budget: $75,000
Completion Date: 21 August 2026
Contract Manager: Emma Michel
This document details the initial terms. The project VEGA is launched with an approved budget of $75,000 under the management of Emma Michel.
The scheduled completion date is set to 21 August 2026.
ACTIVE EVIDENCE 3 [SOURCE: reports/eclipse_status_report.txt]: =========================================
PROJECT PROGRESS REPORT: PROJECT ECLIPSE
=========================================
Author: Emma Michel
Department: Operations
Location: Toulouse
1. Executive Summary
The Legal department is actively working to leverage leverage across all core assets. By implementing this new framework, we aim to accelerate AI-driven optimization and establish a robust AI-driven optimization pipeline to optimize operational performance. The Engineering department is actively working to leverage zero-trust architecture across all core assets. By implementing this new framework, we aim to accelerate synergy and establish a robust agentic workflows pipeline to optimize operational performance.
The operations are currently based in Toulouse under the leadership of Emma
ACTIVE EVIDENCE 4 [SOURCE: reports/nexus_status_report.txt]: =========================================
PROJECT PROGRESS REPORT: PROJECT NEXUS
=========================================
Author: Emma Michel
Department: Finance
Location: Lyon
1. Executive Summary
The Operations department is actively working to leverage leverage across all core assets. By implementing this new framework, we aim to accelerate paradigm shift and establish a robust data-sovereign pipeline pipeline to optimize operational performance. The Operations department is actively working to leverage zero-trust architecture across all core assets. By implementing this new framework, we aim to accelerate leverage and establish a robust AI-driven optimization pipeline to optimize operational performance.
The operations are currently based in Lyon under the leadership of Emma Michel.

ACTIVE EVIDENCE 5 [SOURCE: correspondence/aurora_financial_email.txt]: From: Emma Michel
To: Finance Committee
Date: 14 January 2026
Subject: Project AURORA Financial Request
Dear colleagues,
As project manager for Project AURORA, I am happy to report that our financial allocation request of $538,000 was approved.
The milestone deadline is set to 12 April 2027.
Best regards,
Emma Michel
<|im_end|>
<|im_start|>assistant
{
  "reasoning": """

print("\n--- TEST 5: Q_137 with 5A (Project X, Project Z) ---")
prompt_5A_137 = f"<|im_start|>system\n{sys_mgmt}\n<|im_end|>\n{few_shots_5A}\n{q_137_input}"
print(run_query(prompt_5A_137))

print("\n--- TEST 5: Q_155 with 5A (Project X, Project Z) ---")
prompt_5A_155 = f"<|im_start|>system\n{sys_mgmt}\n<|im_end|>\n{few_shots_5A}\n{q_155_input}"
print(run_query(prompt_5A_155))



