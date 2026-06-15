import os
import sys
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# Set environment to use GPU
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

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

# Let's test q_137 (Comparison) and q_155 (Aggregation) with our current compiler
sys.path.append("src")
from highway.runtime.compiler import ContextCompiler
import json

compiler = ContextCompiler()

# Test q_137
q_137_ir = {
  "query": {
    "id": "q_137",
    "question": "Which project has a higher budget: Project JUPITER or Project ORION?",
    "target_entities": ["JUPITER", "ORION"],
    "required_fields": ["budget"],
    "intent": "comparison"
  },
  "evidence": [
    {
      "source_file": "reports/jupiter_status_report.txt",
      "text": "2. Financial Overview\nThe financial allocation is fully approved. The official budget for Project JUPITER is $967,000.\nNo supplementary cost overrun is expected at this stage."
    },
    {
      "source_file": "reports/orion_status_report.txt",
      "text": "2. Financial Overview\nThe financial allocation is fully approved. The official budget for Project ORION is $987,000.\nNo supplementary cost overrun is expected at this stage."
    }
  ],
  "suppressed_evidence": [],
  "forbidden_matches": [],
  "output_schema": {
    "example": "Project NAME (budget of BUDGET)"
  }
}

prompt_137 = compiler.compile(q_137_ir)
print("\n=== RUNNING Q_137 ===")
print("Prompt suffix:\n", "\n".join(prompt_137.split("\n")[-15:]))
res_137 = run_query(prompt_137)
print("Response:\n", res_137)

# Test q_155
q_155_ir = {
  "query": {
    "id": "q_155",
    "question": "List all project names managed by Emma Michel.",
    "target_entities": ["Emma Michel"],
    "required_fields": [],
    "intent": "aggregation"
  },
  "evidence": [
    {
      "source_file": "specs/aurora_specifications.txt",
      "text": "=========================================\nTECHNICAL SPECIFICATIONS: PROJECT AURORA\n=========================================\nProject Owner: Emma Michel\nLead Unit: Finance\nThis document describes the technical architecture for project AURORA.\nThe project is overseen by the Finance unit, with Emma Michel as director."
    },
    {
      "source_file": "contracts/vega_base_contract.txt",
      "text": "=========================================\nSERVICE CONTRACT: PROJECT VEGA\n=========================================\nDate: 21 August 2026\nApproved Budget: $75,000\nCompletion Date: 21 August 2026\nContract Manager: Emma Michel\nThis document details the initial terms. The project VEGA is launched with an approved budget of $75,000 under the management of Emma Michel.\nThe scheduled completion date is set to 21 August 2026."
    },
    {
      "source_file": "reports/eclipse_status_report.txt",
      "text": "=========================================\nPROJECT PROGRESS REPORT: PROJECT ECLIPSE\n=========================================\nAuthor: Emma Michel\nDepartment: Operations\nLocation: Toulouse\n1. Executive Summary\nThe Legal department is actively working to leverage leverage across all core assets. By implementing this new framework, we aim to accelerate AI-driven optimization and establish a robust AI-driven optimization pipeline to optimize operational performance. The Engineering department is actively working to leverage zero-trust architecture across all core assets. By implementing this new framework, we aim to accelerate synergy and establish a robust agentic workflows pipeline to optimize operational performance.\nThe operations are currently based in Toulouse under the leadership of Emma"
    },
    {
      "source_file": "reports/nexus_status_report.txt",
      "text": "=========================================\nPROJECT PROGRESS REPORT: PROJECT NEXUS\n=========================================\nAuthor: Emma Michel\nDepartment: Finance\nLocation: Lyon\n1. Executive Summary\nThe Operations department is actively working to leverage leverage across all core assets. By implementing this new framework, we aim to accelerate paradigm shift and establish a robust data-sovereign pipeline pipeline to optimize operational performance. The Operations department is actively working to leverage zero-trust architecture across all core assets. By implementing this new framework, we aim to accelerate leverage and establish a robust AI-driven optimization pipeline to optimize operational performance.\nThe operations are currently based in Lyon under the leadership of Emma Michel."
    }
  ],
  "suppressed_evidence": [],
  "forbidden_matches": [],
  "output_schema": {
    "example": "PROJECT_1, PROJECT_2"
  }
}

prompt_155 = compiler.compile(q_155_ir)
print("\n=== RUNNING Q_155 ===")
print("Prompt suffix:\n", "\n".join(prompt_155.split("\n")[-15:]))
res_155 = run_query(prompt_155)
print("Response:\n", res_155)



