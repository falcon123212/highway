import os
import json
import random
from highway.ingestion.ingest import ingest_corpus

def main():
    print("=== Generating POC 2.3 Stress Test Corpus ===")
    
    corpus_dir = "corpus_poc2_stress"
    doc_dir = os.path.join(corpus_dir, "documents")
    q_dir = os.path.join(corpus_dir, "questions")
    
    os.makedirs(doc_dir, exist_ok=True)
    os.makedirs(q_dir, exist_ok=True)
    os.makedirs(os.path.join(doc_dir, "reports"), exist_ok=True)
    os.makedirs(os.path.join(doc_dir, "contracts"), exist_ok=True)
    os.makedirs(os.path.join(doc_dir, "noise"), exist_ok=True)
    
    # 1. Case 1: Suffix Distractor (FALCON)
    # Write 5 distractor reports
    distractors_content = ""
    for suffix, budget in [("Legacy", "$100,000"), ("Mobile", "$200,000"), ("Core", "$300,000"), ("Backup", "$400,000"), ("Cloud", "$500,000")]:
        distractors_content += f"""
=========================================
REPORT FOR PROJECT FALCON-{suffix}
=========================================
Author: Thomas Petit
Department: HR
The official budget of Project FALCON-{suffix} is set to {budget}.
We are building a robust latency culling mechanism.
"""
    with open(os.path.join(doc_dir, "reports", "falcon_distractors.txt"), "w", encoding="utf-8") as f:
        f.write(distractors_content)
        
    # Write Gold FALCON report
    falcon_gold = """
=========================================
PROJECT PROGRESS REPORT: PROJECT FALCON
=========================================
Author: Julie Thomas
Department: Sales
The official budget for Project FALCON is $600,000.
The final deadline of Project FALCON is set to 26 January 2027.
"""
    with open(os.path.join(doc_dir, "reports", "falcon_gold.txt"), "w", encoding="utf-8") as f:
        f.write(falcon_gold)

    # 2. Case 2: Temporal Supersession Cascade (JUPITER)
    # Document 1: Jan 2026
    doc1 = """
=========================================
CONTRACT SPECIFICATION: PROJECT JUPITER
=========================================
Date: 12 January 2026
Author: Marie Dubois
The approved budget for Project JUPITER is $100,000.
"""
    with open(os.path.join(doc_dir, "contracts", "jupiter_contract_base.txt"), "w", encoding="utf-8") as f:
        f.write(doc1)
        
    # Document 2: Jun 2026
    doc2 = """
=========================================
CONTRACT AMENDMENT: PROJECT JUPITER
=========================================
Date: 15 June 2026
This document supersedes the original contract.
The updated budget for Project JUPITER is increased to $120,000.
"""
    with open(os.path.join(doc_dir, "contracts", "jupiter_amendment_1.txt"), "w", encoding="utf-8") as f:
        f.write(doc2)
        
    # Document 3: Sept 2026
    doc3 = """
=========================================
CONTRACT AMENDMENT: PROJECT JUPITER (REV 2)
=========================================
Date: 18 September 2026
This document supersedes the amendment from June 2026.
The updated budget for Project JUPITER is $110,000.
"""
    with open(os.path.join(doc_dir, "contracts", "jupiter_amendment_2.txt"), "w", encoding="utf-8") as f:
        f.write(doc3)
        
    # Document 4: Dec 2026
    doc4 = """
=========================================
CONTRACT REVISION: PROJECT JUPITER (FINAL SPEC)
=========================================
Date: 22 December 2026
This document supersedes the amendment from September 2026.
The final budget of Project JUPITER is set to $130,000.
"""
    with open(os.path.join(doc_dir, "contracts", "jupiter_revision_final.txt"), "w", encoding="utf-8") as f:
        f.write(doc4)

    # 3. Case 3: Obsolete Project with No resurrection (ZENITH)
    zenith_base = """
=========================================
PROJECT PROGRESS REPORT: PROJECT ZENITH
=========================================
Date: 10 March 2026
Author: Sophie Richard
The official budget of Project ZENITH is $500,000.
"""
    with open(os.path.join(doc_dir, "reports", "zenith_base.txt"), "w", encoding="utf-8") as f:
        f.write(zenith_base)
        
    zenith_retire = """
=========================================
PROJECT STATUS NOTICE: PROJECT ZENITH RETIRED
=========================================
Date: 15 November 2026
This document supersedes all reports on ZENITH.
Project ZENITH is retired and obsolete. All active project operations are cancelled.
"""
    with open(os.path.join(doc_dir, "reports", "zenith_retire.txt"), "w", encoding="utf-8") as f:
        f.write(zenith_retire)

    # 4. Case 4: Budgets for NEPTUNE, KRONOS, ECLIPSE (for multi-project comparison)
    neptune_report = """
=========================================
PROJECT PROGRESS REPORT: PROJECT NEPTUNE
=========================================
Author: Alice Martin
The official budget for Project NEPTUNE is $125,000.
"""
    with open(os.path.join(doc_dir, "reports", "neptune_report.txt"), "w", encoding="utf-8") as f:
        f.write(neptune_report)
        
    kronos_report = """
=========================================
PROJECT PROGRESS REPORT: PROJECT KRONOS
=========================================
Author: Nicolas Durand
The official budget for Project KRONOS is $189,000.
"""
    with open(os.path.join(doc_dir, "reports", "kronos_report.txt"), "w", encoding="utf-8") as f:
        f.write(kronos_report)
        
    eclipse_report = """
=========================================
PROJECT PROGRESS REPORT: PROJECT ECLIPSE
=========================================
Author: Emma Michel
The official budget for Project ECLIPSE is $127,000.
"""
    with open(os.path.join(doc_dir, "reports", "eclipse_report.txt"), "w", encoding="utf-8") as f:
        f.write(eclipse_report)

    # 5. Case 5: Low SNR (Noisy Context) (SPECTRUM)
    # Write 100 blocks of text in one file (using ~10,000 words total)
    noise_content = []
    # Injected signal in block 42 (0-indexed)
    for i in range(100):
        if i == 42:
            noise_content.append(f"""
Block {i+1}: Project SPECTRUM is overseen by Nicolas Durand.
Nicolas Durand is the main project manager responsible for all operations of SPECTRUM.
""")
        else:
            noise_content.append(f"""
Block {i+1}: The Marketing department is actively working to leverage synergy across all spec spec assets.
We aim to establish AI-driven optimization to optimize operational performance of general systems.
""")
    with open(os.path.join(doc_dir, "noise", "spectrum_noise.txt"), "w", encoding="utf-8") as f:
        f.write("\n\n".join(noise_content))

    # 6. Generate QA gold set
    qa_data = [
        {
            "id": "stress_001",
            "question": "What is the budget of Project FALCON?",
            "expected_answer": "$600,000",
            "category": "F",
            "difficulty": "stress_test",
            "reasoning": "Tests exact project match boundary culling on suffix distractor projects."
        },
        {
            "id": "stress_002",
            "question": "What is the active budget of Project JUPITER?",
            "expected_answer": "$130,000",
            "category": "C",
            "difficulty": "stress_test",
            "reasoning": "Tests 4-hop temporal supersession cascade tracking."
        },
        {
            "id": "stress_003",
            "question": "What is the budget of Project ZENITH?",
            "expected_answer": "NOT_FOUND",
            "category": "E",
            "difficulty": "stress_test",
            "reasoning": "Tests complete amnesia / obsolescence culling."
        },
        {
            "id": "stress_004",
            "question": "Which project has the highest budget: NEPTUNE, KRONOS, ECLIPSE, or FALCON?",
            "expected_answer": "Project FALCON (budget of $600,000)",
            "category": "G",
            "difficulty": "stress_test",
            "reasoning": "Tests 4-project budget comparison (complex reasoning)."
        },
        {
            "id": "stress_005",
            "question": "Who manages Project SPECTRUM?",
            "expected_answer": "Nicolas Durand",
            "category": "A",
            "difficulty": "stress_test",
            "reasoning": "Tests extreme low SNR (1 signal block out of 100 blocks, i.e., 1% SNR)."
        }
    ]
    
    with open(os.path.join(q_dir, "qa_gold.json"), "w", encoding="utf-8") as f:
        json.dump(qa_data, f, indent=2)
        
    print(f"Saved stress_test_gold QA dataset with {len(qa_data)} queries.")
    
    # 7. Run ingestion
    index_dir = os.path.join(corpus_dir, "index")
    ingest_corpus(corpus_dir, index_dir)
    print("=== Stress Test Corpus Generated and Ingested successfully ===")

if __name__ == "__main__":
    main()


