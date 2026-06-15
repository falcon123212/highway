import os
import json
import random
import string
from typing import List, Dict, Any, Tuple

# Set seed for reproducibility
random.seed(42)

# Vocabulary and Names for synthetic data generation
PROJECT_NAMES = [
    "NEPTUNE", "KRONOS", "ECLIPSE", "FALCON", "IRIS", "JUPITER", "METEOR", "NEXUS",
    "ORION", "PHOENIX", "QUASAR", "SIRIUS", "TITAN", "VEGA", "ZENITH", "AURORA",
    "BEACON", "CHRONOS", "DAWN", "GENESIS", "HELIOS", "LUNA", "PEGASE", "SOLARIS"
]

PEOPLE = [
    "Jean Dupont", "Alice Martin", "Pierre Leroy", "Marie Dubois", "Thomas Petit",
    "Sophie Richard", "Michel Bernard", "Julie Thomas", "Nicolas Durand", "Emma Michel"
]

DEPARTMENTS = ["Engineering", "Finance", "HR", "Legal", "Marketing", "Operations", "Sales"]

LOCATIONS = ["Paris", "Lyon", "Marseille", "Toulouse", "Nice", "Nantes", "Strasbourg", "Bordeaux"]

BUZZWORDS = [
    "synergy", "paradigm shift", "scalability", "leverage", "robustness", "cloud-native",
    "blockchain integration", "AI-driven optimization", "operational efficiency", "framework",
    "zero-trust architecture", "standardization", "data-sovereign pipeline", "latency culling",
    "KV cache optimization", "throughput acceleration", "redundancy elimination", "agentic workflows"
]

MONTHS = [
    "January", "February", "March", "April", "May", "June",
    "July", "August", "September", "October", "November", "December"
]

def generate_noise_text(word_count: int = 100) -> str:
    """Generates random business-sounding noise paragraphs."""
    sentences = []
    current_words = 0
    while current_words < word_count:
        dept = random.choice(DEPARTMENTS)
        b1 = random.choice(BUZZWORDS)
        b2 = random.choice(BUZZWORDS)
        b3 = random.choice(BUZZWORDS)
        sentence = f"The {dept} department is actively working to leverage {b1} across all core assets. By implementing this new framework, we aim to accelerate {b2} and establish a robust {b3} pipeline to optimize operational performance."
        sentences.append(sentence)
        current_words += len(sentence.split())
    return " ".join(sentences)

def generate_date(year: int) -> str:
    day = random.randint(1, 28)
    month = random.choice(MONTHS)
    return f"{day:02d} {month} {year}"

class SyntheticCorpusGenerator:
    def __init__(self, output_dir: str):
        self.output_dir = output_dir
        self.doc_dir = os.path.join(output_dir, "documents")
        self.questions_dir = os.path.join(output_dir, "questions")
        
        # Subdirectories for document types
        self.categories = ["reports", "contracts", "specs", "correspondence", "noise"]
        
        self.documents_written = []
        self.qa_pairs = []
        self.global_question_id = 0
        
    def setup_dirs(self):
        os.makedirs(self.doc_dir, exist_ok=True)
        os.makedirs(self.questions_dir, exist_ok=True)
        for cat in self.categories:
            os.makedirs(os.path.join(self.doc_dir, cat), exist_ok=True)

    def write_document(self, category: str, filename: str, content: str) -> str:
        relative_path = f"{category}/{filename}"
        full_path = os.path.join(self.doc_dir, relative_path)
        with open(full_path, "w", encoding="utf-8") as f:
            f.write(content)
        self.documents_written.append(relative_path)
        return relative_path

    def generate_all(self, total_noise_docs: int = 100):
        self.setup_dirs()
        
        # 1. Injected Project Facts Database
        # We will build structured facts for each project to ensure consistency
        project_db = {}
        for proj in PROJECT_NAMES:
            project_db[proj] = {
                "owner": random.choice(PEOPLE),
                "budget": f"${random.randint(100, 999)},000",
                "location": random.choice(LOCATIONS),
                "deadline": generate_date(2027),
                "department": random.choice(DEPARTMENTS),
                "original_budget": f"${random.randint(50, 99)},000",
                "original_deadline": generate_date(2026)
            }
            
        print(f"Generated a project facts database for {len(PROJECT_NAMES)} projects.")
        
        # Keep track of active projects and files to build QA
        injected_files = {} # project -> list of files it was injected in
        
        # --- PHASE 1: GENERATE SIGNAL DOCUMENTS ---
        
        # Category A & B: Reports & Specs containing simple facts
        for proj in PROJECT_NAMES[:10]: # First 10 projects get simple lookups / dates
            db = project_db[proj]
            
            # Report Document
            report_text = f"""=========================================
PROJECT PROGRESS REPORT: PROJECT {proj}
=========================================
Author: {db['owner']}
Department: {db['department']}
Location: {db['location']}

1. Executive Summary
{generate_noise_text(50)}
The operations are currently based in {db['location']} under the leadership of {db['owner']}.

2. Financial Overview
The financial allocation is fully approved. The official budget for Project {proj} is {db['budget']}.
No supplementary cost overrun is expected at this stage.

3. Schedule and Timeline
We are targeting the key delivery milestones. The final deadline of Project {proj} is set to {db['deadline']}.
All task teams are aligned to meet this date.
"""
            rel_path = self.write_document("reports", f"{proj.lower()}_status_report.txt", report_text)
            injected_files[proj] = [rel_path]
            
            # Generate QA Category A (Simple Lookup)
            self.add_qa(
                question=f"Who is the manager or owner of Project {proj}?",
                expected_answer=db['owner'],
                category="A",
                difficulty="easy",
                source_file=rel_path,
                evidence_quote=f"Author: {db['owner']}",
                reasoning=f"Simple extraction of the author/owner field for project {proj}."
            )
            
            self.add_qa(
                question=f"Which department is responsible for Project {proj}?",
                expected_answer=db['department'],
                category="A",
                difficulty="easy",
                source_file=rel_path,
                evidence_quote=f"Department: {db['department']}",
                reasoning=f"Simple lookup of the department."
            )
            
            # Generate QA Category B (Date/Value)
            self.add_qa(
                question=f"What is the final budget allocated to Project {proj}?",
                expected_answer=db['budget'],
                category="B",
                difficulty="easy",
                source_file=rel_path,
                evidence_quote=f"The official budget for Project {proj} is {db['budget']}.",
                reasoning=f"Numeric budget value extraction."
            )
            
            self.add_qa(
                question=f"What is the final delivery deadline of Project {proj}?",
                expected_answer=db['deadline'],
                category="B",
                difficulty="easy",
                source_file=rel_path,
                evidence_quote=f"The final deadline of Project {proj} is set to {db['deadline']}.",
                reasoning=f"Date deadline extraction."
            )

        # Category C: Status/Evolution (Contradictions & Temporal updates)
        # We write a contract and then an amendment (or report vs email)
        for proj in PROJECT_NAMES[10:15]: # Next 5 projects get temporal/evolution
            db = project_db[proj]
            
            # Base document (obsolete fact)
            base_contract = f"""=========================================
SERVICE CONTRACT: PROJECT {proj}
=========================================
Date: {db['original_deadline']}
Approved Budget: {db['original_budget']}
Completion Date: {db['original_deadline']}
Contract Manager: {db['owner']}

This document details the initial terms. The project {proj} is launched with an approved budget of {db['original_budget']} under the management of {db['owner']}.
The scheduled completion date is set to {db['original_deadline']}.
"""
            rel_base = self.write_document("contracts", f"{proj.lower()}_base_contract.txt", base_contract)
            
            # Updated document (active fact)
            amendment = f"""=========================================
AMENDMENT NO. 1 TO PROJECT {proj} CONTRACT
=========================================
Date: 12 October 2026
Supersedes: Contract of {db['original_deadline']}

Pursuant to board resolutions, terms for Project {proj} are amended as follows:
- The updated budget is increased to {db['budget']}. This supersedes the old budget of {db['original_budget']}.
- The active deadline for project {proj} is officially extended to {db['deadline']}. The previous date of {db['original_deadline']} is now obsolete.
"""
            rel_amend = self.write_document("contracts", f"{proj.lower()}_amendment_v1.txt", amendment)
            injected_files[proj] = [rel_base, rel_amend]
            
            # Generate QA Category C (Status/Evolution)
            self.add_qa(
                question=f"What is the active budget of Project {proj}?",
                expected_answer=db['budget'],
                category="C",
                difficulty="medium",
                source_file=rel_amend,
                evidence_quote=f"The updated budget is increased to {db['budget']}. This supersedes the old budget of {db['original_budget']}.",
                reasoning=f"Must resolve the conflict between base contract ({db['original_budget']}) and amendment ({db['budget']}) using supersession clues."
            )
            
            self.add_qa(
                question=f"What is the current active deadline of Project {proj}?",
                expected_answer=db['deadline'],
                category="C",
                difficulty="medium",
                source_file=rel_amend,
                evidence_quote=f"- The active deadline for project {proj} is officially extended to {db['deadline']}. The previous date of {db['original_deadline']} is now obsolete.",
                reasoning=f"Must identify that the old deadline ({db['original_deadline']}) is obsolete and the new deadline is {db['deadline']}."
            )

        # Category D: Multi-fact Questions
        # We split the owner and the budget/deadline into two different files (e.g. Report and Email)
        for proj in PROJECT_NAMES[15:20]: # Next 5 projects get multi-fact split
            db = project_db[proj]
            
            # Document 1: Spec (contains Owner and Department)
            spec_text = f"""=========================================
TECHNICAL SPECIFICATIONS: PROJECT {proj}
=========================================
Project Owner: {db['owner']}
Lead Unit: {db['department']}

This document describes the technical architecture for project {proj}.
The project is overseen by the {db['department']} unit, with {db['owner']} as director.
"""
            rel_spec = self.write_document("specs", f"{proj.lower()}_specifications.txt", spec_text)
            
            # Document 2: Email (contains budget and deadline)
            email_text = f"""From: {db['owner']}
To: Finance Committee
Date: 14 January 2026
Subject: Project {proj} Financial Request

Dear colleagues,
As project manager for Project {proj}, I am happy to report that our financial allocation request of {db['budget']} was approved.
The milestone deadline is set to {db['deadline']}.

Best regards,
{db['owner']}
"""
            rel_email = self.write_document("correspondence", f"{proj.lower()}_financial_email.txt", email_text)
            injected_files[proj] = [rel_spec, rel_email]
            
            # Generate QA Category D (Multi-fact)
            self.add_qa(
                question=f"Who manages Project {proj} and what is its approved budget?",
                expected_answer=f"{db['owner']} and {db['budget']}",
                category="D",
                difficulty="hard",
                source_file=f"{rel_spec} and {rel_email}",
                evidence_quote=f"Project Owner: {db['owner']} | As project manager for Project {proj}, I am happy to report that our financial allocation request of {db['budget']} was approved.",
                reasoning=f"Requires extracting facts from two separate files: owner from spec, budget from email."
            )
            
            self.add_qa(
                question=f"What is the department and final deadline for Project {proj}?",
                expected_answer=f"{db['department']} and {db['deadline']}",
                category="D",
                difficulty="hard",
                source_file=f"{rel_spec} and {rel_email}",
                evidence_quote=f"Lead Unit: {db['department']} | The milestone deadline is set to {db['deadline']}.",
                reasoning=f"Requires extracting department from spec and deadline from email."
            )

        # Category F: Suffix Distractor Questions
        # We inject a base project name, and multiple suffix-distractors (e.g. Project-Legacy, Project-Mobile, Project-Beta)
        # We query the base project, but the distractor projects have different budgets/deadlines in the text.
        for proj in PROJECT_NAMES[20:24]: # Last 4 projects get suffix distractors
            db = project_db[proj]
            
            # Target project document (active/correct)
            target_report = f"""=========================================
STATUS UPDATE: PROJECT {proj}
=========================================
Target Entity: Project {proj}
Approved Budget: {db['budget']}
Delivery Date: {db['deadline']}
Project Director: {db['owner']}

The main Project {proj} is active. The director is {db['owner']}. The budget is {db['budget']} and completion is set to {db['deadline']}.
"""
            rel_target = self.write_document("reports", f"{proj.lower()}_status.txt", target_report)
            
            # Distractor 1: Legacy
            legacy_report = f"""=========================================
LEGACY ARCHIVE: PROJECT {proj}-Legacy
=========================================
Target Entity: Project {proj}-Legacy
Approved Budget: {db['original_budget']}
Delivery Date: {db['original_deadline']}

This details Project {proj}-Legacy, which was retired. The budget of {proj}-Legacy was {db['original_budget']}.
"""
            rel_legacy = self.write_document("reports", f"{proj.lower()}_legacy.txt", legacy_report)
            
            # Distractor 2: Mobile
            mobile_report = f"""=========================================
MOBILE DIVISION: PROJECT {proj}-Mobile
=========================================
Target Entity: Project {proj}-Mobile
Approved Budget: $999,999
Delivery Date: 31 December 2029

This details the sub-branch Project {proj}-Mobile. The budget is $999,999.
"""
            rel_mobile = self.write_document("reports", f"{proj.lower()}_mobile.txt", mobile_report)
            injected_files[proj] = [rel_target, rel_legacy, rel_mobile]
            
            # Generate QA Category F (Suffix Distractor)
            self.add_qa(
                question=f"What is the approved budget of Project {proj}?",
                expected_answer=db['budget'],
                category="F",
                difficulty="medium",
                source_file=rel_target,
                evidence_quote=f"The main Project {proj} is active. The budget is {db['budget']} and completion is set to {db['deadline']}.",
                reasoning=f"Tests suffix boundary check: the model must ignore Project {proj}-Legacy (${db['original_budget']}) and Project {proj}-Mobile ($999,999)."
            )
            
            self.add_qa(
                question=f"What is the delivery date of Project {proj}?",
                expected_answer=db['deadline'],
                category="F",
                difficulty="medium",
                source_file=rel_target,
                evidence_quote=f"The main Project {proj} is active. The budget is {db['budget']} and completion is set to {db['deadline']}.",
                reasoning=f"Tests suffix boundary check for dates: must ignore legacy/mobile deadlines."
            )

        # Category E: Absent Entities (Abstention)
        # We query projects that DO NOT exist in the database or documents.
        # We create 60 questions with absent project names.
        absent_projects = ["ORC", "PEGASUS", "TITANIUM", "VOYAGER", "GALAXY", "HYDRA", "ODYSSEY", "VALKYRIE"]
        for idx, fake_proj in enumerate(absent_projects * 8):
            if len(self.qa_pairs) >= 440: # Leave space for cross-entity / cross-doc / absent
                break
            self.add_qa(
                question=f"What is the approved budget and lead manager of Project {fake_proj}_{idx}?",
                expected_answer="NOT_FOUND",
                category="E",
                difficulty="medium",
                source_file="None",
                evidence_quote="None",
                reasoning="The requested project does not exist in any document. The system must abstain."
            )

        # Category G: Cross-entity comparison (Exploratory)
        # E.g. "Which project has the higher budget between NEPTUNE and KRONOS?"
        # Let's generate a few of these comparison questions based on the database.
        for idx in range(15):
            proj1 = PROJECT_NAMES[idx % len(PROJECT_NAMES)]
            proj2 = PROJECT_NAMES[(idx + 3) % len(PROJECT_NAMES)]
            db1 = project_db[proj1]
            db2 = project_db[proj2]
            
            # Simple parsing of budget to compare
            b1_val = int(db1['budget'].replace('$', '').replace(',', ''))
            b2_val = int(db2['budget'].replace('$', '').replace(',', ''))
            higher_proj = proj1 if b1_val > b2_val else proj2
            higher_val = db1['budget'] if b1_val > b2_val else db2['budget']
            
            self.add_qa(
                question=f"Which project has a higher budget: Project {proj1} or Project {proj2}?",
                expected_answer=f"Project {higher_proj} (budget of {higher_val})",
                category="G",
                difficulty="hard",
                source_file="None", # Derived from multiple sources
                evidence_quote=f"Project {proj1} budget: {db1['budget']} | Project {proj2} budget: {db2['budget']}",
                reasoning=f"Cross-entity comparison: NEPTUNE ({db1['budget']}) vs KRONOS ({db2['budget']})."
            )

        # Category H: Cross-document summary / Aggregation (Exploratory)
        # E.g. "List all projects managed by Alice Martin."
        # We query the manager name, and list all projects assigned to them.
        for person in PEOPLE:
            managed = [proj for proj, db in project_db.items() if db['owner'] == person]
            if not managed:
                continue
            self.add_qa(
                question=f"List all project names managed by {person}.",
                expected_answer=", ".join(managed),
                category="H",
                difficulty="hard",
                source_file="None", # Multiple
                evidence_quote=f"Projects managed by {person}: " + ", ".join([f"{p} (managed by {person})" for p in managed]),
                reasoning=f"Cross-document aggregation: find all projects where manager is {person}."
            )

        # Top up Category E / F to reach exactly 500 questions if needed
        while len(self.qa_pairs) < 500:
            fake_id = len(self.qa_pairs)
            self.add_qa(
                question=f"Who is the lead technician for Project VOID_PROJ_{fake_id}?",
                expected_answer="NOT_FOUND",
                category="E",
                difficulty="easy",
                source_file="None",
                evidence_quote="None",
                reasoning="Abstention test for a non-existent project."
            )

        # --- PHASE 2: GENERATE NOISE DOCUMENTS ---
        # Write 100 purely noise documents to dilute the search space
        for i in range(total_noise_docs):
            noise_text = f"""=========================================
OPERATIONAL OPTIMIZATION SUMMARY: DOC_NOISE_{i:03d}
=========================================
Security Classification: INTERNAL ONLY
Author: Operations Analyst

1. Context & Scope
{generate_noise_text(120)}

2. Detailed Analysis
{generate_noise_text(150)}

3. Key Takeaways
{generate_noise_text(80)}
"""
            self.write_document("noise", f"noise_doc_{i:03d}.txt", noise_text)

        # Save the QA JSON
        qa_path = os.path.join(self.questions_dir, "qa_gold.json")
        with open(qa_path, "w", encoding="utf-8") as f:
            json.dump(self.qa_pairs, f, indent=2, ensure_ascii=False)
            
        print(f"Successfully generated synthetic corpus under {self.output_dir}:")
        print(f"  - Total documents written: {len(self.documents_written)}")
        print(f"  - Total QA pairs generated: {len(self.qa_pairs)}")
        print(f"    - Category A: {sum(1 for q in self.qa_pairs if q['category'] == 'A')}")
        print(f"    - Category B: {sum(1 for q in self.qa_pairs if q['category'] == 'B')}")
        print(f"    - Category C: {sum(1 for q in self.qa_pairs if q['category'] == 'C')}")
        print(f"    - Category D: {sum(1 for q in self.qa_pairs if q['category'] == 'D')}")
        print(f"    - Category E: {sum(1 for q in self.qa_pairs if q['category'] == 'E')}")
        print(f"    - Category F: {sum(1 for q in self.qa_pairs if q['category'] == 'F')}")
        print(f"    - Category G: {sum(1 for q in self.qa_pairs if q['category'] == 'G')}")
        print(f"    - Category H: {sum(1 for q in self.qa_pairs if q['category'] == 'H')}")

    def add_qa(self, question: str, expected_answer: str, category: str, difficulty: str, source_file: str, evidence_quote: str, reasoning: str):
        self.qa_pairs.append({
            "id": f"q_{self.global_question_id:03d}",
            "question": question,
            "expected_answer": expected_answer,
            "category": category,
            "difficulty": difficulty,
            "source_file": source_file,
            "evidence_quote": evidence_quote,
            "reasoning": reasoning
        })
        self.global_question_id += 1

if __name__ == "__main__":
    generator = SyntheticCorpusGenerator("data/corpus_poc2")
    generator.generate_all(total_noise_docs=100)


