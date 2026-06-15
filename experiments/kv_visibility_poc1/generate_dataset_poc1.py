import json
import random
import os
import argparse
import string
from typing import List, Dict, Any

# Set seed for reproducibility
random.seed(42)

BUZZWORDS = [
    "integration", "synergy", "paradigm", "scalability", "leverage", "robust",
    "deployment", "optimization", "framework", "architecture", "operational",
    "infrastructure", "standardization", "compliance", "methodology", "pipeline",
    "collaboration", "bandwidth", "virtualization", "containerization", "governance",
    "analytics", "monetization", "efficiency", "redundancy", "throughput", "orchestration"
]

DEPARTMENTS = ["HR", "Finance", "Legal", "Engineering", "Marketing", "Operations", "Sales"]

PROJECT_NAMES = [
    "APEX", "BEACON", "CHRONOS", "DAWN", "ECLIPSE", "FALCON", "GENESIS", "HELIOS",
    "IRIS", "JUPITER", "KRONOS", "LUNA", "METEOR", "NEXUS", "ORION", "PHOENIX",
    "QUASAR", "RADAR", "SIRIUS", "TITAN", "URANUS", "VEGA", "WARP", "XENON", "ZENITH",
    "VANTAGE", "SAKURA", "NEBULA", "COSMOS", "COMET", "POLARIS", "AURORA", "HESTIA"
]

MONTHS = [
    "January", "February", "March", "April", "May", "June",
    "July", "August", "September", "October", "November", "December"
]

def generate_noise_paragraph(word_count: int = 80) -> str:
    sentences = []
    current_words = 0
    while current_words < word_count:
        dept = random.choice(DEPARTMENTS)
        b1 = random.choice(BUZZWORDS)
        b2 = random.choice(BUZZWORDS)
        b3 = random.choice(BUZZWORDS)
        sentence = f"The {dept} department is optimizing its {b1} strategies to achieve {b2} and standard {b3} processes across all business units."
        sentence_words = len(sentence.split())
        sentences.append(sentence)
        current_words += sentence_words
    return " ".join(sentences)

def generate_doc_id(index: int) -> str:
    return f"DOC_{index:04d}"

def generate_date(year: int) -> str:
    day = random.randint(1, 28)
    month = random.choice(MONTHS)
    return f"{day:02d} {month} {year}"

def parse_date_to_comparable(date_str: str) -> tuple:
    parts = date_str.split()
    day = int(parts[0])
    month = MONTHS.index(parts[1])
    year = int(parts[2])
    return (year, month, day)

def build_dataset(num_samples: int = 1000, out_dir: str = "data", num_blocks: int = 50, abstention_rate: float = 0.10):
    os.makedirs(out_dir, exist_ok=True)
    
    categories = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J"]
    samples_per_category = num_samples // len(categories)
    print(f"Generating {num_samples} samples ({samples_per_category} per category, {num_blocks} base blocks)...")
    
    corpus_list = []
    questions_list = []
    answers_list = []
    
    global_sample_id = 0
    
    for category in categories:
        for i in range(samples_per_category):
            sample_id = f"Q{global_sample_id:04d}"
            project = PROJECT_NAMES[global_sample_id % len(PROJECT_NAMES)] + f"-{global_sample_id}"
            
            # Determine if this sample is an abstention check (no-answer)
            is_abstention = (random.random() < abstention_rate)
            
            active_date = generate_date(2027)
            deprecated_date = generate_date(2026)
            budget = f"${random.randint(100, 999)},000"
            
            docs = []
            gold_block_ids = []
            deprecated_block_ids = []
            expected = ""
            question = ""
            
            # Build blocks for each category
            if category == "A":
                # Fact simple
                gold_index = random.randint(5, num_blocks - 5)
                for d_idx in range(num_blocks):
                    doc_id = generate_doc_id(d_idx)
                    if d_idx == gold_index and not is_abstention:
                        text = f"{doc_id}:\nFACT_ID: F-{global_sample_id:04d}\nProject: {project}\nActive delivery date: {active_date}\nStatus: ACTIVE"
                        docs.append({"doc_id": doc_id, "text": text, "contains_gold_fact": True, "contains_deprecated_fact": False})
                        gold_block_ids.append(d_idx)
                    else:
                        text = f"{doc_id}:\n{generate_noise_paragraph()}"
                        docs.append({"doc_id": doc_id, "text": text, "contains_gold_fact": False, "contains_deprecated_fact": False})
                question = f"What is the active delivery date for Project {project}?"
                expected = active_date if not is_abstention else "I cannot answer this question based on the provided context."
                
            elif category == "B":
                # Positional placement (start, middle, end)
                pos_type = i % 3
                if pos_type == 0:
                    gold_index = random.randint(1, 3)
                elif pos_type == 1:
                    gold_index = num_blocks // 2
                else:
                    gold_index = num_blocks - 3
                    
                for d_idx in range(num_blocks):
                    doc_id = generate_doc_id(d_idx)
                    if d_idx == gold_index and not is_abstention:
                        text = f"{doc_id}:\nFACT_ID: F-{global_sample_id:04d}\nProject: {project}\nActive delivery date: {active_date}\nStatus: ACTIVE"
                        docs.append({"doc_id": doc_id, "text": text, "contains_gold_fact": True, "contains_deprecated_fact": False})
                        gold_block_ids.append(d_idx)
                    else:
                        text = f"{doc_id}:\n{generate_noise_paragraph()}"
                        docs.append({"doc_id": doc_id, "text": text, "contains_gold_fact": False, "contains_deprecated_fact": False})
                question = f"What is the active delivery date for Project {project}?"
                expected = active_date if not is_abstention else "I cannot answer this question based on the provided context."
                
            elif category == "C":
                # Contradiction
                dep_index = random.randint(2, num_blocks // 2 - 2)
                gold_index = random.randint(num_blocks // 2 + 2, num_blocks - 3)
                for d_idx in range(num_blocks):
                    doc_id = generate_doc_id(d_idx)
                    if d_idx == gold_index and not is_abstention:
                        text = f"{doc_id}:\nFACT_ID: F-{global_sample_id:04d}\nProject: {project}\nActive delivery date: {active_date}\nStatus: ACTIVE"
                        docs.append({"doc_id": doc_id, "text": text, "contains_gold_fact": True, "contains_deprecated_fact": False})
                        gold_block_ids.append(d_idx)
                    elif d_idx == dep_index and not is_abstention:
                        text = f"{doc_id}:\nOld delivery date for Project {project}: {deprecated_date}\nStatus: DEPRECATED"
                        docs.append({"doc_id": doc_id, "text": text, "contains_gold_fact": False, "contains_deprecated_fact": True})
                        deprecated_block_ids.append(d_idx)
                    else:
                        text = f"{doc_id}:\n{generate_noise_paragraph()}"
                        docs.append({"doc_id": doc_id, "text": text, "contains_gold_fact": False, "contains_deprecated_fact": False})
                question = f"What is the active delivery date for Project {project}?"
                expected = active_date if not is_abstention else "I cannot answer this question based on the provided context."
                
            elif category == "D":
                # Multi-facts
                gold_index1 = random.randint(2, num_blocks // 2 - 2)
                gold_index2 = random.randint(num_blocks // 2 + 2, num_blocks - 3)
                for d_idx in range(num_blocks):
                    doc_id = generate_doc_id(d_idx)
                    if d_idx == gold_index1 and not is_abstention:
                        text = f"{doc_id}:\nFACT_ID: F-{global_sample_id:04d}-A\nProject: {project}\nActive delivery date: {active_date}\nStatus: ACTIVE"
                        docs.append({"doc_id": doc_id, "text": text, "contains_gold_fact": True, "contains_deprecated_fact": False})
                        gold_block_ids.append(d_idx)
                    elif d_idx == gold_index2 and not is_abstention:
                        text = f"{doc_id}:\nFACT_ID: F-{global_sample_id:04d}-B\nProject: {project}\nBudget: {budget}\nStatus: ACTIVE"
                        docs.append({"doc_id": doc_id, "text": text, "contains_gold_fact": True, "contains_deprecated_fact": False})
                        gold_block_ids.append(d_idx)
                    else:
                        text = f"{doc_id}:\n{generate_noise_paragraph()}"
                        docs.append({"doc_id": doc_id, "text": text, "contains_gold_fact": False, "contains_deprecated_fact": False})
                question = f"What is the active delivery date and budget for Project {project}?"
                expected = f"{active_date} and {budget}" if not is_abstention else "I cannot answer this question based on the provided context."
                
            elif category == "E":
                # Suffix distractors
                distractors = [f"{project}-A", f"{project}-B", f"{project}-Legacy", f"{project}-Mobile"]
                indices = random.sample(range(num_blocks), 5)
                gold_index = indices[0]
                distractor_indices = indices[1:]
                for d_idx in range(num_blocks):
                    doc_id = generate_doc_id(d_idx)
                    if d_idx == gold_index and not is_abstention:
                        text = f"{doc_id}:\nFACT_ID: F-{global_sample_id:04d}\nProject: {project}\nActive delivery date: {active_date}\nStatus: ACTIVE"
                        docs.append({"doc_id": doc_id, "text": text, "contains_gold_fact": True, "contains_deprecated_fact": False})
                        gold_block_ids.append(d_idx)
                    elif d_idx in distractor_indices:
                        d_proj = distractors[distractor_indices.index(d_idx)]
                        d_date = generate_date(2027)
                        text = f"{doc_id}:\nFACT_ID: F-{global_sample_id:04d}-D\nProject: {d_proj}\nActive delivery date: {d_date}\nStatus: ACTIVE"
                        docs.append({"doc_id": doc_id, "text": text, "contains_gold_fact": False, "contains_deprecated_fact": False})
                    else:
                        text = f"{doc_id}:\n{generate_noise_paragraph()}"
                        docs.append({"doc_id": doc_id, "text": text, "contains_gold_fact": False, "contains_deprecated_fact": False})
                question = f"What is the active delivery date for Project {project}?"
                expected = active_date if not is_abstention else "I cannot answer this question based on the provided context."
                
            elif category == "F":
                # Chronological reasoning: find the earliest active project date
                # Generate 3 projects in the context
                proj_indices = random.sample(range(5, num_blocks - 5), 3)
                dates = [generate_date(2027) for _ in range(3)]
                proj_names = [f"CHRONO_{global_sample_id}_{k}" for k in range(3)]
                
                # Sort dates to find the earliest
                parsed_dates = [parse_date_to_comparable(d) for d in dates]
                earliest_idx = parsed_dates.index(min(parsed_dates))
                earliest_date = dates[earliest_idx]
                
                for d_idx in range(num_blocks):
                    doc_id = generate_doc_id(d_idx)
                    if d_idx in proj_indices and not is_abstention:
                        k = proj_indices.index(d_idx)
                        text = f"{doc_id}:\nProject: {proj_names[k]}\nActive delivery date: {dates[k]}\nStatus: ACTIVE"
                        docs.append({"doc_id": doc_id, "text": text, "contains_gold_fact": (k == earliest_idx), "contains_deprecated_fact": False})
                        if k == earliest_idx:
                            gold_block_ids.append(d_idx)
                    else:
                        text = f"{doc_id}:\n{generate_noise_paragraph()}"
                        docs.append({"doc_id": doc_id, "text": text, "contains_gold_fact": False, "contains_deprecated_fact": False})
                question = f"What is the active delivery date of the earliest active project?"
                expected = earliest_date if not is_abstention else "I cannot answer this question based on the provided context."
                
            elif category == "G":
                # Local Summary of a specific block
                gold_index = random.randint(5, num_blocks - 5)
                dept_to_summarize = random.choice(DEPARTMENTS)
                buzz_to_find = [random.choice(BUZZWORDS) for _ in range(3)]
                summary_fact = f"The {dept_to_summarize} department utilizes {', '.join(buzz_to_find)} to improve systems."
                for d_idx in range(num_blocks):
                    doc_id = generate_doc_id(d_idx)
                    if d_idx == gold_index and not is_abstention:
                        text = f"{doc_id}:\nLocal Info:\nProject: {project}\nSummary: {summary_fact}"
                        docs.append({"doc_id": doc_id, "text": text, "contains_gold_fact": True, "contains_deprecated_fact": False})
                        gold_block_ids.append(d_idx)
                    else:
                        text = f"{doc_id}:\n{generate_noise_paragraph()}"
                        docs.append({"doc_id": doc_id, "text": text, "contains_gold_fact": False, "contains_deprecated_fact": False})
                question = f"Summarize the systems improvements of the {dept_to_summarize} department in Project {project}."
                expected = summary_fact if not is_abstention else "I cannot answer this question based on the provided context."
                
            elif category == "H":
                # Global Summary: list all active projects in the context
                # Generate 4 active projects across the context
                proj_indices = random.sample(range(2, num_blocks - 2), 4)
                sub_projects = [f"SUB_PROJ_{global_sample_id}_{k}" for k in range(4)]
                for d_idx in range(num_blocks):
                    doc_id = generate_doc_id(d_idx)
                    if d_idx in proj_indices and not is_abstention:
                        k = proj_indices.index(d_idx)
                        text = f"{doc_id}:\nProject Name: {sub_projects[k]}\nActive delivery date: {generate_date(2027)}\nStatus: ACTIVE"
                        docs.append({"doc_id": doc_id, "text": text, "contains_gold_fact": True, "contains_deprecated_fact": False})
                        gold_block_ids.append(d_idx)
                    else:
                        text = f"{doc_id}:\n{generate_noise_paragraph()}"
                        docs.append({"doc_id": doc_id, "text": text, "contains_gold_fact": False, "contains_deprecated_fact": False})
                question = "List all the active Project Names mentioned in the context."
                expected = ", ".join(sub_projects) if not is_abstention else "I cannot answer this question based on the provided context."
                
            elif category == "I":
                # Long Chat memory: multi-turn conversation
                gold_index = random.randint(5, num_blocks - 5)
                transcript = (
                    f"User: Hey, I want to check details for {project}.\n"
                    f"Assistant: Sure! What details do you need?\n"
                    f"User: What is the active delivery date?\n"
                    f"Assistant: Let me check. It is scheduled for {active_date}.\n"
                    f"User: Excellent, thank you!"
                )
                for d_idx in range(num_blocks):
                    doc_id = generate_doc_id(d_idx)
                    if d_idx == gold_index and not is_abstention:
                        text = f"{doc_id}:\nChat Transcript:\n{transcript}"
                        docs.append({"doc_id": doc_id, "text": text, "contains_gold_fact": True, "contains_deprecated_fact": False})
                        gold_block_ids.append(d_idx)
                    else:
                        text = f"{doc_id}:\n{generate_noise_paragraph()}"
                        docs.append({"doc_id": doc_id, "text": text, "contains_gold_fact": False, "contains_deprecated_fact": False})
                question = f"What is the active delivery date of Project {project} according to our conversation?"
                expected = active_date if not is_abstention else "I cannot answer this question based on the provided context."
                
            elif category == "J":
                # Massive Noise: 1 gold fact block, but context scale is handled by scaling tool
                gold_index = random.randint(2, 4) # placed near the beginning
                for d_idx in range(num_blocks):
                    doc_id = generate_doc_id(d_idx)
                    if d_idx == gold_index and not is_abstention:
                        text = f"{doc_id}:\nFACT_ID: F-{global_sample_id:04d}\nProject: {project}\nActive delivery date: {active_date}\nStatus: ACTIVE"
                        docs.append({"doc_id": doc_id, "text": text, "contains_gold_fact": True, "contains_deprecated_fact": False})
                        gold_block_ids.append(d_idx)
                    else:
                        text = f"{doc_id}:\n{generate_noise_paragraph()}"
                        docs.append({"doc_id": doc_id, "text": text, "contains_gold_fact": False, "contains_deprecated_fact": False})
                question = f"What is the active delivery date for Project {project}?"
                expected = active_date if not is_abstention else "I cannot answer this question based on the provided context."
            
            # If it's an abstention check, overwrite the project name in the question to ensure it doesn't match
            if is_abstention:
                fake_project = f"MYSTIC_{global_sample_id}_VOID"
                question = question.replace(project, fake_project)
                project = fake_project
                gold_block_ids = []
                deprecated_block_ids = []
            
            corpus_list.append({
                "question_id": sample_id,
                "category": category,
                "project": project,
                "question": question,
                "documents": docs
            })
            
            questions_list.append({
                "question_id": sample_id,
                "category": category,
                "question": question
            })
            
            answers_list.append({
                "question_id": sample_id,
                "category": category,
                "expected_answer": expected,
                "gold_block_ids": gold_block_ids,
                "deprecated_block_ids": deprecated_block_ids,
                "is_abstention": is_abstention
            })
            
            global_sample_id += 1
            
    # Save files
    with open(os.path.join(out_dir, "corpus.jsonl"), "w") as f:
        for item in corpus_list:
            f.write(json.dumps(item) + "\n")
            
    with open(os.path.join(out_dir, "questions.jsonl"), "w") as f:
        for item in questions_list:
            f.write(json.dumps(item) + "\n")
            
    with open(os.path.join(out_dir, "answers.jsonl"), "w") as f:
        for item in answers_list:
            f.write(json.dumps(item) + "\n")
            
    print(f"Dataset generated successfully under {out_dir}/")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-samples", type=int, default=1000)
    parser.add_argument("--out-dir", type=str, default="data")
    parser.add_argument("--num-blocks", type=int, default=50)
    parser.add_argument("--abstention-rate", type=float, default=0.10)
    args = parser.parse_args()
    build_dataset(args.num_samples, args.out_dir, args.num_blocks, args.abstention_rate)


