import json
import random
import os
import argparse
from typing import List, Dict, Any

# Fix seed for reproducibility
random.seed(42)

# Business filler words to generate noise docs
BUZZWORDS = [
    "integration", "synergy", "paradigm", "scalability", "leverage", "robust",
    "deployment", "optimization", "framework", "architecture", "operational",
    "infrastructure", "standardization", "compliance", "methodology", "pipeline",
    "collaboration", "bandwidth", "virtualization", "containerization", "governance",
    "analytics", "monetization", "efficiency", "redundancy", "throughput", "orchestration"
]

DEPARTMENTS = ["HR", "Finance", "Legal", "Engineering", "Marketing", "Operations", "Sales"]

def generate_noise_paragraph(word_count: int = 85) -> str:
    """Generates a block of business-like noise text roughly word_count long."""
    sentences = []
    current_words = 0
    while current_words < word_count:
        dept = random.choice(DEPARTMENTS)
        b1 = random.choice(BUZZWORDS)
        b2 = random.choice(BUZZWORDS)
        b3 = random.choice(BUZZWORDS)
        sentence = f"The {dept} department is optimizing its {b1} strategies to achieve better {b2} and standard {b3} processes across all business units."
        sentence_words = len(sentence.split())
        sentences.append(sentence)
        current_words += sentence_words
    return " ".join(sentences)

def generate_doc_id(index: int) -> str:
    return f"DOC_{index:04d}"

def generate_date(year: int) -> str:
    months = [
        "January", "February", "March", "April", "May", "June",
        "July", "August", "September", "October", "November", "December"
    ]
    day = random.randint(1, 28)
    month = random.choice(months)
    return f"{day:02d} {month} {year}"

TRAIN_PROJECT_NAMES = [
    "APEX", "BEACON", "CHRONOS", "DAWN", "ECLIPSE", "FALCON", "GENESIS", "HELIOS",
    "IRIS", "JUPITER", "KRONOS", "LUNA", "METEOR", "NEXUS", "ORION", "PHOENIX"
]
TEST_PROJECT_NAMES = [
    "QUASAR", "RADAR", "SIRIUS", "TITAN", "URANUS", "VEGA", "WARP", "XENON", "ZENITH",
    "VANTAGE", "SAKURA", "NEBULA", "COSMOS", "COMET", "POLARIS", "AURORA"
]

def build_dataset(num_samples: int = 500, out_dir: str = "data", num_blocks: int = 30):
    os.makedirs(out_dir, exist_ok=True)
    
    samples_per_category = num_samples // 5
    print(f"Generating {num_samples} samples ({samples_per_category} per category, {num_blocks} blocks each)...")
    
    corpus_list = []
    questions_list = []
    answers_list = []
    
    global_sample_id = 0
    
    # We will generate 5 categories: A, B, C, D, E
    categories = ["A", "B", "C", "D", "E"]
    
    for category in categories:
        for i in range(samples_per_category):
            sample_id = f"Q{global_sample_id:04d}"
            if global_sample_id < 400:
                project = TRAIN_PROJECT_NAMES[global_sample_id % len(TRAIN_PROJECT_NAMES)] + f"-{global_sample_id}"
            else:
                project = TEST_PROJECT_NAMES[(global_sample_id - 400) % len(TEST_PROJECT_NAMES)] + f"-{global_sample_id}"
            
            # Generate a unique set of dates and budgets
            active_date = generate_date(2027)
            deprecated_date = generate_date(2026)
            budget = f"${random.randint(100, 999)},000"
            deprecated_budget = f"${random.randint(100, 999)},000"
            
            docs = []
            gold_block_ids = []
            deprecated_block_ids = []
            
            # Category A: Needle simple (1 fact, 0 contradictions, num_blocks blocks of noise)
            # Gold placed randomly
            if category == "A":
                gold_index = random.randint(num_blocks // 5, num_blocks * 4 // 5)
                for d_idx in range(num_blocks):
                    doc_id = generate_doc_id(d_idx)
                    if d_idx == gold_index:
                        text = f"{doc_id}:\nFACT_ID: F-{global_sample_id:04d}\nProject: {project}\nActive delivery date: {active_date}\nStatus: ACTIVE"
                        docs.append({"doc_id": doc_id, "text": text, "contains_gold_fact": True, "contains_deprecated_fact": False})
                        gold_block_ids.append(d_idx)
                    else:
                        text = f"{doc_id}:\n{generate_noise_paragraph()}"
                        docs.append({"doc_id": doc_id, "text": text, "contains_gold_fact": False, "contains_deprecated_fact": False})
                question = f"What is the active delivery date for Project {project}?"
                expected = active_date
                
            # Category B: Needle position (start / middle / end)
            elif category == "B":
                # Varies position of the needle based on num_blocks
                pos_type = i % 3
                if pos_type == 0:
                    gold_index = random.randint(1, min(5, num_blocks - 2))
                elif pos_type == 1:
                    gold_index = random.randint(num_blocks // 2 - 3, num_blocks // 2 + 3)
                else:
                    gold_index = random.randint(num_blocks - 6, num_blocks - 2)
                    
                for d_idx in range(num_blocks):
                    doc_id = generate_doc_id(d_idx)
                    if d_idx == gold_index:
                        text = f"{doc_id}:\nFACT_ID: F-{global_sample_id:04d}\nProject: {project}\nActive delivery date: {active_date}\nStatus: ACTIVE"
                        docs.append({"doc_id": doc_id, "text": text, "contains_gold_fact": True, "contains_deprecated_fact": False})
                        gold_block_ids.append(d_idx)
                    else:
                        text = f"{doc_id}:\n{generate_noise_paragraph()}"
                        docs.append({"doc_id": doc_id, "text": text, "contains_gold_fact": False, "contains_deprecated_fact": False})
                question = f"What is the active delivery date for Project {project}?"
                expected = active_date
                
            # Category C: Contradiction (deprecated vs active date)
            elif category == "C":
                dep_index = random.randint(2, num_blocks // 2 - 2)
                gold_index = random.randint(num_blocks // 2 + 2, num_blocks - 3)
                    
                for d_idx in range(num_blocks):
                    doc_id = generate_doc_id(d_idx)
                    if d_idx == gold_index:
                        text = f"{doc_id}:\nFACT_ID: F-{global_sample_id:04d}\nProject: {project}\nActive delivery date: {active_date}\nStatus: ACTIVE"
                        docs.append({"doc_id": doc_id, "text": text, "contains_gold_fact": True, "contains_deprecated_fact": False})
                        gold_block_ids.append(d_idx)
                    elif d_idx == dep_index:
                        text = f"{doc_id}:\nOld delivery date for Project {project}: {deprecated_date}\nStatus: DEPRECATED"
                        docs.append({"doc_id": doc_id, "text": text, "contains_gold_fact": False, "contains_deprecated_fact": True})
                        deprecated_block_ids.append(d_idx)
                    else:
                        text = f"{doc_id}:\n{generate_noise_paragraph()}"
                        docs.append({"doc_id": doc_id, "text": text, "contains_gold_fact": False, "contains_deprecated_fact": False})
                question = f"What is the active delivery date for Project {project}?"
                expected = active_date

            # Category D: Multi-facts (Requires active date AND budget)
            elif category == "D":
                gold_index1 = random.randint(2, num_blocks // 2 - 2)
                gold_index2 = random.randint(num_blocks // 2 + 2, num_blocks - 3)
                
                for d_idx in range(num_blocks):
                    doc_id = generate_doc_id(d_idx)
                    if d_idx == gold_index1:
                        text = f"{doc_id}:\nFACT_ID: F-{global_sample_id:04d}-A\nProject: {project}\nActive delivery date: {active_date}\nStatus: ACTIVE"
                        docs.append({"doc_id": doc_id, "text": text, "contains_gold_fact": True, "contains_deprecated_fact": False})
                        gold_block_ids.append(d_idx)
                    elif d_idx == gold_index2:
                        text = f"{doc_id}:\nFACT_ID: F-{global_sample_id:04d}-B\nProject: {project}\nBudget: {budget}\nStatus: ACTIVE"
                        docs.append({"doc_id": doc_id, "text": text, "contains_gold_fact": True, "contains_deprecated_fact": False})
                        gold_block_ids.append(d_idx)
                    else:
                        text = f"{doc_id}:\n{generate_noise_paragraph()}"
                        docs.append({"doc_id": doc_id, "text": text, "contains_gold_fact": False, "contains_deprecated_fact": False})
                question = f"What is the active delivery date and budget for Project {project}?"
                expected = f"{active_date} and {budget}"
                
            # Category E: Distractor (Semantic distractors, e.g., NOVA-A, NOVA-B, etc.)
            elif category == "E":
                distractor_projects = [
                    f"{project}-A",
                    f"{project}-B",
                    f"{project}-Legacy",
                    f"{project}-Mobile"
                ]
                indices = random.sample(range(num_blocks), 5)
                gold_index = indices[0]
                distractor_indices = indices[1:]
                
                for d_idx in range(num_blocks):
                    doc_id = generate_doc_id(d_idx)
                    if d_idx == gold_index:
                        text = f"{doc_id}:\nFACT_ID: F-{global_sample_id:04d}\nProject: {project}\nActive delivery date: {active_date}\nStatus: ACTIVE"
                        docs.append({"doc_id": doc_id, "text": text, "contains_gold_fact": True, "contains_deprecated_fact": False})
                        gold_block_ids.append(d_idx)
                    elif d_idx in distractor_indices:
                        d_proj = distractor_projects[distractor_indices.index(d_idx)]
                        d_date = generate_date(2027)
                        text = f"{doc_id}:\nFACT_ID: F-{global_sample_id:04d}-D\nProject: {d_proj}\nActive delivery date: {d_date}\nStatus: ACTIVE"
                        docs.append({"doc_id": doc_id, "text": text, "contains_gold_fact": False, "contains_deprecated_fact": False})
                    else:
                        text = f"{doc_id}:\n{generate_noise_paragraph()}"
                        docs.append({"doc_id": doc_id, "text": text, "contains_gold_fact": False, "contains_deprecated_fact": False})
                question = f"What is the active delivery date for Project {project}?"
                expected = active_date
            
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
                "deprecated_block_ids": deprecated_block_ids
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
    parser.add_argument("--num-samples", type=int, default=500)
    parser.add_argument("--out-dir", type=str, default="data")
    parser.add_argument("--num-blocks", type=int, default=30)
    args = parser.parse_args()
    build_dataset(args.num_samples, args.out_dir, args.num_blocks)


