import os
from datasets import load_dataset
from grpo_trainer import MinimalGRPOTrainer, combined_reward
from trl.trainer.grpo_config import GRPOConfig
import torch
import regex as re
import json

os.environ["CUDA_VISIBLE_DEVICES"] = "0,2,3"
device = "cuda" if torch.cuda.is_available() else "cpu"

SYSTEM_PROMPT = (
    "You are an AI assistant for Visual Question Answering (VQA). "
    "Your task is to answer the given question based on the provided image. "
    "Your response MUST strictly follow this format:\n\n"
    "<think> Reasoning process here (optional) </think><answer> Final answer (MUST be a NUMBER) </answer>\n\n"
    "**Strict Rules:**\n"
    "1. Your response MUST always include both <think> and <answer> tags.\n"
    "2. The <answer> section MUST contain a SINGLE NUMBER (e.g., 3, 12, 45.7) and NOTHING ELSE.\n"
    "3. NEVER include words, explanations, or additional text inside the <answer> tags.\n"
    "4. If you cannot determine the answer, you MUST still return a number (e.g., 0 or -1), but never leave it blank.\n"
    "5. Violation of these rules will result in rejection of your response."
)


QUESTION_TEMPLATE = (
    "{Question} Your response MUST follow this exact format:\n\n"
    "<think> Provide reasoning here if necessary </think><answer> Provide a SINGLE NUMBER here </answer>\n\n"
    "⚠️ IMPORTANT: \n"
    "- The answer inside <answer> MUST be a NUMBER (e.g., 1, 5, 12, 42.3). \n"
    "- NO words or explanations are allowed in <answer>.\n"
    "- If unsure, output the closest estimate as a number."
)

if __name__ == "__main__":
    config = GRPOConfig()   
    config.num_iterations = 1 
    # number of outputs G in grpo, reduce it would lead to faster training and smaller memory cost but higher variance
    config.num_generations = 2 #8
    config.max_steps = 1000
    config.per_device_train_batch_size = 4 
    config.gradient_accumulation_steps = 1 #2
    config.gradient_checkpointing=True
    config.report_to = "wandb"
    config.run_name = "smolvlm2_counting_batch4"
    config.logging_steps= 1

    model_id = "HuggingFaceTB/SmolVLM2-2.2B-Instruct"

    dataset = load_dataset("leonardPKU/clevr_cogen_a_train")
    total_samples = len(dataset["train"])

    train_dataset = []
    for idx, entry in enumerate(dataset["train"]):
        if idx % 5000 == 0 or idx == total_samples - 1:
            print(f"🔍 {idx+1}/{total_samples} 데이터 로딩 중...")
        new_entry = {
            "prompt": entry["problem"],
            "image_path": entry["image"],
            "ground_truth": entry["solution"],
            "conversations": [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "url": entry["image"]},
                        {"type": "text", "text": QUESTION_TEMPLATE.format(Question=entry["solution"])},
                    ]
                }
            ]
        }
        train_dataset.append(new_entry)

    eval_dataset = []
    with open("./prompts/superclevr_test200_counting_problems.jsonl", "r") as f:
        for line in f:
            entry = json.loads(line)
            new_entry = {
                "prompt": QUESTION_TEMPLATE.format(Question=entry["question"]),
                "image_path": entry["image_path"],
                "ground_truth": entry["ground_truth"],
                "conversations": [
                    {
                        "role": "user",
                        "content": [
                            {"type": "image", "url": entry["image_path"]},
                            {"type": "text", "text": QUESTION_TEMPLATE.format(Question=entry["question"])},
                        ]
                    }
                ]
            }
            eval_dataset.append(new_entry)

    trainer = MinimalGRPOTrainer(
        model_name=model_id,
        reward_func=combined_reward,  
        args=config,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset  
    )

    trainer.train()  
    trainer.evaluate()  