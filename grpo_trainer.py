import torch
import re
import base64
from datetime import datetime
from transformers import AutoProcessor, GenerationConfig, Trainer, AutoModelForImageTextToText
from PIL import Image
from torch.utils.data import Sampler
from accelerate.utils import set_seed
import wandb
import json
from tqdm import tqdm

class RepeatRandomSampler(Sampler):
    def __init__(
        self,
        data_source,
        mini_repeat_count: int,
        batch_size: int = 1,
        repeat_count: int = 1,
        seed = None,
    ):
        self.data_source = data_source
        self.mini_repeat_count = mini_repeat_count
        self.batch_size = batch_size
        self.repeat_count = repeat_count
        self.num_samples = len(data_source)
        self.seed = seed
        self.generator = torch.Generator()  # Create a local random generator
        if seed is not None:
            self.generator.manual_seed(seed)

    def __iter__(self):
        indexes = torch.randperm(self.num_samples, generator=self.generator).tolist()
        indexes = [indexes[i : i + self.batch_size] for i in range(0, len(indexes), self.batch_size)]
        indexes = [chunk for chunk in indexes if len(chunk) == self.batch_size]
        for chunk in indexes:
            for _ in range(self.repeat_count):
                for index in chunk:
                    for _ in range(self.mini_repeat_count):
                        yield index

    def __len__(self) -> int:
        return self.num_samples * self.mini_repeat_count * self.repeat_count

def accuracy_reward(prompt, completion, solution, **kwargs):
    
    reward = 0.0
    try:
        if isinstance(solution, list):
            solution = solution[0]

        sol_match = re.search(r'<answer>\s*(\d+)\s*</answer>', str(solution))
        ground_truth = sol_match.group(1).strip() if sol_match else str(solution).strip()

        content_match = re.search(r'<answer>\s*(\d+)\s*</answer>', completion)
        if content_match:
            student_answer = content_match.group(1).strip()
        else:
            student_answer = completion.strip()

        try:
            if float(student_answer) == float(ground_truth):
                reward = 1.0
        except ValueError:
            if student_answer.lower() == ground_truth.lower():
                reward = 1.0

    except Exception:
        pass

    return reward


def format_reward(prompt, completion, **kwargs):
    pattern = r"<think>.*?</think>\s*<answer>.*?</answer>"
    return 1.0 if re.fullmatch(pattern, completion, re.DOTALL) else 0.0


def combined_reward(prompt, completion, solution, **kwargs):
    format_score = format_reward(prompt, completion)
    accuracy_score = accuracy_reward(prompt, completion, solution)


    return accuracy_score


class MinimalGRPOTrainer(Trainer):
    def __init__(self, model_name, reward_func, args, train_dataset, eval_dataset=None, **kwargs):
        self.model = AutoModelForImageTextToText.from_pretrained(
            model_name,
            device_map="cuda:0",
            torch_dtype=torch.bfloat16
        )
        self.processing_class = AutoProcessor.from_pretrained(model_name, padding_side="left")


        self.train_generation_config = GenerationConfig(
            max_new_tokens=args.max_completion_length,
            do_sample=True,
            temperature=1,
            pad_token_id=self.processing_class.tokenizer.pad_token_id
        )
        self.eval_generation_config = GenerationConfig(
            max_new_tokens=args.max_completion_length,
            do_sample=False,
            temperature=1,
            pad_token_id=self.processing_class.tokenizer.pad_token_id
        )

        self.global_step = 1
        self.reward_func = reward_func
        self.num_generations = args.num_generations
        self.num_iterations = args.num_iterations
        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset
        
        super().__init__(
            model=self.model,
            processing_class=self.processing_class,
            args=args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            data_collator=lambda x: x,
            **kwargs
        )

    def _get_train_sampler(self):
        effective_batch_size = (
            self.args.per_device_train_batch_size
            * self.args.gradient_accumulation_steps
        )
        return RepeatRandomSampler(
            data_source=self.train_dataset,
            mini_repeat_count=self.num_generations,
            batch_size=effective_batch_size // self.num_generations,
            repeat_count=self.num_iterations,
            seed=self.args.seed,
        )
        
    def _generate_and_score(self, batch, is_eval=False):
    
        prompts = [example["prompt"] for example in batch]
        conversations = [example["conversations"] for example in batch]
        solutions = [example["ground_truth"] for example in batch]
        image_paths = [example["image_path"] for example in batch]

        # Replicate each prompt num_generations times
        expanded_prompts = [prompt for prompt in prompts for _ in range(self.num_generations)]
        expanded_conversations = [conv for conv in conversations for _ in range(self.num_generations)]
        expanded_solutions = [sol for sol in solutions for _ in range(self.num_generations)]
        expanded_image_paths = [img_path for img_path in image_paths for _ in range(self.num_generations)]

        inputs = self.processing_class.apply_chat_template(
            expanded_conversations,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
            text_kwargs={
                "padding": "longest",
                "truncation": True
            }
        ).to(self.model.device, dtype=torch.bfloat16)


        gen_config = self.eval_generation_config if is_eval else self.train_generation_config
        generated_ids = self.model.generate(**inputs, generation_config=gen_config)

        prompt_len = inputs["input_ids"].shape[1]
        completion_ids = generated_ids[:, prompt_len:]
        completions = self.processing_class.batch_decode(completion_ids, skip_special_tokens=True)


        rewards = torch.tensor(
            [combined_reward(prompt, completion, solution)
            for prompt, completion, solution in zip(expanded_prompts, completions, expanded_solutions)],
            device=self.model.device
        )
        
        self.global_step += 1
        if self.global_step % 5 == 0 and not is_eval:
            print("="*20)
            print("Step", self.global_step)
            print("-"*20)
            print(expanded_prompts[0])
            print(completions[0])

        return inputs["input_ids"], inputs["attention_mask"], completion_ids, rewards
    
    def compute_loss(self, model, batch, return_outputs=False, num_items_in_batch=None):
        prompt_ids, prompt_mask, completion_ids, rewards = self._generate_and_score(batch)

        # Reshape tensors to group completions by their original prompt
        batch_size = len(batch)
        prompt_ids = prompt_ids.view(batch_size, self.num_generations, -1)
        prompt_mask = prompt_mask.view(batch_size, self.num_generations, -1)
        completion_ids = completion_ids.view(batch_size, self.num_generations, -1)
        rewards = rewards.view(batch_size, self.num_generations)

        # Log reward statistics
        self.log({
            "reward_mean": rewards.mean().item(),
            "reward_std": rewards.std().item()
        })

        # Flatten tensors for model input
        flat_prompt_ids = prompt_ids.view(-1, prompt_ids.size(-1))
        flat_prompt_mask = prompt_mask.view(-1, prompt_mask.size(-1))
        flat_completion_ids = completion_ids.view(-1, completion_ids.size(-1))

        input_ids = torch.cat([flat_prompt_ids, flat_completion_ids], dim=1)
        attention_mask = torch.cat([flat_prompt_mask, torch.ones_like(flat_completion_ids)], dim=1)

        outputs = model(input_ids, attention_mask=attention_mask)
        completion_logits = outputs.logits[:, flat_prompt_ids.shape[1]:, :]
        log_probs = torch.log_softmax(completion_logits, dim=-1)
        target_log_probs = log_probs.gather(dim=-1, index=flat_completion_ids.unsqueeze(-1)).squeeze(-1)

        target_log_probs = target_log_probs.view(batch_size, self.num_generations, -1)
        mean_log_probs = target_log_probs.mean(dim=-1)

        advantages = rewards
        loss = - (mean_log_probs * advantages).mean()
        return loss
    
    def evaluate(self, eval_dataset, batch_size=64, output_path="./logs/superclevr_eval_results.json"):
    
        if eval_dataset is None:
            print("⚠️ No evaluation dataset provided. Skipping evaluation.")
            return

        self.model.eval()
        total_correct = 0
        total_samples = len(eval_dataset)
        all_outputs = []

        print("\n🔍 Running Evaluation...")

        # Batch-wise evaluation
        for i in tqdm(range(0, len(eval_dataset), batch_size)):
            batch_data = eval_dataset[i:i + batch_size]
            batch_messages = []

            for sample in batch_data:
                message = [{
                    "role": "user",
                    "content": [
                        {"type": "image", "image": f"file://{sample['image_path']}"},
                        {"type": "text", "text": sample["prompt"]}
                    ]
                }]
                batch_messages.append(message)

            text = [self.processing_class.apply_chat_template(msg, tokenize=False, add_generation_prompt=True) for msg in batch_messages]

            inputs = self.processing_class(
                text=text,
                images=[sample["image_path"] for sample in batch_data],
                padding=True,
                return_tensors="pt"
            )
            inputs = inputs.to("cuda")
            
            with torch.no_grad():
                generated_ids = self.model.generate(**inputs, generation_config=self.eval_generation_config)

            batch_output_text = self.processing_class.batch_decode(
                generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
            )

            all_outputs.extend(batch_output_text)

        # Compare outputs with ground truth
        accuracy = sum(1 for sample, output in zip(eval_dataset, all_outputs)
                       if self.extract_number_answer(output) == sample['ground_truth']) / total_samples * 100

        print(f"\n🏆 Evaluation Completed! Accuracy: {accuracy:.2f}%")

        with open(output_path, "w") as f:
            json.dump({'accuracy': accuracy, 'results': all_outputs}, f, indent=2)

        print(f"Results saved to {output_path}")

    @staticmethod
    def extract_number_answer(output_str):
        answer_pattern = r'<answer>\s*(\d+)\s*</answer>'
        match = re.search(answer_pattern, output_str)
        return int(match.group(1)) if match else None






