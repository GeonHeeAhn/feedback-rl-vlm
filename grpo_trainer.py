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
    """
    PyTorch의 Sampler를 상속받은 커스텀 샘플러(데이터셋에서 어떤 순서로 데이터를 뽑아올지)
    - mini_repeat_count: 하나의 index를 batch 안에서 몇 번 반복할지
    - batch_size: 한 배치당 몇 개의 고유한 index를 뽑을지 (중복을 세지 않고)
    - repeat_count: 전체 샘플링 과정을 몇 번 반복할지
    - seed: 랜덤 시드를 고정해서 항상 같은 결과가 나오게 할지
    
    1. batch_size만큼의 "고유 index"를 랜덤하게 뽑고
       batch_size=3이면, [4, 3, 0] 같이 3개의 index를 랜덤하게 뽑음
    2. 뽑힌 index들을 mini_repeat_count만큼 반복
       mini_repeat_count=2면, [4, 4, 3, 3, 0, 0] 이런 식으로 같은 index를 2번씩 반복
    3. 위 과정을 repeat_count번 반복
    4. 최종적으로 뽑힌 index들을 차례로 반환
    
    GRPO는 동일한 데이터를 여러 번 사용하면서 안정적으로 policy를 개선하는게 목표
    : 같은 샘플을 여러번 학습 / batch 내에서도 복제된 데이터가 필요하기에 기존 Sampler 말고 RepeatRandomSampler()이용
    """
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
        #batch_size만큼 index를 잘라서 chunk 단위로 묶음
        indexes = [indexes[i : i + self.batch_size] for i in range(0, len(indexes), self.batch_size)]
        indexes = [chunk for chunk in indexes if len(chunk) == self.batch_size]
        for chunk in indexes:
            for _ in range(self.repeat_count):
                for index in chunk:
                    for _ in range(self.mini_repeat_count):
                        yield index

    #샘플러의 총 길이 반환 - DataLoader가 몇 개의 샘플을 꺼낼지
    def __len__(self) -> int:
        return self.num_samples * self.mini_repeat_count * self.repeat_count

def accuracy_reward(prompt, completion, solution, **kwargs):
    
    reward = 0.0
    try:
        if isinstance(solution, list): #e.g) [5]
            solution = solution[0]
        #re.search() : finds the first matching substring and returns a Match object.
        sol_match = re.search(r'<answer>\s*(\d+)\s*</answer>', str(solution))
        ground_truth = sol_match.group(1).strip() if sol_match else str(solution).strip()

        content_match = re.search(r'<answer>\s*(\d+)\s*</answer>', completion)
        if content_match:
            student_answer = content_match.group(1).strip()
        else: #without tag
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


    return format_score * 0.3 + accuracy_score * 0.7


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
        self.num_generations = args.num_generations #하나의 prompt로 몇 개의 샘플을 생성할지
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

    #train()시 자동 호출(hook처럼)
    def _get_train_sampler(self):
        #모델이 한 번의 optimizer.step() 전에 처리하는 총 샘플 수
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

        # Replicate each prompt num_generations times: currently 2
        expanded_prompts = [prompt for prompt in prompts for _ in range(self.num_generations)]
        expanded_conversations = [conv for conv in conversations for _ in range(self.num_generations)]
        expanded_solutions = [sol for sol in solutions for _ in range(self.num_generations)]
        #expanded_image_paths = [img_path for img_path in image_paths for _ in range(self.num_generations)]
        
        inputs = self.processing_class.apply_chat_template(
            expanded_conversations,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
            text_kwargs={"padding": "longest", "truncation": True}
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
        
        eos_token_id = self.processing_class.tokenizer.eos_token_id
        is_eos = completion_ids == eos_token_id
        eos_idx = torch.full((is_eos.size(0),), is_eos.size(1), dtype=torch.long, device=self.model.device)
        eos_idx[is_eos.any(dim=1)] = is_eos.int().argmax(dim=1)[is_eos.any(dim=1)]
        seq_idx = torch.arange(is_eos.size(1), device=self.model.device).unsqueeze(0)
        completion_mask = (seq_idx <= eos_idx.unsqueeze(1)).int()

        
        self.global_step += 1
        if self.global_step % 5 == 0 and not is_eval:
            print("="*20)
            print("Step", self.global_step)
            print("-"*20)
            print(expanded_prompts[0])
            print(completions[0])

        return inputs["input_ids"], inputs["attention_mask"], completion_ids, completion_mask, rewards
    
    def compute_loss(self, model, batch, return_outputs=False, num_items_in_batch=None):
        prompt_ids, prompt_mask, completion_ids, completion_mask, rewards = self._generate_and_score(batch)

        # Reshape tensors to group completions by their original prompt
        batch_size = len(batch)
        prompt_ids = prompt_ids.view(batch_size, self.num_generations, -1)
        prompt_mask = prompt_mask.view(batch_size, self.num_generations, -1)
        completion_ids = completion_ids.view(batch_size, self.num_generations, -1)
        completion_mask = completion_mask.view(batch_size, self.num_generations, -1)
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
        flat_completion_mask = completion_mask.view(-1, completion_mask.size(-1))

        input_ids = torch.cat([flat_prompt_ids, flat_completion_ids], dim=1)
        attention_mask = torch.cat([flat_prompt_mask, flat_completion_mask], dim=1)

        outputs = model(input_ids, attention_mask=attention_mask)
        completion_logits = outputs.logits[:, flat_prompt_ids.shape[1]:, :]
        log_probs = torch.log_softmax(completion_logits, dim=-1)
        target_log_probs = log_probs.gather(dim=-1, index=flat_completion_ids.unsqueeze(-1)).squeeze(-1)
        
        #reward를 prompt 그룹 단위로 묶어서 평균/표준편차 계산
        mean_rewards = rewards.mean(dim=1, keepdim=True) #dim=1 : num_generations diemension 기준으로 그룹화
        std_rewards = rewards.std(dim=1, keepdim=True) + 1e-4
        advantages = (rewards - mean_rewards) / std_rewards
        advantages = advantages.view(-1, 1)

        target_log_probs = target_log_probs.sum(dim=1, keepdim=True)
        loss = - (target_log_probs * advantages).mean()
        return loss
    

    def evaluate(self, batch_size=16, output_path="./logs/superclevr_eval_results.json"):

        if self.eval_dataset is None:
            print("⚠️ No evaluation dataset provided. Skipping evaluation.")
            return

        self.model.eval()
        total_samples = len(self.eval_dataset)
        all_outputs = []
        accuracy = 0
        correct_count = 0
        
        print("\n🔍 Running Evaluation...")


        for i in tqdm(range(0, len(self.eval_dataset), batch_size)):
            batch_data = self.eval_dataset[i:i + batch_size]
            batch_conversations = [sample["conversations"] for sample in batch_data]
            inputs = self.processing_class.apply_chat_template(
                batch_conversations,
                add_generation_prompt=True,
                tokenize=True,
                return_dict=True,
                return_tensors="pt",
                padding="longest"
            ).to(self.model.device,dtype=torch.bfloat16)

            with torch.no_grad():
                generated_ids = self.model.generate(**inputs, generation_config=self.eval_generation_config)
            prompt_len = inputs["input_ids"].shape[1]
            generated_ids_trimmed = generated_ids[:, prompt_len:]

            batch_output_text = self.processing_class.batch_decode(
                generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
            )
            all_outputs.extend(batch_output_text)
            
            for sample, output in zip(batch_data, batch_output_text):
                #predicted = self.extract_number_answer(output)
                #if <answer> tag included
                match = re.search(r'<answer>\s*(\d+)\s*</answer>', output)
                
                if match:
                    predicted = int(match.group(1))
                else:
                    #if no tag
                    num_match = re.search(r'\d+', output)
                    predicted = int(num_match.group(0)) if num_match else None

                ground_truth = int(sample['ground_truth'])
                is_correct = predicted == ground_truth

                if is_correct:
                    correct_count += 1

                print("=" * 40)
                print(f"🔹 Question: {sample['prompt']}")
                print(f"🤖 Model Output: {output.strip()}")
                print(f"✅ Ground Truth: {ground_truth}")
                print(f"🎯 Correct? {'Yes ✅' if is_correct else 'No ❌'}")
                print(f"📊 Running Accuracy: {correct_count / (len(all_outputs)) * 100:.2f}%")
                accuracy = correct_count / total_samples * 100
        """
        accuracy = sum(
            1 for sample, output in zip(self.eval_dataset, all_outputs)
            #if self.extract_number_answer(output) == sample['ground_truth']
            if output.strip() == sample['ground_truth']
        ) / total_samples * 100
        """

        print(f"\n🏆 Evaluation Completed! Accuracy: {accuracy:.2f}%")

        with open(output_path, "w") as f:
            json.dump({'accuracy': accuracy, 'results': all_outputs}, f, indent=2)

        print(f"Results saved to {output_path}")

    @staticmethod
    def extract_number_answer(output_str):
        answer_pattern = r'<answer>\s*(\d+)\s*</answer>'
        match = re.search(answer_pattern, output_str)
        return int(match.group(1)) if match else None






