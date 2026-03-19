"""SkillRL: Recursive Skill-Augmented RL for Text Summarization.

Based on: "SkillRL: Evolving Agents via Recursive Skill-Augmented
Reinforcement Learning" (Xia et al., arXiv:2602.08234)

Key idea: Build a hierarchical SkillBank of summarization strategies,
inject relevant skills into prompts during GRPO training, and recursively
evolve the skill library based on validation failures.

Adapted for summarization:
- General Skills: universal summarization strategies (conciseness, key info extraction, etc.)
- Task-Specific Skills: category-level heuristics (sports, tech, science, etc.)
- Recursive Evolution: after each validation, update skills based on low-ROUGE failures
"""

import argparse
import copy
import json
from pathlib import Path

import torch
from datasets import Dataset
from rouge_score import rouge_scorer
from trl import GRPOConfig, GRPOTrainer

from data_utils import build_dataset, format_prompt
from model_utils import get_model_name, load_tokenizer, load_model


# ---------------------------------------------------------------------------
# SkillBank: Hierarchical skill library for summarization
# ---------------------------------------------------------------------------

DEFAULT_GENERAL_SKILLS = [
    {
        "skill_id": "G1",
        "title": "Trích xuất thông tin chính",
        "principle": "Xác định và giữ lại các sự kiện, con số, tên riêng quan trọng nhất trong văn bản gốc.",
        "when_to_apply": "Luôn áp dụng cho mọi bài tóm tắt.",
    },
    {
        "skill_id": "G2",
        "title": "Kiểm soát độ dài",
        "principle": "Bản tóm tắt nên ngắn gọn, khoảng 1-2 câu, tập trung vào nội dung cốt lõi. Tránh lặp lại hoặc thêm thông tin không có trong văn bản gốc.",
        "when_to_apply": "Luôn áp dụng.",
    },
    {
        "skill_id": "G3",
        "title": "Diễn đạt lại thay vì sao chép",
        "principle": "Tóm tắt bằng cách diễn đạt lại nội dung chính, không sao chép nguyên văn các câu dài từ văn bản gốc.",
        "when_to_apply": "Khi văn bản gốc có các câu dài và phức tạp.",
    },
    {
        "skill_id": "G4",
        "title": "Giữ tính chính xác",
        "principle": "Không thay đổi ý nghĩa, không thêm suy luận hoặc thông tin không có trong văn bản gốc.",
        "when_to_apply": "Luôn áp dụng, đặc biệt với tin tức và khoa học.",
    },
]

DEFAULT_TASK_SPECIFIC_SKILLS = {
    "khoa_hoc": [
        {
            "skill_id": "S-KH1",
            "title": "Tóm tắt nghiên cứu khoa học",
            "principle": "Ưu tiên: phát hiện chính → phương pháp → kết quả. Giữ lại các con số thống kê quan trọng.",
            "when_to_apply": "Bài báo/nghiên cứu khoa học.",
        },
    ],
    "kinh_te": [
        {
            "skill_id": "S-KT1",
            "title": "Tóm tắt tin kinh tế",
            "principle": "Ưu tiên: sự kiện chính → tác động → bối cảnh. Giữ lại các con số (%, tỷ lệ, giá trị).",
            "when_to_apply": "Tin tức kinh tế, tài chính.",
        },
    ],
    "the_thao": [
        {
            "skill_id": "S-TT1",
            "title": "Tóm tắt tin thể thao",
            "principle": "Ưu tiên: kết quả → người ghi bàn/điểm → ý nghĩa giải đấu. Giữ tỷ số chính xác.",
            "when_to_apply": "Tin thể thao, kết quả thi đấu.",
        },
    ],
    "cong_nghe": [
        {
            "skill_id": "S-CN1",
            "title": "Tóm tắt tin công nghệ",
            "principle": "Ưu tiên: sản phẩm/tính năng mới → thông số kỹ thuật chính → giá/ngày ra mắt.",
            "when_to_apply": "Ra mắt sản phẩm, cập nhật công nghệ.",
        },
    ],
    "phap_luat": [
        {
            "skill_id": "S-PL1",
            "title": "Tóm tắt tin pháp luật",
            "principle": "Ưu tiên: quyết định/luật mới → thay đổi chính → tác động. Giữ chính xác số liệu biểu quyết.",
            "when_to_apply": "Tin pháp luật, chính sách.",
        },
    ],
}

DEFAULT_COMMON_MISTAKES = [
    {
        "mistake_id": "M1",
        "description": "Tóm tắt quá ngắn, thiếu thông tin quan trọng",
        "avoidance": "Đảm bảo tóm tắt chứa ít nhất 2-3 thông tin chính từ văn bản gốc.",
    },
    {
        "mistake_id": "M2",
        "description": "Tóm tắt quá dài, chứa chi tiết không cần thiết",
        "avoidance": "Loại bỏ các chi tiết phụ, tập trung vào nội dung cốt lõi.",
    },
    {
        "mistake_id": "M3",
        "description": "Thêm thông tin không có trong văn bản gốc",
        "avoidance": "Chỉ sử dụng thông tin có trong văn bản gốc, không suy luận thêm.",
    },
]


class SkillBank:
    """Hierarchical skill library for summarization."""

    def __init__(
        self,
        general_skills: list[dict] | None = None,
        task_specific_skills: dict[str, list[dict]] | None = None,
        common_mistakes: list[dict] | None = None,
    ):
        self.general_skills = general_skills or copy.deepcopy(DEFAULT_GENERAL_SKILLS)
        self.task_specific_skills = task_specific_skills or copy.deepcopy(DEFAULT_TASK_SPECIFIC_SKILLS)
        self.common_mistakes = common_mistakes or copy.deepcopy(DEFAULT_COMMON_MISTAKES)

    def retrieve_skills(self, category: str | None = None, top_k: int = 4) -> str:
        """Retrieve relevant skills as a formatted string for prompt injection."""
        skills_text = "### Kỹ năng tóm tắt:\n"

        # General skills (always included)
        for skill in self.general_skills[:top_k]:
            skills_text += f"- [{skill['skill_id']}] {skill['title']}: {skill['principle']}\n"

        # Task-specific skills
        if category and category in self.task_specific_skills:
            for skill in self.task_specific_skills[category]:
                skills_text += f"- [{skill['skill_id']}] {skill['title']}: {skill['principle']}\n"

        # Common mistakes (include top ones)
        if self.common_mistakes:
            skills_text += "### Lỗi thường gặp:\n"
            for mistake in self.common_mistakes[:2]:
                skills_text += f"- {mistake['description']} → {mistake['avoidance']}\n"

        return skills_text

    def evolve(
        self,
        failures: list[dict],
        model=None,
        tokenizer=None,
        max_new_skills: int = 2,
    ):
        """Evolve the skill library based on validation failures.

        Analyzes failure patterns and generates new skills or updates
        existing ones. If model/tokenizer are provided, uses the model
        to generate new skills (online evolution). Otherwise, uses
        rule-based evolution.

        Args:
            failures: List of failed examples with 'input', 'output',
                      'generated', 'rouge2', 'category'.
            model: Optional model for generating new skills.
            tokenizer: Optional tokenizer.
            max_new_skills: Maximum new skills to add per evolution step.
        """
        if not failures:
            return

        # Analyze failure patterns by category
        category_failures: dict[str, list] = {}
        for f in failures:
            cat = f.get("category", "general")
            category_failures.setdefault(cat, []).append(f)

        for cat, cat_failures in category_failures.items():
            if len(cat_failures) < 2:
                continue

            # Analyze common patterns in failures
            too_short = sum(1 for f in cat_failures if len(f.get("generated", "").split()) < 10)
            too_long = sum(1 for f in cat_failures if len(f.get("generated", "").split()) > 100)
            low_rouge = sum(1 for f in cat_failures if f.get("rouge2", 0) < 0.1)

            new_skills = []

            if too_short > len(cat_failures) * 0.5:
                new_skills.append({
                    "skill_id": f"S-EV-{cat}-short",
                    "title": f"Tránh tóm tắt quá ngắn ({cat})",
                    "principle": "Bản tóm tắt cần đủ chi tiết, chứa ít nhất 15-30 từ với các thông tin chính.",
                    "when_to_apply": f"Khi tóm tắt bài viết thuộc loại {cat}.",
                })

            if too_long > len(cat_failures) * 0.5:
                new_skills.append({
                    "skill_id": f"S-EV-{cat}-long",
                    "title": f"Kiểm soát độ dài ({cat})",
                    "principle": "Giới hạn tóm tắt trong 1-2 câu, loại bỏ chi tiết phụ.",
                    "when_to_apply": f"Khi tóm tắt bài viết thuộc loại {cat}.",
                })

            if low_rouge > len(cat_failures) * 0.5:
                new_skills.append({
                    "skill_id": f"S-EV-{cat}-accuracy",
                    "title": f"Cải thiện độ chính xác ({cat})",
                    "principle": "Sử dụng chính xác các từ khóa và cụm từ quan trọng từ văn bản gốc.",
                    "when_to_apply": f"Khi tóm tắt bài viết thuộc loại {cat}.",
                })

            # Add new skills (limited)
            for skill in new_skills[:max_new_skills]:
                if cat not in self.task_specific_skills:
                    self.task_specific_skills[cat] = []
                # Avoid duplicate skill IDs
                existing_ids = {s["skill_id"] for s in self.task_specific_skills[cat]}
                if skill["skill_id"] not in existing_ids:
                    self.task_specific_skills[cat].append(skill)
                    print(f"  [SkillBank] Added new skill: {skill['skill_id']} - {skill['title']}")

    def save(self, path: str):
        """Save the skill bank to a JSON file."""
        data = {
            "general_skills": self.general_skills,
            "task_specific_skills": self.task_specific_skills,
            "common_mistakes": self.common_mistakes,
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    @classmethod
    def load(cls, path: str) -> "SkillBank":
        """Load skill bank from a JSON file."""
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return cls(
            general_skills=data.get("general_skills"),
            task_specific_skills=data.get("task_specific_skills"),
            common_mistakes=data.get("common_mistakes"),
        )


# ---------------------------------------------------------------------------
# SkillRL-augmented prompt formatting
# ---------------------------------------------------------------------------

def format_prompt_with_skills(
    text: str,
    category: str | None,
    skill_bank: SkillBank,
    top_k: int = 4,
) -> str:
    """Format prompt with skill injection for SkillRL."""
    skills_text = skill_bank.retrieve_skills(category, top_k=top_k)
    if category:
        return (
            f"{skills_text}\n"
            f"### Category: {category}\n"
            f"### Document:\n{text}\n\n"
            f"### Summary:\n"
        )
    return (
        f"{skills_text}\n"
        f"### Document:\n{text}\n\n"
        f"### Summary:\n"
    )


# ---------------------------------------------------------------------------
# SkillRL Trainer
# ---------------------------------------------------------------------------

class SkillRLTrainer:
    """SkillRL trainer for summarization with recursive skill evolution."""

    def __init__(
        self,
        model,
        tokenizer,
        raw_data: list[dict],
        skill_bank: SkillBank,
        grpo_config: GRPOConfig,
        evolution_threshold: float = 0.4,
        max_new_skills_per_evolution: int = 2,
        top_k_skills: int = 4,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.raw_data = raw_data
        self.skill_bank = skill_bank
        self.grpo_config = grpo_config
        self.evolution_threshold = evolution_threshold
        self.max_new_skills = max_new_skills_per_evolution
        self.top_k = top_k_skills
        self.scorer = rouge_scorer.RougeScorer(["rouge2"], use_stemmer=True)

    def _build_dataset_with_skills(self) -> tuple[Dataset, dict[str, str]]:
        """Build the training dataset with skill-augmented prompts."""
        prompts = []
        references = {}

        for item in self.raw_data:
            cat = item.get("category", None)
            prompt = format_prompt_with_skills(
                item["input"], cat, self.skill_bank, top_k=self.top_k
            )
            prompts.append({"prompt": prompt})
            references[prompt] = item["output"]

        return Dataset.from_list(prompts), references

    def _build_reward_functions(self, references: dict[str, str]):
        """Build reward functions."""
        scorer = self.scorer

        def rouge2_fn(completions, prompts=None, **kwargs):
            rewards = []
            prompt_list = prompts if prompts else [None] * len(completions)
            for completion, prompt in zip(completions, prompt_list):
                ref = references.get(prompt, "")
                if not ref or not completion.strip():
                    rewards.append(0.0)
                    continue
                score = scorer.score(ref, completion)
                rewards.append(score["rouge2"].fmeasure)
            return rewards

        def length_fn(completions, prompts=None, **kwargs):
            rewards = []
            for c in completions:
                n = len(c.split())
                if n == 0:
                    rewards.append(-1.0)
                elif n <= 80:
                    rewards.append(n / 80)
                elif n <= 200:
                    rewards.append(1.0 - (n - 80) / 120)
                else:
                    rewards.append(-0.5)
            return rewards

        return [rouge2_fn, length_fn], [0.7, 0.3]

    def _validate_and_collect_failures(self) -> list[dict]:
        """Run validation and collect failure cases for skill evolution."""
        print("  [SkillRL] Running validation for skill evolution...")
        failures = []
        self.model.eval()

        for item in self.raw_data[:50]:  # Validate on subset
            cat = item.get("category", None)
            prompt = format_prompt_with_skills(
                item["input"], cat, self.skill_bank, top_k=self.top_k
            )
            inputs = self.tokenizer(
                prompt, return_tensors="pt", truncation=True, max_length=512
            )
            inputs = {k: v.to(self.model.device) for k, v in inputs.items()}

            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=128,
                    temperature=0.7,
                    do_sample=True,
                    pad_token_id=self.tokenizer.pad_token_id,
                )

            generated = self.tokenizer.decode(
                outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True
            ).strip()

            ref = item["output"]
            score = self.scorer.score(ref, generated) if generated else None
            rouge2 = score["rouge2"].fmeasure if score else 0.0

            if rouge2 < self.evolution_threshold:
                failures.append({
                    "input": item["input"],
                    "output": ref,
                    "generated": generated,
                    "rouge2": rouge2,
                    "category": cat or "general",
                })

        self.model.train()
        print(f"  [SkillRL] Found {len(failures)}/{min(50, len(self.raw_data))} failures "
              f"(threshold={self.evolution_threshold})")
        return failures

    def train(self, num_evolution_rounds: int = 3):
        """Run SkillRL training with recursive skill evolution.

        Training alternates between:
        1. GRPO training with current skills
        2. Validation + failure analysis
        3. Skill evolution based on failures
        """
        for round_idx in range(num_evolution_rounds):
            print(f"\n{'='*50}")
            print(f"SkillRL Round {round_idx + 1}/{num_evolution_rounds}")
            print(f"{'='*50}")

            # Rebuild dataset with current skills
            train_dataset, references = self._build_dataset_with_skills()
            reward_fns, reward_weights = self._build_reward_functions(references)

            # Adjust output dir per round
            round_config = copy.deepcopy(self.grpo_config)
            round_config.output_dir = f"{self.grpo_config.output_dir}/round_{round_idx}"

            # Train
            trainer = GRPOTrainer(
                model=self.model,
                args=round_config,
                processing_class=self.tokenizer,
                train_dataset=train_dataset,
                reward_funcs=reward_fns,
                reward_weights=reward_weights,
            )
            trainer.train()

            # Validate and evolve skills (except last round)
            if round_idx < num_evolution_rounds - 1:
                failures = self._validate_and_collect_failures()
                if failures:
                    self.skill_bank.evolve(
                        failures,
                        model=self.model,
                        tokenizer=self.tokenizer,
                        max_new_skills=self.max_new_skills,
                    )

        print("\nSkillRL training complete!")

    def save(self, output_dir: str):
        """Save model and skill bank."""
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        self.model.save_pretrained(output_dir)
        self.tokenizer.save_pretrained(output_dir)
        self.skill_bank.save(f"{output_dir}/skill_bank.json")
        print(f"Model and SkillBank saved to {output_dir}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="SkillRL Training for Text Summarization"
    )
    parser.add_argument("--model", type=str, default="gpt2")
    parser.add_argument("--train_data", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="outputs/skillrl")
    parser.add_argument("--skill_bank", type=str, default=None,
                        help="Path to existing skill bank JSON")
    parser.add_argument("--num_evolution_rounds", type=int, default=3,
                        help="Number of train+evolve rounds")
    parser.add_argument("--evolution_threshold", type=float, default=0.4,
                        help="ROUGE-2 threshold for failure detection")
    parser.add_argument("--max_new_skills", type=int, default=2,
                        help="Max new skills per evolution round")
    parser.add_argument("--top_k_skills", type=int, default=4,
                        help="Number of general skills to inject per prompt")
    parser.add_argument("--num_epochs", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=8)
    parser.add_argument("--learning_rate", type=float, default=1e-5)
    parser.add_argument("--num_generations", type=int, default=4)
    parser.add_argument("--max_completion_length", type=int, default=256)
    parser.add_argument("--max_prompt_length", type=int, default=768)
    parser.add_argument("--use_lora", action="store_true", default=True)
    parser.add_argument("--no_lora", action="store_true")
    parser.add_argument("--lora_r", type=int, default=16)
    parser.add_argument("--lora_alpha", type=int, default=32)
    parser.add_argument("--load_in_4bit", action="store_true")
    parser.add_argument("--logging_steps", type=int, default=5)
    parser.add_argument("--save_steps", type=int, default=100)
    parser.add_argument("--report_to", type=str, default="none")
    args = parser.parse_args()

    use_lora = not args.no_lora
    model_name = get_model_name(args.model)
    print(f"Loading model: {model_name}")

    tokenizer = load_tokenizer(model_name)
    model = load_model(
        model_name,
        use_lora=use_lora,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        load_in_4bit=args.load_in_4bit,
    )

    # Load or create skill bank
    if args.skill_bank and Path(args.skill_bank).exists():
        skill_bank = SkillBank.load(args.skill_bank)
        print(f"Loaded SkillBank from {args.skill_bank}")
    else:
        skill_bank = SkillBank()
        print("Using default SkillBank")

    # Load data
    raw_dataset = build_dataset(args.train_data)
    raw_data = [raw_dataset[i] for i in range(len(raw_dataset))]

    grpo_config = GRPOConfig(
        output_dir=args.output_dir,
        num_train_epochs=args.num_epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        save_total_limit=3,
        num_generations=args.num_generations,
        max_completion_length=args.max_completion_length,
        max_prompt_length=args.max_prompt_length,
        report_to=args.report_to,
        bf16=True,
        gradient_checkpointing=True,
    )

    trainer = SkillRLTrainer(
        model=model,
        tokenizer=tokenizer,
        raw_data=raw_data,
        skill_bank=skill_bank,
        grpo_config=grpo_config,
        evolution_threshold=args.evolution_threshold,
        max_new_skills_per_evolution=args.max_new_skills,
        top_k_skills=args.top_k_skills,
    )

    trainer.train(num_evolution_rounds=args.num_evolution_rounds)
    trainer.save(f"{args.output_dir}/final")


if __name__ == "__main__":
    main()
