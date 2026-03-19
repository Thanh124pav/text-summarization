"""GRPO (Group Relative Policy Optimization) training for text summarization.

Rewards:
  1. Length reward: penalizes summaries that are too long or too short
  2. ROUGE-2 reward: measures bigram overlap with reference summary
"""

import argparse

from datasets import Dataset
from rouge_score import rouge_scorer
from transformers import AutoTokenizer
from trl import GRPOConfig, GRPOTrainer

from data_utils import build_dataset, format_prompt
from model_utils import get_model_name, load_tokenizer, load_model


def make_reward_functions(
    tokenizer: AutoTokenizer,
    references: dict[str, str],
    target_length: int = 80,
    max_length: int = 200,
    length_weight: float = 0.3,
    rouge_weight: float = 0.7,
):
    """Create reward functions for GRPO training.

    Args:
        tokenizer: Tokenizer for decoding.
        references: Mapping from prompt to reference summary.
        target_length: Ideal summary length in tokens.
        max_length: Maximum acceptable length in tokens.
        length_weight: Weight for length reward.
        rouge_weight: Weight for ROUGE-2 reward.

    Returns:
        List of reward functions.
    """
    scorer = rouge_scorer.RougeScorer(["rouge2"], use_stemmer=True)

    def length_reward_fn(completions: list[str], prompts: list[str] | None = None, **kwargs) -> list[float]:
        """Reward based on summary length — penalizes too short or too long."""
        rewards = []
        for completion in completions:
            tokens = tokenizer.encode(completion, add_special_tokens=False)
            n_tokens = len(tokens)

            if n_tokens == 0:
                rewards.append(-1.0)
            elif n_tokens <= target_length:
                # Linearly scale from 0 to 1 as length approaches target
                rewards.append(n_tokens / target_length)
            elif n_tokens <= max_length:
                # Linearly decay from 1 to 0 between target and max
                rewards.append(1.0 - (n_tokens - target_length) / (max_length - target_length))
            else:
                # Penalty for exceeding max length
                rewards.append(-0.5)

        return rewards

    def rouge2_reward_fn(completions: list[str], prompts: list[str] | None = None, **kwargs) -> list[float]:
        """Reward based on ROUGE-2 F1 score with reference summaries."""
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

    return [length_reward_fn, rouge2_reward_fn], [length_weight, rouge_weight]


def build_grpo_dataset(dataset: Dataset) -> tuple[Dataset, dict[str, str]]:
    """Build prompt dataset and reference mapping for GRPO."""
    prompts = []
    references = {}

    for item in dataset:
        cat = item.get("category", None)
        prompt = format_prompt(item["input"], cat)
        prompts.append({"prompt": prompt})
        references[prompt] = item["output"]

    return Dataset.from_list(prompts), references


def main():
    parser = argparse.ArgumentParser(description="GRPO Training for Text Summarization")
    parser.add_argument("--model", type=str, default="gpt2", help="Model name or key")
    parser.add_argument("--train_data", type=str, required=True, help="Path to training JSONL file")
    parser.add_argument("--output_dir", type=str, default="outputs/grpo", help="Output directory")
    parser.add_argument("--num_epochs", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=8)
    parser.add_argument("--learning_rate", type=float, default=1e-5)
    parser.add_argument("--num_generations", type=int, default=4, help="Number of completions per prompt in GRPO")
    parser.add_argument("--max_completion_length", type=int, default=256)
    parser.add_argument("--max_prompt_length", type=int, default=512)
    parser.add_argument("--target_length", type=int, default=80, help="Target summary length in tokens")
    parser.add_argument("--max_summary_length", type=int, default=200, help="Max summary length for length reward")
    parser.add_argument("--length_weight", type=float, default=0.3, help="Weight for length reward")
    parser.add_argument("--rouge_weight", type=float, default=0.7, help="Weight for ROUGE-2 reward")
    parser.add_argument("--use_lora", action="store_true", default=True)
    parser.add_argument("--no_lora", action="store_true")
    parser.add_argument("--lora_r", type=int, default=16)
    parser.add_argument("--lora_alpha", type=int, default=32)
    parser.add_argument("--load_in_4bit", action="store_true")
    parser.add_argument("--load_in_8bit", action="store_true")
    parser.add_argument("--logging_steps", type=int, default=5)
    parser.add_argument("--save_steps", type=int, default=100)
    parser.add_argument("--report_to", type=str, default="none", choices=["none", "wandb"])
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
        load_in_8bit=args.load_in_8bit,
    )

    # Build dataset
    print("Loading training data...")
    raw_dataset = build_dataset(args.train_data)
    train_dataset, references = build_grpo_dataset(raw_dataset)

    # Create reward functions
    reward_fns, reward_weights = make_reward_functions(
        tokenizer=tokenizer,
        references=references,
        target_length=args.target_length,
        max_length=args.max_summary_length,
        length_weight=args.length_weight,
        rouge_weight=args.rouge_weight,
    )

    # GRPO config
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

    trainer = GRPOTrainer(
        model=model,
        args=grpo_config,
        processing_class=tokenizer,
        train_dataset=train_dataset,
        reward_funcs=reward_fns,
        reward_weights=reward_weights,
    )

    print("Starting GRPO training...")
    print(f"  Reward weights: length={args.length_weight}, rouge2={args.rouge_weight}")
    trainer.train()

    # Save
    final_path = f"{args.output_dir}/final"
    trainer.save_model(final_path)
    tokenizer.save_pretrained(final_path)
    print(f"Model saved to {final_path}")


if __name__ == "__main__":
    main()
