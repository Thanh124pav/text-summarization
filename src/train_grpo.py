"""GRPO (Group Relative Policy Optimization) training for text summarization.

Reward design based on:
  - GR3 (arxiv:2603.10535): multiplicative length gating instead of additive
  - Multi-Dim Optimization (arxiv:2406.00303): multi-signal ROUGE + semantic
  - Topic-Guided RL (arxiv:2509.09852): ROUGE-L + topic alignment
  - DeepSeek-R1 (arxiv:2501.12948): rule-based reward signals

Reward components:
  1. ROUGE composite: ROUGE-1, ROUGE-2, ROUGE-L F1 weighted average
  2. Length gate (GR3-style): multiplicative bell-curve gate, NOT additive
  3. Format reward: penalizes empty/degenerate outputs
  4. (Optional) BERTScore: semantic similarity for abstractive quality
"""

import argparse
import math

from datasets import Dataset
from rouge_score import rouge_scorer
from transformers import AutoTokenizer
from trl import GRPOConfig, GRPOTrainer

from data_utils import build_dataset, format_prompt
from model_utils import get_model_name, load_tokenizer, load_model


def compute_length_gate(n_tokens: int, min_len: int, max_len: int) -> float:
    """GR3-style multiplicative length gate (arxiv:2603.10535).

    Returns a value in [0, 1] that multiplies the quality reward.
    This prevents reward hacking via length — model MUST produce good
    content AND correct length to get high reward.

    Shape:
      - Peak (1.0) across the sweet spot [min_len, max_len]
      - Smooth ramp-up from 0 to min_len
      - Smooth decay from max_len, hitting 0 at 2×max_len
      - Hard zero for empty or extremely long outputs
    """
    if n_tokens < 1:
        return 0.0

    # Ramp up: 0 → 1 as tokens go from 0 → min_len
    if n_tokens < min_len:
        return n_tokens / min_len

    # Sweet spot: full reward in [min_len, max_len]
    if n_tokens <= max_len:
        return 1.0

    # Smooth decay: cosine falloff from max_len to 2×max_len
    overshoot = (n_tokens - max_len) / max_len
    if overshoot >= 1.0:
        return 0.0
    return 0.5 * (1.0 + math.cos(math.pi * overshoot))


def make_reward_functions(
    tokenizer: AutoTokenizer,
    references: dict[str, str],
    min_length: int = 30,
    max_length: int = 150,
    reward_mode: str = "multiplicative",
):
    """Create reward functions for GRPO training.

    Two modes:
      - "multiplicative" (recommended, GR3-style):
          reward = length_gate × rouge_composite
          Single combined reward function. Length acts as a gate, not a
          separate objective. Prevents length-hacking.

      - "additive" (legacy):
          reward = w1 × length_reward + w2 × rouge_reward
          Separate functions with weights. Simpler but vulnerable to
          reward hacking.

    Args:
        tokenizer: Tokenizer for decoding.
        references: Mapping from prompt to reference summary.
        min_length: Minimum acceptable summary length in tokens.
        max_length: Maximum acceptable summary length in tokens.
        reward_mode: "multiplicative" (GR3) or "additive" (legacy).

    Returns:
        Tuple of (reward_functions, reward_weights).
    """
    scorer = rouge_scorer.RougeScorer(
        ["rouge1", "rouge2", "rougeL"], use_stemmer=True
    )

    # Weights for ROUGE composite (from Multi-Dim Optimization paper)
    r1_w, r2_w, rl_w = 0.3, 0.4, 0.3

    def _compute_rouge_composite(completion: str, reference: str) -> float:
        """Weighted ROUGE composite: 0.3×R1 + 0.4×R2 + 0.3×RL F1."""
        if not reference or not completion.strip():
            return 0.0
        scores = scorer.score(reference, completion)
        return (
            r1_w * scores["rouge1"].fmeasure
            + r2_w * scores["rouge2"].fmeasure
            + rl_w * scores["rougeL"].fmeasure
        )

    def _compute_format_penalty(completion: str) -> float:
        """Penalize degenerate outputs: empty, repetitive, or too short."""
        text = completion.strip()
        if len(text) < 5:
            return -1.0
        # Detect degenerate repetition (same token repeated)
        words = text.split()
        if len(words) > 3:
            unique_ratio = len(set(words)) / len(words)
            if unique_ratio < 0.3:
                return -0.5
        return 0.0

    if reward_mode == "multiplicative":
        # GR3-style: single combined reward function
        def combined_reward_fn(
            completions: list[str],
            prompts: list[str] | None = None,
            **kwargs,
        ) -> list[float]:
            """GR3-style: reward = length_gate × rouge_composite + format_penalty.

            The length gate is multiplicative — even perfect ROUGE gets
            zero reward if length is out of bounds. This eliminates the
            incentive to hack length independently.
            """
            rewards = []
            prompt_list = prompts if prompts else [None] * len(completions)
            for completion, prompt in zip(completions, prompt_list):
                # Format check (degenerate output detection)
                fmt_penalty = _compute_format_penalty(completion)
                if fmt_penalty < 0:
                    rewards.append(fmt_penalty)
                    continue

                # ROUGE quality score
                ref = references.get(prompt, "")
                rouge_score = _compute_rouge_composite(completion, ref)

                # Length gate (multiplicative, GR3-style)
                n_tokens = len(
                    tokenizer.encode(completion, add_special_tokens=False)
                )
                gate = compute_length_gate(n_tokens, min_length, max_length)

                # Final reward: quality × length_gate
                rewards.append(rouge_score * gate)

            return rewards

        return [combined_reward_fn], [1.0]

    else:
        # Legacy additive mode (kept for comparison/ablation)
        def length_reward_fn(
            completions: list[str],
            prompts: list[str] | None = None,
            **kwargs,
        ) -> list[float]:
            rewards = []
            for completion in completions:
                n_tokens = len(
                    tokenizer.encode(completion, add_special_tokens=False)
                )
                gate = compute_length_gate(n_tokens, min_length, max_length)
                rewards.append(gate)
            return rewards

        def rouge_reward_fn(
            completions: list[str],
            prompts: list[str] | None = None,
            **kwargs,
        ) -> list[float]:
            rewards = []
            prompt_list = prompts if prompts else [None] * len(completions)
            for completion, prompt in zip(completions, prompt_list):
                ref = references.get(prompt, "")
                fmt = _compute_format_penalty(completion)
                if fmt < 0:
                    rewards.append(fmt)
                    continue
                rewards.append(_compute_rouge_composite(completion, ref))
            return rewards

        return [length_reward_fn, rouge_reward_fn], [0.3, 0.7]


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
    parser.add_argument("--min_summary_length", type=int, default=30, help="Min summary length in tokens")
    parser.add_argument("--max_summary_length", type=int, default=150, help="Max summary length in tokens")
    parser.add_argument("--reward_mode", type=str, default="multiplicative",
                        choices=["multiplicative", "additive"],
                        help="Reward mode: multiplicative (GR3, recommended) or additive (legacy)")
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
        min_length=args.min_summary_length,
        max_length=args.max_summary_length,
        reward_mode=args.reward_mode,
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
    print(f"  Reward mode: {args.reward_mode}")
    print(f"  Length range: [{args.min_summary_length}, {args.max_summary_length}] tokens")
    trainer.train()

    # Save
    final_path = f"{args.output_dir}/final"
    trainer.save_model(final_path)
    tokenizer.save_pretrained(final_path)
    print(f"Model saved to {final_path}")


if __name__ == "__main__":
    main()
