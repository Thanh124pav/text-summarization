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
  4. Style reward: P(target_style | completion) from a pre-trained classifier
"""

import argparse
import math

import torch
from datasets import Dataset
from rouge_score import rouge_scorer
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from trl import GRPOConfig, GRPOTrainer

from data_utils import build_dataset, format_prompt, STYLE_NAMES
from model_utils import get_model_name, load_tokenizer, load_model


class StyleRewardModel:
    """Wraps a pre-trained PhoBERT/BERT style classifier as a reward signal.

    The classifier predicts P(style | text) for 5 Vietnamese writing styles.
    The reward is the probability assigned to the *target* style specified
    in the prompt.

    Args:
        model_path: Path to HuggingFace checkpoint or local directory.
        label_map: Mapping from style name -> classifier label index.
            Example: {"bao_chi": 0, "hanh_chinh": 1, "khoa_hoc": 2,
                       "chinh_luan": 3, "sinh_hoat": 4}
        device: "cuda", "cpu", or "auto".
        max_length: Max tokens for the classifier tokenizer.
    """

    def __init__(
        self,
        model_path: str,
        label_map: dict[str, int] | None = None,
        device: str = "auto",
        max_length: int = 256,
    ):
        self.max_length = max_length

        # Default label map (user can override)
        self.label_map = label_map or {
            name: i for i, name in enumerate(STYLE_NAMES)
        }

        # Load classifier
        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_path, trust_remote_code=True
        ).to(self.device)
        self.model.eval()

        print(f"Style classifier loaded from {model_path}")
        print(f"  Labels: {self.label_map}")
        print(f"  Device: {self.device}")

    @torch.no_grad()
    def predict_proba(self, texts: list[str]) -> torch.Tensor:
        """Get softmax probabilities for each style class.

        Returns:
            Tensor of shape (batch_size, num_classes).
        """
        inputs = self.tokenizer(
            texts,
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=self.max_length,
        ).to(self.device)

        logits = self.model(**inputs).logits
        return torch.softmax(logits, dim=-1).cpu()

    def get_style_reward(
        self,
        completions: list[str],
        target_styles: list[str],
    ) -> list[float]:
        """Compute P(target_style | completion) for each sample.

        Args:
            completions: Generated summaries.
            target_styles: Target style name for each completion.

        Returns:
            List of probabilities (0-1) as reward scores.
        """
        if not completions:
            return []

        probs = self.predict_proba(completions)  # (B, num_classes)

        rewards = []
        for i, style in enumerate(target_styles):
            label_idx = self.label_map.get(style)
            if label_idx is None:
                rewards.append(0.0)
            else:
                rewards.append(probs[i, label_idx].item())

        return rewards


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
    style_reward_model: StyleRewardModel | None = None,
    prompt_styles: dict[str, str] | None = None,
    style_weight: float = 0.3,
):
    """Create reward functions for GRPO training.

    Three reward modes:
      - "multiplicative" (GR3-style, no style):
          reward = length_gate × rouge_composite

      - "multiplicative_style" (GR3 + style classifier):
          reward = length_gate × (w_rouge × ROUGE + w_style × P(style))
          The style classifier probability is blended with ROUGE, then
          gated by length. This ensures all 3 objectives are coupled.

      - "additive" (legacy):
          reward = w1 × length + w2 × rouge [+ w3 × style]

    Args:
        tokenizer: Tokenizer for decoding.
        references: Mapping from prompt -> reference summary.
        min_length: Min acceptable summary length in tokens.
        max_length: Max acceptable summary length in tokens.
        reward_mode: "multiplicative", "multiplicative_style", or "additive".
        style_reward_model: Pre-trained StyleRewardModel instance (required
            for multiplicative_style mode).
        prompt_styles: Mapping from prompt -> target style name. Required
            when using style reward.
        style_weight: Weight for style reward in the quality blend.
            ROUGE weight becomes (1 - style_weight).

    Returns:
        Tuple of (reward_functions, reward_weights).
    """
    scorer = rouge_scorer.RougeScorer(
        ["rouge1", "rouge2", "rougeL"], use_stemmer=True
    )

    # Weights for ROUGE composite (from Multi-Dim Optimization paper)
    r1_w, r2_w, rl_w = 0.3, 0.4, 0.3
    rouge_weight = 1.0 - style_weight

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
        words = text.split()
        if len(words) > 3:
            unique_ratio = len(set(words)) / len(words)
            if unique_ratio < 0.3:
                return -0.5
        return 0.0

    if reward_mode == "multiplicative_style":
        # GR3 + Style: reward = length_gate × (w_r × ROUGE + w_s × P(style))
        assert style_reward_model is not None, (
            "style_reward_model is required for multiplicative_style mode"
        )
        assert prompt_styles is not None, (
            "prompt_styles mapping is required for multiplicative_style mode"
        )

        def combined_style_reward_fn(
            completions: list[str],
            prompts: list[str] | None = None,
            **kwargs,
        ) -> list[float]:
            """GR3 + Style reward.

            reward = length_gate × (w_rouge × ROUGE + w_style × P(target_style))

            Flow:
              1. Check format (degenerate → penalty)
              2. Compute ROUGE composite with reference
              3. Compute P(target_style) via classifier
              4. Blend: quality = w_rouge × ROUGE + w_style × style_prob
              5. Gate by length: final = length_gate × quality
            """
            prompt_list = prompts if prompts else [None] * len(completions)

            # Batch classify all completions for style at once (efficient)
            target_styles = [
                prompt_styles.get(p, STYLE_NAMES[0]) for p in prompt_list
            ]
            style_probs = style_reward_model.get_style_reward(
                completions, target_styles
            )

            rewards = []
            for i, (completion, prompt) in enumerate(
                zip(completions, prompt_list)
            ):
                # Format check
                fmt_penalty = _compute_format_penalty(completion)
                if fmt_penalty < 0:
                    rewards.append(fmt_penalty)
                    continue

                # ROUGE quality
                ref = references.get(prompt, "")
                rouge_score = _compute_rouge_composite(completion, ref)

                # Style quality
                style_prob = style_probs[i]

                # Blended quality score
                quality = rouge_weight * rouge_score + style_weight * style_prob

                # Length gate (multiplicative)
                n_tokens = len(
                    tokenizer.encode(completion, add_special_tokens=False)
                )
                gate = compute_length_gate(n_tokens, min_length, max_length)

                rewards.append(quality * gate)

            return rewards

        return [combined_style_reward_fn], [1.0]

    elif reward_mode == "multiplicative":
        # GR3-style without style: reward = length_gate × rouge_composite
        def combined_reward_fn(
            completions: list[str],
            prompts: list[str] | None = None,
            **kwargs,
        ) -> list[float]:
            """GR3-style: reward = length_gate × rouge_composite."""
            rewards = []
            prompt_list = prompts if prompts else [None] * len(completions)
            for completion, prompt in zip(completions, prompt_list):
                fmt_penalty = _compute_format_penalty(completion)
                if fmt_penalty < 0:
                    rewards.append(fmt_penalty)
                    continue

                ref = references.get(prompt, "")
                rouge_score = _compute_rouge_composite(completion, ref)

                n_tokens = len(
                    tokenizer.encode(completion, add_special_tokens=False)
                )
                gate = compute_length_gate(n_tokens, min_length, max_length)

                rewards.append(rouge_score * gate)

            return rewards

        return [combined_reward_fn], [1.0]

    else:
        # Legacy additive mode
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

        reward_fns = [length_reward_fn, rouge_reward_fn]
        weights = [0.2, 0.5]

        if style_reward_model is not None and prompt_styles is not None:
            def style_reward_fn(
                completions: list[str],
                prompts: list[str] | None = None,
                **kwargs,
            ) -> list[float]:
                prompt_list = prompts if prompts else [None] * len(completions)
                target_styles = [
                    prompt_styles.get(p, STYLE_NAMES[0]) for p in prompt_list
                ]
                return style_reward_model.get_style_reward(
                    completions, target_styles
                )

            reward_fns.append(style_reward_fn)
            weights.append(style_weight)

        return reward_fns, weights


def build_grpo_dataset(
    dataset: Dataset,
    with_style: bool = False,
    seed: int = 42,
) -> tuple[Dataset, dict[str, str], dict[str, str] | None]:
    """Build prompt dataset and reference/style mappings for GRPO.

    Args:
        dataset: Raw dataset with 'input', 'output', optional 'category'.
        with_style: If True, randomly assign a style to each prompt.
        seed: Random seed for style assignment.

    Returns:
        (prompt_dataset, references_map, prompt_styles_map)
        prompt_styles_map is None when with_style=False.
    """
    import random

    rng = random.Random(seed)

    prompts = []
    references = {}
    prompt_styles = {} if with_style else None

    for item in dataset:
        cat = item.get("category", None)
        style = rng.choice(STYLE_NAMES) if with_style else None
        prompt = format_prompt(item["input"], cat, style)
        prompts.append({"prompt": prompt})
        references[prompt] = item["output"]
        if with_style:
            prompt_styles[prompt] = style

    return Dataset.from_list(prompts), references, prompt_styles


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
                        choices=["multiplicative", "multiplicative_style", "additive"],
                        help="Reward mode: multiplicative (GR3), multiplicative_style (GR3+style), additive (legacy)")

    # Style reward options
    parser.add_argument("--style_classifier", type=str, default=None,
                        help="Path to pre-trained PhoBERT/BERT style classifier checkpoint")
    parser.add_argument("--style_weight", type=float, default=0.3,
                        help="Weight for style reward in quality blend (ROUGE gets 1-style_weight)")
    parser.add_argument("--style_max_length", type=int, default=256,
                        help="Max token length for style classifier input")
    parser.add_argument("--label_map", type=str, default=None,
                        help='JSON string for label mapping, e.g. \'{"bao_chi":0,"hanh_chinh":1,...}\'')
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

    # Determine if style reward is used
    use_style = args.style_classifier is not None or args.reward_mode == "multiplicative_style"

    # Build dataset (with style assignment if needed)
    print("Loading training data...")
    raw_dataset = build_dataset(args.train_data)
    train_dataset, references, prompt_styles = build_grpo_dataset(
        raw_dataset, with_style=use_style,
    )

    # Load style classifier if requested
    style_model = None
    if args.style_classifier:
        import json as _json
        label_map = None
        if args.label_map:
            label_map = _json.loads(args.label_map)
        style_model = StyleRewardModel(
            model_path=args.style_classifier,
            label_map=label_map,
            max_length=args.style_max_length,
        )
    elif args.reward_mode == "multiplicative_style":
        parser.error("--style_classifier is required for multiplicative_style mode")

    # Create reward functions
    reward_fns, reward_weights = make_reward_functions(
        tokenizer=tokenizer,
        references=references,
        min_length=args.min_summary_length,
        max_length=args.max_summary_length,
        reward_mode=args.reward_mode,
        style_reward_model=style_model,
        prompt_styles=prompt_styles,
        style_weight=args.style_weight,
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
    if use_style:
        print(f"  Style classifier: {args.style_classifier}")
        print(f"  Style weight: {args.style_weight} (ROUGE weight: {1 - args.style_weight})")
    trainer.train()

    # Save
    final_path = f"{args.output_dir}/final"
    trainer.save_model(final_path)
    tokenizer.save_pretrained(final_path)
    print(f"Model saved to {final_path}")


if __name__ == "__main__":
    main()
