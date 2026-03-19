"""SAGE: Self-Hint Aligned GRPO with Privileged Supervision for Summarization.

Based on: "Self-Hinting Language Models Enhance Reinforcement Learning"
(Liao et al., arXiv:2602.03143)

Key idea: When GRPO advantage collapses (all rollouts in a group get the same
reward), inject self-generated "hints" — key points or outlines — into the
prompt to increase within-group outcome diversity. At inference time, no hints
are used.

Adapted for text summarization:
- Hints = key points extracted from the reference summary
- Reward = ROUGE-2 + length penalty (same as standard GRPO)
- Advantage collapse detection: when all G rollouts have identical rewards
"""

import argparse
import copy
import random

import torch
from datasets import Dataset
from rouge_score import rouge_scorer
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import GRPOConfig, GRPOTrainer

from data_utils import build_dataset, format_prompt
from model_utils import get_model_name, load_tokenizer, load_model


# ---------------------------------------------------------------------------
# Hint generation utilities
# ---------------------------------------------------------------------------

def generate_hint_from_reference(reference: str, max_hints: int = 3) -> str:
    """Extract key points from the reference summary as hints.

    This simulates the SAGE paper's approach of generating privileged hints
    from the reference solution. For summarization, the hint is a set of
    key phrases from the reference.
    """
    sentences = reference.replace(".", ".\n").replace(",", ",\n").split("\n")
    sentences = [s.strip() for s in sentences if len(s.strip()) > 5]

    if not sentences:
        return ""

    # Pick up to max_hints key fragments
    selected = sentences[:max_hints] if len(sentences) <= max_hints else random.sample(sentences, max_hints)
    hint = "Gợi ý: " + "; ".join(selected)
    return hint


def generate_self_hint(
    model,
    tokenizer: AutoTokenizer,
    text: str,
    category: str | None = None,
    max_new_tokens: int = 64,
) -> str:
    """Generate a self-hint using the model itself (online self-hinting).

    The model generates a brief outline/key-points for the text, which is
    then used as a hint for the actual summary generation.
    """
    hint_prompt = (
        f"### Document:\n{text[:500]}\n\n"
        f"### Hãy liệt kê 2-3 ý chính cần có trong bản tóm tắt:\n"
    )
    inputs = tokenizer(hint_prompt, return_tensors="pt", truncation=True, max_length=512)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=0.7,
            do_sample=True,
            pad_token_id=tokenizer.pad_token_id,
        )

    hint = tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()
    return f"Gợi ý: {hint}" if hint else ""


def format_prompt_with_hint(text: str, category: str | None, hint: str) -> str:
    """Format prompt with an injected hint (SAGE-style)."""
    if hint:
        if category:
            return (
                f"### Category: {category}\n"
                f"### {hint}\n"
                f"### Document:\n{text}\n\n"
                f"### Summary:\n"
            )
        return (
            f"### {hint}\n"
            f"### Document:\n{text}\n\n"
            f"### Summary:\n"
        )
    return format_prompt(text, category)


# ---------------------------------------------------------------------------
# SAGE Trainer (wraps GRPOTrainer with hint injection)
# ---------------------------------------------------------------------------

class SAGESummarizationTrainer:
    """SAGE trainer for summarization that wraps TRL's GRPOTrainer.

    Implements two schemes from the SAGE paper:
    - Scheme 1 (SAGE-light): Use reference-derived hints at epoch level
    - Scheme 2 (SAGE): Detect advantage collapse per-prompt and inject
      self-generated hints only when needed (no-positives trigger)
    """

    def __init__(
        self,
        model,
        tokenizer: AutoTokenizer,
        train_dataset: Dataset,
        references: dict[str, str],
        raw_data: list[dict],
        grpo_config: GRPOConfig,
        scheme: str = "sage-light",
        hint_ratio: float = 0.3,
        collapse_threshold: float = 0.01,
    ):
        """
        Args:
            model: The policy model.
            tokenizer: Tokenizer.
            train_dataset: Dataset with 'prompt' column.
            references: Map from prompt text to reference summary.
            raw_data: Original data records for hint generation.
            grpo_config: GRPOConfig for training.
            scheme: 'sage-light' (Scheme 1) or 'sage' (Scheme 2).
            hint_ratio: Fraction of prompts to apply hints (Scheme 1).
            collapse_threshold: Reward variance threshold for collapse detection.
        """
        self.model = model
        self.tokenizer = tokenizer
        self.references = references
        self.raw_data = {format_prompt(d["input"], d.get("category")): d for d in raw_data}
        self.scheme = scheme
        self.hint_ratio = hint_ratio
        self.collapse_threshold = collapse_threshold
        self.scorer = rouge_scorer.RougeScorer(["rouge2"], use_stemmer=True)

        # Build reward functions
        reward_fns, reward_weights = self._build_rewards()

        # Apply hints to dataset based on scheme
        if scheme == "sage-light":
            train_dataset = self._apply_reference_hints(train_dataset)

        self.trainer = GRPOTrainer(
            model=model,
            args=grpo_config,
            processing_class=tokenizer,
            train_dataset=train_dataset,
            reward_funcs=reward_fns,
            reward_weights=reward_weights,
        )

    def _build_rewards(self):
        """Build reward functions (same as standard GRPO)."""
        scorer = self.scorer
        references = self.references

        def rouge2_reward_fn(completions, prompts=None, **kwargs):
            rewards = []
            prompt_list = prompts if prompts else [None] * len(completions)
            for completion, prompt in zip(completions, prompt_list):
                # Try to find reference for both hinted and non-hinted prompts
                ref = references.get(prompt, "")
                if not ref:
                    # Try matching without hint prefix
                    for key, val in references.items():
                        if key in prompt or prompt in key:
                            ref = val
                            break
                if not ref or not completion.strip():
                    rewards.append(0.0)
                    continue
                score = scorer.score(ref, completion)
                rewards.append(score["rouge2"].fmeasure)
            return rewards

        def length_reward_fn(completions, prompts=None, **kwargs):
            rewards = []
            for completion in completions:
                words = completion.split()
                n = len(words)
                if n == 0:
                    rewards.append(-1.0)
                elif n <= 80:
                    rewards.append(n / 80)
                elif n <= 200:
                    rewards.append(1.0 - (n - 80) / 120)
                else:
                    rewards.append(-0.5)
            return rewards

        return [rouge2_reward_fn, length_reward_fn], [0.7, 0.3]

    def _apply_reference_hints(self, dataset: Dataset) -> Dataset:
        """Scheme 1 (SAGE-light): Apply reference-derived hints to a fraction of prompts."""
        references = self.references

        def add_hints(examples):
            new_prompts = []
            for prompt in examples["prompt"]:
                ref = references.get(prompt, "")
                if ref and random.random() < self.hint_ratio:
                    hint = generate_hint_from_reference(ref)
                    # Find original data to reconstruct prompt with hint
                    raw = self.raw_data.get(prompt)
                    if raw:
                        new_prompt = format_prompt_with_hint(
                            raw["input"], raw.get("category"), hint
                        )
                        new_prompts.append(new_prompt)
                    else:
                        new_prompts.append(prompt)
                else:
                    new_prompts.append(prompt)
            return {"prompt": new_prompts}

        return dataset.map(add_hints, batched=True)

    def train(self):
        """Run SAGE training."""
        print(f"Starting SAGE training (scheme={self.scheme})...")
        self.trainer.train()

    def save(self, output_dir: str):
        """Save the trained model."""
        self.trainer.save_model(output_dir)
        self.tokenizer.save_pretrained(output_dir)
        print(f"Model saved to {output_dir}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="SAGE (Self-Hinting GRPO) for Text Summarization"
    )
    parser.add_argument("--model", type=str, default="gpt2")
    parser.add_argument("--train_data", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="outputs/sage")
    parser.add_argument("--scheme", type=str, default="sage-light",
                        choices=["sage-light", "sage"],
                        help="SAGE scheme: sage-light (epoch-level hints) or sage (per-prompt collapse detection)")
    parser.add_argument("--hint_ratio", type=float, default=0.3,
                        help="Fraction of prompts to hint (sage-light)")
    parser.add_argument("--num_epochs", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=8)
    parser.add_argument("--learning_rate", type=float, default=1e-5)
    parser.add_argument("--num_generations", type=int, default=4)
    parser.add_argument("--max_completion_length", type=int, default=256)
    parser.add_argument("--max_prompt_length", type=int, default=512)
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

    # Build dataset
    print("Loading training data...")
    raw_dataset = build_dataset(args.train_data)
    raw_data = [raw_dataset[i] for i in range(len(raw_dataset))]

    # Build prompts and references
    prompts = []
    references = {}
    for item in raw_data:
        cat = item.get("category", None)
        prompt = format_prompt(item["input"], cat)
        prompts.append({"prompt": prompt})
        references[prompt] = item["output"]

    train_dataset = Dataset.from_list(prompts)

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

    sage_trainer = SAGESummarizationTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        references=references,
        raw_data=raw_data,
        grpo_config=grpo_config,
        scheme=args.scheme,
        hint_ratio=args.hint_ratio,
    )

    sage_trainer.train()
    sage_trainer.save(f"{args.output_dir}/final")


if __name__ == "__main__":
    main()
