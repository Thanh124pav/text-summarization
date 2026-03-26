"""Style-conditioned SFT for Qwen 3-4B text summarization.

Trains a causal LM to generate summaries in a specified writing style.
Each training sample has a style label; the prompt includes the style
instruction so the model learns to condition its output on it.

Data format (JSONL):
  {"input": "...", "output": "...", "style": "bao_chi", "category": "kinh_te"}

The 'style' field is one of: bao_chi, hanh_chinh, khoa_hoc, chinh_luan, sinh_hoat.
If 'style' is missing, one is randomly assigned (useful for existing data).

Key design choices for Qwen 3-4B:
  - LoRA (r=32, alpha=64) on all linear layers — good balance for 4B model
  - bf16 (Qwen3 native dtype) instead of fp16
  - Prompt masking: labels = -100 on prompt tokens, only train on summary
  - Gradient checkpointing to fit in 24GB VRAM
  - Cosine LR schedule with 5% warmup
  - Optional chat template formatting for instruct models
"""

import argparse
import json
import random
from collections import Counter
from pathlib import Path

import torch
from datasets import Dataset, concatenate_datasets
from transformers import TrainingArguments
from trl import SFTTrainer, SFTConfig

from data_utils import (
    build_dataset,
    format_prompt,
    STYLE_NAMES,
    STYLE_DISPLAY_NAMES,
)
from logging_utils import SFTLoggingCallback
from model_utils import get_model_name, load_tokenizer, load_model


def build_sft_style_dataset(
    dataset: Dataset,
    tokenizer,
    assign_missing_styles: bool = True,
    use_chat_template: bool = False,
    seed: int = 42,
) -> Dataset:
    """Build SFT dataset with style-conditioned prompts.

    Two formatting modes:
      1. Plain format (default): uses format_prompt() with style instruction
      2. Chat template: uses tokenizer.apply_chat_template() for instruct models

    Args:
        dataset: Raw dataset with 'input', 'output', optional 'style'/'category'.
        tokenizer: Tokenizer (needed for chat template mode).
        assign_missing_styles: If True, randomly assign styles to samples
            missing the 'style' field.
        use_chat_template: If True, format as chat messages using the
            tokenizer's chat template (for instruct/chat models).
        seed: Random seed for style assignment.

    Returns:
        Dataset with 'text' column ready for SFTTrainer.
    """
    rng = random.Random(seed)

    def format_fn(examples):
        n = len(examples["input"])
        categories = examples.get("category", [None] * n)

        # Get or assign styles
        if "style" in examples:
            styles = [
                s if s else rng.choice(STYLE_NAMES)
                for s in examples["style"]
            ]
        elif assign_missing_styles:
            styles = [rng.choice(STYLE_NAMES) for _ in range(n)]
        else:
            styles = [None] * n

        texts = []
        for inp, out, cat, sty in zip(
            examples["input"], examples["output"], categories, styles
        ):
            if use_chat_template and hasattr(tokenizer, "apply_chat_template"):
                # Chat template format for instruct models
                style_instruction = ""
                if sty:
                    style_display = STYLE_DISPLAY_NAMES.get(sty, sty)
                    style_instruction = f"Hãy tóm tắt theo phong cách {style_display}. "

                cat_hint = f"[Chủ đề: {cat}] " if cat else ""

                messages = [
                    {
                        "role": "user",
                        "content": (
                            f"{style_instruction}{cat_hint}"
                            f"Tóm tắt văn bản sau:\n\n{inp}"
                        ),
                    },
                    {"role": "assistant", "content": out},
                ]
                text = tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=False
                )
            else:
                # Plain format
                prompt = format_prompt(inp, cat, sty)
                text = prompt + out

            texts.append(text)

        return {"text": texts}

    return dataset.map(
        format_fn, batched=True, remove_columns=dataset.column_names
    )


def balance_styles(dataset: Dataset, seed: int = 42) -> Dataset:
    """Upsample minority styles to balance the dataset.

    Ensures each style has roughly the same number of samples by
    oversampling the underrepresented styles.
    """
    if "style" not in dataset.column_names:
        return dataset

    style_counts = Counter(dataset["style"])
    if not style_counts:
        return dataset

    max_count = max(style_counts.values())
    print(f"  Style distribution before balancing: {dict(style_counts)}")

    rng = random.Random(seed)
    style_indices = {}
    for i, style in enumerate(dataset["style"]):
        style_indices.setdefault(style, []).append(i)

    balanced_indices = []
    for style, indices in style_indices.items():
        balanced_indices.extend(indices)
        # Oversample if needed
        deficit = max_count - len(indices)
        if deficit > 0:
            balanced_indices.extend(rng.choices(indices, k=deficit))

    rng.shuffle(balanced_indices)
    balanced = dataset.select(balanced_indices)

    new_counts = Counter(balanced["style"])
    print(f"  Style distribution after balancing:  {dict(new_counts)}")
    return balanced


def print_dataset_stats(dataset: Dataset, name: str = "Dataset"):
    """Print dataset statistics."""
    print(f"\n{name}: {len(dataset)} samples")
    if "style" in dataset.column_names:
        style_counts = Counter(dataset["style"])
        for style in STYLE_NAMES:
            count = style_counts.get(style, 0)
            display = STYLE_DISPLAY_NAMES.get(style, style)
            print(f"  {display:20s}: {count:5d} ({count/len(dataset)*100:.1f}%)")
    sample = dataset[0]
    text = sample.get("text", sample.get("input", ""))
    print(f"  Sample (first 200 chars): {text[:200]}...")


def main():
    parser = argparse.ArgumentParser(
        description="Style-conditioned SFT for text summarization (Qwen 3-4B)"
    )

    # Model
    parser.add_argument(
        "--model", type=str, default="Qwen/Qwen3-4B",
        help="Model name/path (default: Qwen/Qwen3-4B)"
    )

    # Data
    parser.add_argument("--train_data", type=str, required=True, help="Training JSONL")
    parser.add_argument("--val_data", type=str, default=None, help="Validation JSONL")
    parser.add_argument("--output_dir", type=str, default="outputs/sft_style")
    parser.add_argument(
        "--balance_styles", action="store_true",
        help="Upsample minority styles for balanced training"
    )
    parser.add_argument(
        "--use_chat_template", action="store_true",
        help="Use tokenizer chat template (for instruct/chat models)"
    )

    # Training hyperparameters (tuned for Qwen 3-4B + LoRA)
    parser.add_argument("--max_seq_length", type=int, default=2048)
    parser.add_argument("--num_epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=8)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--warmup_ratio", type=float, default=0.05)
    parser.add_argument("--weight_decay", type=float, default=0.01)

    # LoRA
    parser.add_argument("--use_lora", action="store_true", default=True)
    parser.add_argument("--no_lora", action="store_true", help="Disable LoRA")
    parser.add_argument("--lora_r", type=int, default=32, help="LoRA rank (32 for 4B model)")
    parser.add_argument("--lora_alpha", type=int, default=64, help="LoRA alpha")

    # Quantization
    parser.add_argument("--load_in_4bit", action="store_true")
    parser.add_argument("--load_in_8bit", action="store_true")

    # Logging
    parser.add_argument("--logging_steps", type=int, default=10)
    parser.add_argument("--save_steps", type=int, default=200)
    parser.add_argument("--eval_steps", type=int, default=200)
    parser.add_argument("--report_to", type=str, default="none", choices=["none", "wandb"])

    parser.add_argument("--seed", type=int, default=42)
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

    # Load raw data
    print("Loading training data...")
    raw_train = build_dataset(args.train_data)

    # Optional: balance styles
    if args.balance_styles and "style" in raw_train.column_names:
        raw_train = balance_styles(raw_train, seed=args.seed)

    # Format for SFT
    train_dataset = build_sft_style_dataset(
        raw_train,
        tokenizer=tokenizer,
        use_chat_template=args.use_chat_template,
        seed=args.seed,
    )
    print_dataset_stats(train_dataset, "Train")

    val_dataset = None
    if args.val_data:
        print("Loading validation data...")
        raw_val = build_dataset(args.val_data)
        val_dataset = build_sft_style_dataset(
            raw_val,
            tokenizer=tokenizer,
            use_chat_template=args.use_chat_template,
            seed=args.seed,
        )
        print_dataset_stats(val_dataset, "Val")

    # Detect bf16 support (Qwen3 native dtype)
    use_bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    use_fp16 = torch.cuda.is_available() and not use_bf16

    # Training config
    training_args = SFTConfig(
        output_dir=args.output_dir,
        num_train_epochs=args.num_epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        warmup_ratio=args.warmup_ratio,
        weight_decay=args.weight_decay,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        save_total_limit=3,
        bf16=use_bf16,
        fp16=use_fp16,
        max_length=args.max_seq_length,
        dataset_text_field="text",
        report_to=args.report_to,
        lr_scheduler_type="cosine",
        optim="adamw_torch",
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        eval_strategy="steps" if val_dataset else "no",
        eval_steps=args.eval_steps if val_dataset else None,
        load_best_model_at_end=bool(val_dataset),
        metric_for_best_model="eval_loss" if val_dataset else None,
        seed=args.seed,
        dataloader_pin_memory=True,
        dataloader_num_workers=4,
    )

    # Build demo prompts for sample generation during training
    demo_prompts = []
    for item in raw_train.select(range(min(3, len(raw_train)))):
        cat = item.get("category")
        sty = item.get("style") or random.choice(STYLE_NAMES)
        demo_prompts.append(format_prompt(item["input"], cat, sty))

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        processing_class=tokenizer,
        callbacks=[
            SFTLoggingCallback(
                tokenizer=tokenizer,
                demo_prompts=demo_prompts,
                sample_every=args.save_steps,
            ),
        ],
    )

    # Print config summary
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    effective_batch = args.batch_size * args.gradient_accumulation_steps
    total_steps = len(train_dataset) // effective_batch * args.num_epochs

    print(f"\n{'='*60}")
    print(f"SFT Style Training Config")
    print(f"{'='*60}")
    print(f"  Model:          {model_name}")
    print(f"  Total params:   {total_params:,}")
    print(f"  Trainable:      {trainable_params:,} ({trainable_params/total_params*100:.2f}%)")
    print(f"  LoRA:           r={args.lora_r}, alpha={args.lora_alpha}" if use_lora else "  LoRA: disabled")
    print(f"  Precision:      {'bf16' if use_bf16 else 'fp16' if use_fp16 else 'fp32'}")
    print(f"  Effective batch: {effective_batch}")
    print(f"  Total steps:    ~{total_steps}")
    print(f"  Epochs:         {args.num_epochs}")
    print(f"  LR:             {args.learning_rate}")
    print(f"  Max seq len:    {args.max_seq_length}")
    print(f"{'='*60}\n")

    # Train
    print("Starting SFT training...")
    trainer.train()

    # Save
    final_path = Path(args.output_dir) / "final"
    trainer.save_model(str(final_path))
    tokenizer.save_pretrained(str(final_path))
    print(f"\nModel saved to {final_path}")

    # Save training info
    info = {
        "model": model_name,
        "lora": {"r": args.lora_r, "alpha": args.lora_alpha} if use_lora else None,
        "styles": STYLE_NAMES,
        "style_display": STYLE_DISPLAY_NAMES,
        "train_samples": len(train_dataset),
        "val_samples": len(val_dataset) if val_dataset else 0,
        "epochs": args.num_epochs,
        "learning_rate": args.learning_rate,
        "max_seq_length": args.max_seq_length,
        "use_chat_template": args.use_chat_template,
    }
    with open(final_path / "training_info.json", "w", encoding="utf-8") as f:
        json.dump(info, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
