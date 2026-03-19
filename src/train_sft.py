"""Supervised Finetuning (SFT) for text summarization."""

import argparse
from pathlib import Path

from datasets import Dataset
from transformers import TrainingArguments
from trl import SFTTrainer, SFTConfig

from data_utils import build_dataset, format_prompt
from model_utils import get_model_name, load_tokenizer, load_model


def build_sft_dataset(dataset: Dataset) -> Dataset:
    """Convert dataset to text format for SFTTrainer."""

    def format_fn(examples):
        texts = []
        categories = examples.get("category", [None] * len(examples["input"]))
        for inp, out, cat in zip(examples["input"], examples["output"], categories):
            prompt = format_prompt(inp, cat)
            texts.append(prompt + out)
        return {"text": texts}

    return dataset.map(format_fn, batched=True, remove_columns=dataset.column_names)


def main():
    parser = argparse.ArgumentParser(description="SFT Training for Text Summarization")
    parser.add_argument("--model", type=str, default="gpt2", help="Model name or key from SUPPORTED_MODELS")
    parser.add_argument("--train_data", type=str, required=True, help="Path to training JSONL file")
    parser.add_argument("--val_data", type=str, default=None, help="Path to validation JSONL file")
    parser.add_argument("--output_dir", type=str, default="outputs/sft", help="Output directory")
    parser.add_argument("--max_seq_length", type=int, default=1024, help="Maximum sequence length")
    parser.add_argument("--num_epochs", type=int, default=3, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=4, help="Per-device batch size")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4)
    parser.add_argument("--learning_rate", type=float, default=2e-4)
    parser.add_argument("--use_lora", action="store_true", default=True)
    parser.add_argument("--no_lora", action="store_true", help="Disable LoRA")
    parser.add_argument("--lora_r", type=int, default=16)
    parser.add_argument("--lora_alpha", type=int, default=32)
    parser.add_argument("--load_in_4bit", action="store_true")
    parser.add_argument("--load_in_8bit", action="store_true")
    parser.add_argument("--warmup_ratio", type=float, default=0.05)
    parser.add_argument("--logging_steps", type=int, default=10)
    parser.add_argument("--save_steps", type=int, default=200)
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

    # Load and format datasets
    print("Loading training data...")
    train_dataset = build_sft_dataset(build_dataset(args.train_data))

    val_dataset = None
    if args.val_data:
        print("Loading validation data...")
        val_dataset = build_sft_dataset(build_dataset(args.val_data))

    # Training config
    training_args = SFTConfig(
        output_dir=args.output_dir,
        num_train_epochs=args.num_epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        warmup_ratio=args.warmup_ratio,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        save_total_limit=3,
        fp16=True,
        max_length=args.max_seq_length,
        dataset_text_field="text",
        report_to=args.report_to,
        lr_scheduler_type="cosine",
        optim="adamw_torch",
        gradient_checkpointing=True,
    )

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        processing_class=tokenizer,
    )

    print("Starting SFT training...")
    trainer.train()

    # Save final model
    final_path = Path(args.output_dir) / "final"
    trainer.save_model(str(final_path))
    tokenizer.save_pretrained(str(final_path))
    print(f"Model saved to {final_path}")


if __name__ == "__main__":
    main()
