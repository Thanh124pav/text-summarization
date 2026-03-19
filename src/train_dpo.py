"""DPO (Direct Preference Optimization) training for text summarization.

Requires a dataset with 'input', 'output' (chosen), and 'rejected' fields.
If no rejected summaries exist, use generate_dpo_pairs.py to create them.
"""

import argparse
from pathlib import Path

from datasets import Dataset
from trl import DPOConfig, DPOTrainer

from data_utils import build_dataset, format_prompt
from model_utils import get_model_name, load_tokenizer, load_model


def build_dpo_dataset(dataset: Dataset) -> Dataset:
    """Build dataset for DPO training.

    Each item needs: prompt, chosen (good summary), rejected (bad summary).
    """

    def format_fn(examples):
        prompts = []
        chosen_list = []
        rejected_list = []

        categories = examples.get("category", [None] * len(examples["input"]))
        for inp, out, rej, cat in zip(
            examples["input"],
            examples["output"],
            examples["rejected"],
            categories,
        ):
            prompt = format_prompt(inp, cat)
            prompts.append(prompt)
            chosen_list.append(out)
            rejected_list.append(rej)

        return {"prompt": prompts, "chosen": chosen_list, "rejected": rejected_list}

    return dataset.map(format_fn, batched=True, remove_columns=dataset.column_names)


def main():
    parser = argparse.ArgumentParser(description="DPO Training for Text Summarization")
    parser.add_argument("--model", type=str, default="gpt2", help="Model name or key")
    parser.add_argument("--train_data", type=str, required=True, help="Path to training JSONL (with 'rejected' field)")
    parser.add_argument("--val_data", type=str, default=None, help="Path to validation JSONL")
    parser.add_argument("--output_dir", type=str, default="outputs/dpo", help="Output directory")
    parser.add_argument("--num_epochs", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=8)
    parser.add_argument("--learning_rate", type=float, default=5e-6)
    parser.add_argument("--beta", type=float, default=0.1, help="DPO beta parameter")
    parser.add_argument("--max_length", type=int, default=1024)
    parser.add_argument("--max_prompt_length", type=int, default=512)
    parser.add_argument("--use_lora", action="store_true", default=True)
    parser.add_argument("--no_lora", action="store_true")
    parser.add_argument("--lora_r", type=int, default=16)
    parser.add_argument("--lora_alpha", type=int, default=32)
    parser.add_argument("--load_in_4bit", action="store_true")
    parser.add_argument("--load_in_8bit", action="store_true")
    parser.add_argument("--logging_steps", type=int, default=10)
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

    # Load dataset (must have 'rejected' field)
    print("Loading training data...")
    raw_train = build_dataset(args.train_data)
    assert "rejected" in raw_train.column_names, (
        "DPO training requires a 'rejected' field. Use generate_dpo_pairs.py first."
    )
    train_dataset = build_dpo_dataset(raw_train)

    val_dataset = None
    if args.val_data:
        raw_val = build_dataset(args.val_data)
        val_dataset = build_dpo_dataset(raw_val)

    # DPO config
    dpo_config = DPOConfig(
        output_dir=args.output_dir,
        num_train_epochs=args.num_epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        beta=args.beta,
        max_length=args.max_length,
        max_prompt_length=args.max_prompt_length,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        save_total_limit=3,
        bf16=True,
        gradient_checkpointing=True,
        report_to=args.report_to,
        lr_scheduler_type="cosine",
        warmup_ratio=0.1,
    )

    trainer = DPOTrainer(
        model=model,
        args=dpo_config,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        processing_class=tokenizer,
    )

    print("Starting DPO training...")
    print(f"  Beta: {args.beta}")
    trainer.train()

    # Save
    final_path = Path(args.output_dir) / "final"
    trainer.save_model(str(final_path))
    tokenizer.save_pretrained(str(final_path))
    print(f"Model saved to {final_path}")


if __name__ == "__main__":
    main()
