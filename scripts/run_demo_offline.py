"""Offline demo: End-to-end text summarization pipeline test.

Creates a tiny model from scratch (no download needed) and runs the full
SFT training pipeline on CPU with BKAINewsCorpus synthetic data.

Usage:
    python scripts/run_demo_offline.py
"""

import json
import sys
import os
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import torch
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    PreTrainedTokenizerFast,
)
from tokenizers import Tokenizer, models, pre_tokenizers, trainers
from trl import SFTTrainer, SFTConfig
from datasets import Dataset

from data_utils import build_dataset, format_prompt


def create_tiny_tokenizer(save_dir: str) -> PreTrainedTokenizerFast:
    """Create a minimal tokenizer for testing."""
    # Build a simple BPE tokenizer
    tokenizer = Tokenizer(models.BPE())
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)

    # Train on Vietnamese text samples
    trainer = trainers.BpeTrainer(
        vocab_size=1000,
        special_tokens=["<pad>", "<s>", "</s>", "<unk>"],
        min_frequency=1,
    )

    vi_corpus = [
        "Các nhà khoa học đã phát triển phương pháp mới để phát hiện ung thư sớm",
        "Ngân hàng Nhà nước Việt Nam công bố quyết định điều chỉnh lãi suất",
        "Đội tuyển bóng đá Việt Nam giành chiến thắng trong trận chung kết",
        "Apple vừa ra mắt sản phẩm mới với nhiều cải tiến đáng chú ý",
        "Quốc hội thông qua luật sửa đổi với đa số đại biểu tán thành",
        "Xuất khẩu thủy sản Việt Nam đạt mức tăng trưởng ấn tượng",
        "Bệnh viện thực hiện thành công ca phẫu thuật ghép gan đầu tiên",
        "Bộ Giáo dục công bố kết quả kỳ thi tốt nghiệp THPT",
        "Thị trường bất động sản ghi nhận sự phục hồi mạnh mẽ",
        "Nghiên cứu ứng dụng trí tuệ nhân tạo trong chẩn đoán bệnh",
        "Document Summary Category kinh tế khoa học thể thao công nghệ",
    ]

    tokenizer.train_from_iterator(vi_corpus, trainer=trainer)

    # Convert to HuggingFace tokenizer
    os.makedirs(save_dir, exist_ok=True)
    tokenizer.save(f"{save_dir}/tokenizer.json")

    hf_tokenizer = PreTrainedTokenizerFast(
        tokenizer_file=f"{save_dir}/tokenizer.json",
        bos_token="<s>",
        eos_token="</s>",
        unk_token="<unk>",
        pad_token="<pad>",
    )
    hf_tokenizer.save_pretrained(save_dir)
    return hf_tokenizer


def create_tiny_model(tokenizer, save_dir: str) -> AutoModelForCausalLM:
    """Create a tiny GPT-2 model for testing (no download needed)."""
    config = AutoConfig.for_model(
        "gpt2",
        vocab_size=len(tokenizer),
        n_positions=512,
        n_embd=64,
        n_layer=2,
        n_head=2,
        n_inner=128,
        bos_token_id=tokenizer.bos_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )
    model = AutoModelForCausalLM.from_config(config)
    print(f"Tiny model created: {sum(p.numel() for p in model.parameters()):,} parameters")
    return model


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
    project_dir = Path(__file__).parent.parent
    os.chdir(project_dir)

    print("=" * 60)
    print("Text Summarization - Offline Demo")
    print("BKAINewsCorpus (synthetic) + Tiny GPT-2 + CPU")
    print("=" * 60)

    # Step 1: Prepare data
    print("\n[Step 1/4] Preparing BKAINewsCorpus data...")
    os.system("python3 src/prepare_bkai_data.py --output_dir data/bkai --max_samples 30 --demo")

    # Step 2: Create tiny model
    print("\n[Step 2/4] Creating tiny model (no download)...")
    model_dir = "outputs/demo_tiny_model"
    tokenizer = create_tiny_tokenizer(model_dir)
    model = create_tiny_model(tokenizer, model_dir)

    # Step 3: Train
    print("\n[Step 3/4] Running SFT training...")
    train_dataset = build_sft_dataset(build_dataset("data/bkai/train.jsonl"))
    val_dataset = build_sft_dataset(build_dataset("data/bkai/val.jsonl"))

    training_args = SFTConfig(
        output_dir="outputs/demo_sft",
        num_train_epochs=2,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=1,
        learning_rate=5e-4,
        warmup_ratio=0.1,
        logging_steps=5,
        save_steps=999999,
        save_total_limit=1,
        fp16=False,
        max_length=256,
        dataset_text_field="text",
        report_to="none",
        lr_scheduler_type="cosine",
        optim="adamw_torch",
        use_cpu=True,
    )

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        processing_class=tokenizer,
    )

    train_result = trainer.train()
    print(f"\nTraining loss: {train_result.training_loss:.4f}")

    # Step 4: Test inference
    print("\n[Step 4/4] Testing inference...")
    with open("data/bkai/val.jsonl", "r", encoding="utf-8") as f:
        sample = json.loads(f.readline())

    prompt = format_prompt(sample["input"], sample.get("category"))
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=200)

    model.eval()
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=50,
            do_sample=False,
            num_beams=1,
        )
    generated = tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)

    print(f"\nInput (trích): {sample['input'][:150]}...")
    print(f"\nReference:     {sample['output']}")
    print(f"\nGenerated:     {generated[:200]}")
    print(f"  (Note: Random model output - meaningful results require real model + GPU training)")

    # Eval
    print("\n[Evaluation]")
    eval_result = trainer.evaluate()
    print(f"  Eval loss: {eval_result['eval_loss']:.4f}")

    print("\n" + "=" * 60)
    print("Demo completed successfully!")
    print("=" * 60)
    print("\nTo train with real BKAINewsCorpus + GPU:")
    print("  python src/prepare_bkai_data.py --output_dir data/bkai --max_samples 5000")
    print("  python src/train_sft.py --model qwen2-0.5b --train_data data/bkai/train.jsonl \\")
    print("      --val_data data/bkai/val.jsonl --output_dir outputs/sft --use_lora")


if __name__ == "__main__":
    main()
