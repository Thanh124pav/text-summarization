"""Data utilities for loading and preprocessing JSONL summarization datasets."""

import json
from pathlib import Path

from datasets import Dataset
from transformers import PreTrainedTokenizer


def load_jsonl(file_path: str) -> list[dict]:
    """Load a JSONL file with fields: input, output, category."""
    records = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            assert "input" in record and "output" in record, (
                f"Each record must have 'input' and 'output' fields. Got: {list(record.keys())}"
            )
            records.append(record)
    return records


def build_dataset(file_path: str) -> Dataset:
    """Load JSONL file and return a HuggingFace Dataset."""
    records = load_jsonl(file_path)
    return Dataset.from_list(records)


def format_prompt(
    text: str,
    category: str | None = None,
    style: str | None = None,
) -> str:
    """Format input text into a summarization prompt.

    Args:
        text: The document to summarize.
        category: Optional topic category (e.g., khoa_hoc, the_thao).
        style: Optional writing style instruction. One of:
            bao_chi, hanh_chinh, khoa_hoc, chinh_luan, sinh_hoat.
    """
    parts = []
    if category:
        parts.append(f"### Category: {category}")
    if style:
        style_display = STYLE_DISPLAY_NAMES.get(style, style)
        parts.append(f"### Phong cách viết: {style_display}")
    parts.append(f"### Document:\n{text}\n")
    parts.append("### Summary:\n")
    return "\n".join(parts)


# 5 writing styles for Vietnamese text
STYLE_NAMES = ["bao_chi", "hanh_chinh", "khoa_hoc", "chinh_luan", "sinh_hoat"]

STYLE_DISPLAY_NAMES = {
    "bao_chi": "Báo chí",
    "hanh_chinh": "Hành chính",
    "khoa_hoc": "Khoa học",
    "chinh_luan": "Chính luận",
    "sinh_hoat": "Sinh hoạt hàng ngày",
}


def preprocess_for_sft(
    dataset: Dataset,
    tokenizer: PreTrainedTokenizer,
    max_source_len: int = 1024,
    max_target_len: int = 256,
) -> Dataset:
    """Preprocess dataset for supervised finetuning.

    Concatenates prompt + summary into a single sequence with labels
    masked on the prompt portion.
    """

    def tokenize_fn(examples):
        prompts = [
            format_prompt(inp, cat)
            for inp, cat in zip(examples["input"], examples.get("category", [None] * len(examples["input"])))
        ]
        summaries = [out + tokenizer.eos_token for out in examples["output"]]

        model_inputs = {"input_ids": [], "attention_mask": [], "labels": []}

        for prompt, summary in zip(prompts, summaries):
            prompt_ids = tokenizer(
                prompt, truncation=True, max_length=max_source_len, add_special_tokens=True
            )["input_ids"]
            summary_ids = tokenizer(
                summary, truncation=True, max_length=max_target_len, add_special_tokens=False
            )["input_ids"]

            input_ids = prompt_ids + summary_ids
            labels = [-100] * len(prompt_ids) + summary_ids

            # Pad or truncate to max length
            total_max = max_source_len + max_target_len
            if len(input_ids) > total_max:
                input_ids = input_ids[:total_max]
                labels = labels[:total_max]

            attention_mask = [1] * len(input_ids)

            model_inputs["input_ids"].append(input_ids)
            model_inputs["attention_mask"].append(attention_mask)
            model_inputs["labels"].append(labels)

        return model_inputs

    return dataset.map(tokenize_fn, batched=True, remove_columns=dataset.column_names)


def preprocess_for_dpo(dataset: Dataset) -> Dataset:
    """Preprocess dataset for DPO training.

    Expects each record to have 'input', 'output' (chosen), and 'rejected' fields.
    If 'rejected' is missing, it will be left empty (must be generated separately).
    """

    def format_fn(examples):
        prompts = [
            format_prompt(inp, cat)
            for inp, cat in zip(examples["input"], examples.get("category", [None] * len(examples["input"])))
        ]
        chosen = examples["output"]
        rejected = examples.get("rejected", [""] * len(chosen))

        return {"prompt": prompts, "chosen": chosen, "rejected": rejected}

    return dataset.map(format_fn, batched=True, remove_columns=dataset.column_names)


def preprocess_for_grpo(
    dataset: Dataset,
    with_style: bool = False,
    seed: int = 42,
) -> Dataset:
    """Preprocess dataset for GRPO training.

    Args:
        dataset: Raw dataset with 'input', 'output', optional 'category'.
        with_style: If True, randomly assign a writing style to each prompt
            and include it in the prompt text. The assigned style name is
            stored in the 'style' column for reward computation.
        seed: Random seed for style assignment.

    Returns prompts and reference summaries.
    """
    import random

    rng = random.Random(seed)

    def format_fn(examples):
        n = len(examples["input"])
        categories = examples.get("category", [None] * n)

        if with_style:
            styles = [rng.choice(STYLE_NAMES) for _ in range(n)]
        else:
            styles = [None] * n

        prompts = [
            format_prompt(inp, cat, sty)
            for inp, cat, sty in zip(examples["input"], categories, styles)
        ]

        result = {"prompt": prompts, "reference": examples["output"]}
        if with_style:
            result["style"] = styles
        return result

    return dataset.map(format_fn, batched=True, remove_columns=dataset.column_names)


def collate_with_padding(batch: list[dict], tokenizer: PreTrainedTokenizer) -> dict:
    """Collate function that pads sequences to the same length within a batch."""
    import torch

    max_len = max(len(item["input_ids"]) for item in batch)
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0

    input_ids = []
    attention_mask = []
    labels = []

    for item in batch:
        pad_len = max_len - len(item["input_ids"])
        input_ids.append(item["input_ids"] + [pad_id] * pad_len)
        attention_mask.append(item["attention_mask"] + [0] * pad_len)
        labels.append(item["labels"] + [-100] * pad_len)

    return {
        "input_ids": torch.tensor(input_ids, dtype=torch.long),
        "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
        "labels": torch.tensor(labels, dtype=torch.long),
    }
