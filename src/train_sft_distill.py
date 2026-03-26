"""Style-conditioned SFT with Knowledge Distillation for text summarization.

Two distillation modes:

  1. Soft distillation (--distill_mode soft):
     Teacher model hosted via vLLM. For each training batch, we forward
     the same input through the teacher to get its logits, then compute
     KL divergence between student and teacher output distributions.

     Loss = (1-α) × CE(student, ground_truth) + α × T² × KL(student || teacher)

     Requires: teacher model accessible via vLLM (local or remote).

  2. Hard distillation (--distill_mode hard):
     Teacher outputs are pre-generated (via distill_generate.py) and stored
     in the JSONL dataset as 'teacher_output'. Training uses a weighted
     cross-entropy between ground truth and teacher output:

     Loss = (1-α) × CE(student, ground_truth) + α × CE(student, teacher_output)

     Requires: JSONL with both 'output' and 'teacher_output' fields.

Data format:
  # For hard distillation (teacher_output pre-generated):
  {"input": "...", "output": "GT summary", "teacher_output": "teacher summary",
   "style": "bao_chi", "category": "kinh_te"}

  # For soft distillation (teacher_output not needed in data):
  {"input": "...", "output": "GT summary", "style": "bao_chi"}
"""

import argparse
import json
import random
from collections import Counter
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from datasets import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)
from peft import get_peft_model, LoraConfig, prepare_model_for_kbit_training

from data_utils import (
    build_dataset,
    format_prompt,
    STYLE_NAMES,
    STYLE_DISPLAY_NAMES,
)
from logging_utils import DistillLoggingCallback
from model_utils import get_model_name, load_tokenizer, load_model


# ---------------------------------------------------------------------------
# Dataset builders
# ---------------------------------------------------------------------------

def build_distill_dataset(
    dataset: Dataset,
    tokenizer: AutoTokenizer,
    distill_mode: str,
    max_prompt_len: int = 1024,
    max_target_len: int = 512,
    use_chat_template: bool = False,
    seed: int = 42,
) -> Dataset:
    """Build tokenized dataset for distillation training.

    For both soft and hard modes, we tokenize prompt + ground_truth as the
    primary sequence. For hard mode, we additionally tokenize prompt +
    teacher_output as the distillation target.

    Returns dataset with columns:
      - input_ids, attention_mask, labels: prompt + GT (labels masked on prompt)
      - teacher_input_ids, teacher_attention_mask, teacher_labels:
            prompt + teacher_output (hard mode only)
    """
    rng = random.Random(seed)

    def tokenize_sequence(prompt_text: str, target_text: str):
        """Tokenize prompt + target, mask labels on prompt portion."""
        prompt_ids = tokenizer(
            prompt_text, truncation=True, max_length=max_prompt_len,
            add_special_tokens=True,
        )["input_ids"]
        target_ids = tokenizer(
            target_text + tokenizer.eos_token,
            truncation=True, max_length=max_target_len,
            add_special_tokens=False,
        )["input_ids"]

        input_ids = prompt_ids + target_ids
        labels = [-100] * len(prompt_ids) + target_ids
        attention_mask = [1] * len(input_ids)

        # Truncate to total max
        total_max = max_prompt_len + max_target_len
        input_ids = input_ids[:total_max]
        labels = labels[:total_max]
        attention_mask = attention_mask[:total_max]

        return input_ids, attention_mask, labels

    def tokenize_fn(examples):
        n = len(examples["input"])
        categories = examples.get("category", [None] * n)

        if "style" in examples:
            styles = [s or rng.choice(STYLE_NAMES) for s in examples["style"]]
        else:
            styles = [rng.choice(STYLE_NAMES) for _ in range(n)]

        result = {
            "input_ids": [], "attention_mask": [], "labels": [],
        }

        has_teacher = distill_mode == "hard" and "teacher_output" in examples
        if has_teacher:
            result["teacher_input_ids"] = []
            result["teacher_attention_mask"] = []
            result["teacher_labels"] = []

        for i in range(n):
            inp = examples["input"][i]
            out = examples["output"][i]
            cat = categories[i]
            sty = styles[i]

            if use_chat_template and hasattr(tokenizer, "apply_chat_template"):
                style_display = STYLE_DISPLAY_NAMES.get(sty, sty)
                cat_hint = f"[Chủ đề: {cat}] " if cat else ""
                messages = [
                    {
                        "role": "user",
                        "content": (
                            f"Hãy tóm tắt theo phong cách {style_display}. "
                            f"{cat_hint}Tóm tắt văn bản sau:\n\n{inp}"
                        ),
                    },
                ]
                prompt_text = tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                )
            else:
                prompt_text = format_prompt(inp, cat, sty)

            # Primary: prompt + ground truth
            ids, mask, labs = tokenize_sequence(prompt_text, out)
            result["input_ids"].append(ids)
            result["attention_mask"].append(mask)
            result["labels"].append(labs)

            # Hard distillation: prompt + teacher output
            if has_teacher:
                teacher_out = examples["teacher_output"][i]
                t_ids, t_mask, t_labs = tokenize_sequence(prompt_text, teacher_out)
                result["teacher_input_ids"].append(t_ids)
                result["teacher_attention_mask"].append(t_mask)
                result["teacher_labels"].append(t_labs)

        return result

    return dataset.map(
        tokenize_fn, batched=True, remove_columns=dataset.column_names,
    )


def collate_distill(batch: list[dict], tokenizer: AutoTokenizer) -> dict:
    """Pad sequences in a batch to uniform length."""
    pad_id = tokenizer.pad_token_id or 0

    def pad_field(items, pad_value):
        max_len = max(len(x) for x in items)
        return torch.tensor(
            [x + [pad_value] * (max_len - len(x)) for x in items],
            dtype=torch.long,
        )

    result = {
        "input_ids": pad_field([b["input_ids"] for b in batch], pad_id),
        "attention_mask": pad_field([b["attention_mask"] for b in batch], 0),
        "labels": pad_field([b["labels"] for b in batch], -100),
    }

    if "teacher_input_ids" in batch[0]:
        result["teacher_input_ids"] = pad_field(
            [b["teacher_input_ids"] for b in batch], pad_id
        )
        result["teacher_attention_mask"] = pad_field(
            [b["teacher_attention_mask"] for b in batch], 0
        )
        result["teacher_labels"] = pad_field(
            [b["teacher_labels"] for b in batch], -100
        )

    return result


# ---------------------------------------------------------------------------
# Soft distillation: vLLM teacher logits
# ---------------------------------------------------------------------------

class VLLMTeacherClient:
    """Client to get teacher logits from a vLLM-served model.

    vLLM must be started with --return-logprobs or we use the completions
    API with logprobs. For full logit distillation, we forward through a
    locally loaded teacher model instead.
    """

    def __init__(
        self,
        model_name_or_path: str,
        device: str = "auto",
        dtype: str = "auto",
    ):
        """Load teacher model locally for logit access.

        For true soft distillation we need full logit vectors, which the
        vLLM API doesn't expose directly. So we load the teacher model
        in inference mode (no grad) on a separate device or share GPU.
        """
        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        print(f"Loading teacher model: {model_name_or_path}")
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name_or_path, trust_remote_code=True
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        torch_dtype = torch.bfloat16 if dtype == "auto" else getattr(torch, dtype)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name_or_path,
            trust_remote_code=True,
            torch_dtype=torch_dtype,
            device_map=device,
        )
        self.model.eval()
        print(f"  Teacher loaded on {self.device}, dtype={torch_dtype}")

    @torch.no_grad()
    def get_logits(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Forward pass through teacher, return logits.

        Args:
            input_ids: (batch, seq_len)
            attention_mask: (batch, seq_len)

        Returns:
            logits: (batch, seq_len, vocab_size)
        """
        inputs = input_ids.to(self.device)
        mask = attention_mask.to(self.device)
        outputs = self.model(input_ids=inputs, attention_mask=mask)
        return outputs.logits.cpu()


# ---------------------------------------------------------------------------
# Custom Trainer with distillation loss
# ---------------------------------------------------------------------------

class DistillationTrainer(Trainer):
    """HuggingFace Trainer with knowledge distillation support.

    Computes a combined loss:
      loss = (1-α) × CE_gt + α × distill_loss

    Where distill_loss is:
      - Soft mode: T² × KL(student_softmax/T || teacher_softmax/T)
      - Hard mode: CE(student, teacher_output_labels)
    """

    def __init__(
        self,
        *args,
        distill_mode: str = "hard",
        distill_alpha: float = 0.5,
        distill_temperature: float = 2.0,
        teacher_client: VLLMTeacherClient | None = None,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.distill_mode = distill_mode
        self.distill_alpha = distill_alpha
        self.distill_temperature = distill_temperature
        self.teacher_client = teacher_client

        if distill_mode == "soft" and teacher_client is None:
            raise ValueError("teacher_client required for soft distillation")

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        """Custom loss with distillation."""
        # --- Ground truth CE loss ---
        labels = inputs.pop("labels")

        # Remove teacher fields from model forward
        teacher_input_ids = inputs.pop("teacher_input_ids", None)
        teacher_attention_mask = inputs.pop("teacher_attention_mask", None)
        teacher_labels = inputs.pop("teacher_labels", None)

        outputs = model(**inputs)
        logits = outputs.logits

        # Shift for causal LM: predict next token
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()

        ce_loss = F.cross_entropy(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1),
            ignore_index=-100,
        )

        # --- Distillation loss ---
        if self.distill_mode == "soft":
            distill_loss = self._soft_distill_loss(
                inputs["input_ids"], inputs["attention_mask"], logits
            )
        elif self.distill_mode == "hard" and teacher_labels is not None:
            distill_loss = self._hard_distill_loss(
                model, teacher_input_ids, teacher_attention_mask, teacher_labels
            )
        else:
            distill_loss = torch.tensor(0.0, device=ce_loss.device)

        # --- Combined loss ---
        alpha = self.distill_alpha
        loss = (1 - alpha) * ce_loss + alpha * distill_loss

        # Log components
        if self.state.global_step % self.args.logging_steps == 0:
            self.log({
                "ce_loss": ce_loss.item(),
                "distill_loss": distill_loss.item(),
                "total_loss": loss.item(),
            })

        return (loss, outputs) if return_outputs else loss

    def _soft_distill_loss(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        student_logits: torch.Tensor,
    ) -> torch.Tensor:
        """KL divergence between student and teacher logit distributions.

        Loss = T² × KL(student_softmax(logits/T) || teacher_softmax(logits/T))

        Only computed on non-padding positions.
        """
        T = self.distill_temperature

        # Get teacher logits (no grad, possibly different device)
        teacher_logits = self.teacher_client.get_logits(input_ids, attention_mask)
        teacher_logits = teacher_logits.to(student_logits.device)

        # Align sequence lengths (teacher may have different padding)
        min_len = min(student_logits.size(1), teacher_logits.size(1))
        s_logits = student_logits[:, :min_len, :]
        t_logits = teacher_logits[:, :min_len, :]

        # Shift for causal LM
        s_logits = s_logits[:, :-1, :].contiguous()
        t_logits = t_logits[:, :-1, :].contiguous()

        # Align vocab sizes (student may have different vocab with LoRA)
        min_vocab = min(s_logits.size(-1), t_logits.size(-1))
        s_logits = s_logits[..., :min_vocab]
        t_logits = t_logits[..., :min_vocab]

        # Temperature-scaled softmax
        student_log_probs = F.log_softmax(s_logits / T, dim=-1)
        teacher_probs = F.softmax(t_logits / T, dim=-1)

        # KL divergence (batchmean reduction)
        kl_loss = F.kl_div(
            student_log_probs, teacher_probs,
            reduction="batchmean", log_target=False,
        )

        return T * T * kl_loss

    def _hard_distill_loss(
        self,
        model,
        teacher_input_ids: torch.Tensor,
        teacher_attention_mask: torch.Tensor,
        teacher_labels: torch.Tensor,
    ) -> torch.Tensor:
        """Cross-entropy loss on teacher-generated output.

        Forward student model on the same prompt but with teacher_output
        as the target sequence. This teaches the student to mimic the
        teacher's generation.
        """
        teacher_outputs = model(
            input_ids=teacher_input_ids,
            attention_mask=teacher_attention_mask,
        )
        t_logits = teacher_outputs.logits

        # Shift for causal LM
        shift_logits = t_logits[..., :-1, :].contiguous()
        shift_labels = teacher_labels[..., 1:].contiguous()

        return F.cross_entropy(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1),
            ignore_index=-100,
        )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="SFT + Knowledge Distillation for style summarization"
    )

    # Model
    parser.add_argument("--model", type=str, default="Qwen/Qwen3-4B")
    parser.add_argument("--train_data", type=str, required=True)
    parser.add_argument("--val_data", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default="outputs/sft_distill")

    # Distillation
    parser.add_argument("--distill_mode", type=str, required=True,
                        choices=["soft", "hard"],
                        help="soft: KL with live teacher logits; hard: CE with teacher outputs")
    parser.add_argument("--distill_alpha", type=float, default=0.5,
                        help="Distillation weight: 0=pure SFT, 1=pure distillation")
    parser.add_argument("--distill_temperature", type=float, default=2.0,
                        help="Temperature for soft distillation (ignored for hard)")

    # Teacher (soft mode)
    parser.add_argument("--teacher_model", type=str, default=None,
                        help="Teacher model path for soft distillation")
    parser.add_argument("--teacher_device", type=str, default="auto",
                        help="Device for teacher model (auto, cuda:0, cuda:1, cpu)")
    parser.add_argument("--teacher_dtype", type=str, default="auto",
                        choices=["auto", "float16", "bfloat16", "float32"])

    # Style & formatting
    parser.add_argument("--use_chat_template", action="store_true")
    parser.add_argument("--max_prompt_len", type=int, default=1024)
    parser.add_argument("--max_target_len", type=int, default=512)

    # Training
    parser.add_argument("--num_epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=8)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--warmup_ratio", type=float, default=0.05)
    parser.add_argument("--weight_decay", type=float, default=0.01)

    # LoRA
    parser.add_argument("--use_lora", action="store_true", default=True)
    parser.add_argument("--no_lora", action="store_true")
    parser.add_argument("--lora_r", type=int, default=32)
    parser.add_argument("--lora_alpha", type=int, default=64)

    # Quantization
    parser.add_argument("--load_in_4bit", action="store_true")
    parser.add_argument("--load_in_8bit", action="store_true")

    # Logging
    parser.add_argument("--logging_steps", type=int, default=10)
    parser.add_argument("--save_steps", type=int, default=200)
    parser.add_argument("--eval_steps", type=int, default=200)
    parser.add_argument("--report_to", type=str, default="none")
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()

    use_lora = not args.no_lora
    model_name = get_model_name(args.model)

    # --- Load student model ---
    print(f"Loading student model: {model_name}")
    tokenizer = load_tokenizer(model_name)
    model = load_model(
        model_name,
        use_lora=use_lora,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        load_in_4bit=args.load_in_4bit,
        load_in_8bit=args.load_in_8bit,
    )

    # --- Load teacher (soft mode) ---
    teacher_client = None
    if args.distill_mode == "soft":
        if not args.teacher_model:
            parser.error("--teacher_model required for soft distillation")
        teacher_client = VLLMTeacherClient(
            model_name_or_path=args.teacher_model,
            device=args.teacher_device,
            dtype=args.teacher_dtype,
        )

    # --- Build dataset ---
    print("Loading training data...")
    raw_train = build_dataset(args.train_data)

    if args.distill_mode == "hard" and "teacher_output" not in raw_train.column_names:
        raise ValueError(
            "Hard distillation requires 'teacher_output' field in data. "
            "Run distill_generate.py first to generate teacher outputs."
        )

    train_dataset = build_distill_dataset(
        raw_train, tokenizer,
        distill_mode=args.distill_mode,
        max_prompt_len=args.max_prompt_len,
        max_target_len=args.max_target_len,
        use_chat_template=args.use_chat_template,
        seed=args.seed,
    )

    val_dataset = None
    if args.val_data:
        raw_val = build_dataset(args.val_data)
        val_dataset = build_distill_dataset(
            raw_val, tokenizer,
            distill_mode=args.distill_mode,
            max_prompt_len=args.max_prompt_len,
            max_target_len=args.max_target_len,
            use_chat_template=args.use_chat_template,
            seed=args.seed,
        )

    # --- Training args ---
    use_bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    use_fp16 = torch.cuda.is_available() and not use_bf16

    training_args = TrainingArguments(
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
        report_to=args.report_to,
        lr_scheduler_type="cosine",
        optim="adamw_torch",
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        eval_strategy="steps" if val_dataset else "no",
        eval_steps=args.eval_steps if val_dataset else None,
        load_best_model_at_end=bool(val_dataset),
        seed=args.seed,
        remove_unused_columns=False,
        dataloader_pin_memory=True,
    )

    # --- Trainer ---
    trainer = DistillationTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=lambda batch: collate_distill(batch, tokenizer),
        processing_class=tokenizer,
        distill_mode=args.distill_mode,
        distill_alpha=args.distill_alpha,
        distill_temperature=args.distill_temperature,
        teacher_client=teacher_client,
        callbacks=[
            DistillLoggingCallback(
                distill_mode=args.distill_mode,
                alpha=args.distill_alpha,
                temperature=args.distill_temperature,
            ),
        ],
    )

    # --- Print summary ---
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"\n{'='*60}")
    print(f"SFT + Distillation Training")
    print(f"{'='*60}")
    print(f"  Student:        {model_name}")
    print(f"  Params:         {trainable_params:,} / {total_params:,} trainable")
    print(f"  Distill mode:   {args.distill_mode}")
    print(f"  Alpha (α):      {args.distill_alpha}")
    if args.distill_mode == "soft":
        print(f"  Teacher:        {args.teacher_model}")
        print(f"  Temperature:    {args.distill_temperature}")
        print(f"  Loss = (1-{args.distill_alpha})×CE_gt + {args.distill_alpha}×T²×KL(s||t)")
    else:
        print(f"  Loss = (1-{args.distill_alpha})×CE_gt + {args.distill_alpha}×CE_teacher")
    print(f"  Train samples:  {len(train_dataset)}")
    print(f"{'='*60}\n")

    # --- Train ---
    print("Starting training...")
    trainer.train()

    # --- Save ---
    final_path = Path(args.output_dir) / "final"
    trainer.save_model(str(final_path))
    tokenizer.save_pretrained(str(final_path))
    print(f"\nModel saved to {final_path}")

    info = {
        "model": model_name,
        "distill_mode": args.distill_mode,
        "distill_alpha": args.distill_alpha,
        "distill_temperature": args.distill_temperature if args.distill_mode == "soft" else None,
        "teacher_model": args.teacher_model,
        "train_samples": len(train_dataset),
        "epochs": args.num_epochs,
        "learning_rate": args.learning_rate,
    }
    with open(final_path / "training_info.json", "w", encoding="utf-8") as f:
        json.dump(info, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
