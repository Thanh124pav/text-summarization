#!/bin/bash
# Soft Distillation: Student + Teacher on same machine
#
# Teacher model is loaded in inference mode alongside the student.
# For each batch: teacher provides logits → KL divergence loss.
#
# Loss = (1-α) × CE(student, GT) + α × T² × KL(student || teacher)
#
# VRAM requirements:
#   Teacher 8B (bf16)  + Student 4B (LoRA 4bit) ≈ 16+6 = ~22GB → fits A100 40GB
#   Teacher 14B (bf16) + Student 4B (LoRA 4bit) ≈ 28+6 = ~34GB → needs A100 80GB
#
# Usage:
#   # Teacher and student on same GPU (needs enough VRAM)
#   bash scripts/run_distill_soft.sh --teacher_model Qwen/Qwen3-8B
#
#   # Teacher on GPU:1, student on GPU:0
#   bash scripts/run_distill_soft.sh \
#       --teacher_model Qwen/Qwen3-8B \
#       --teacher_device cuda:1
#
#   # Custom alpha and temperature
#   bash scripts/run_distill_soft.sh \
#       --teacher_model Qwen/Qwen3-8B \
#       --distill_alpha 0.7 \
#       --distill_temperature 3.0

set -e
cd "$(dirname "$0")/.."

python3 src/train_sft_distill.py \
    --model qwen3-4b \
    --train_data data/bkai/train.jsonl \
    --val_data data/bkai/val.jsonl \
    --output_dir outputs/sft_distill_soft \
    --distill_mode soft \
    --distill_alpha 0.5 \
    --distill_temperature 2.0 \
    --max_prompt_len 1024 \
    --max_target_len 512 \
    --num_epochs 3 \
    --batch_size 1 \
    --gradient_accumulation_steps 16 \
    --learning_rate 1e-4 \
    --lora_r 32 \
    --lora_alpha 64 \
    --load_in_4bit \
    "$@"
