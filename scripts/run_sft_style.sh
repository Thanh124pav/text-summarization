#!/bin/bash
# Style-conditioned SFT training for Qwen 3-4B
#
# Usage:
#   # Basic: Qwen3-4B + LoRA (needs ~20GB VRAM)
#   bash scripts/run_sft_style.sh
#
#   # 4-bit quantized (needs ~12GB VRAM)
#   bash scripts/run_sft_style.sh --load_in_4bit
#
#   # Custom data
#   bash scripts/run_sft_style.sh --train_data data/my_style_data.jsonl
#
#   # With chat template (for instruct models)
#   bash scripts/run_sft_style.sh --model Qwen/Qwen3-4B-Instruct --use_chat_template
#
#   # Full pipeline: SFT → GRPO
#   bash scripts/run_sft_style.sh
#   bash scripts/run_grpo_style.sh --model outputs/sft_style/final --style_classifier /path/to/clf

set -e
cd "$(dirname "$0")/.."

python3 src/train_sft_style.py \
    --model qwen3-4b \
    --train_data data/bkai/train.jsonl \
    --val_data data/bkai/val.jsonl \
    --output_dir outputs/sft_style \
    --max_seq_length 2048 \
    --num_epochs 3 \
    --batch_size 2 \
    --gradient_accumulation_steps 8 \
    --learning_rate 1e-4 \
    --warmup_ratio 0.05 \
    --lora_r 32 \
    --lora_alpha 64 \
    --balance_styles \
    --logging_steps 10 \
    --save_steps 200 \
    --eval_steps 200 \
    "$@"
