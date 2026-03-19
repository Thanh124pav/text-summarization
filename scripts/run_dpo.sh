#!/bin/bash
# Run DPO training
set -e

cd "$(dirname "$0")/.."

python src/train_dpo.py \
    --model qwen2-0.5b \
    --train_data data/sample_dpo.jsonl \
    --output_dir outputs/dpo \
    --num_epochs 1 \
    --batch_size 2 \
    --gradient_accumulation_steps 8 \
    --learning_rate 5e-6 \
    --beta 0.1 \
    --max_length 1024 \
    --max_prompt_length 512 \
    --lora_r 16 \
    --lora_alpha 32 \
    --logging_steps 10 \
    --save_steps 100
