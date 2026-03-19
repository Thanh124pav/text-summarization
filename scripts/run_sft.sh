#!/bin/bash
# Run SFT training
set -e

cd "$(dirname "$0")/.."

python src/train_sft.py \
    --model qwen2-0.5b \
    --train_data data/sample_train.jsonl \
    --output_dir outputs/sft \
    --num_epochs 3 \
    --batch_size 4 \
    --gradient_accumulation_steps 4 \
    --learning_rate 2e-4 \
    --max_seq_length 1024 \
    --lora_r 16 \
    --lora_alpha 32 \
    --logging_steps 10 \
    --save_steps 200
