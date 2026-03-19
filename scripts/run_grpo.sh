#!/bin/bash
# Run GRPO training
set -e

cd "$(dirname "$0")/.."

python src/train_grpo.py \
    --model qwen2-0.5b \
    --train_data data/sample_train.jsonl \
    --output_dir outputs/grpo \
    --num_epochs 1 \
    --batch_size 2 \
    --gradient_accumulation_steps 8 \
    --learning_rate 1e-5 \
    --num_generations 4 \
    --max_completion_length 256 \
    --max_prompt_length 512 \
    --target_length 80 \
    --max_summary_length 200 \
    --length_weight 0.3 \
    --rouge_weight 0.7 \
    --lora_r 16 \
    --lora_alpha 32 \
    --logging_steps 5 \
    --save_steps 100
