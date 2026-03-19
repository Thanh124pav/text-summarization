#!/bin/bash
# Run GRPO training with verl library
set -e

cd "$(dirname "$0")/.."

# Step 1: Prepare data (convert JSONL to parquet)
python src/prepare_verl_data.py \
    --input data/sample_train.jsonl \
    --output data/train.parquet

# Step 2: Run verl GRPO training
python src/train_grpo_verl.py \
    --model Qwen/Qwen2-0.5B \
    --train_data data/train.parquet \
    --output_dir outputs/grpo_verl \
    --num_epochs 1 \
    --train_batch_size 64 \
    --num_generations 4 \
    --max_prompt_length 512 \
    --max_response_length 256 \
    --kl_coef 0.001 \
    --reward_function src/verl_reward.py \
    --lora_r 16 \
    --lora_alpha 32
