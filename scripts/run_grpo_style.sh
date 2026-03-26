#!/bin/bash
# GRPO + Style Reward training
#
# Reward formula:
#   reward = length_gate × (0.7 × ROUGE_composite + 0.3 × P(target_style))
#
# Usage:
#   bash scripts/run_grpo_style.sh --style_classifier /path/to/checkpoint
#   bash scripts/run_grpo_style.sh --style_classifier /path/to/checkpoint --style_weight 0.4
#   bash scripts/run_grpo_style.sh --style_classifier /path/to/checkpoint --label_map '{"bao_chi":2,"hanh_chinh":0,...}'

set -e
cd "$(dirname "$0")/.."

python3 src/train_grpo.py \
    --model qwen2-0.5b \
    --train_data data/bkai/train.jsonl \
    --output_dir outputs/grpo_style \
    --reward_mode multiplicative_style \
    --min_summary_length 30 \
    --max_summary_length 150 \
    --style_weight 0.3 \
    --num_generations 4 \
    --batch_size 2 \
    --gradient_accumulation_steps 8 \
    --learning_rate 1e-5 \
    --num_epochs 1 \
    --logging_steps 5 \
    --save_steps 100 \
    "$@"
