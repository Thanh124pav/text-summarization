#!/bin/bash
# Hard Distillation Pipeline
#
# Step 1: Generate teacher outputs from endpoint
# Step 2: Train student with CE(gt) + CE(teacher)
#
# Usage:
#   # Step 1: Generate teacher data
#   bash scripts/run_distill_hard.sh generate \
#       --teacher_endpoint http://localhost:8000/v1 \
#       --teacher_model Qwen/Qwen3-32B
#
#   # Step 2: Train student
#   bash scripts/run_distill_hard.sh train
#
#   # Both steps at once
#   bash scripts/run_distill_hard.sh all \
#       --teacher_endpoint http://localhost:8000/v1 \
#       --teacher_model Qwen/Qwen3-32B

set -e
cd "$(dirname "$0")/.."

STEP="${1:-all}"
shift 2>/dev/null || true

case "$STEP" in
    generate)
        echo "=== Step 1: Generating teacher outputs ==="
        python3 src/distill_generate.py \
            --input_data data/bkai/train.jsonl \
            --output_data data/bkai/train_distill.jsonl \
            --with_style \
            --temperature 0.3 \
            --max_tokens 512 \
            "$@"
        ;;

    train)
        echo "=== Step 2: Training with hard distillation ==="
        python3 src/train_sft_distill.py \
            --model qwen3-4b \
            --train_data data/bkai/train_distill.jsonl \
            --val_data data/bkai/val.jsonl \
            --output_dir outputs/sft_distill_hard \
            --distill_mode hard \
            --distill_alpha 0.5 \
            --max_prompt_len 1024 \
            --max_target_len 512 \
            --num_epochs 3 \
            --batch_size 2 \
            --gradient_accumulation_steps 8 \
            --learning_rate 1e-4 \
            --lora_r 32 \
            --lora_alpha 64 \
            "$@"
        ;;

    all)
        echo "=== Full hard distillation pipeline ==="
        echo ""
        echo "--- Step 1: Generating teacher outputs ---"
        python3 src/distill_generate.py \
            --input_data data/bkai/train.jsonl \
            --output_data data/bkai/train_distill.jsonl \
            --with_style \
            --temperature 0.3 \
            "$@"

        echo ""
        echo "--- Step 2: Training with hard distillation ---"
        python3 src/train_sft_distill.py \
            --model qwen3-4b \
            --train_data data/bkai/train_distill.jsonl \
            --val_data data/bkai/val.jsonl \
            --output_dir outputs/sft_distill_hard \
            --distill_mode hard \
            --distill_alpha 0.5 \
            --num_epochs 3 \
            --batch_size 2 \
            --gradient_accumulation_steps 8
        ;;

    *)
        echo "Usage: $0 {generate|train|all} [extra args]"
        exit 1
        ;;
esac
