#!/bin/bash
# Full training pipeline: SFT -> Generate DPO pairs -> DPO -> GRPO
set -e

cd "$(dirname "$0")/.."

echo "============================================"
echo "Step 1: SFT Training"
echo "============================================"
python src/train_sft.py \
    --model qwen2-0.5b \
    --train_data data/sample_train.jsonl \
    --output_dir outputs/sft \
    --num_epochs 3 \
    --batch_size 4 \
    --gradient_accumulation_steps 4 \
    --learning_rate 2e-4

echo ""
echo "============================================"
echo "Step 2: Generate DPO pairs from SFT model"
echo "============================================"
python src/generate_dpo_pairs.py \
    --model qwen2-0.5b \
    --adapter_path outputs/sft/final \
    --input_data data/sample_train.jsonl \
    --output_data data/generated_dpo.jsonl \
    --temperature 1.2 \
    --num_samples 3

echo ""
echo "============================================"
echo "Step 3: DPO Training"
echo "============================================"
python src/train_dpo.py \
    --model qwen2-0.5b \
    --train_data data/generated_dpo.jsonl \
    --output_dir outputs/dpo \
    --num_epochs 1 \
    --batch_size 2 \
    --learning_rate 5e-6

echo ""
echo "============================================"
echo "Step 4: GRPO Training"
echo "============================================"
python src/train_grpo.py \
    --model qwen2-0.5b \
    --train_data data/sample_train.jsonl \
    --output_dir outputs/grpo \
    --num_epochs 1 \
    --batch_size 2 \
    --learning_rate 1e-5 \
    --length_weight 0.3 \
    --rouge_weight 0.7

echo ""
echo "============================================"
echo "Step 5: Inference & Evaluation"
echo "============================================"
python src/inference.py \
    --model qwen2-0.5b \
    --adapter_path outputs/grpo/final \
    --input_file data/sample_train.jsonl \
    --output_file outputs/predictions.jsonl

python src/evaluate.py \
    --predictions outputs/predictions.jsonl

echo ""
echo "Pipeline complete!"
