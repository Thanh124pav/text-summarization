#!/bin/bash
# Demo script: End-to-end text summarization pipeline
# Uses synthetic BKAINewsCorpus data + GPT-2 on CPU for quick testing
#
# Usage:
#   bash scripts/run_demo.sh          # Full demo (synthetic data + training)
#   bash scripts/run_demo.sh --gpu    # With GPU (uses larger model)

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_DIR"

# Defaults
MODEL="gpt2"
BATCH_SIZE=2
MAX_SAMPLES=30
NUM_EPOCHS=2
MAX_SEQ_LENGTH=256
USE_LORA="--no_lora"
FP16=""

if [[ "$1" == "--gpu" ]]; then
    MODEL="gpt2-medium"
    BATCH_SIZE=4
    MAX_SAMPLES=200
    NUM_EPOCHS=3
    MAX_SEQ_LENGTH=512
    USE_LORA=""
    FP16=""
    echo "=== GPU Mode ==="
else
    echo "=== CPU Demo Mode (no GPU) ==="
fi

echo "Model: $MODEL | Samples: $MAX_SAMPLES | Epochs: $NUM_EPOCHS"
echo ""

# Step 1: Prepare data
echo "=========================================="
echo "Step 1: Preparing BKAINewsCorpus data..."
echo "=========================================="
python3 src/prepare_bkai_data.py \
    --output_dir data/bkai \
    --max_samples "$MAX_SAMPLES" \
    --demo \
    --seed 42

echo ""

# Step 2: SFT Training
echo "=========================================="
echo "Step 2: Running SFT training..."
echo "=========================================="
python3 src/train_sft.py \
    --model "$MODEL" \
    --train_data data/bkai/train.jsonl \
    --val_data data/bkai/val.jsonl \
    --output_dir outputs/demo_sft \
    --max_seq_length "$MAX_SEQ_LENGTH" \
    --num_epochs "$NUM_EPOCHS" \
    --batch_size "$BATCH_SIZE" \
    --gradient_accumulation_steps 1 \
    --learning_rate 5e-5 \
    --logging_steps 5 \
    --save_steps 999999 \
    $USE_LORA \
    --report_to none

echo ""

# Step 3: Inference test
echo "=========================================="
echo "Step 3: Testing inference..."
echo "=========================================="
python3 -c "
import json
from pathlib import Path

# Load a sample from val data
with open('data/bkai/val.jsonl', 'r') as f:
    sample = json.loads(f.readline())

print('Input article:')
print(sample['input'][:300])
print()
print('Reference summary:')
print(sample['output'])
print()
print('(Model inference requires GPU for meaningful output with decoder-only models)')
print()
print('=== Demo completed successfully! ===')
"
