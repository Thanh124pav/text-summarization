# Text Summarization with SFT, GRPO & DPO

Project tóm tắt văn bản sử dụng các phương pháp finetuning khác nhau trên các model decoder-only nhỏ từ HuggingFace.

## Cấu trúc project

```
text-summarization/
├── src/
│   ├── data_utils.py          # Tiện ích load và xử lý dữ liệu JSONL
│   ├── model_utils.py         # Load model, tokenizer, LoRA, quantization
│   ├── train_sft.py           # Supervised Finetuning
│   ├── train_grpo.py          # GRPO với reward: độ dài + ROUGE-2
│   ├── train_dpo.py           # Direct Preference Optimization
│   ├── generate_dpo_pairs.py  # Tạo cặp chosen/rejected cho DPO
│   ├── inference.py           # Inference (đơn lẻ hoặc batch)
│   └── evaluate.py            # Đánh giá ROUGE scores
├── data/
│   ├── sample_train.jsonl     # Dữ liệu mẫu cho SFT/GRPO
│   └── sample_dpo.jsonl       # Dữ liệu mẫu cho DPO
├── configs/                   # File cấu hình YAML
├── scripts/                   # Shell scripts chạy training
├── requirements.txt
└── README.md
```

## Định dạng dữ liệu

File JSONL với các trường:

```json
{
  "input": "Văn bản cần tóm tắt...",
  "output": "Bản tóm tắt...",
  "category": "khoa_hoc"
}
```

Với DPO, cần thêm trường `rejected`:

```json
{
  "input": "Văn bản cần tóm tắt...",
  "output": "Tóm tắt tốt (chosen)...",
  "rejected": "Tóm tắt kém (rejected)...",
  "category": "khoa_hoc"
}
```

## Các model được hỗ trợ

| Key | Model | Params |
|-----|-------|--------|
| `gpt2` | GPT-2 | 124M |
| `gpt2-medium` | GPT-2 Medium | 355M |
| `phi-1.5` | Phi-1.5 | 1.3B |
| `phi-2` | Phi-2 | 2.7B |
| `qwen2-0.5b` | Qwen2-0.5B | 0.5B |
| `qwen2-1.5b` | Qwen2-1.5B | 1.5B |
| `tinyllama` | TinyLlama-1.1B | 1.1B |
| `gemma-2b` | Gemma-2B | 2B |
| `pythia-410m` | Pythia-410M | 410M |
| `pythia-1b` | Pythia-1B | 1B |
| `bloom-560m` | BLOOM-560M | 560M |
| `opt-350m` | OPT-350M | 350M |

Hoặc truyền trực tiếp tên model trên HuggingFace.

## Cài đặt

```bash
pip install -r requirements.txt
```

## Sử dụng

### 1. SFT (Supervised Finetuning)

```bash
python src/train_sft.py \
    --model qwen2-0.5b \
    --train_data data/sample_train.jsonl \
    --output_dir outputs/sft \
    --num_epochs 3 \
    --batch_size 4 \
    --learning_rate 2e-4
```

### 2. GRPO (Group Relative Policy Optimization)

GRPO sử dụng 2 hàm reward:
- **Length reward** (weight=0.3): Thưởng cho tóm tắt có độ dài phù hợp, phạt quá ngắn/dài
- **ROUGE-2 reward** (weight=0.7): Đo bigram overlap với bản tóm tắt tham chiếu

```bash
python src/train_grpo.py \
    --model qwen2-0.5b \
    --train_data data/sample_train.jsonl \
    --output_dir outputs/grpo \
    --length_weight 0.3 \
    --rouge_weight 0.7 \
    --target_length 80 \
    --num_generations 4
```

### 3. DPO (Direct Preference Optimization)

#### Bước 3a: Tạo cặp preference (nếu chưa có trường `rejected`)

```bash
python src/generate_dpo_pairs.py \
    --model qwen2-0.5b \
    --adapter_path outputs/sft/final \
    --input_data data/sample_train.jsonl \
    --output_data data/generated_dpo.jsonl
```

#### Bước 3b: Train DPO

```bash
python src/train_dpo.py \
    --model qwen2-0.5b \
    --train_data data/sample_dpo.jsonl \
    --output_dir outputs/dpo \
    --beta 0.1 \
    --learning_rate 5e-6
```

### 4. Inference

```bash
# Đơn lẻ
python src/inference.py \
    --model qwen2-0.5b \
    --adapter_path outputs/sft/final \
    --input_text "Văn bản cần tóm tắt..."

# Batch
python src/inference.py \
    --model qwen2-0.5b \
    --adapter_path outputs/sft/final \
    --input_file data/sample_train.jsonl \
    --output_file outputs/predictions.jsonl
```

### 5. Đánh giá

```bash
python src/evaluate.py --predictions outputs/predictions.jsonl
```

### Pipeline đầy đủ

```bash
bash scripts/run_pipeline.sh
```

## Pipeline đề xuất

```
SFT → Generate DPO pairs → DPO → GRPO → Evaluate
```

1. **SFT**: Finetune model cơ bản trên dữ liệu tóm tắt
2. **Generate pairs**: Dùng model SFT để sinh rejected samples
3. **DPO**: Train model phân biệt tóm tắt tốt/xấu
4. **GRPO**: Tối ưu thêm với reward dựa trên độ dài và ROUGE-2
5. **Evaluate**: Đánh giá bằng ROUGE scores

## Tùy chọn nâng cao

- **LoRA**: Mặc định bật, tắt với `--no_lora`
- **4-bit quantization**: `--load_in_4bit`
- **8-bit quantization**: `--load_in_8bit`
- **Wandb logging**: `--report_to wandb`
