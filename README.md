# Text Summarization with SFT, GRPO, DPO, SAGE & SkillRL

Project tóm tắt văn bản sử dụng các phương pháp finetuning và RL khác nhau trên các model decoder-only nhỏ từ HuggingFace.

## Các phương pháp

| Phương pháp | Paper | Mô tả |
|-------------|-------|-------|
| **SFT** | - | Supervised Finetuning cơ bản |
| **GRPO** (TRL) | [DeepSeekMath](https://arxiv.org/abs/2402.03300) | Group Relative Policy Optimization với reward: ROUGE-2 + độ dài |
| **GRPO** (verl) | [HybridFlow](https://github.com/volcengine/verl) | GRPO sử dụng thư viện verl của ByteDance, hỗ trợ scale lớn |
| **DPO** | [DPO](https://arxiv.org/abs/2305.18290) | Direct Preference Optimization với cặp chosen/rejected |
| **SAGE** | [arXiv:2602.03143](https://arxiv.org/abs/2602.03143) | Self-Hint Aligned GRPO — inject hints khi advantage collapse |
| **SkillRL** | [arXiv:2602.08234](https://arxiv.org/abs/2602.08234) | Recursive Skill-Augmented RL — skill bank tiến hóa theo training |

## Cấu trúc project

```
text-summarization/
├── src/
│   ├── data_utils.py          # Tiện ích load và xử lý dữ liệu JSONL
│   ├── model_utils.py         # Load model, tokenizer, LoRA, quantization
│   ├── train_sft.py           # Supervised Finetuning
│   ├── train_grpo.py          # GRPO (TRL) với reward: độ dài + ROUGE-2
│   ├── train_grpo_verl.py     # GRPO (verl) — scalable RL training
│   ├── train_sage.py          # SAGE: Self-Hinting GRPO
│   ├── train_skillrl.py       # SkillRL: Recursive Skill-Augmented RL
│   ├── train_dpo.py           # Direct Preference Optimization
│   ├── generate_dpo_pairs.py  # Tạo cặp chosen/rejected cho DPO
│   ├── verl_reward.py         # Custom reward function cho verl
│   ├── prepare_verl_data.py   # Convert JSONL → parquet cho verl
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

### 4. GRPO với verl (Scalable RL)

[verl](https://github.com/volcengine/verl) là thư viện RL của ByteDance, hỗ trợ scale training lên hàng trăm GPU.

```bash
# Bước 1: Convert dữ liệu sang parquet
python src/prepare_verl_data.py --input data/sample_train.jsonl --output data/train.parquet

# Bước 2: Train GRPO với verl
python src/train_grpo_verl.py \
    --model Qwen/Qwen2-0.5B \
    --train_data data/train.parquet \
    --output_dir outputs/grpo_verl \
    --num_generations 4 \
    --kl_coef 0.001
```

### 5. SAGE — Self-Hinting GRPO (arXiv:2602.03143)

SAGE giải quyết vấn đề **advantage collapse** trong GRPO bằng cách inject hints vào prompt khi tất cả rollouts trong group nhận cùng reward.

**Ý tưởng chính:**
- Khi GRPO bị stall (advantage = 0 cho mọi sample), SAGE inject "gợi ý" (key points) vào prompt
- Hints tăng diversity trong group → advantage không collapse
- Tại inference: không dùng hint, chỉ dùng policy đã được cải thiện

**2 schemes:**
- `sage-light` (Scheme 1): Inject hints cho một phần prompts ở mức epoch
- `sage` (Scheme 2): Detect advantage collapse per-prompt, inject hint khi cần

```bash
python src/train_sage.py \
    --model qwen2-0.5b \
    --train_data data/sample_train.jsonl \
    --output_dir outputs/sage \
    --scheme sage-light \
    --hint_ratio 0.3 \
    --num_generations 4
```

### 6. SkillRL — Recursive Skill-Augmented RL (arXiv:2602.08234)

SkillRL xây dựng **SkillBank** (thư viện kỹ năng) cho tóm tắt văn bản và tiến hóa nó trong quá trình training.

**Pipeline:**
1. Khởi tạo SkillBank với các kỹ năng tóm tắt mặc định (General + Task-Specific)
2. GRPO training với skills inject vào prompt
3. Validation → thu thập failures (ROUGE-2 < threshold)
4. Skill Evolution: phân tích failure patterns → tạo skills mới
5. Lặp lại bước 2-4 (recursive evolution)

**SkillBank bao gồm:**
- **General Skills**: Kỹ năng tóm tắt chung (trích xuất info, kiểm soát độ dài, ...)
- **Task-Specific Skills**: Kỹ năng theo loại bài (khoa học, kinh tế, thể thao, ...)
- **Common Mistakes**: Các lỗi thường gặp cần tránh

```bash
python src/train_skillrl.py \
    --model qwen2-0.5b \
    --train_data data/sample_train.jsonl \
    --output_dir outputs/skillrl \
    --num_evolution_rounds 3 \
    --evolution_threshold 0.4 \
    --top_k_skills 4
```

### 7. Inference & Đánh giá

```bash
# Inference
python src/inference.py \
    --model qwen2-0.5b \
    --adapter_path outputs/sft/final \
    --input_file data/sample_train.jsonl \
    --output_file outputs/predictions.jsonl

# Đánh giá ROUGE
python src/evaluate.py --predictions outputs/predictions.jsonl
```

### Pipeline đầy đủ

```bash
bash scripts/run_pipeline.sh
```

## Pipelines đề xuất

### Pipeline cơ bản
```
SFT → Generate DPO pairs → DPO → GRPO → Evaluate
```

### Pipeline nâng cao (SAGE + SkillRL)
```
SFT → SAGE (Self-Hinting GRPO) → SkillRL (Skill Evolution) → Evaluate
```

1. **SFT**: Finetune model cơ bản trên dữ liệu tóm tắt
2. **SAGE**: GRPO với self-hinting để tránh advantage collapse
3. **SkillRL**: GRPO với recursive skill evolution để cải thiện liên tục
4. **Evaluate**: Đánh giá bằng ROUGE scores

## Tùy chọn nâng cao

- **LoRA**: Mặc định bật, tắt với `--no_lora`
- **4-bit quantization**: `--load_in_4bit`
- **8-bit quantization**: `--load_in_8bit`
- **Wandb logging**: `--report_to wandb`

## References

- **GRPO**: [DeepSeekMath: Pushing the Limits of Mathematical Reasoning](https://arxiv.org/abs/2402.03300)
- **verl**: [HybridFlow: A Flexible and Efficient RLHF Framework](https://github.com/volcengine/verl)
- **SAGE**: [Self-Hinting Language Models Enhance Reinforcement Learning](https://arxiv.org/abs/2602.03143) — Liao et al., 2026
- **SkillRL**: [Evolving Agents via Recursive Skill-Augmented RL](https://arxiv.org/abs/2602.08234) — Xia et al., 2026
- **DPO**: [Direct Preference Optimization](https://arxiv.org/abs/2305.18290)
