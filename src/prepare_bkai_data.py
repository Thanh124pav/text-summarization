"""Download and prepare BKAINewsCorpus for text summarization training.

BKAINewsCorpus (bkai-foundation-models/BKAINewsCorpus) is a Vietnamese news
corpus with articles and their summaries across multiple categories.

Usage:
    python prepare_bkai_data.py --output_dir data/bkai --max_samples 5000
    python prepare_bkai_data.py --output_dir data/bkai --max_samples 100 --demo
"""

import argparse
import json
import random
from pathlib import Path


def load_bkai_from_huggingface(max_samples: int | None = None):
    """Load BKAINewsCorpus from HuggingFace."""
    from datasets import load_dataset

    print("Downloading BKAINewsCorpus from HuggingFace...")
    ds = load_dataset("bkai-foundation-models/BKAINewsCorpus", split="train")

    if max_samples and max_samples < len(ds):
        ds = ds.shuffle(seed=42).select(range(max_samples))
        print(f"Selected {max_samples} samples from {len(ds)} total")

    return ds


def convert_to_jsonl(dataset, output_path: str):
    """Convert HuggingFace dataset to our JSONL format (input, output, category)."""
    # BKAINewsCorpus typical fields: 'text'/'content' (article), 'summary'/'title',
    # 'category'/'topic'
    # We detect the actual field names from the dataset
    sample = dataset[0]
    columns = list(sample.keys())
    print(f"Dataset columns: {columns}")

    # Map fields - try common field names
    text_field = None
    for f in ["text", "content", "article", "body", "document"]:
        if f in columns:
            text_field = f
            break

    summary_field = None
    for f in ["summary", "abstract", "title", "headline"]:
        if f in columns:
            summary_field = f
            break

    category_field = None
    for f in ["category", "topic", "label", "section"]:
        if f in columns:
            category_field = f
            break

    if not text_field or not summary_field:
        raise ValueError(
            f"Cannot detect text/summary fields from columns: {columns}. "
            f"Detected text={text_field}, summary={summary_field}"
        )

    print(f"Field mapping: text={text_field}, summary={summary_field}, category={category_field}")

    records = []
    skipped = 0
    for item in dataset:
        text = item[text_field]
        summary = item[summary_field]

        if not text or not summary or len(text.strip()) < 50 or len(summary.strip()) < 10:
            skipped += 1
            continue

        record = {
            "input": text.strip(),
            "output": summary.strip(),
        }
        if category_field and item.get(category_field):
            record["category"] = str(item[category_field]).strip()

        records.append(record)

    print(f"Converted {len(records)} records ({skipped} skipped due to short text)")
    return records


def create_synthetic_bkai_data(num_samples: int = 50) -> list[dict]:
    """Create synthetic Vietnamese news data mimicking BKAINewsCorpus format.

    Used for testing when HuggingFace is not accessible.
    """
    categories = {
        "kinh_te": [
            (
                "Ngân hàng Nhà nước Việt Nam vừa công bố báo cáo tài chính quý III năm 2024 "
                "cho thấy tăng trưởng tín dụng đạt 8.5%, vượt kỳ vọng của thị trường. "
                "Lãi suất cho vay bình quân giảm 0.3% so với quý trước, hỗ trợ doanh nghiệp "
                "vừa và nhỏ tiếp cận nguồn vốn. Chỉ số VN-Index cũng ghi nhận mức tăng 12% "
                "trong quý, phản ánh niềm tin của nhà đầu tư vào triển vọng kinh tế.",
                "Tăng trưởng tín dụng quý III đạt 8.5%, lãi suất cho vay giảm 0.3%, VN-Index tăng 12%."
            ),
            (
                "Xuất khẩu thủy sản Việt Nam trong 9 tháng đầu năm 2024 đạt 7.2 tỷ USD, "
                "tăng 15% so với cùng kỳ năm ngoái. Tôm vẫn là mặt hàng chủ lực với kim ngạch "
                "3.1 tỷ USD, tiếp theo là cá tra với 1.5 tỷ USD. Thị trường Mỹ, EU và Nhật Bản "
                "chiếm hơn 60% tổng kim ngạch xuất khẩu thủy sản.",
                "Xuất khẩu thủy sản 9 tháng đạt 7.2 tỷ USD, tăng 15%, tôm dẫn đầu với 3.1 tỷ USD."
            ),
            (
                "Thị trường bất động sản TP.HCM ghi nhận sự phục hồi trong quý III với "
                "số lượng giao dịch tăng 25% so với quý trước. Phân khúc căn hộ tầm trung "
                "có mức giá từ 2-4 tỷ đồng chiếm 65% tổng giao dịch. Các chuyên gia dự báo "
                "thị trường sẽ tiếp tục ấm lên trong quý IV nhờ chính sách hỗ trợ lãi suất.",
                "Bất động sản TP.HCM phục hồi, giao dịch tăng 25%, căn hộ tầm trung chiếm 65%."
            ),
        ],
        "khoa_hoc": [
            (
                "Viện Hàn lâm Khoa học và Công nghệ Việt Nam vừa công bố kết quả nghiên cứu "
                "về ứng dụng trí tuệ nhân tạo trong chẩn đoán bệnh lý phổi. Hệ thống AI được "
                "phát triển có khả năng phát hiện 15 loại bệnh phổi từ ảnh X-quang với độ chính "
                "xác 96.7%. Nghiên cứu đã được đánh giá trên 50,000 ảnh X-quang từ các bệnh viện "
                "lớn tại Việt Nam.",
                "Viện Hàn lâm phát triển AI chẩn đoán 15 loại bệnh phổi từ X-quang, độ chính xác 96.7%."
            ),
            (
                "Nhóm nghiên cứu Đại học Bách khoa Hà Nội đã phát triển thành công vật liệu "
                "nano composite mới có khả năng lọc nước hiệu quả gấp 3 lần so với vật liệu "
                "truyền thống. Vật liệu được chế tạo từ graphene oxide kết hợp với hạt nano bạc, "
                "có thể loại bỏ 99.5% kim loại nặng và 99.9% vi khuẩn trong nước.",
                "ĐHBK Hà Nội phát triển vật liệu nano lọc nước hiệu quả gấp 3 lần, loại bỏ 99.5% kim loại nặng."
            ),
        ],
        "the_thao": [
            (
                "Đội tuyển bóng đá nữ Việt Nam đã giành chiến thắng 2-0 trước Thái Lan trong "
                "trận bán kết SEA Games 2024. Hai bàn thắng được ghi bởi Huỳnh Như ở phút 34 "
                "và Nguyễn Thị Bích Thùy ở phút 67. Đây là chiến thắng thứ 5 liên tiếp của "
                "đội tuyển nữ Việt Nam trước Thái Lan tại SEA Games.",
                "Tuyển nữ Việt Nam thắng Thái Lan 2-0 bán kết SEA Games, bàn thắng của Huỳnh Như và Bích Thùy."
            ),
            (
                "VĐV Nguyễn Huy Hoàng đã phá kỷ lục quốc gia ở nội dung 1500m tự do tại giải "
                "bơi vô địch châu Á 2024. Anh về đích với thời gian 14 phút 58 giây 23, cải thiện "
                "0.5 giây so với kỷ lục cũ do chính anh thiết lập năm 2023. Thành tích này giúp "
                "anh giành huy chương đồng tại giải đấu.",
                "Nguyễn Huy Hoàng phá kỷ lục QG 1500m tự do, đạt 14:58.23 tại vô địch châu Á 2024."
            ),
        ],
        "cong_nghe": [
            (
                "VinAI Research vừa công bố mô hình ngôn ngữ lớn PhoGPT phiên bản 2.0 với "
                "7.5 tỷ tham số, được huấn luyện trên 500GB dữ liệu tiếng Việt. Mô hình đạt "
                "kết quả state-of-the-art trên các bài đánh giá tiếng Việt bao gồm tóm tắt văn bản, "
                "trả lời câu hỏi và phân loại cảm xúc. PhoGPT 2.0 được phát hành miễn phí cho "
                "cộng đồng nghiên cứu.",
                "VinAI ra mắt PhoGPT 2.0 với 7.5 tỷ tham số, đạt SOTA trên các benchmark tiếng Việt."
            ),
            (
                "FPT Software đã ký hợp đồng trị giá 200 triệu USD với đối tác Nhật Bản để "
                "phát triển hệ thống quản lý chuỗi cung ứng sử dụng blockchain và AI. Dự án "
                "kéo dài 3 năm, dự kiến triển khai tại 15 quốc gia trong khu vực châu Á - "
                "Thái Bình Dương. Đây là hợp đồng lớn nhất trong lịch sử FPT Software.",
                "FPT Software ký hợp đồng 200 triệu USD phát triển hệ thống chuỗi cung ứng blockchain-AI."
            ),
        ],
        "phap_luat": [
            (
                "Bộ Công an vừa triệt phá đường dây lừa đảo qua mạng xuyên quốc gia, bắt giữ "
                "25 đối tượng tại 3 tỉnh thành. Đường dây hoạt động từ năm 2022, lừa đảo hơn "
                "500 nạn nhân với tổng số tiền chiếm đoạt khoảng 150 tỷ đồng. Các đối tượng sử dụng "
                "các ứng dụng giả mạo ngân hàng và sàn đầu tư để thực hiện hành vi phạm tội.",
                "Bộ Công an triệt phá đường dây lừa đảo mạng xuyên quốc gia, bắt 25 người, chiếm đoạt 150 tỷ."
            ),
        ],
        "giao_duc": [
            (
                "Bộ Giáo dục và Đào tạo công bố kết quả kỳ thi tốt nghiệp THPT 2024 với tỷ lệ "
                "đỗ toàn quốc đạt 98.5%. Điểm trung bình môn Toán là 6.8, Ngữ văn 7.2 và "
                "Tiếng Anh 5.5. Hà Nội dẫn đầu cả nước với điểm trung bình 3 môn chính đạt 7.1. "
                "Năm nay có 5 thí sinh đạt điểm tuyệt đối 30/30.",
                "Tỷ lệ tốt nghiệp THPT 2024 đạt 98.5%, Toán TB 6.8, Văn 7.2, Anh 5.5, 5 thí sinh đạt 30 điểm."
            ),
        ],
        "suc_khoe": [
            (
                "Bệnh viện Chợ Rẫy TP.HCM vừa thực hiện thành công ca ghép gan từ người cho "
                "sống đầu tiên bằng phương pháp nội soi. Ca phẫu thuật kéo dài 12 giờ với sự "
                "tham gia của 30 bác sĩ. Bệnh nhân là nam giới 45 tuổi bị xơ gan giai đoạn cuối. "
                "Sau 2 tuần, bệnh nhân hồi phục tốt và dự kiến xuất viện trong tuần tới.",
                "BV Chợ Rẫy ghép gan nội soi đầu tiên thành công, ca mổ 12 giờ, bệnh nhân phục hồi tốt."
            ),
        ],
    }

    records = []
    for cat, examples in categories.items():
        for text, summary in examples:
            records.append({
                "input": text,
                "output": summary,
                "category": cat,
            })

    # Augment by creating variations
    while len(records) < num_samples:
        base = random.choice(records[:len(records) // 2 + 1])
        record = {
            "input": base["input"],
            "output": base["output"],
            "category": base["category"],
        }
        records.append(record)

    random.shuffle(records)
    return records[:num_samples]


def save_jsonl(records: list[dict], output_path: str):
    """Save records to JSONL file."""
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    print(f"Saved {len(records)} records to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Prepare BKAINewsCorpus for summarization")
    parser.add_argument("--output_dir", type=str, default="data/bkai")
    parser.add_argument("--max_samples", type=int, default=5000, help="Max training samples")
    parser.add_argument("--val_ratio", type=float, default=0.1, help="Validation split ratio")
    parser.add_argument("--demo", action="store_true", help="Use synthetic data (offline mode)")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.demo:
        print("Demo mode: creating synthetic Vietnamese news data...")
        records = create_synthetic_bkai_data(args.max_samples)
    else:
        try:
            ds = load_bkai_from_huggingface(args.max_samples)
            records = convert_to_jsonl(ds, str(output_dir / "train.jsonl"))
        except Exception as e:
            print(f"Failed to load from HuggingFace: {e}")
            print("Falling back to synthetic data...")
            records = create_synthetic_bkai_data(args.max_samples)

    # Split into train/val
    random.shuffle(records)
    val_size = int(len(records) * args.val_ratio)
    val_records = records[:val_size]
    train_records = records[val_size:]

    save_jsonl(train_records, str(output_dir / "train.jsonl"))
    if val_records:
        save_jsonl(val_records, str(output_dir / "val.jsonl"))

    # Print stats
    print(f"\nDataset statistics:")
    print(f"  Train: {len(train_records)} samples")
    print(f"  Val:   {len(val_records)} samples")

    categories = {}
    for r in train_records:
        cat = r.get("category", "unknown")
        categories[cat] = categories.get(cat, 0) + 1
    print(f"  Categories: {dict(sorted(categories.items()))}")


if __name__ == "__main__":
    main()
