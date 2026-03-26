"""Generate hard distillation dataset from a teacher model endpoint.

Calls a teacher model (via OpenAI-compatible API, e.g. vLLM served model)
to generate summaries for each (input, style) pair. The output is a JSONL
file with both the ground truth and teacher-generated summary, ready for
hard distillation training.

Output format (JSONL):
  {
    "input": "...",
    "output": "ground truth summary",
    "teacher_output": "teacher-generated summary",
    "style": "bao_chi",
    "category": "kinh_te"
  }

Usage:
  # Generate from vLLM endpoint
  python src/distill_generate.py \
      --input_data data/bkai/train.jsonl \
      --output_data data/bkai/train_distill.jsonl \
      --teacher_endpoint http://localhost:8000/v1 \
      --teacher_model Qwen/Qwen3-32B \
      --with_style

  # Generate from any OpenAI-compatible API
  python src/distill_generate.py \
      --input_data data/bkai/train.jsonl \
      --output_data data/bkai/train_distill.jsonl \
      --teacher_endpoint https://api.example.com/v1 \
      --teacher_model my-model \
      --api_key sk-xxx
"""

import argparse
import json
import random
import time
from pathlib import Path

from tqdm import tqdm

from data_utils import STYLE_NAMES, STYLE_DISPLAY_NAMES


def load_jsonl(path: str) -> list[dict]:
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def build_teacher_prompt(
    text: str,
    style: str | None = None,
    category: str | None = None,
) -> str:
    """Build the prompt sent to the teacher model."""
    parts = []
    if style:
        style_display = STYLE_DISPLAY_NAMES.get(style, style)
        parts.append(
            f"Hãy tóm tắt văn bản sau theo phong cách {style_display}."
        )
    else:
        parts.append("Hãy tóm tắt văn bản sau.")

    if category:
        parts.append(f"Chủ đề: {category}.")

    parts.append(f"\nVăn bản:\n{text}\n\nTóm tắt:")
    return " ".join(parts)


def call_teacher_openai(
    client,
    model: str,
    prompt: str,
    max_tokens: int = 512,
    temperature: float = 0.3,
) -> str:
    """Call teacher model via OpenAI-compatible API."""
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=max_tokens,
        temperature=temperature,
    )
    return response.choices[0].message.content.strip()


def call_teacher_requests(
    endpoint: str,
    model: str,
    prompt: str,
    max_tokens: int = 512,
    temperature: float = 0.3,
    api_key: str | None = None,
) -> str:
    """Call teacher model via raw HTTP requests (no openai dependency)."""
    import urllib.request

    url = f"{endpoint.rstrip('/')}/chat/completions"
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": temperature,
    }

    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(url, data=data, headers=headers)
    resp = urllib.request.urlopen(req, timeout=120)
    result = json.loads(resp.read().decode("utf-8"))
    return result["choices"][0]["message"]["content"].strip()


def main():
    parser = argparse.ArgumentParser(
        description="Generate hard distillation data from teacher model"
    )
    parser.add_argument("--input_data", type=str, required=True)
    parser.add_argument("--output_data", type=str, required=True)
    parser.add_argument("--teacher_endpoint", type=str, required=True,
                        help="OpenAI-compatible API base URL (e.g. http://localhost:8000/v1)")
    parser.add_argument("--teacher_model", type=str, required=True,
                        help="Model name at the endpoint")
    parser.add_argument("--api_key", type=str, default=None)
    parser.add_argument("--with_style", action="store_true",
                        help="Assign random styles and include in teacher prompt")
    parser.add_argument("--max_tokens", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=0.3)
    parser.add_argument("--batch_pause", type=float, default=0.0,
                        help="Seconds to pause between requests (rate limiting)")
    parser.add_argument("--max_retries", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    rng = random.Random(args.seed)
    records = load_jsonl(args.input_data)
    print(f"Loaded {len(records)} records from {args.input_data}")

    # Try to use openai client, fallback to raw requests
    use_openai = False
    try:
        from openai import OpenAI
        client = OpenAI(
            base_url=args.teacher_endpoint,
            api_key=args.api_key or "dummy",
        )
        use_openai = True
        print("Using OpenAI client")
    except ImportError:
        print("openai not installed, using raw HTTP requests")

    output_path = Path(args.output_data)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    results = []
    failed = 0

    with open(output_path, "w", encoding="utf-8") as f_out:
        for record in tqdm(records, desc="Generating teacher outputs"):
            text = record["input"]
            category = record.get("category")
            style = None
            if args.with_style:
                style = record.get("style") or rng.choice(STYLE_NAMES)

            prompt = build_teacher_prompt(text, style=style, category=category)

            # Call teacher with retries
            teacher_output = None
            for attempt in range(args.max_retries):
                try:
                    if use_openai:
                        teacher_output = call_teacher_openai(
                            client, args.teacher_model, prompt,
                            max_tokens=args.max_tokens,
                            temperature=args.temperature,
                        )
                    else:
                        teacher_output = call_teacher_requests(
                            args.teacher_endpoint, args.teacher_model, prompt,
                            max_tokens=args.max_tokens,
                            temperature=args.temperature,
                            api_key=args.api_key,
                        )
                    break
                except Exception as e:
                    if attempt < args.max_retries - 1:
                        wait = 2 ** attempt
                        print(f"  Retry {attempt+1}/{args.max_retries} after {wait}s: {e}")
                        time.sleep(wait)
                    else:
                        print(f"  Failed after {args.max_retries} attempts: {e}")
                        failed += 1

            if teacher_output is None:
                continue

            out_record = {
                "input": text,
                "output": record["output"],
                "teacher_output": teacher_output,
            }
            if category:
                out_record["category"] = category
            if style:
                out_record["style"] = style

            f_out.write(json.dumps(out_record, ensure_ascii=False) + "\n")
            results.append(out_record)

            if args.batch_pause > 0:
                time.sleep(args.batch_pause)

    print(f"\nDone: {len(results)} generated, {failed} failed")
    print(f"Saved to {output_path}")

    # Print sample
    if results:
        s = results[0]
        print(f"\nSample:")
        print(f"  Input:   {s['input'][:100]}...")
        print(f"  GT:      {s['output'][:100]}...")
        print(f"  Teacher: {s['teacher_output'][:100]}...")
        if "style" in s:
            print(f"  Style:   {s['style']}")


if __name__ == "__main__":
    main()
