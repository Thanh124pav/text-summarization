"""Inference script for text summarization models."""

import argparse
import json

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

from data_utils import format_prompt
from model_utils import get_model_name, load_tokenizer


def generate_summary(
    model,
    tokenizer: AutoTokenizer,
    text: str,
    category: str | None = None,
    max_new_tokens: int = 256,
    temperature: float = 0.7,
    top_p: float = 0.9,
    do_sample: bool = True,
) -> str:
    """Generate a summary for the given text."""
    prompt = format_prompt(text, category)
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1024)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            do_sample=do_sample,
            pad_token_id=tokenizer.pad_token_id,
        )

    generated = outputs[0][inputs["input_ids"].shape[1]:]
    return tokenizer.decode(generated, skip_special_tokens=True).strip()


def main():
    parser = argparse.ArgumentParser(description="Inference for Text Summarization")
    parser.add_argument("--model", type=str, required=True, help="Base model name or key")
    parser.add_argument("--adapter_path", type=str, default=None, help="Path to LoRA adapter")
    parser.add_argument("--input_file", type=str, default=None, help="Input JSONL file for batch inference")
    parser.add_argument("--input_text", type=str, default=None, help="Single text to summarize")
    parser.add_argument("--category", type=str, default=None, help="Category for single text")
    parser.add_argument("--output_file", type=str, default=None, help="Output file for batch results")
    parser.add_argument("--max_new_tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top_p", type=float, default=0.9)
    args = parser.parse_args()

    model_name = get_model_name(args.model)
    tokenizer = load_tokenizer(model_name)

    print(f"Loading model: {model_name}")
    model = AutoModelForCausalLM.from_pretrained(
        model_name, device_map="auto", trust_remote_code=True, torch_dtype=torch.float16
    )

    if args.adapter_path:
        print(f"Loading LoRA adapter: {args.adapter_path}")
        model = PeftModel.from_pretrained(model, args.adapter_path)

    model.eval()

    if args.input_text:
        # Single inference
        summary = generate_summary(
            model, tokenizer, args.input_text,
            category=args.category,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
        )
        print(f"\nInput: {args.input_text[:200]}...")
        print(f"\nSummary: {summary}")

    elif args.input_file:
        # Batch inference
        results = []
        with open(args.input_file, "r", encoding="utf-8") as f:
            records = [json.loads(line) for line in f if line.strip()]

        print(f"Processing {len(records)} examples...")
        for i, record in enumerate(records):
            summary = generate_summary(
                model, tokenizer, record["input"],
                category=record.get("category"),
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
                top_p=args.top_p,
            )
            result = {
                "input": record["input"],
                "generated_summary": summary,
            }
            if "output" in record:
                result["reference_summary"] = record["output"]
            results.append(result)

            if (i + 1) % 10 == 0:
                print(f"  Processed {i + 1}/{len(records)}")

        output_path = args.output_file or "predictions.jsonl"
        with open(output_path, "w", encoding="utf-8") as f:
            for r in results:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"Results saved to {output_path}")

    else:
        print("Please provide --input_text or --input_file")


if __name__ == "__main__":
    main()
