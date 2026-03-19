"""Generate DPO preference pairs from an SFT-trained model.

Takes a JSONL dataset with 'input' and 'output' fields and generates
rejected summaries using the model, creating pairs for DPO training.
"""

import argparse
import json

import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

from data_utils import load_jsonl, format_prompt
from model_utils import get_model_name, load_tokenizer


def generate_rejected(
    model,
    tokenizer: AutoTokenizer,
    prompt: str,
    max_new_tokens: int = 256,
    temperature: float = 1.2,
    top_p: float = 0.9,
    num_samples: int = 3,
) -> str:
    """Generate a lower-quality summary by sampling with high temperature.

    Generates multiple samples and picks the one with lowest quality
    (shortest or most divergent from typical output).
    """
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    outputs = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_p=top_p,
        do_sample=True,
        num_return_sequences=num_samples,
        pad_token_id=tokenizer.pad_token_id,
    )

    # Decode and pick the worst (shortest non-empty) summary
    summaries = []
    for output in outputs:
        text = tokenizer.decode(output[inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()
        if text:
            summaries.append(text)

    if not summaries:
        return "No summary generated."

    # Pick the shortest as "rejected" (typically lower quality)
    return min(summaries, key=len)


def main():
    parser = argparse.ArgumentParser(description="Generate DPO pairs using a trained model")
    parser.add_argument("--model", type=str, required=True, help="Base model name or key")
    parser.add_argument("--adapter_path", type=str, default=None, help="Path to LoRA adapter (if used)")
    parser.add_argument("--input_data", type=str, required=True, help="Input JSONL file")
    parser.add_argument("--output_data", type=str, required=True, help="Output JSONL file with rejected pairs")
    parser.add_argument("--temperature", type=float, default=1.2)
    parser.add_argument("--max_new_tokens", type=int, default=256)
    parser.add_argument("--num_samples", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=1)
    args = parser.parse_args()

    model_name = get_model_name(args.model)
    tokenizer = load_tokenizer(model_name)

    print(f"Loading model: {model_name}")
    model = AutoModelForCausalLM.from_pretrained(
        model_name, device_map="auto", trust_remote_code=True, torch_dtype=torch.float16
    )

    if args.adapter_path:
        print(f"Loading LoRA adapter from: {args.adapter_path}")
        model = PeftModel.from_pretrained(model, args.adapter_path)

    model.eval()

    # Load data
    records = load_jsonl(args.input_data)
    print(f"Generating rejected pairs for {len(records)} examples...")

    results = []
    for record in tqdm(records):
        cat = record.get("category", None)
        prompt = format_prompt(record["input"], cat)

        with torch.no_grad():
            rejected = generate_rejected(
                model, tokenizer, prompt,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
                num_samples=args.num_samples,
            )

        result = {
            "input": record["input"],
            "output": record["output"],
            "rejected": rejected,
        }
        if cat:
            result["category"] = cat
        results.append(result)

    # Save
    with open(args.output_data, "w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"Saved {len(results)} DPO pairs to {args.output_data}")


if __name__ == "__main__":
    main()
