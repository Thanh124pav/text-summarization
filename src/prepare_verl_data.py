"""Prepare data for verl GRPO training.

verl expects parquet files with specific columns.
This script converts JSONL summarization data to the required format.
"""

import argparse
import json

import pandas as pd

from data_utils import format_prompt


def convert_jsonl_to_parquet(input_path: str, output_path: str):
    """Convert JSONL with input/output/category to parquet for verl."""
    records = []
    with open(input_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)
            cat = item.get("category", None)
            prompt = format_prompt(item["input"], cat)
            records.append({
                "prompt": prompt,
                "reference": item["output"],
                "category": cat or "",
            })

    df = pd.DataFrame(records)
    df.to_parquet(output_path, index=False)
    print(f"Converted {len(records)} records to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Convert JSONL to parquet for verl")
    parser.add_argument("--input", type=str, required=True, help="Input JSONL file")
    parser.add_argument("--output", type=str, required=True, help="Output parquet file")
    args = parser.parse_args()

    convert_jsonl_to_parquet(args.input, args.output)


if __name__ == "__main__":
    main()
