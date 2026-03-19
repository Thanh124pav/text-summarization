"""Evaluation script: computes ROUGE scores on predictions."""

import argparse
import json

from rouge_score import rouge_scorer


def evaluate_file(predictions_file: str):
    """Evaluate predictions from a JSONL file with 'generated_summary' and 'reference_summary'."""
    scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)

    scores = {"rouge1": [], "rouge2": [], "rougeL": []}

    with open(predictions_file, "r", encoding="utf-8") as f:
        records = [json.loads(line) for line in f if line.strip()]

    n_evaluated = 0
    for record in records:
        ref = record.get("reference_summary", "")
        gen = record.get("generated_summary", "")
        if not ref or not gen:
            continue

        result = scorer.score(ref, gen)
        for key in scores:
            scores[key].append(result[key].fmeasure)
        n_evaluated += 1

    if n_evaluated == 0:
        print("No valid prediction-reference pairs found.")
        return

    print(f"\nEvaluation Results ({n_evaluated} samples):")
    print("-" * 40)
    for key in scores:
        avg = sum(scores[key]) / len(scores[key])
        print(f"  {key:10s}: {avg:.4f}")
    print("-" * 40)


def main():
    parser = argparse.ArgumentParser(description="Evaluate summarization predictions")
    parser.add_argument("--predictions", type=str, required=True, help="Path to predictions JSONL file")
    args = parser.parse_args()

    evaluate_file(args.predictions)


if __name__ == "__main__":
    main()
