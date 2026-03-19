"""Custom reward functions for verl GRPO training on summarization.

verl uses a compute_score function that receives data_source, solution_str,
ground_truth, and extra_info, and returns a float reward score.
"""

from rouge_score import rouge_scorer

_scorer = rouge_scorer.RougeScorer(["rouge2"], use_stemmer=True)


def compute_score(
    data_source: str,
    solution_str: str,
    ground_truth: dict,
    extra_info: dict | None = None,
) -> float:
    """Compute reward score for a generated summary.

    Combines ROUGE-2 F1 score with a length penalty.

    Args:
        data_source: Dataset identifier (unused).
        solution_str: Generated summary text.
        ground_truth: Dict with 'reference' key containing the reference summary.
        extra_info: Optional extra information.

    Returns:
        Float reward score in [−1, 1].
    """
    reference = ground_truth.get("reference", "")
    if not solution_str.strip() or not reference:
        return -1.0

    # ROUGE-2 score (0 to 1)
    rouge_result = _scorer.score(reference, solution_str)
    rouge2_f1 = rouge_result["rouge2"].fmeasure

    # Length reward
    words = solution_str.split()
    n_words = len(words)
    ref_words = len(reference.split())
    target_len = ref_words  # Aim for similar length as reference

    if n_words == 0:
        length_score = -1.0
    elif n_words <= target_len:
        length_score = n_words / max(target_len, 1)
    elif n_words <= target_len * 2:
        length_score = 1.0 - (n_words - target_len) / max(target_len, 1)
    else:
        length_score = -0.5

    # Weighted combination
    reward = 0.7 * rouge2_f1 + 0.3 * max(length_score, 0.0)

    return reward
