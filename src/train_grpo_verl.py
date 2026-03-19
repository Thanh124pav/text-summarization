"""GRPO training for text summarization using the verl library.

verl (Volcano Engine Reinforcement Learning) is a scalable RL training
framework that supports GRPO, PPO, DAPO, and more.

Installation: pip install verl

Usage:
    # 1. Prepare data
    python src/prepare_verl_data.py --input data/sample_train.jsonl --output data/train.parquet

    # 2. Run GRPO training
    python src/train_grpo_verl.py \\
        --model Qwen/Qwen2-0.5B \\
        --train_data data/train.parquet \\
        --output_dir outputs/grpo_verl
"""

import argparse
import subprocess
import sys
from pathlib import Path


def build_verl_command(args) -> list[str]:
    """Build the verl training command with Hydra overrides."""
    cmd = [
        sys.executable, "-m", "verl.trainer.main_ppo",
        # Algorithm: GRPO
        "algorithm.adv_estimator=grpo",
        # Model
        f"actor_rollout_ref.model.path={args.model}",
        # Data
        f"data.train_files={args.train_data}",
        f"data.max_prompt_length={args.max_prompt_length}",
        f"data.max_response_length={args.max_response_length}",
        f"data.train_batch_size={args.train_batch_size}",
        # GRPO specific: number of generations per prompt
        f"actor_rollout_ref.rollout.n={args.num_generations}",
        # KL loss for GRPO
        "actor_rollout_ref.actor.use_kl_loss=True",
        f"actor_rollout_ref.actor.kl_loss_coef={args.kl_coef}",
        # Training
        f"trainer.total_epochs={args.num_epochs}",
        f"trainer.save_freq={args.save_steps}",
        f"trainer.project_name={args.project_name}",
        f"trainer.experiment_name={args.experiment_name}",
        f"trainer.default_local_dir={args.output_dir}",
        # Reward: custom function
        f"reward.custom_reward_function.path={args.reward_function}",
        # PPO mini batch
        f"algorithm.ppo_mini_batch_size={args.ppo_mini_batch_size}",
    ]

    if args.use_lora:
        cmd.extend([
            "actor_rollout_ref.actor.use_lora=True",
            f"actor_rollout_ref.actor.lora_rank={args.lora_r}",
            f"actor_rollout_ref.actor.lora_alpha={args.lora_alpha}",
        ])

    return cmd


def main():
    parser = argparse.ArgumentParser(description="GRPO Training with verl for Summarization")
    parser.add_argument("--model", type=str, default="Qwen/Qwen2-0.5B",
                        help="HuggingFace model path")
    parser.add_argument("--train_data", type=str, required=True,
                        help="Path to training parquet file")
    parser.add_argument("--output_dir", type=str, default="outputs/grpo_verl")
    parser.add_argument("--num_epochs", type=int, default=1)
    parser.add_argument("--train_batch_size", type=int, default=64)
    parser.add_argument("--ppo_mini_batch_size", type=int, default=16)
    parser.add_argument("--num_generations", type=int, default=4,
                        help="Number of completions per prompt (group size G)")
    parser.add_argument("--max_prompt_length", type=int, default=512)
    parser.add_argument("--max_response_length", type=int, default=256)
    parser.add_argument("--kl_coef", type=float, default=0.001,
                        help="KL loss coefficient for GRPO")
    parser.add_argument("--reward_function", type=str,
                        default="src/verl_reward.py",
                        help="Path to custom reward function")
    parser.add_argument("--use_lora", action="store_true", default=True)
    parser.add_argument("--no_lora", action="store_true")
    parser.add_argument("--lora_r", type=int, default=16)
    parser.add_argument("--lora_alpha", type=int, default=32)
    parser.add_argument("--save_steps", type=int, default=50)
    parser.add_argument("--project_name", type=str, default="text-summarization")
    parser.add_argument("--experiment_name", type=str, default="grpo-verl")
    args = parser.parse_args()

    if args.no_lora:
        args.use_lora = False

    # Ensure output directory
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    cmd = build_verl_command(args)
    print("Running verl GRPO training:")
    print(" ".join(cmd))
    print()

    result = subprocess.run(cmd)
    sys.exit(result.returncode)


if __name__ == "__main__":
    main()
