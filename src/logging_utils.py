"""Logging callbacks and utilities for all training modules.

Provides TrainerCallback classes for rich debug logging during training:
  - SFTLoggingCallback: loss curves, perplexity, sample generation demos
  - GRPOLoggingCallback: reward stats, entropy, length stats, sample completions
  - DistillLoggingCallback: CE/KL/total loss breakdown, alpha tracking

Each callback prints human-readable tables to stdout and optionally logs
to wandb (via trainer.log()).
"""

import time
from collections import defaultdict

import torch
from transformers import TrainerCallback, TrainerState, TrainerControl


# ---------------------------------------------------------------------------
# Formatting helpers
# ---------------------------------------------------------------------------

def _fmt(value, precision=4):
    """Format a numeric value for display."""
    if isinstance(value, float):
        if abs(value) < 0.001 and value != 0:
            return f"{value:.2e}"
        return f"{value:.{precision}f}"
    return str(value)


def _bar(value, max_val=1.0, width=20):
    """Simple ASCII progress bar."""
    ratio = min(max(value / max_val, 0), 1.0) if max_val > 0 else 0
    filled = int(ratio * width)
    return "█" * filled + "░" * (width - filled)


def _header(title: str, width: int = 70):
    return f"\n{'─'*width}\n  {title}\n{'─'*width}"


# ---------------------------------------------------------------------------
# SFT Logging Callback
# ---------------------------------------------------------------------------

class SFTLoggingCallback(TrainerCallback):
    """Rich logging for SFT training.

    Logs at every `sample_every` steps:
      - Loss, perplexity, learning rate, grad norm
      - EMA of loss for trend detection
      - Generates sample summaries from a few prompts (demo_prompts)

    Args:
        tokenizer: Tokenizer for decoding generated samples.
        demo_prompts: List of prompt strings to generate samples from.
            If None, sample generation is skipped.
        sample_every: Generate samples every N steps.
        max_new_tokens: Max tokens for sample generation.
    """

    def __init__(
        self,
        tokenizer=None,
        demo_prompts: list[str] | None = None,
        sample_every: int = 100,
        max_new_tokens: int = 128,
    ):
        self.tokenizer = tokenizer
        self.demo_prompts = demo_prompts or []
        self.sample_every = sample_every
        self.max_new_tokens = max_new_tokens

        self._loss_ema = None
        self._ema_alpha = 0.1
        self._step_times = []
        self._last_time = None

    def on_train_begin(self, args, state, control, **kwargs):
        self._last_time = time.time()
        print(_header("SFT Training Started"))
        print(f"  Total steps: {state.max_steps}")
        print(f"  Sample generation every {self.sample_every} steps")
        print(f"  Demo prompts: {len(self.demo_prompts)}")

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs is None:
            return

        # Track step timing
        now = time.time()
        if self._last_time:
            dt = now - self._last_time
            self._step_times.append(dt)
        self._last_time = now

        loss = logs.get("loss")
        if loss is None:
            return

        # EMA loss
        if self._loss_ema is None:
            self._loss_ema = loss
        else:
            self._loss_ema = self._ema_alpha * loss + (1 - self._ema_alpha) * self._loss_ema

        perplexity = min(torch.exp(torch.tensor(loss)).item(), 1e6)
        lr = logs.get("learning_rate", 0)
        grad_norm = logs.get("grad_norm", None)
        epoch = logs.get("epoch", 0)

        # Trend indicator
        trend = "→"
        if self._loss_ema < loss * 0.98:
            trend = "↑"  # loss above EMA = getting worse
        elif self._loss_ema > loss * 1.02:
            trend = "↓"  # loss below EMA = improving

        # Speed
        speed = ""
        if len(self._step_times) > 1:
            avg_time = sum(self._step_times[-10:]) / len(self._step_times[-10:])
            speed = f" | {avg_time:.1f}s/log"

        step = state.global_step
        pct = step / state.max_steps * 100 if state.max_steps > 0 else 0

        line = (
            f"  [{step:6d}/{state.max_steps}] ({pct:5.1f}%) "
            f"loss={_fmt(loss)} {trend} ppl={_fmt(perplexity)} "
            f"lr={lr:.2e}"
        )
        if grad_norm is not None:
            line += f" |∇|={_fmt(grad_norm)}"
        line += f" ep={epoch:.2f}{speed}"
        print(line)

    def on_step_end(self, args, state, control, model=None, **kwargs):
        # Sample generation at intervals
        if (
            state.global_step > 0
            and state.global_step % self.sample_every == 0
            and self.demo_prompts
            and model is not None
            and self.tokenizer is not None
        ):
            self._generate_samples(model, state.global_step)

    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        if metrics:
            eval_loss = metrics.get("eval_loss")
            if eval_loss is not None:
                eval_ppl = min(torch.exp(torch.tensor(eval_loss)).item(), 1e6)
                print(f"\n  ✦ Eval @ step {state.global_step}: "
                      f"loss={_fmt(eval_loss)} ppl={_fmt(eval_ppl)}")

    def on_train_end(self, args, state, control, **kwargs):
        print(_header("SFT Training Complete"))
        if self._step_times:
            total = sum(self._step_times)
            print(f"  Total time: {total:.0f}s ({total/60:.1f}min)")

    @torch.no_grad()
    def _generate_samples(self, model, step):
        model.eval()
        print(f"\n  ── Sample Generation @ step {step} ──")
        for i, prompt in enumerate(self.demo_prompts[:3]):
            inputs = self.tokenizer(
                prompt, return_tensors="pt", truncation=True, max_length=512
            ).to(model.device)
            outputs = model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
            )
            generated = self.tokenizer.decode(
                outputs[0][inputs["input_ids"].shape[1]:],
                skip_special_tokens=True,
            )
            prompt_short = prompt.replace("\n", " ")[:80]
            gen_short = generated.replace("\n", " ")[:150]
            print(f"  [{i+1}] Prompt: {prompt_short}...")
            print(f"      Output: {gen_short}")
        model.train()


# ---------------------------------------------------------------------------
# GRPO Logging Callback
# ---------------------------------------------------------------------------

class GRPOLoggingCallback(TrainerCallback):
    """Rich logging for GRPO reinforcement learning training.

    Logs:
      - Reward statistics: mean, std, min, max per batch
      - Policy loss, value loss, entropy
      - Completion length statistics
      - Sample completions periodically
    """

    def __init__(self, sample_every: int = 50):
        self.sample_every = sample_every
        self._reward_history = []
        self._last_time = None

    def on_train_begin(self, args, state, control, **kwargs):
        self._last_time = time.time()
        print(_header("GRPO Training Started"))
        print(f"  Total steps: {state.max_steps}")

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs is None:
            return

        step = state.global_step
        pct = step / state.max_steps * 100 if state.max_steps > 0 else 0

        # Extract GRPO-specific metrics
        loss = logs.get("loss")
        reward_mean = logs.get("reward", logs.get("rewards/mean"))
        reward_std = logs.get("reward_std", logs.get("rewards/std"))
        policy_loss = logs.get("policy_loss", logs.get("loss/policy"))
        entropy = logs.get("entropy", logs.get("policy_entropy"))
        kl = logs.get("kl", logs.get("kl_divergence"))
        lr = logs.get("learning_rate", 0)
        completion_len = logs.get("completion_length", logs.get("completions/mean_length"))

        # Track reward history
        if reward_mean is not None:
            self._reward_history.append(reward_mean)

        # Reward trend
        reward_trend = ""
        if len(self._reward_history) >= 10:
            recent = sum(self._reward_history[-5:]) / 5
            older = sum(self._reward_history[-10:-5]) / 5
            if recent > older * 1.05:
                reward_trend = " ↑"
            elif recent < older * 0.95:
                reward_trend = " ↓"
            else:
                reward_trend = " →"

        # Build log line
        parts = [f"  [{step:6d}/{state.max_steps}] ({pct:5.1f}%)"]

        if loss is not None:
            parts.append(f"loss={_fmt(loss)}")
        if reward_mean is not None:
            r_str = f"R={_fmt(reward_mean)}"
            if reward_std is not None:
                r_str += f"±{_fmt(reward_std, 3)}"
            r_str += reward_trend
            parts.append(r_str)
        if entropy is not None:
            parts.append(f"H={_fmt(entropy)}")
        if kl is not None:
            parts.append(f"KL={_fmt(kl)}")
        if policy_loss is not None:
            parts.append(f"π_loss={_fmt(policy_loss)}")
        if completion_len is not None:
            parts.append(f"len={_fmt(completion_len, 1)}")

        parts.append(f"lr={lr:.2e}")
        print(" | ".join(parts))

        # Detailed reward breakdown periodically
        if step > 0 and step % self.sample_every == 0:
            self._print_reward_summary(logs, step)

    def _print_reward_summary(self, logs, step):
        """Print detailed reward/generation stats."""
        print(f"\n  ── GRPO Details @ step {step} ──")

        # Reward histogram from history
        if len(self._reward_history) >= 5:
            recent = self._reward_history[-10:]
            r_min = min(recent)
            r_max = max(recent)
            r_mean = sum(recent) / len(recent)
            print(f"  Reward (last {len(recent)} steps): "
                  f"mean={_fmt(r_mean)} min={_fmt(r_min)} max={_fmt(r_max)}")
            print(f"  Reward bar: [{_bar(r_mean, max_val=1.0)}] {_fmt(r_mean)}")

        # Print any available per-reward-function breakdowns
        for key in sorted(logs.keys()):
            if "reward" in key.lower() and key not in ("reward", "reward_std", "rewards/mean", "rewards/std"):
                print(f"  {key}: {_fmt(logs[key])}")

        # Completion samples (if trl logs them)
        completions = logs.get("completions", None)
        if completions and isinstance(completions, list):
            print(f"  Sample completions:")
            for i, c in enumerate(completions[:2]):
                c_short = str(c).replace("\n", " ")[:150]
                print(f"    [{i+1}] {c_short}")

    def on_train_end(self, args, state, control, **kwargs):
        print(_header("GRPO Training Complete"))
        if self._reward_history:
            final_mean = sum(self._reward_history[-10:]) / min(len(self._reward_history), 10)
            start_mean = sum(self._reward_history[:10]) / min(len(self._reward_history), 10)
            print(f"  Reward: {_fmt(start_mean)} → {_fmt(final_mean)} "
                  f"(Δ={_fmt(final_mean - start_mean, 3)})")


# ---------------------------------------------------------------------------
# Distillation Logging Callback
# ---------------------------------------------------------------------------

class DistillLoggingCallback(TrainerCallback):
    """Rich logging for knowledge distillation training.

    Logs:
      - CE loss (ground truth), distillation loss (KL or teacher CE)
      - Total combined loss with alpha weighting
      - CE/distill ratio tracking
      - Perplexity for both GT and teacher paths
    """

    def __init__(self, distill_mode: str = "hard", alpha: float = 0.5,
                 temperature: float = 2.0):
        self.distill_mode = distill_mode
        self.alpha = alpha
        self.temperature = temperature
        self._ce_history = []
        self._distill_history = []
        self._last_time = None

    def on_train_begin(self, args, state, control, **kwargs):
        self._last_time = time.time()
        mode_str = f"soft (T={self.temperature})" if self.distill_mode == "soft" else "hard"
        print(_header(f"Distillation Training Started [{mode_str}]"))
        print(f"  α={self.alpha}: "
              f"loss = {1-self.alpha:.1f}×CE_gt + {self.alpha:.1f}×"
              f"{'T²×KL(s||t)' if self.distill_mode == 'soft' else 'CE_teacher'}")
        print(f"  Total steps: {state.max_steps}")

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs is None:
            return

        step = state.global_step
        pct = step / state.max_steps * 100 if state.max_steps > 0 else 0
        lr = logs.get("learning_rate", 0)

        ce_loss = logs.get("ce_loss")
        distill_loss = logs.get("distill_loss")
        total_loss = logs.get("total_loss", logs.get("loss"))

        # Track histories
        if ce_loss is not None:
            self._ce_history.append(ce_loss)
        if distill_loss is not None:
            self._distill_history.append(distill_loss)

        # Perplexity from CE loss
        ce_ppl = ""
        if ce_loss is not None:
            ppl = min(torch.exp(torch.tensor(ce_loss)).item(), 1e6)
            ce_ppl = f" ppl={_fmt(ppl)}"

        # CE/distill ratio
        ratio_str = ""
        if ce_loss is not None and distill_loss is not None and distill_loss > 0:
            ratio = ce_loss / distill_loss
            ratio_str = f" ce/dist={_fmt(ratio, 2)}"

        # Trend arrows
        ce_trend = self._trend(self._ce_history)
        dist_trend = self._trend(self._distill_history)

        # Build line
        parts = [f"  [{step:6d}/{state.max_steps}] ({pct:5.1f}%)"]

        if total_loss is not None:
            parts.append(f"total={_fmt(total_loss)}")
        if ce_loss is not None:
            parts.append(f"CE_gt={_fmt(ce_loss)}{ce_trend}{ce_ppl}")
        if distill_loss is not None:
            dist_label = "KL" if self.distill_mode == "soft" else "CE_t"
            parts.append(f"{dist_label}={_fmt(distill_loss)}{dist_trend}")
        parts.append(f"lr={lr:.2e}{ratio_str}")

        print(" | ".join(parts))

        # Detailed summary periodically
        if step > 0 and step % (args.logging_steps * 10) == 0:
            self._print_summary(step)

    def _trend(self, history, window=5):
        if len(history) < window * 2:
            return ""
        recent = sum(history[-window:]) / window
        older = sum(history[-window*2:-window]) / window
        if recent < older * 0.98:
            return " ↓"
        elif recent > older * 1.02:
            return " ↑"
        return " →"

    def _print_summary(self, step):
        print(f"\n  ── Distillation Summary @ step {step} ──")
        if self._ce_history:
            n = min(len(self._ce_history), 20)
            recent_ce = self._ce_history[-n:]
            print(f"  CE_gt  (last {n}): mean={_fmt(sum(recent_ce)/n)} "
                  f"min={_fmt(min(recent_ce))} max={_fmt(max(recent_ce))}")
            print(f"    [{_bar(sum(recent_ce)/n, max_val=10.0, width=30)}]")

        if self._distill_history:
            n = min(len(self._distill_history), 20)
            recent_d = self._distill_history[-n:]
            label = "KL_div" if self.distill_mode == "soft" else "CE_teacher"
            print(f"  {label} (last {n}): mean={_fmt(sum(recent_d)/n)} "
                  f"min={_fmt(min(recent_d))} max={_fmt(max(recent_d))}")
            print(f"    [{_bar(sum(recent_d)/n, max_val=10.0, width=30)}]")

        # Effective loss contribution
        if self._ce_history and self._distill_history:
            ce_contrib = (1 - self.alpha) * self._ce_history[-1]
            dist_contrib = self.alpha * self._distill_history[-1]
            total = ce_contrib + dist_contrib
            if total > 0:
                ce_pct = ce_contrib / total * 100
                dist_pct = dist_contrib / total * 100
                print(f"  Loss share: CE_gt={ce_pct:.1f}% | "
                      f"{'KL' if self.distill_mode == 'soft' else 'CE_t'}={dist_pct:.1f}%")

    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        if metrics:
            eval_loss = metrics.get("eval_loss")
            if eval_loss is not None:
                ppl = min(torch.exp(torch.tensor(eval_loss)).item(), 1e6)
                print(f"\n  ✦ Eval @ step {state.global_step}: "
                      f"loss={_fmt(eval_loss)} ppl={_fmt(ppl)}")

    def on_train_end(self, args, state, control, **kwargs):
        print(_header("Distillation Training Complete"))
        if self._ce_history:
            print(f"  CE_gt:  {_fmt(self._ce_history[0])} → {_fmt(self._ce_history[-1])}")
        if self._distill_history:
            label = "KL_div" if self.distill_mode == "soft" else "CE_t  "
            print(f"  {label}: {_fmt(self._distill_history[0])} → {_fmt(self._distill_history[-1])}")
