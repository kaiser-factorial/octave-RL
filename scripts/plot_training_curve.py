#!/usr/bin/env python3
"""Parse archived prime-rl logs and render reward/optimizer curves."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import matplotlib.pyplot as plt

ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")
REWARD_RE = re.compile(
    r"Step (\d+) \|.*?Reward ([+-]?\d+(?:\.\d+)?) \| Trainable"
)
TRAIN_RE = re.compile(
    r"Step (\d+) \|.*?Loss ([+-]?\d+(?:\.\d+)?) \| "
    r"Entropy ([+-]?\d+(?:\.\d+)?) \| Mismatch KL ([+-]?\d+(?:\.\d+)?) "
    r"\| Grad\. Norm ([+-]?\d+(?:\.\d+)?)"
)
EVAL_RE = re.compile(
    r"Evaluated .*?\(Step (\d+)\).*?Reward ([+-]?\d+(?:\.\d+)?)"
)


def clean(path: Path) -> str:
    return ANSI_RE.sub("", path.read_text(errors="replace"))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("orchestrator_log", type=Path)
    parser.add_argument("trainer_log", type=Path)
    parser.add_argument(
        "--output", type=Path, default=Path("artifacts/training_curve.png")
    )
    parser.add_argument(
        "--summary", type=Path, default=Path("artifacts/training_summary.json")
    )
    args = parser.parse_args()

    orch = clean(args.orchestrator_log)
    trainer = clean(args.trainer_log)
    rewards = [
        {"step": int(step), "reward": float(value)}
        for step, value in REWARD_RE.findall(orch)
    ]
    evals = [
        {"step": int(step), "reward": float(value)}
        for step, value in EVAL_RE.findall(orch)
    ]
    optimization = [
        {
            "step": int(step),
            "loss": float(loss),
            "entropy": float(entropy),
            "mismatch_kl": float(kl),
            "grad_norm": float(grad),
        }
        for step, loss, entropy, kl, grad in TRAIN_RE.findall(trainer)
    ]
    if not rewards:
        raise SystemExit("no completed reward steps found")

    fig, (reward_axis, diagnostic_axis) = plt.subplots(
        2, 1, figsize=(8.6, 7.2), sharex=True, constrained_layout=True
    )
    reward_axis.plot(
        [row["step"] for row in rewards],
        [row["reward"] for row in rewards],
        marker="o",
        linewidth=2,
        color="#4863A0",
        label="train batch reward",
    )
    if evals:
        reward_axis.scatter(
            [row["step"] for row in evals],
            [row["reward"] for row in evals],
            marker="D",
            s=55,
            color="#D2691E",
            label="held-out reward",
            zorder=3,
        )
    reward_axis.set(title="Native prime-rl Octave run", ylabel="Reward")
    reward_axis.grid(alpha=0.2)
    reward_axis.legend(frameon=False)

    if optimization:
        diagnostic_axis.plot(
            [row["step"] for row in optimization],
            [row["entropy"] for row in optimization],
            marker="o",
            label="entropy",
            color="#348A6B",
        )
        diagnostic_axis.plot(
            [row["step"] for row in optimization],
            [row["mismatch_kl"] for row in optimization],
            marker=".",
            label="mismatch KL",
            color="#8D5A97",
        )
    diagnostic_axis.set(xlabel="Optimizer step", ylabel="Diagnostic value")
    diagnostic_axis.grid(alpha=0.2)
    diagnostic_axis.legend(frameon=False)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=180)
    payload = {
        "completed_steps": len(rewards),
        "reward_non_flat": len({row["reward"] for row in rewards}) > 1,
        "train_reward": rewards,
        "heldout_reward": evals,
        "optimizer": optimization,
        "source_logs": {
            "orchestrator": str(args.orchestrator_log),
            "trainer": str(args.trainer_log),
        },
    }
    args.summary.write_text(json.dumps(payload, indent=2) + "\n")


if __name__ == "__main__":
    main()
