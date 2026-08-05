#!/usr/bin/env python3
"""Render the model-size calibration and multi-turn repair results."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt

ARTIFACTS = Path("artifacts")


def main() -> None:
    ARTIFACTS.mkdir(exist_ok=True)

    sizes = ["0.8B", "2B", "4B"]
    means = [0.0, 0.0, 0.1]
    fig, axis = plt.subplots(figsize=(7.2, 4.4), constrained_layout=True)
    bars = axis.bar(sizes, means, color=["#A7B0BE", "#77869B", "#4863A0"])
    axis.axhspan(0.10, 0.35, color="#72B388", alpha=0.18, label="target baseline range")
    axis.set(
        title="Qwen model-size ladder · Octave Level 1",
        xlabel="Qwen3.5 model size",
        ylabel="Mean raw case reward",
        ylim=(0, 0.4),
    )
    axis.bar_label(bars, labels=["0%", "0%", "10%"], padding=4)
    axis.legend(frameon=False, loc="upper left")
    fig.savefig(ARTIFACTS / "calibration_ladder.png", dpi=180)
    plt.close(fig)

    labels = ["Attempt 1", "Attempt 2\n(no guide)", "Attempt 3\n(with guide)"]
    rewards = [0.20, 0.20, 0.40]
    multipliers = [1.00, 0.85, 0.60]
    fig, axis = plt.subplots(figsize=(7.2, 4.4), constrained_layout=True)
    bars = axis.bar(labels, rewards, color=["#4863A0", "#D69A52", "#348A6B"])
    axis.set(
        title="Matched Level 1 repair sample · Qwen3.5-4B",
        ylabel="Mean raw case reward",
        ylim=(0, 0.5),
    )
    axis.bar_label(
        bars,
        labels=[
            f"{reward:.0%} raw\n×{multiplier:.2f} if solved here"
            for reward, multiplier in zip(rewards, multipliers)
        ],
        padding=4,
    )
    fig.savefig(ARTIFACTS / "retry_lift.png", dpi=180)
    plt.close(fig)

    payload = {
        "model_size_ladder": [
            {"model": f"Qwen/Qwen3.5-{size}", "level": 1, "n": 10, "raw_mean": mean}
            for size, mean in zip(sizes, means)
        ],
        "matched_retry_sample": {
            "model": "Qwen/Qwen3.5-4B",
            "level": 1,
            "n": 10,
            "raw_mean_by_attempt": rewards,
            "correctness_multiplier_by_attempt": multipliers,
        },
    }
    (ARTIFACTS / "calibration_summary.json").write_text(
        json.dumps(payload, indent=2) + "\n"
    )


if __name__ == "__main__":
    main()
