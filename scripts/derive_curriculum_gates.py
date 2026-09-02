"""Derive curriculum promotion gates from measured per-level rates.

The gates in ``curriculum_controller.py`` are five numbers that decide when the
training mix moves to harder material. They were chosen against the 0.4.x task
pool and are not transferable: parameterisation changed the level ladder, and
the numbers were never re-derived. This script does the derivation from
measurement so the result is reproducible rather than asserted.

It reports three things the controller's own configuration makes easy to get
wrong:

**The nominal threshold is not the bar.** A gate passes on a one-sided 95%
Wilson lower bound, so a nominal 0.55 at the default 20-rollout evaluation
demands an observed 0.75. Every stated threshold is roughly 0.15-0.20 higher in
practice than it reads, and the size of that gap depends on the evaluation size
rather than on anything about the curriculum.

**Whether the base policy already passes.** A gate the untrained policy clears
at step 0 is not a stage, and a promotion sequence that runs through it is not
evidence that difficulty was approached in order.

**Whether the level still teaches.** GRPO advantage is zero inside a group whose
rollouts all score alike, so a level's value is its non-unanimous group share at
the training group size -- measured here, not assumed, because per-task rates
cluster and the independence estimate 2p(1-p) overstates signal at high rates.

Usage::

    uv run python scripts/derive_curriculum_gates.py \\
        --level 1 outputs/threeturn-band/qwen3.5-4b-sampled-g4-l1/traces.jsonl \\
        --level 2 outputs/threeturn-band/qwen3.5-4b-sampled-g4-l2/traces.jsonl \\
        --level 3 outputs/threeturn-band/qwen3.5-4b-sampled-g4-l3/traces.jsonl \\
        --eval-examples 20 --group-size 2

Traces must come from the scaffold the controller evaluates under -- three
attempts with the guide -- because that scaffold was worth 3-4x on the old pool.
Single-turn rates would set every gate against an instrument the controller
never uses.
"""

from __future__ import annotations

import argparse
import itertools
import json
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from curriculum_controller import wilson_lower  # noqa: E402

# The gates as they stand, in controller order: name, level, nominal threshold.
CURRENT_GATES = (
    ("level1_mastery", 1, 0.55),
    ("level2_signal", 2, 0.20),
    ("level2_mastery", 2, 0.45),
    ("level2_advanced", 2, 0.60),
    ("level3_signal", 3, 0.10),
)


@dataclass(frozen=True)
class LevelReading:
    level: int
    tasks: int
    rollouts: int
    rate: float
    stderr: float
    signal: float
    independent_signal: float


def read_level(path: Path, level: int, group_size: int) -> LevelReading:
    by_task: dict[str, list[float]] = defaultdict(list)
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        # raw_case_fraction is what the gates threshold: undiscounted, so a
        # success on attempt 2 or 3 counts, unlike the reward.
        by_task[row["task"]["data"]["name"]].append(
            row["metrics"].get("raw_case_fraction", 0.0)
        )

    per_task = [sum(v) / len(v) for v in by_task.values()]
    rate = sum(per_task) / len(per_task)
    variance = sum((x - rate) ** 2 for x in per_task) / (len(per_task) - 1)
    stderr = (variance / len(per_task)) ** 0.5

    # Exact expectation over every unordered group the rollouts of one task can
    # form, rather than a sampled estimate.
    usable = [v for v in by_task.values() if len(v) >= group_size]
    signal = 0.0
    for scores in usable:
        groups = list(itertools.combinations(scores, group_size))
        varied = sum(1 for g in groups if max(g) - min(g) > 1e-9)
        signal += varied / len(groups)
    signal /= len(usable)

    return LevelReading(
        level=level,
        tasks=len(per_task),
        rollouts=sum(len(v) for v in by_task.values()),
        rate=rate,
        stderr=stderr,
        signal=signal,
        independent_signal=1.0 - rate**group_size - (1.0 - rate) ** group_size,
    )


def observed_rate_needed(threshold: float, examples: int) -> float | None:
    """Smallest observed rate whose Wilson lower bound clears ``threshold``."""
    for successes in range(examples + 1):
        if wilson_lower(successes, examples) >= threshold:
            return successes / examples
    return None


def threshold_for_observed(observed: float, examples: int) -> float:
    """The nominal threshold that makes ``observed`` exactly sufficient."""
    return wilson_lower(round(observed * examples), examples)


def smallest_sample_for_gap(gap: float, *, ceiling: int = 4000) -> int | None:
    """Smallest evaluation size whose Wilson penalty is under ``gap``.

    The penalty is the distance between a nominal threshold and the observed
    rate it demands. Once it exceeds the spacing between levels, the gates stop
    ranking the curriculum and start ranking sampling noise.
    """
    for examples in range(20, ceiling + 1, 10):
        needed = observed_rate_needed(0.50, examples)
        if needed is not None and needed - 0.50 < gap:
            return examples
    return None


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--level",
        action="append",
        nargs=2,
        metavar=("LEVEL", "TRACES"),
        required=True,
        help="level number and its traces.jsonl; repeat for each level",
    )
    parser.add_argument("--eval-examples", type=int, default=20)
    parser.add_argument("--group-size", type=int, default=2)
    parser.add_argument("--json", type=Path, help="write the reading as JSON")
    args = parser.parse_args()

    readings = {
        int(level): read_level(Path(path), int(level), args.group_size)
        for level, path in args.level
    }
    n = args.eval_examples

    print("Measured ladder (three attempts with the guide, the scaffold the gates see)\n")
    print(f"{'level':>5s} {'tasks':>6s} {'rate':>16s} {'signal@g'+str(args.group_size):>12s} {'if independent':>15s}")
    for level in sorted(readings):
        r = readings[level]
        print(
            f"{level:5d} {r.tasks:6d} {r.rate:8.3f} ± {r.stderr:5.3f} "
            f"{r.signal:12.3f} {r.independent_signal:15.3f}"
        )

    print(f"\n\nCurrent gates at --eval-examples {n}\n")
    print(f"{'gate':18s} {'L':>2s} {'nominal':>8s} {'observed needed':>16s} {'base rate':>10s} {'passes at step 0':>17s}")
    vacuous, unreachable = [], []
    for name, level, threshold in CURRENT_GATES:
        reading = readings.get(level)
        needed = observed_rate_needed(threshold, n)
        needed_text = "impossible" if needed is None else f"{needed:.3f}"
        if reading is None:
            print(f"{name:18s} {level:2d} {threshold:8.2f} {needed_text:>16s} {'-':>10s} {'unmeasured':>17s}")
            continue
        passes = needed is not None and reading.rate >= needed
        if passes:
            vacuous.append(name)
        if needed is None:
            unreachable.append(name)
        print(
            f"{name:18s} {level:2d} {threshold:8.2f} {needed_text:>16s} "
            f"{reading.rate:10.3f} {('YES -- vacuous' if passes else 'no'):>17s}"
        )

    penalty = None
    needed_half = observed_rate_needed(0.50, n)
    if needed_half is not None:
        penalty = needed_half - 0.50
        print(f"\nWilson penalty at n={n}: {penalty:+.3f} on a 0.50 threshold.")

    levels = sorted(readings)
    gaps = [
        (a, b, abs(readings[a].rate - readings[b].rate))
        for a, b in itertools.combinations(levels, 2)
    ]
    if gaps:
        smallest = min(gaps, key=lambda g: g[2])
        print(
            f"Smallest gap between levels: {smallest[2]:.3f} "
            f"(L{smallest[0]} vs L{smallest[1]})."
        )
        if penalty is not None and penalty >= smallest[2]:
            enough = smallest_sample_for_gap(smallest[2])
            print(
                f"\n  The penalty exceeds the ladder's own spacing, so at this "
                f"evaluation size\n  the gates separate sampling noise rather "
                f"than curriculum difficulty."
            )
            print(
                "  Smallest evaluation size that puts the penalty under that gap: "
                f"{enough if enough is not None else 'more than 4000'} rollouts."
            )

    if vacuous:
        print(f"\nVacuous at step 0 (base policy already passes): {', '.join(vacuous)}")
    if unreachable:
        print(f"Unreachable at n={n}: {', '.join(unreachable)}")

    if args.json:
        args.json.write_text(
            json.dumps(
                {
                    "eval_examples": n,
                    "group_size": args.group_size,
                    "wilson_penalty_at_0.50": penalty,
                    "levels": {
                        str(level): {
                            "tasks": r.tasks,
                            "rollouts": r.rollouts,
                            "rate": round(r.rate, 4),
                            "stderr": round(r.stderr, 4),
                            "signal": round(r.signal, 4),
                            "independent_signal": round(r.independent_signal, 4),
                        }
                        for level, r in sorted(readings.items())
                    },
                    "current_gates": [
                        {
                            "name": name,
                            "level": level,
                            "nominal": threshold,
                            "observed_needed": observed_rate_needed(threshold, n),
                            "vacuous_at_step_0": name in vacuous,
                        }
                        for name, level, threshold in CURRENT_GATES
                    ],
                },
                indent=2,
            )
            + "\n"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
