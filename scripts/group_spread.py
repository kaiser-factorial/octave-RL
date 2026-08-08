"""Measure how often a sampled group carries any GRPO advantage.

GRPO's advantage is ``rewards - rewards.mean()`` with no std normalisation, so a
group whose rollouts all score the same contributes exactly zero gradient. With
a near-binary reward the question "what fraction of groups teach anything" is
therefore the one that decides how much of a rollout budget is real.

The usual way to answer it is a model: assume rollouts within a group are
independent draws at the marginal pass rate ``p``, and the degenerate fraction
is ``p**g + (1-p)**g``. That assumption is doing a lot of work and it errs in a
known direction — rollouts from one policy on one task are positively
correlated, so real groups are *more* unanimous than independence predicts, and
the model understates waste.

This script replaces the model with an observation. Given cells run at
``num_rollouts = g``, it reports, per level:

- the observed degenerate fraction at the run's own group size;
- the exact expected degenerate fraction at every smaller group size, by
  averaging over all ``C(g, k)`` sub-groups rather than resampling (a group with
  ``s`` successes out of ``g`` has ``[C(s,k) + C(g-s,k)] / C(g,k)`` unanimous
  sub-groups of size ``k``);
- the independence prediction beside it, so the size of the modelling error is
  visible rather than assumed;
- the dispersion ratio Var(s) / Var_binomial(s), which is >1 exactly when
  rollouts are correlated within a task.

Usage:
    uv run python scripts/group_spread.py --root outputs/nemo-g8
"""

from __future__ import annotations

import argparse
import json
import statistics
from collections import defaultdict
from math import comb
from pathlib import Path


def load_groups(path: Path) -> dict[str, list[float]]:
    """Task name -> its rollout scores, for one cell."""
    groups: dict[str, list[float]] = defaultdict(list)
    for line in path.open():
        row = json.loads(line)
        data = (row.get("task") or {}).get("data") or {}
        groups[data.get("name")].append(row["metrics"]["raw_case_fraction"])
    return dict(groups)


def degenerate_fraction(groups: dict[str, list[float]]) -> float:
    """Share of groups whose rollouts all scored identically."""
    return sum(len(set(v)) == 1 for v in groups.values()) / len(groups)


def expected_degenerate_at(groups: dict[str, list[float]], k: int) -> float | None:
    """Exact expected degenerate fraction for sub-groups of size `k`.

    Averaged over every C(g, k) subset of each observed group, so this is a
    property of the data rather than a resampling estimate. Binary scores only:
    a partial score makes "unanimous" ill-defined here, and they are rare enough
    (1.1% of rollouts in the 2026-08-08 run) that dropping the affected groups
    is more honest than bucketing them.
    """
    total = 0.0
    counted = 0
    for scores in groups.values():
        if any(s not in (0.0, 1.0) for s in scores):
            continue
        g = len(scores)
        if k > g:
            return None
        s = int(sum(scores))
        total += (comb(s, k) + comb(g - s, k)) / comb(g, k)
        counted += 1
    return total / counted if counted else None


def dispersion(groups: dict[str, list[float]]) -> tuple[float, float]:
    """(marginal pass rate, Var(successes) / binomial variance).

    Ratio > 1 means rollouts within a task agree more than independent draws
    would — the direction that makes the independence model optimistic.
    """
    counts = [sum(v) for v in groups.values()]
    sizes = {len(v) for v in groups.values()}
    g = sizes.pop() if len(sizes) == 1 else statistics.mean(len(v) for v in groups.values())
    p = sum(counts) / (len(counts) * g)
    binom_var = g * p * (1 - p)
    observed = statistics.variance(counts) if len(counts) > 1 else 0.0
    return p, (observed / binom_var if binom_var else float("nan"))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--sizes", type=int, nargs="+", default=[2, 4, 8])
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    report = {}
    header = f"{'cell':<28}{'p':>7}{'disp':>7}"
    header += "".join(f"{'obs g=' + str(k):>10}{'ind g=' + str(k):>10}" for k in args.sizes)
    print(header)
    print("-" * len(header))

    for path in sorted(args.root.glob("*/traces.jsonl")):
        cell = path.parent.name
        groups = load_groups(path)
        p, disp = dispersion(groups)
        row = f"{cell[-24:]:<28}{p:>7.3f}{disp:>7.2f}"
        entry = {"p": round(p, 4), "dispersion": round(disp, 3),
                 "n_groups": len(groups), "observed": {}, "independent": {}}
        for k in args.sizes:
            obs = expected_degenerate_at(groups, k)
            ind = p**k + (1 - p) ** k
            entry["observed"][k] = None if obs is None else round(obs, 4)
            entry["independent"][k] = round(ind, 4)
            row += f"{'--':>10}" if obs is None else f"{obs:>10.3f}"
            row += f"{ind:>10.3f}"
        print(row)
        report[cell] = entry

    print("\nobs = measured from the run's own groups; ind = the independence model.")
    print("disp = Var(successes) / binomial variance; > 1 means correlated rollouts.")
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2) + "\n")
        print(f"\nreport: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
