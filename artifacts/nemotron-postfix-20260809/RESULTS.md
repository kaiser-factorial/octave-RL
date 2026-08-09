# Nemotron on the repaired taskset, at both turn budgets — 2026-08-09

Nemotron-3-Nano-30B-A3B-BF16, all three levels, at `max_turns = 1` and
`max_turns = 3`. The three-turn cell has never been measurable for this model:
the guide credential was broken until this morning, so any earlier attempt
would have lost a fifth of its rollouts to `UserError`.

## Setup

32 tasks per level, 8 rollouts, seed `20260808`, T = 1.0, thinking off,
1,536-token cap, scored against pinned GNU Octave 10.2.0 under `unshare --net`.
**1,536 rollouts, zero infrastructure errors, no GPU.** One CPU Sandbox,
about $0.35 all in.

Single-turn cells use `guide_enabled = false`; three-turn cells enable the
guide (`Qwen/Qwen3.5-35B-A3B`) before attempt 3.

## The result

| cell | reward | solve | execution | format_ok | token trunc | mean turns |
|---|---:|---:|---:|---:|---:|---:|
| 1 turn, L1 | 0.573 | 0.570 | 0.736 | 0.98 | 0.0% | 1.00 |
| 1 turn, L2 | 0.517 | 0.504 | 0.644 | 0.97 | 0.4% | 1.00 |
| 1 turn, L3 | 0.312 | 0.309 | 0.548 | 0.99 | 0.4% | 1.00 |
| 3 turns, L1 | **0.778** | 0.602 | 0.900 | 0.99 | 0.0% | 1.65 |
| 3 turns, L2 | **0.690** | 0.484 | 0.840 | 0.99 | 0.0% | 1.85 |
| 3 turns, L3 | **0.481** | 0.312 | 0.695 | 0.99 | 0.5% | 2.24 |

**Reward rises by +0.169 to +0.205. Solve rate moves by −0.020 to +0.031.**

| level | reward | solve |
|---|---|---|
| L1 | 0.573 → 0.778 (**+0.205**) | 0.570 → 0.602 (+0.031) |
| L2 | 0.517 → 0.690 (**+0.173**) | 0.504 → 0.484 (−0.020) |
| L3 | 0.312 → 0.481 (**+0.169**) | 0.309 → 0.312 (+0.004) |

## What the retry loop actually buys

Not solutions — **execution**. The fraction of hidden cases that run without
raising goes 0.736 → 0.900 (L1), 0.644 → 0.840 (L2), 0.548 → 0.695 (L3).

The mechanism is straightforward once separated: reward is the fraction of
hidden cases passed, discounted 0.85 on attempt 2 and 0.60 on attempt 3, so
partial credit accrues across retries. A rollout that goes from "throws on
every case" to "runs and passes three of six" earns real reward without ever
becoming a solution. Solve rate, which counts only all-six-pass, sees none of
that.

So the diagnostic feedback teaches Nemotron to produce *runnable* Octave. It
does not teach it the algorithm.

**Per-family deltas are noise.** The largest are +0.042 (`sliding_window`,
`signal_identity`, `reshape_permute`) and −0.056 (`sequence_recurrence`) at
n ≈ 72 per family, where the standard error is about 0.06. Solve rate cannot
truly fall with more attempts — a solved rollout stops — so the negative deltas
are sampling variance and should be read as zero. The aggregate, near zero
across all ten families, is the real finding.

## Why this matters beyond Nemotron

Any comparison between a one-turn and a three-turn configuration is comparing
two different measurements, not two capability levels. **A reported reward is
not portable across turn budgets.** Every hosted evaluation in this repository
used `max_turns = 1`; the training scaffold uses 3. The 4B smoke's 0.925 and
the eval's 0.400 on the same cell are both correct and not comparable.

If the question is "did the policy get better at Octave", the metric is
`solve_rate`, and it should be reported at a fixed turn budget.

## Nemotron against the models being trained

| model | params | format_ok | execution | solve (L1) | token trunc |
|---|---|---:|---:|---:|---:|
| Nemotron-3-Nano | 30B-A3B | **0.98** | 0.736 | 0.570 | 0.0% |
| Qwen3.5-4B | 4B | 0.92 | 0.816 | 0.328 | 3.0% |
| Qwen3.5-2B | 2B | 0.59 | 0.243 | 0.086 | 10.0% |
| Qwen3.5-0.8B | 0.8B | 0.64 | 0.171 | 0.078 | 2.5% |

Qwen figures are the three-attempt scaffold; Nemotron's solve is single-turn.
Nemotron would saturate Level 1 immediately — at 0.570 it has less headroom
than 4B, which was already judged too easy. It is a reference point, not a
training candidate for this cell.

## Every Nemotron measurement on record

| date | rollouts | what it established |
|---|---:|---|
| 2026-08-08 pre-repair | 960 | per-family spread 0.030–0.734; the 24x range that exposed the orientation defect |
| 2026-08-08 group spread | 768 | 50% of L2 groups unanimous; independence model understated waste 14x |
| 2026-08-09 post-repair | 768 | +0.162 overall (SE 0.033); `linsolve_tolerance` 0.030 → 0.752; L2 degenerate 50% → 10% |
| 2026-08-09 turn budgets | 1,536 | this document |

**4,032 Nemotron rollouts total**, all on pinned Octave 10.2.0, all with
thinking verified off from traces rather than config.

## Files

- `raw-outputs.tgz` — traces, configs and logs for all six cells.
