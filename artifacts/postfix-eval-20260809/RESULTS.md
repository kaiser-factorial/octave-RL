# Post-repair re-measurement — 2026-08-09

Paired re-measurement of the 0.2.0 taskset repair against the 0.1.0 baseline in
`artifacts/group-spread-20260808/`.

## Setup

Identical to the 2026-08-08 group-spread run in every respect that was held
fixed: both models, levels 1–3, **32 tasks per level, 8 rollouts, seed
`20260808`**, T = 1.0, single turn, no guide, thinking off, 1,536-token cap,
scored against the pinned GNU Octave 10.2.0. **1,536 rollouts, zero
infrastructure errors.**

What differs is the taskset itself — that is the thing being measured.
Task ids, family names, level structure, hidden expected values and reward
multipliers are unchanged, so **family-level and task-level pairing both hold**
(the tasks at each index are the same tasks; their prompts, and for two families
their input orientation, changed).

Ran on a Prime CPU Sandbox (`gnuoctave/octave:10.2.0`, 8 cores / 16 GB, native
x86_64) rather than the previous host, because the only local option was
`linux/amd64` emulation on an arm64 Mac at roughly 20x the cost per rollout.
Candidate isolation was the full `unshare --net` path, and
`_child_environment` keeps `PRIME_API_KEY` out of candidate processes by
construction.

Cost: ~$0.25 Prime Inference plus ~$0.55 of sandbox time.

## Headline

**Every family is now reachable.** The lowest family pass rate rose from 0.030
to 0.213 for Nemotron and from 0.000 to 0.056 for Qwen. A family that no model
ever passes contributes zero GRPO advantage at any group size, so this is the
property that mattered.

| | Nemotron | Qwen |
|---|---:|---:|
| overall change, paired on 96 tasks | **+0.162** (SE 0.033, t = 4.84) | **+0.105** (SE 0.022, t = 4.87) |
| lowest family pass rate, before → after | 0.030 → 0.213 | 0.000 → 0.056 |
| family spread (max/min) | 24.4x → 3.5x | ∞ → 7.0x |

## Per family, paired on task

`delta` is the mean per-task change; `SE` is across tasks, not rollouts (the
2026-08-08 standard-error defect). `*` marks |t| > 2.

### Nemotron-3-Nano-30B-A3B-BF16

| family | tasks | before | after | delta | SE | t |
|---|---:|---:|---:|---:|---:|---:|
| `linsolve_tolerance` | 9 | 0.030 | 0.752 | **+0.722** | 0.076 | 9.56 * |
| `reshape_permute` | 9 | 0.306 | 0.569 | **+0.264** | 0.089 | 2.95 * |
| `broadcast_arith` | 9 | 0.236 | 0.458 | **+0.222** | 0.102 | 2.19 * |
| `struct_cell_wrangle` | 9 | 0.528 | 0.736 | +0.208 | 0.165 | 1.26 |
| `sequence_recurrence` | 9 | 0.417 | 0.574 | **+0.157** | 0.060 | 2.60 * |
| `string_parse` | 9 | 0.264 | 0.366 | +0.102 | 0.074 | 1.38 |
| `sliding_window` | 9 | 0.148 | 0.213 | +0.065 | 0.058 | 1.12 |
| `signal_identity` | 9 | 0.174 | 0.234 | +0.060 | 0.092 | 0.65 |
| `reduce_along_dim` | 12 | 0.351 | 0.389 | +0.038 | 0.052 | 0.74 |
| `logical_index` | 12 | 0.734 | 0.639 | −0.095 | 0.059 | −1.61 |

### Qwen3.5-4B

| family | tasks | before | after | delta | SE | t |
|---|---:|---:|---:|---:|---:|---:|
| `sequence_recurrence` | 9 | 0.058 | 0.366 | **+0.308** | 0.089 | 3.47 * |
| `linsolve_tolerance` | 9 | 0.000 | 0.292 | **+0.292** | 0.059 | 4.95 * |
| `reduce_along_dim` | 12 | 0.137 | 0.326 | **+0.189** | 0.052 | 3.66 * |
| `sliding_window` | 9 | 0.014 | 0.190 | **+0.176** | 0.075 | 2.35 * |
| `broadcast_arith` | 9 | 0.014 | 0.083 | +0.069 | 0.037 | 1.89 |
| `reshape_permute` | 9 | 0.236 | 0.292 | +0.056 | 0.047 | 1.18 |
| `signal_identity` | 9 | 0.044 | 0.086 | +0.042 | 0.034 | 1.23 |
| `string_parse` | 9 | 0.032 | 0.056 | +0.023 | 0.035 | 0.65 |
| `struct_cell_wrangle` | 9 | 0.375 | 0.389 | +0.014 | 0.061 | 0.23 |
| `logical_index` | 12 | 0.398 | 0.312 | −0.085 | 0.062 | −1.38 |

The diagnosis predicted the ranking. `linsolve_tolerance`, the family whose
natural solution could not execute, moves furthest by a wide margin, and its
execution fraction goes 0.058 → 0.778 for Nemotron. `signal_identity` and
`sliding_window`, the two families explicitly identified as *genuinely* hard
rather than defective, barely move for Nemotron — which is the correct outcome,
not a shortfall.

## What the fix did to GRPO group economics

The 2026-08-08 finding that motivated all of this was that at `group_size = 8`
on Level 2, **50% of Nemotron groups were unanimous and therefore contributed
exactly zero gradient**. Measured on the repaired pool with the same script:

**Degenerate (unanimous) group fraction, observed**

| model | level | g=2 before → after | g=4 before → after | g=8 before → after |
|---|---|---|---|---|
| Nemotron | 1 | 0.639 → 0.583 | 0.351 → 0.252 | 0.148 → **0.034** |
| Nemotron | 2 | 0.792 → 0.580 | 0.621 → 0.252 | 0.500 → **0.100** |
| Nemotron | 3 | 0.829 → 0.658 | 0.693 → 0.385 | 0.593 → **0.208** |
| Qwen | 1 | 0.720 → 0.620 | 0.494 → 0.320 | 0.333 → **0.143** |
| Qwen | 2 | 0.903 → 0.681 | 0.824 → 0.410 | 0.759 → **0.143** |
| Qwen | 3 | 0.896 → 0.813 | 0.807 → 0.653 | 0.710 → **0.484** |

**Dispersion** (Var(successes) / binomial variance; > 1 means correlated
rollouts within a task):

| model | L1 | L2 | L3 |
|---|---|---|---|
| Nemotron | 3.20 → 1.75 | 4.60 → 2.16 | 2.01 → **3.24** |
| Qwen | 3.00 → 2.52 | 3.16 → 1.54 | 3.18 → 1.91 |

Dispersion falls everywhere except Nemotron Level 3, where it rises while the
degenerate fraction still falls by two thirds (0.593 → 0.208). Those are
compatible: Level 3's pass rate rose from 0.143 to 0.362, moving mass off the
all-fail wall, and the degenerate fraction — not dispersion — is the quantity
that decides how much of a rollout budget produces gradient.

## The one thing that moved the wrong way

`logical_index` fell in **both** models, by 0.095 and 0.085. Neither drop is
individually significant (|t| = 1.61 and 1.38), but the direction agrees across
two models on the same tasks, so it should not be written off as noise.

The only change to that family was cosmetic: its level-1 description went from
"Return a row vector containing the positive elements of x, in original order."
to "Return the positive elements of x, in original order." with the row-vector
claim moved to the generated shape line. If a real effect, the most likely
cause is that splitting the shape requirement onto its own line makes it easier
to skim past than having it inline — which would be worth knowing, since every
family now carries that line. **Unresolved; worth one targeted A/B before
anyone reads much into small per-family deltas.**

## Files

- `raw-outputs.tgz` — configs, traces and logs for all six cells.
- `family_breakdown.json` — per-family aggregates for both runs.
