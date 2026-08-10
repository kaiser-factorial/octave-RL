# Nemotron across turn budgets on the repaired taskset — 2026-08-09

Nemotron-3-Nano-30B-A3B-BF16, all three levels, at `max_turns` 1, 2 and 3.
The multi-turn cells had never been measurable for any model outside GPU
training: the guide credential was broken until this morning.

> **Correction, same day.** An earlier version of this document concluded that
> retries raise reward without raising solve rate. That was wrong. "Solve" had
> been computed from `rewards.case_fraction`, the *discounted* reward — a
> correct answer earns 0.85 on attempt 2 and 0.60 on attempt 3, so the measure
> could not count a solve that arrived after the first attempt, which is exactly
> the population in question. The right field is `raw_case_fraction`, which the
> environment already reports. Everything below uses it. See `PIPELINE_LOG.md`,
> entry "RETRACTED: retries buy execution, not solutions".

## Setup

32 tasks per level, 8 rollouts, seed `20260808`, T = 1.0, thinking off,
1,536-token cap, scored against pinned GNU Octave 10.2.0 under `unshare --net`.
**2,304 rollouts, zero infrastructure errors, no GPU**, about $0.5 all in.

One- and two-turn cells run `guide_enabled = false`; at `max_attempts = 2` the
guide cannot fire anyway (the code requires 3), so the 1→2 hop isolates the
diagnostic and 2→3 isolates the hint.

## Retries are among the most effective things in the environment

| level | turns | reward | **solve rate** | execution | mean turns |
|---|---:|---:|---:|---:|---:|
| L1 | 1 | 0.573 | 0.570 | 0.736 | 1.00 |
| L1 | 2 | 0.704 | **0.723** | 0.820 | 1.43 |
| L1 | 3 | 0.778 | **0.828** | 0.900 | 1.65 |
| L2 | 1 | 0.517 | 0.504 | 0.644 | 1.00 |
| L2 | 2 | 0.676 | **0.688** | 0.783 | 1.43 |
| L2 | 3 | 0.690 | **0.750** | 0.840 | 1.85 |
| L3 | 1 | 0.312 | 0.309 | 0.548 | 1.00 |
| L3 | 2 | 0.408 | **0.422** | 0.644 | 1.69 |
| L3 | 3 | 0.481 | **0.527** | 0.695 | 2.24 |

**Solve rate rises 22–26 points from one turn to three** (+0.258, +0.246,
+0.219). Execution rises alongside it, monotonically at every level. Of hinted
rollouts that rewrote their function, **21.6% went on to fully solve**.

The split across the two hops:

| level | 1→2 (diagnostic only) | 2→3 (adds the hint) |
|---|---:|---:|
| L1 | +0.153 | +0.105 |
| L2 | +0.184 | +0.062 |
| L3 | +0.113 | +0.105 |

Both hops earn their place. The first retry is worth more than the second, as
you would expect from a declining-returns process, but neither is decoration.

## Reward is not a capability measure when the turn budget varies

This is the durable lesson. `case_fraction` multiplies correctness by an
attempt discount, so it mixes *how well* with *how much help*. Across a fixed
turn budget it is a fine training signal. Across different budgets it is not
comparable, and thresholding it to define "solved" is simply wrong.

**Use `raw_case_fraction` for any claim of the form "the policy got better".**
The two fields coincide only at one turn, which is why every single-turn
evaluation in this repository was unaffected — and why the defect survived.

## Changing what the feedback *says* changed nothing

A paired A/B on the 2-turn cells: identical model, tasks, seed and turn budget,
`guide_enabled = false`, with the only difference being the rewritten retry
diagnostic (blob stripped, errors deduplicated, failure mode named).

| level | old feedback | new feedback | delta | paired SE | t |
|---|---:|---:|---:|---:|---:|
| L1 | 0.723 | 0.719 | −0.004 | 0.023 | −0.17 |
| L2 | 0.688 | 0.672 | −0.016 | 0.033 | −0.47 |
| L3 | 0.422 | 0.441 | +0.020 | 0.039 | +0.51 |

**No effect.** The rewrite is still worth keeping — it cut a six-case syntax
failure from 1,084 characters to 110, removed an internal protocol blob from the
model's context, and gave the previously-silent "ran but wrong" case something
to say — but on this model it buys no additional solves.

Taken with the turn-budget result, that is a specific claim: **the retry
mechanism is valuable and the retry *content* is not**, at least for a model
this strong. The obvious explanation is that a retry is mostly another sample
at T = 1.0, and the diagnostic is close to inert. That is testable and is the
next experiment on the list.

## Nemotron against the models being trained

| model | params | format_ok | execution | solve, L1 | turns | token trunc |
|---|---|---:|---:|---:|---:|---:|
| Nemotron-3-Nano | 30B-A3B | **0.98** | 0.736 | 0.570 | 1 | 0.0% |
| Qwen3.5-4B | 4B | 0.92 | 0.816 | 0.727* | 3 | 3.0% |
| Qwen3.5-2B | 2B | 0.59 | 0.243 | 0.148* | 3 | 10.0% |
| Qwen3.5-0.8B | 0.8B | 0.64 | 0.171 | 0.117* | 3 | 2.5% |

\* Qwen figures come from the 3-attempt scaffold, so they are **not** comparable
to Nemotron's single-turn cell — see the turn-budget tables above. They were
first published as 0.328 / 0.086 / 0.078, computed with the flawed discounted
measure; recomputed here from `raw_case_fraction`. Nemotron's is single-turn,
where the two measures coincide.

Nemotron would saturate Level 1 immediately. It is a reference point, not a
training candidate for this cell.

## Every Nemotron measurement on record

| date | rollouts | what it established |
|---|---:|---|
| 2026-08-08 pre-repair | 960 | per-family spread 0.030–0.734; the 24x range that exposed the orientation defect |
| 2026-08-08 group spread | 768 | 50% of L2 groups unanimous; independence model understated waste 14x |
| 2026-08-09 post-repair | 768 | +0.162 overall (SE 0.033); `linsolve_tolerance` 0.030 → 0.752; L2 degenerate 50% → 10% |
| 2026-08-09 turn budgets | 2,304 | this document |
| 2026-08-09 feedback A/B | 768 | rewritten diagnostic: no effect on solve rate |

**5,568 Nemotron rollouts total**, all on pinned Octave 10.2.0, thinking
verified off from traces rather than config.

## Files

- `raw-outputs.tgz` — 1- and 3-turn cells.
- `two-turn-outputs.tgz` — 2-turn cells (old feedback).
- `../feedback-rewrite-20260809/raw-outputs.tgz` — 2-turn cells (new feedback).
