# What a retry is worth, and what the feedback adds — 2026-08-09

The retry scaffold was rebuilt this session (blob stripped, errors
deduplicated, failure mode named, guide fired on need). This is the measurement
of whether any of it matters, decomposed into the parts.

All cells: 32 tasks per level, 8 rollouts, seed `20260808`, T = 1.0, thinking
off, pinned GNU Octave 10.2.0. Solve rate from **`raw_case_fraction`** — see the
correction note at the end. ~5,400 rollouts on CPU sandboxes, no GPU.

## The decomposition

Nemotron-3-Nano-30B-A3B, solve rate:

| component | L1 | L2 | L3 |
|---|---:|---:|---:|
| one turn (baseline) | 0.570 | 0.504 | 0.309 |
| + a retry saying only "That answer was not correct." | **+0.118** | **+0.152** | **+0.125** |
| + the full diagnostic instead of that sentence | +0.031 | +0.016 | +0.008 |
| + a third turn, still no guide | +0.047 | +0.055 | +0.063 |
| + the LLM guide hint | **+0.062** | +0.023 | +0.023 |
| **three turns, guided** | **0.828** | **0.750** | **0.527** |

## What each part is worth

**The extra attempt is nearly everything.** A content-free retry captures 79%,
90% and 95% of the total one-to-two-turn gain. Whether the model is told what
went wrong is worth +0.031 / +0.016 / +0.008 — none distinguishable from zero
(|t| = 0.98, 0.49, 0.21 paired across 32 tasks).

**Not quite pure resampling, though.** If attempts were independent draws at the
one-turn rate, two turns would give `1-(1-p)^2` = 0.815 / 0.754 / 0.523. Observed
is 0.688 / 0.656 / 0.434 — *below* independence at every level, which is what
correlated attempts look like. The model tends to repeat its own mistakes,
consistent with the 1.4–2.2 rollout dispersion measured elsewhere.

**The guide is the one component that pays**, and it pays more for weaker
models:

| | L1 | L2 | L3 |
|---|---:|---:|---:|
| Nemotron-3-Nano 30B-A3B | **+0.062** (t 3.06→2.37) | +0.023 | +0.023 |
| Qwen3.5-4B | **+0.129** (t 3.06) | +0.004 | +0.031 |

A specific, model-generated diagnosis of the actual bug does something that
generic text does not. Examples from the traces read like *"the function
computes the mean along dimension 2, which returns a column vector, whereas the
task requires the mean of each column"* — an actual diagnosis, not a category.

## Retries by model

Solve rate, one turn → three turns:

| model | L1 | L2 | L3 |
|---|---|---|---|
| Nemotron-3-Nano | 0.570 → 0.828 (**+0.258**) | 0.504 → 0.750 (**+0.246**) | 0.309 → 0.527 (**+0.219**) |
| Qwen3.5-4B | 0.332 → 0.715 (**+0.383**) | 0.230 → 0.562 (**+0.332**) | 0.141 → 0.359 (**+0.218**) |

The effect is a property of the environment, not of one model, and it is larger
where there is more headroom.

## What to do with this

1. **State the turn budget with every score.** A 3-turn number is not comparable
   to a 1-turn number and most of the gap is resampling, not capability.
2. **Keep the guide on.** It is the only part of the feedback apparatus with a
   measured effect, and it is worth most to exactly the small models being
   trained.
3. **Keep the feedback rewrite, but expect nothing from it.** It removed an
   internal protocol blob from the model's context and cut a six-case syntax
   failure from 1,084 characters to 110. That is hygiene, not a lever.
4. **Do not spend more on feedback wording** without a mechanism that changes
   what the model can *do*, rather than what it is told. Showing a failing input
   would qualify and would also leak; that trade has not been evaluated.

## Two corrections this run produced

**Solve rate must come from `raw_case_fraction`.** An earlier analysis defined
it as discounted `case_fraction >= 0.999`, which cannot register a success on
attempt 2 (0.85) or 3 (0.60) — exactly the population under study. That produced
the false conclusion "retries raise reward but not solve rate". Full entry in
`PIPELINE_LOG.md`.

**Qwen3.5-4B, 3 turns, Level 3 is 0.359, not 0.500.** The first measurement of
that cell reached only n = 14 before its sandbox hit the runtime cap, and the
partial figure was briefly published. Redone at n = 256.

## Files

- `raw-outputs.tgz` — 2-turn cells with the new diagnostic.
- `null-control-outputs.tgz` — 2-turn cells with the content-free retry.
- `t3-noguide-outputs.tgz` — Nemotron 3 turns, guide off.
- `../qwen4b-turns-20260809/` — the Qwen sweep and its guide matrix.
- `../nemotron-postfix-20260809/` — the 1- and 3-turn Nemotron cells.
