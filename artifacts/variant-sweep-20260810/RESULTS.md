# Per-variant pass rates on the 0.5.0 pool — PRELIMINARY, NOT PAIRED

Single-turn, no guide, T=1.0, thinking off, seed `20260808`, scored locally
against the pinned Octave 10.2.0. Generation through Prime Inference; no GPU, no
Sandbox.

`per_variant.json` holds one record per (model, family, variant, level) with
solve rate, execution fraction and format validity. Solve rate is the `solved`
metric — undiscounted, and identical in both reward modes — never a threshold on
the discounted reward.

## Read this before using any number here

**The three models did not see the same pool.** The sweep resolved
`--families converted` at launch time, and families were being registered
throughout the session as they were authored. Qwen3.5-0.8B ran when six families
were on the variant form; Qwen3.5-4B ran after all ten were.

| model | families | variants | tasks per variant |
|---|---:|---:|---:|
| Qwen3.5-0.8B | 6 | 48 | 6 |
| Qwen3.5-4B | 10 | 80 | 3–4 |

Two consequences:

1. **No cross-model comparison in this artifact is paired.** A variant absent
   from the 0.8B column was not measured for that model; it did not score zero.
2. **The 4B cells are thinner than intended.** The pool was fixed at 288 tasks
   per level, which is exactly 6 per variant across 48 variants and 3–4 across
   80. Per-variant n for 4B is 12–16 rollouts, not 24.

This is a design error in the sweep, not a defect in the pool: `--families
converted` resolving at run time is right for a stable pool and wrong for one
being extended underneath it. **The definitive run re-measures all three models
on the frozen ten-family pool at 6 tasks per variant.** Treat everything here as
a screen — good enough to spot a variant nobody can solve, not good enough to
compare models or to quote a level rate.

## What the screen supports

**Qwen3.5-4B sits in the training band on the new pool.** Level 1 solve `0.322`,
level 2 `0.276`, against `0.400` and `0.214` on the 0.4.x pool at the same
single-turn design. Level 1 moved from just above the 10–35% band to inside it.

**Qwen3.5-0.8B fell below it.** Level 1 solve `0.017 ± 0.004` against `0.030` on
0.4.x. Format compliance is unchanged (`0.745` against `0.71`), so the longer
descriptions — 53 to 332 characters on average — are not costing output
discipline; the model simply cannot write the functions.

**No variant looks unreadable yet.** Every level-1 failure read by hand at 0.8B
is ordinary incompetence: syntax errors, `help <fn>` inside a function body,
`error('Input ' numel(x) ' must be...')`. A variant is only suspect when it is
near zero for **every** model, and that intersection cannot be computed from an
unpaired sweep.

## What this cannot answer

**Which model to train.** Training runs three attempts with a guide, worth
`0.030 → 0.117` for 0.8B on the old pool. A single-turn number is the wrong
instrument for the band judgement, as `artifacts/smaller-models-20260809`
already argued at length. That needs its own three-turn cell.

**Whether the pool is legible.** A naive solution passing
`validate_natural_solutions.py` proves a prompt satisfiable. Only a model
reading it proves it legible, and only the paired run can say so per variant.
