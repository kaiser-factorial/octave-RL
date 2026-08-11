# Per-variant pass rates on the 0.5.0 pool — PRELIMINARY, NOT PAIRED

**Superseded by the definitive run** (`--num-tasks 480`, four models, 2048-token
cap). Kept as the record of what was measured and why it was redone.

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

## Level rates, all three models (single-turn, 1536-token cap)

Solve rate with standard error across tasks, n = 288 tasks per cell.

| model | L1 | L2 | L3 |
|---|---:|---:|---:|
| Nemotron-3-Nano | 0.571 ± 0.019 | 0.582 ± 0.019 | 0.337 ± 0.021 |
| Qwen3.5-4B | 0.325 ± 0.017 | 0.279 ± 0.017 | 0.174 ± 0.015 |
| Qwen3.5-0.8B | 0.017 ± 0.004 | 0.000 ± 0.000 | 0.001 ± 0.001 |

Level 2 is not harder than level 1 for Nemotron (0.582 against 0.571) — the same
flatness the 0.4.x pool showed between levels 2 and 3, now one rung lower.

## The near-zero intersection, and why it is not a defect

Eleven variant-levels are at or below 0.02 for **both** 4B and Nemotron, out of
240 paired cells. **All eleven are level 3**, and none has the shape that
indicates a broken prompt:

| | level 2 | level 3 |
|---|---|---|
| execution | 0.17–0.82 | 0.00–0.32 |
| solve | nonzero for 9 of 11 | 0.00 |

An unreadable prompt or an undisclosed convention shows up as **high execution
with zero solve** — the model writes running Octave and still disagrees with the
grader. That appears nowhere. What appears instead is execution collapsing when
the loop ban arrives, on exactly the constructions where a loop-free formulation
is awkward: windowed medians under dilation, circular cross-correlation, and
parsing a cell array of records.

Worth noting because it is not obvious: **the loop ban is not enforced by the
reward.** `require_vectorized` drives the `vectorized` metric only, so a model
that ignores the ban and writes a loop still scores a full solve if the answer is
right. These cells are not failing because loops are punished; they are failing
because the models obey the ban and then write worse code.

Two reasons not to act on this yet: n is 3–4 tasks per cell here, and 4B
truncation on these cells runs as high as **0.75** at the 1536-token cap. The
definitive run fixes both.

## The completion cap excludes the prompt — checked, not assumed

Worth writing down because the reverse would change what every truncated cell
means. Across the definitive run, **every call that stopped with
`finish_reason == "length"` spent exactly 2048 completion tokens**, with prompt
tokens reported separately:

| family | truncated calls | completion | prompt |
|---|---:|---:|---:|
| `reshape_permute` | 599 | 2048 | 396 |
| `signal_identity` | 124 | 2048 | 250 |
| `broadcast_arith` | 94 | 2048 | 199 |

`max_tokens` in `[sampling]` is a completion-only cap, so a longer prompt costs
nothing against it. `max_total_tokens` covers the whole conversation but is not
binding at single turn either — 396 + 2048 sits well inside 4096. A family with
a long prompt is not being squeezed by it.

**So the truncation is a property of the task, not of the accounting.**
`reshape_permute` induces long generations: mean completion 899 tokens against
122–365 for the families that never truncate, because models write out index
reasoning and worked examples before the function. At a 2048 cap that costs 25%
of 4B rollouts and 61% of 2B's, which makes its solve rate a floor rather than
an estimate.

`outputs/headroom-reshape` re-runs this family alone at 4096 on the *same tasks
and seed* the definitive run uses — task ids verified identical — so it is a
paired A/B on one variable:

- solve rises, truncation gone → the cap was wrong and the prompt is fine;
- solve flat, truncation gone → the prompt genuinely defeats these models, and
  tightening its three-way restatement of the permutation is the next move
  rather than a guess.

That redundancy is not decoration. This is one of the three families that
shipped the undisclosed-convention defect, and stating the permutation as an
index equation, a dimension mapping and a resulting size is what removes the
inverse reading. Shortening it is the obvious fix for truncation and the exact
way to reintroduce the original defect, which is why it waits on this number.
