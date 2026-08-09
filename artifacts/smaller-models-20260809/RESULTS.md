# Can we train something smaller than 4B? — 2026-08-09

`Qwen/Qwen3.5-4B` was selected long ago as "the first tested Qwen size with
nonzero reward". That selection was made on a taskset where whole families were
unsolvable *regardless of model*, so the criterion no longer means what it did.
The 2026-08-09 repair, and the finding that repaired Level 1 is now saturated
for 4B, make the question live again.

Prime offers `Qwen3.5-0.8B`, `2B`, `4B`, `9B` and `35B-A3B` for both training
and inference. 0.8B and 2B cost **1/5** and **1/2** of 4B per training token.

Everything below ran on one CPU Sandbox, no GPU, for a few cents.

## Single-turn baseline — the number that misleads

Levels 1–3, 32 tasks, 8 rollouts, seed `20260808`, T=1.0, thinking off, no
guide. Same design as `artifacts/postfix-eval-20260809/`, so these sit directly
beside Nemotron and 4B.

| model | L1 | L2 | L3 | overall | format_ok |
|---|---:|---:|---:|---:|---:|
| Qwen3.5-0.8B | 0.030 | 0.006 | 0.008 | **0.015** | 0.71 |
| Qwen3.5-2B | 0.068 | 0.040 | 0.014 | **0.041** | 0.49 |
| Qwen3.5-4B *(reference)* | 0.400 | 0.214 | 0.118 | — | — |

768 rollouts per model, zero infrastructure errors.

Read alone, this says both are far below the 10–35% starting band and the
answer is no. That reading is wrong, because **training does not run
single-turn**.

## Training configuration — the number that decides

Level 1, **three attempts with the guide**, 32 tasks x 4 rollouts = 128
rollouts per model, T=1.0. This is the scaffold `octave-qwen-4b-3step-smoke`
actually uses.

| model | reward | solve rate | execution | format_ok | **truncation** | tokens p50/p95 |
|---|---:|---:|---:|---:|---:|---|
| Qwen3.5-0.8B | **0.107** | 0.078 | 0.171 | 0.64 | **2.5%** | 136 / 695 |
| Qwen3.5-2B | **0.128** | 0.086 | 0.243 | 0.59 | **10.0%** | 152 / 1536 |
| Qwen3.5-4B | **0.641** | 0.328 | 0.816 | 0.92 | 3.0% | 140 / 825 |

**n = 128 rollouts per model** (32 tasks x 4). An earlier version of this file
said 256; that was a double-count from two overlapping globs. The means are
unaffected — the same data was counted twice — but the sample is half as large
as stated.

**Thinking is genuinely off**, verified from traces rather than config: zero
`reasoning_tokens` and zero non-empty `reasoning_content` across all 989 model
calls.

**Truncation matters only for 2B.** Its p95 completion length sits *at* the
1,536-token cap and 10% of calls are cut off; 0.8B and 4B truncate at 2.5-3.0%
with p95 well under the cap. Raising `max_completion_tokens` would buy 2B
something and the other two almost nothing. No length pressure belongs in the
reward — see the 2026-08-08 entry "Efficiency pressure belongs in the advantage,
not in the reward".

**Both smaller models land inside the 10–35% band on Level 1, exactly where 4B
has outgrown it.** The scaffold is worth 3–4x: 2B goes 0.068 → 0.128, 4B goes
0.400 → 0.641.

### A correction to the smoke

The 3-step smoke reported 0.9250 for 4B on Level 1 and this run gives 0.641 on
the same cell. The smoke's batches were 8–12 rollouts; this is 128. The smoke's
figure was small-sample optimism, as its own write-up warned. **Level 1 is
still too easy for 4B at 0.641** — the conclusion stands, the magnitude does
not.

## What limits the small models

Not algorithmic reasoning. `execution_fraction` is 0.171 and 0.243, and
`format_ok` is 0.64 and 0.59 — they frequently cannot emit a well-formed single
fenced Octave function at all, let alone a correct one. 4B sits at 0.816 and
0.92.

That is worth thinking about rather than treating as a disqualification. The
2026-08-08 taxonomy ranked this environment's competencies as: (1) can the model
emit runnable Octave, (2) shape/orientation conventions, (3) prompt-constraint
compliance, (4) algorithmic correctness. A model failing mostly at (1) is
failing at the thing this environment is *best* at teaching, and RL on it has
somewhere obvious to go.

The risk is the mirror image: with `format_ok` near 0.6, roughly 40% of
rollouts score zero for a formatting reason, so a large share of the gradient
would be spent teaching fence discipline rather than Octave.

## The number that actually decides: group economics

Mean reward is not what determines whether a group teaches anything — the
**solve rate** is, because the reward is near-binary. Measured on these same
rollouts, at `group_size = 4`:

| model | solve rate | dispersion | degenerate (zero-gradient) groups |
|---|---:|---:|---:|
| Qwen3.5-0.8B | 0.078 | **1.44** | **0.781** |
| Qwen3.5-2B | 0.086 | **1.15** | 0.719 |
| Qwen3.5-4B | 0.328 | 2.15 | 0.500 |

At `group_size = 4` on Level 1, **78% of 0.8B groups carry no gradient** —
which is the pathology the entire taskset repair existed to remove,
reintroduced by model choice rather than by the pool.

But the dispersion column cuts the other way and is the more interesting
result. 0.8B and 2B sit at 1.44 and 1.15, close to independent, where 4B is at
2.15. Near-independent rollouts mean **larger groups buy nearly what the theory
says they should**. Taking 0.8B to `group_size = 8` at p = 0.078 and near
independence gives a degenerate fraction around `(1-p)^8` = 0.52, against 0.78
at g=4. For 4B, correlation eats much of that gain.

So the small models are not disqualified by group economics; they are
*conditional* on a larger group size, and they respond to it better than 4B
does.

## Recommendation

**Qwen3.5-0.8B on Level 1 at `group_size >= 8`**, with eyes open.

For it: cheapest by 5x, the *highest* `format_ok` of the three (0.64 against
2B's 0.59), the *lowest* truncation (2.5% against 2B's 10%), and the lowest
rollout correlation, so raising the group size actually works.

Against it: a solve rate of 0.078 is at or just below the bottom of the 10-35%
band — and the band should be read on solve rate, not on the fractional reward
of 0.107, which is inflated by partial credit. At `group_size = 4` that leaves
78% of groups gradient-free, so the group size is not optional.

2B buys very little for twice the price: +0.008 solve rate, worse formatting,
and 4x the truncation. If 0.8B stalls, the informative next step is 4B on an
L2/L3 mix, not 2B on Level 1.

If staying on 4B, move to an L2/L3 mix — 0.214 and 0.118 single-turn.

**Unresolved:** whether 0.8B's headroom is reachable. Its ceiling here is
formatting and basic fluency, and nobody has checked whether RL fixes that
quickly (in which case it is an ideal curriculum — this environment's strongest
competency is exactly "can the model emit runnable Octave") or plateaus (in
which case the run measures fence discipline). One 3-step smoke on 0.8B at
`group_size = 8` settles it for about $1, and should report **truncation
alongside reward, execution and format** — omitting it is how 2B's 10% went
unnoticed until asked for.

## Files

- `raw-outputs.tgz` — single-turn baseline traces, both models, all levels.
- `multiturn-outputs.tgz` — training-configuration traces for all three sizes,
  plus the attempts x guide matrix logs from the user-server diagnosis.
