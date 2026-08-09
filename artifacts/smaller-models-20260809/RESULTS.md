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

Level 1, **three attempts with the guide**, 32 tasks x 4 rollouts = 256
rollouts per model, T=1.0. This is the scaffold `octave-qwen-4b-3step-smoke`
actually uses.

| model | reward | fully solved | execution | format_ok |
|---|---:|---:|---:|---:|
| Qwen3.5-0.8B | **0.107** | 0.078 | 0.171 | 0.64 |
| Qwen3.5-2B | **0.128** | 0.086 | 0.243 | 0.59 |
| Qwen3.5-4B | **0.641** | 0.328 | 0.816 | 0.92 |

**Both smaller models land inside the 10–35% band on Level 1, exactly where 4B
has outgrown it.** The scaffold is worth 3–4x: 2B goes 0.068 → 0.128, 4B goes
0.400 → 0.641.

### A correction to the smoke

The 3-step smoke reported 0.9250 for 4B on Level 1 and this run gives 0.641 on
the same cell. The smoke's batches were 8–12 rollouts; this is 256. The smoke's
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

## Recommendation

**Qwen3.5-2B on Level 1**, if the goal is a training run with headroom:

- 0.128 sits in the band where 4B's 0.641 does not;
- half the training cost of 4B;
- more capacity than 0.8B, whose 0.107 comes with worse execution and a
  suspicion of too little capacity to move.

If staying on 4B, move to an L2/L3 mix — 0.214 and 0.118 single-turn.

**Unresolved:** whether 2B's headroom is reachable. Its ceiling here is
formatting, and no one has yet checked whether RL fixes that quickly (in which
case it becomes a good curriculum) or plateaus (in which case the run measures
fence discipline). One 3-step smoke on 2B would tell you, and now costs about
$1.

## Files

- `raw-outputs.tgz` — single-turn baseline traces, both models, all levels.
- `multiturn-outputs.tgz` — training-configuration traces for all three sizes,
  plus the attempts x guide matrix logs from the user-server diagnosis.
