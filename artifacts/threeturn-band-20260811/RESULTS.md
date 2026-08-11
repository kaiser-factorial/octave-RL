# Which Qwen to train: the three-attempt band on the 0.5.0 pool

Level 1, **three attempts with the guide** — the scaffold training actually
runs. 160 tasks x 4 rollouts = 640 rollouts per model, T=1.0, thinking off,
seed `20260808`, 2048-token completion cap, 6144-token conversation budget.
Generation through Prime Inference; scoring local against the pinned Octave
10.2.0. 1,920 rollouts total.

The companion single-turn sweep (`artifacts/variant-sweep-20260811/`) closed by
saying it could not answer which model to train, because a single-turn number is
the wrong instrument for a scaffold worth 3-4x. This is that cell.

## Provenance: pre-tightening `reshape_permute`, same as the sweep

The working tree that produced this run carries the 927-character
`reshape_permute` level-2 description, not the 506-character one now on `main`.
Every `reshape_permute` figure below describes a prompt that no longer exists.
Nothing else in the pool differs.

## The band

Solve rate is the undiscounted `solved` metric, ± standard error across tasks.

| model | solve rate | reward | execution | format_ok | truncation | attempts |
|---|---:|---:|---:|---:|---:|---:|
| Qwen3.5-0.8B | 0.044 ± 0.010 | 0.036 | 0.073 | 0.652 | 0.125 | 2.95 |
| Qwen3.5-2B | 0.127 ± 0.017 | 0.099 | 0.220 | 0.508 | 0.263 | 2.88 |
| Qwen3.5-4B | **0.681 ± 0.023** | 0.578 | 0.752 | 0.906 | 0.097 | 2.13 |

Against the same scaffold on the 0.4.x pool (0.8B 0.117, 2B 0.148, 4B 0.727):
**2B and 4B are essentially unmoved; 0.8B lost two thirds of its score.** The
parameterised pool did not make the work harder for a model that could do it —
it removed the thirty memorisable prompts that were carrying the smallest model.

Read against a 10-35% target band, only **2B** lands inside it. That reading is
wrong, and the next section is why.

## The band is a proxy. Measure the thing it proxies for.

A target band exists to predict *gradient signal*: GRPO computes advantage
within a group, so a group whose rollouts all score the same contributes
nothing, whatever the reward was. At `group_size = 4` that is directly
measurable, and it does not track the band.

| model | groups with signal | all-zero | all-one | unanimous |
|---|---:|---:|---:|---:|
| Qwen3.5-0.8B | 0.131 | 0.869 | 0.000 | 0.869 |
| Qwen3.5-2B | 0.331 | 0.662 | 0.000 | 0.669 |
| Qwen3.5-4B | **0.931** | 0.050 | 0.013 | 0.069 |

**4B wastes 7% of its rollouts. 2B wastes 67%. 0.8B wastes 87%.**

The band heuristic assumes a mid-range solve rate keeps groups mixed, which is
true under independence — at p = 0.68 with four rollouts, chance alone predicts
22% unanimous. 4B observes 6.9%, three times better than independent, because
its per-task rates cluster in the middle rather than splitting the pool into
tasks it always solves and tasks it never does. 0.8B and 2B fail the other way:
their zeros are not spread thinly across many tasks, they are concentrated on
tasks that are dead for every rollout.

A high solve rate is only a problem when it produces all-one groups. 4B produces
those on 1.3% of tasks. It has not run out of headroom.

## Why the small models fail, since it is not the same reason

`format_ok` is 0.65 for 0.8B and **0.51 for 2B**: half of 2B's rollouts never
emit a parseable function at all. Its execution fraction is 0.220, so most of
what remains does not run.

Truncation is the aggravating factor for 2B — 26% of rollouts hit the cap, and
**29% of its zero-scoring rollouts are truncated**, against 18% for 4B and 13%
for 0.8B. Its p95 completion budget is 6144 tokens, the conversation ceiling
exactly, so the three-attempt scaffold is spending 2B's budget on retries it
cannot finish. Some of 2B's gap is a budget artefact rather than capability, but
closing it entirely would still leave it far short of 4B's signal fraction.

Solved on the first attempt: 0.8B 0.017, 2B 0.044, 4B 0.312. The retries are
doing most of the work for the small models and comparatively little for 4B —
which is the shape reported on the 0.4.x pool too.

## Per-family solve rate

| family | 0.8B | 2B | 4B |
|---|---:|---:|---:|
| `sequence_recurrence` | 0.06 | 0.34 | 0.88 |
| `reduce_along_dim` | 0.14 | 0.11 | 0.84 |
| `logical_index` | 0.12 | 0.19 | 0.83 |
| `struct_cell_wrangle` | 0.00 | 0.06 | 0.77 |
| `sliding_window` | 0.00 | 0.11 | 0.73 |
| `linsolve_tolerance` | 0.00 | 0.19 | 0.72 |
| `broadcast_arith` | 0.11 | 0.14 | 0.69 |
| `reshape_permute` | 0.00 | 0.00 | 0.59 |
| `signal_identity` | 0.00 | 0.11 | 0.56 |
| `string_parse` | 0.00 | 0.02 | 0.20 |

**`string_parse` is the hard family at Level 1** even for 4B, and the only one
under 0.5 for it. `reshape_permute` at 0.59 for 4B and 0.00 for both small
models is measured on the untightened prompt and should be re-read after the
re-probe.

Six of ten families are at exactly 0.00 for 0.8B: at this size the model is not
being taught by most of the pool.

## Conclusion

**Train Qwen3.5-4B.** It is the only rung where nearly every group carries an
advantage, its format compliance (0.91) means its zeros are Octave errors rather
than parsing failures, and its 1.3% all-one rate says it has room left to
improve. 2B looks better on the band and is worse on every mechanism the band
was standing in for.

This does not settle **which level** to train on — Level 1 is the only rung
measured here, and the sweep showed the ladder flattened between Levels 1 and 2.
The promotion gates still need re-deriving against the new pool.
