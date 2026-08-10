# Design: parameterised task descriptions

**Status: proposed, not started. Written 2026-08-09 as the handoff into a fresh
session.** Read `OCTAVE_HANDOFF.md` first, then this.

## The problem, in one measurement

A 1,500-task pool contains **30 distinct prompts** — one per (family, level).
A prompt is the signature plus the family/level description plus a generated
shape sentence, and none of those depend on the per-task draw. Training seed `0`
and held-out seed `20260808` share **30 of 30**.

Two consequences, both now measured rather than suspected:

1. **A held-out seed holds out hidden inputs, not questions.** The `families`
   config field (0.3.0) works around this by holding out whole families, but
   that buys at most a 10-way split and costs training coverage.
2. **The multi-turn scaffold is approximately correlated best-of-N sampling.**
   Retries are worth +0.22 to +0.38 solve rate, but a content-free retry
   captures 79–95% of that. With only 30 problems, resampling is most of what a
   multi-turn score measures.

Parameterisation attacks both: more distinct problems means a seed split starts
to mean something, and it dilutes the memorisation channel that makes
resampling so effective.

## What "one problem" should mean

A task's prompt should be determined by a **spec**: a small set of drawn choices
that changes what the function must compute, not merely which numbers test it.
Today the spec is empty — only the hidden cases vary.

Target: **≥ 200 distinct prompts** across the pool, i.e. roughly 6–8 variants per
(family, level) cell. That is enough that a 500-task pool stops being 50 copies
of one question, without exploding the surface a reference solution must cover.

## Per-family proposals

Each row is a suggested spec. The reference solution and the naive solution both
have to be generated *from the same spec* — see "The trap" below.

| family | current (fixed) | proposed spec dimensions | variants |
|---|---|---|---|
| `reduce_along_dim` | mean of each column; k-th largest | statistic ∈ {mean, median, sum, max, min, k-th largest}; axis ∈ {columns, rows} | 12 |
| `logical_index` | positive elements; NaN outside [lo,hi] | predicate ∈ {> 0, < 0, \|x\| > t, outside [lo,hi], even}; action ∈ {extract, set NaN, set 0, clamp} | 20 |
| `reshape_permute` | permute to [2 1 3] / [3 1 2] | permutation ∈ the 6 orderings of 3 dims; output ∈ {flattened row, flattened column} | 12 |
| `broadcast_arith` | outer sum; squared pairwise difference | op ∈ {sum, difference, product, squared difference, absolute difference, max} | 6 |
| `sliding_window` | sums; means; medians | statistic ∈ {sum, mean, median, max, min, range}; stride ∈ {1, drawn} | 12 |
| `linsolve_tolerance` | x; [x; residual norm] | return ∈ {x, [x; ‖Ax−b‖], ‖Ax−b‖ alone, [x; rank]}; system ∈ {square, over-determined} | 8 |
| `sequence_recurrence` | x(i)=x(i−1)+d; two-term linear | order ∈ {1, 2}; return ∈ {terms, cumulative sum, final term}; coefficients as args | 12 |
| `struct_cell_wrangle` | [column minima; column maxima] | statistics ∈ {[min;max], [mean;std], [min;median;max], [sum;count]}; axis ∈ {columns, rows} | 8 |
| `string_parse` | comma-separated numbers | separator ∈ {',', ';', whitespace, mixed}; numbers ∈ {integers, signed decimals}; output ∈ {row, column} | 16 |
| `signal_identity` | circshift by k; FFT autocorrelation | op ∈ {circular shift, autocorrelation, cross-correlation, circular convolution}; direction | 8 |

Rough total: **~114 variants**, spread across three levels, so the pool would
carry well over 200 distinct prompts once the level dimension multiplies in.

Start with **three families**, not ten. `reduce_along_dim`, `broadcast_arith`
and `sliding_window` are the cheapest to parameterise correctly and cover the
three shapes of risk (axis conventions, orientation, windowing semantics).
Measure, then extend.

## The trap this project keeps falling into

Every new variant is a fresh opportunity to ship a task that is only solvable
through an undisclosed convention. That defect has now occurred **three times**
(`linsolve_tolerance`, `broadcast_arith`, `reshape_permute`), and each time both
reference-based validators stayed green, because a reference passes precisely
*because* it contains the convention.

`scripts/validate_natural_solutions.py` was written to catch exactly this — but
**it currently hardcodes one naive solution per (family, level)**. The moment a
family generates several distinct problems, that validator silently covers one
of them and reports PASS for the rest.

**This is the single most important constraint on the design.** The naive
solution must be produced from the spec, alongside the reference:

```python
def variant(spec, rng, level):
    return {
        "description": ...,   # what the model is told
        "reference":   ...,   # known-correct, may use defensive coercion
        "natural":     ...,   # what a competent reader writes, no coercion
        "cases":       ...,   # hidden cases
    }
```

Three artefacts from one spec, so they cannot drift. If a variant cannot express
a `natural` that passes, that variant is not shippable — which is the check the
project has been missing all along, applied at authoring time rather than after.

## Invariants that must survive

Guarded by tests today; do not let parameterisation break them.

- **Every prompt states its graded output shape** (`_shape_sentence` derives it
  from the expected values, so it adapts for free) —
  `test_every_prompt_states_the_shape_the_grader_compares_against`.
- **Level 3 restates its own task**, never "…without loops" alone —
  `test_level_three_descriptions_restate_their_own_task`. This test reads a
  fixed `DESCRIPTIONS` dict and **will need rewriting** against the spec form.
- **A family's k-th task is identical whichever other families are present** —
  `test_a_family_generates_the_same_tasks_whichever_others_are_present`. This is
  what makes the holdout split trustworthy. Draw the spec from the same rng
  stream, in the same order, or this breaks silently.
- **Arguments arrive as their signature implies** —
  `test_orientation_sensitive_arguments_arrive_as_the_prompt_describes`.
- Family names, level structure and task ids should stay stable enough that the
  `families` holdout keeps working.

## Measurement plan

Cheap, and in this order:

1. **Distinct-prompt count.** The definition of success. Free:
   `len({t["prompt"][0]["content"] for t in build_tasks(...)})` per seed, and the
   overlap between two seeds. Target ≥ 200 total, and **seed overlap well below
   100%**.
2. **`validate_natural_solutions.py --num-tasks 500`** must be 9,000/9,000, with
   the validator parameterised. If it is not, the variant set is wrong. Run it
   in the amd64 container or a CPU sandbox — see the repo README.
3. **`validate_local_runtime.py`** for the reference path, same scale.
4. **Per-family, per-variant pass rate at 1 turn**, Nemotron and Qwen3.5-4B,
   32 tasks × 8 rollouts, seed `20260808`, ≈$0.25 on a CPU sandbox. Watch for
   any variant near 0.00 — that is the old defect wearing a new hat.
5. **Re-baseline the training model** (Qwen3.5-0.8B, Level 1, 3 turns) so the
   10–35% band judgement is made on the new pool.
6. **Re-run the retry control** on the new pool. If distinct problems weaken the
   memorisation channel, the content-free retry should capture *less* of the
   gain than the 79–95% measured today. That is the most interesting number this
   whole change can produce.

## Reporting rules that apply throughout

- **Solve rate comes from `raw_case_fraction`, never `rewards.case_fraction`.**
  The reward is discounted 0.85/0.60 by attempt, so thresholding it cannot count
  a success after attempt 1. This produced a false headline on 2026-08-09.
- **State the turn budget with every score.** Multi-turn is mostly resampling.
- **Standard errors across tasks, not rollouts.**
- **Report truncation** — and say which truncation: the trainer's counts
  `max_turns`, the per-call `finish_reason == "length"` counts the token cap.

## Version and comparability

This is a **breaking change to task semantics: version 0.5.0**. Every prompt
changes, so:

- No number measured before it is comparable at task or family level.
- The 2026-08-09 artifacts stay as the record of the 0.2–0.4 pool.
- Re-measure the headline cells before quoting anything.

## Cost and shape of the work

| step | effort | compute |
|---|---|---|
| spec plumbing + three families | half a session | free |
| parameterise `validate_natural_solutions` | small, but do it *first* | free |
| validation passes (both validators, 500/level) | wall-clock | ~$0.30 sandbox |
| per-variant pass-rate sweep, 2 models | wall-clock | ~$0.30 |
| re-baseline 0.8B + retry control | wall-clock | ~$0.40 |
| remaining seven families | second session | ~$0.30 |

Roughly **$1.50 of compute** for the first three families end to end, and the
design work is the expensive part.

## Open questions for the next session

1. **How much variation is too much?** Fifteen variants per cell makes each one
   rare in a 500-task pool, which weakens per-variant statistics. Six to eight
   is the suggestion; it is not measured.
2. **Should the level ladder survive?** Levels currently mean "same task, harder
   constraint". With a spec, difficulty could be a spec dimension instead, which
   would be cleaner but is a larger change and breaks the `level` config field.
3. **Does the family holdout still make sense?** With genuine problem diversity,
   a *variant* holdout inside each family may be a better generalization test —
   and it would not cost training coverage the way holding out two whole
   families does.
4. **Is the guide's value spec-dependent?** It is worth +0.062 (Nemotron) and
   +0.129 (Qwen-4B) at L1 today. On a more diverse pool it may matter more,
   since the model can no longer lean on having seen the problem.
