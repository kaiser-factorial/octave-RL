# Octave RL: a purpose-built training environment

**For:** Codex (or whichever agent picks this up)
**Goal:** an RL environment for training a model to write correct, idiomatic Octave — not a
benchmark, not an Exercism port.
**Status:** independent of Forks A/B/C. Doesn't block on them and isn't blocked by them.

**Related:** `octave_RL/FORKC_OCTAVE_HANDOFF.md` — §6 (images, `LANG_CMDS`, the test-runner
problem), §7 (hidden tests) and §8 (Octave language gotchas) all transfer directly and are not
repeated here. Read those three sections before starting. Everything else in that document is
superseded by this one.

---

## 0. Why not just add Octave to aiderpolyglot

Because aider's 225 exercises were selected by keeping only those solved by **three or fewer of
seven frontier models**, deliberately calibrated so the strongest models land between 5% and 50%.

That's a benchmark property. For training it's a liability:

- The model you'll actually train is small-to-mid, in a language it has seen far less of than
  Python, on problems chosen to defeat frontier models. Expected baseline: ~0.
- Reward is binary pass/fail, so ~0 baseline means **no variance, therefore no gradient**. That's
  the degenerate case — you'd spend real money watching a flat line.
- Thirty exercises is an eval set. Prime's own recipes run `batch_size` 128–256; you need hundreds
  to thousands of tasks.

The one thing inheriting polyglot buys — comparability with aider's published leaderboard — is a
benchmark virtue you don't need when you're training one model rather than ranking many.

So: generate tasks programmatically, control difficulty explicitly, and design the reward to be
dense rather than binary.

---

## 1. The shape of a task

One Octave function, fully specified, verified numerically. Every task is:

- a **prompt** stating the problem and the exact required function signature
- a **fixed function name** the model does not choose
- a set of **precomputed test cases** — inputs and expected outputs, generated in Python at dataset
  build time
- a **tolerance** for numeric comparison
- optional **constraints** (e.g. "must be vectorised")

A dataset row:

```python
{
  "prompt": [{"role": "user", "content": prompt_text}],
  "info": {
    "family": "col_kth_largest",
    "level": 1,
    "fn_name": "col_kth_largest",              # the model never picks this
    "signature": "function out = col_kth_largest(A, k)",
    "cases": [
      {"args": [[[3,1],[2,9],[5,4]], 2], "expected": [3, 4]},
      ...
    ],
    "tolerance": 1e-9,
    "require_vectorized": False,
  },
  "task": "octave-l1-col_kth_largest",
}
```

**Expected outputs are computed at generation time in Python (numpy), not at scoring time.** Two
consequences worth being deliberate about:

1. No reference implementation ever ships to the sandbox, so there is nothing for the model to read
   or overwrite. The ground truth exists only as literal values in a harness written *after* the
   model's code.
2. Scoring is a pure comparison — cheap, deterministic, no judge, no second model. This is the
   single best property Octave gives you as an RL target.

---

## 2. Reward design — graduated, not binary

This is where the environment should be clearly better than aiderpolyglot. Binary pass/fail throws
away almost everything.

Use `vf.Rubric` with a weighted primary and several observed-only metrics:

| Function | Weight | What it captures |
|---|---|---|
| `case_fraction` | **1.0** | fraction of test cases passed, not all-or-nothing |
| `runs_without_error` | 0.1 | did the file even parse and execute |
| `vectorized` | 0.0 → 0.1 later | source contains no `for`/`while` when the task requires it |
| `format_ok` | 0.0 | did the reply contain a parseable code block |

The graduation is the point. A model emitting syntactically valid Octave that runs but returns
wrong numbers **should** score above one emitting garbage — that's the gradient a binary reward
denies you. `case_fraction` alone converts a task from one bit of signal into ~10.

Register the weight-0.0 entries with `Rubric.add_metric(func, weight=0.0)` — a real method in
`verifiers` (`rubrics/rubric.py`), alongside `add_reward_func(func, weight=1.0)`. Observe first,
price later: watch how `vectorized` behaves for a few hundred rollouts before deciding whether
paying for it teaches the right thing.

**Watch for the obvious reward hack in `case_fraction`.** Partial credit means a model that returns
a constant matching one common expected value scores above zero for free. Two mitigations: make
sure generated cases have well-spread expected values (don't let half of them be `0`), and log the
distribution of per-case pass patterns — a model passing exactly the same subset every time is
gaming, not learning.

---

## 3. Task families

Each family is a Python generator with difficulty knobs. Target **8–12 families**; that plus
parameter variation gets you thousands of tasks.

Concrete starting set, chosen because expected values are trivially computable in numpy and the
Octave idioms are genuinely distinctive:

| Family | Task shape | Difficulty knobs |
|---|---|---|
| `reduce_along_dim` | k-th largest / n-th moment / trimmed mean per column | matrix size, `k`, which dim, ties |
| `logical_index` | select/replace elements by predicate | predicate complexity, NaN presence |
| `reshape_permute` | reorder a tensor to a stated layout | ndims, whether dims are ambiguous |
| `broadcast_arith` | compute an outer-product-ish quantity without loops | operand shapes, vectorisation required |
| `sliding_window` | moving statistic with edge handling | window size, edge policy, stride |
| `linsolve_tolerance` | solve/least-squares, report residual | conditioning, over/under-determined |
| `sequence_recurrence` | generate a sequence with a stated recurrence | length, whether closed form exists |
| `struct_cell_wrangle` | build/query a struct array or cell array of char | nesting depth, field count |
| `string_parse` | parse a formatted char array into numbers | format irregularity, error cases |
| `signal_identity` | verify/apply an FFT or filter identity | length, tolerance tightness |

Two design rules for every family:

- **The signature is given in the prompt.** The model should never be guessing the interface — that
  measures mind-reading, not Octave.
- **Include error cases where they're natural.** Document the exact expected `error('id:sub', ...)`
  identifier in the prompt, the way Exercism tracks do. A model can't be expected to invent an
  identifier the test checks for.

---

## 4. Curriculum

Three levels, following the pattern in Prime's search-agents guide.

**Level 1 — diagnostic.** Single function, scalar or vector in, deterministic out, no vectorisation
constraint, generous tolerance. Should be *easy*. Its job is to confirm the pipeline works: if
reward doesn't climb toward 1.0 within ~15 steps, you have a bug, not a hard problem. Do not skip
this level because it looks trivial — it's the cheapest debugging tool you'll have.

**Level 2 — the working set.** Matrix operations, multiple constraints, edge cases (empty input,
NaN, singular matrices, ties). This is where most training time goes.

**Level 3 — the interesting one.** Vectorisation required (rewarded via the `vectorized` metric
promoted to a real weight), tighter numeric tolerance, or multiple cooperating functions.

Generate the levels as separate task pools so you can weight them via `[[env]] ratio` in the
training config, and shift the mix as the model improves.

---

## 5. Security by construction

The lecture's own lesson, applied at design time rather than patched in later.

**The model never supplies a filename.** It returns one code block; the environment writes it to
`info["fn_name"] + ".m"`, a name the environment chose. There is no filename to validate because
there is no filename in the model's output. The entire class of attack that broke aiderpolyglot
doesn't exist here.

**The harness is written after the model's file, and contains the expected values.** Generate
`run_cases.m` from `info["cases"]` with literals inlined, write it *after* the model's function,
and run it. The model has no opportunity to see or modify it — it isn't on disk when the model is
producing output, and the model can't name it anyway.

**Still snapshot the directory.** Cheap insurance, and it gives you a `tampering_detected` metric
for free. If the model somehow produces files you didn't write, you want to know.

If you later relax the single-block design (e.g. multi-function tasks), the allowlist comes back —
and the rule from the Fork B notes applies: enumerate what's *permitted*, never what's forbidden.

---

## 6. Implementation sketch

`SingleTurnEnv` for Level 1, `MultiTurnEnv` with `max_turns=2` for Levels 2–3 — a second attempt
after seeing the failure output teaches debugging, which is a real capability, at roughly double the
rollout cost. Make it a `load_environment` kwarg rather than hardcoding it.

```python
def load_environment(
    level: int = 1,
    num_tasks: int = 500,
    max_turns: int = 2,
    require_vectorized: bool = False,
    seed: int = 0,
    sandbox_provider: str = "prime",
    **kwargs,
) -> vf.Environment:
```

Scoring flow per rollout:

1. Parse a single fenced code block (```octave or ```matlab or bare) from the reply.
2. Write it to `<fn_name>.m` in a fresh temp dir.
3. Generate `run_cases.m` from `info["cases"]` with values inlined, write it.
4. Run Octave in the sandbox.
5. Parse a machine-readable summary line from stdout.

**Have the harness print structured output, not just set an exit code.** Something like:

```
RESULT passed=7 total=10
CASE 3 FAIL expected=4.000000 got=3.000000
```

Then `case_fraction` is a parse rather than an inference. This is a direct improvement on
aiderpolyglot, where `proc.returncode == 0` is the entire signal and a partially-correct solution is
indistinguishable from a syntax error.

For the image, `LANG_CMDS`-style command, exit-code behaviour, stdout-vs-stderr, and quoting: **see
§6 of the Fork C handoff**, and treat its runner incantation as unverified until you've run it. The
five checks listed there apply unchanged.

---

## 7. Calibrate before you scale

**Do not author 12 families and 2,000 tasks against an unverified pipeline.**

1. One family, Level 1, 20 tasks. Get the image building and the harness reporting correctly.
2. `prime eval run` with **the model you actually intend to train** — not a frontier model. The
   number that matters is that model's baseline, and the target is roughly **10–35%**.
3. If it's above ~70%: turn the difficulty knobs up, or move weight to Level 2.
   If it's near 0%: the tasks are too hard, the signature documentation is insufficient, or the
   harness is broken. **Distinguish those three before touching difficulty** — the third is common
   and looks exactly like the first.
4. Only then generate at scale.

Verified model overlaps (present on both the training and inference lists, so you can baseline and
train the same model): `meta-llama/Llama-3.2-1B-Instruct`, `Qwen/Qwen3.5-0.8B`, and the Qwen3.5
2B/4B/9B line. **No model is free for both** — the free-to-train ones aren't servable for
inference, and the free inference model isn't trainable.

---

## 8. Validation

- **Every generated task's reference answer passes its own harness.** Script this as a loop over
  the whole pool. It's the highest-value check here — a task whose expected value is wrong trains
  the model to be wrong.
- **A deliberately wrong solution fails**, and `case_fraction` lands strictly between 0 and 1 for a
  partially-correct one. Verify the partial case explicitly; it's the whole point of the reward
  design.
- **The reward distribution isn't degenerate.** Plot it across a few hundred rollouts. All-0 or
  all-1 means no gradient regardless of how good the tasks look individually.
- **Check for the constant-output hack** described in §2 before running anything long.
- **A 10-step calibration run** (`--override max_steps=10`) before committing to a full run.

---

## 9. Deliverables

- A task-generation module: 8–12 families, difficulty knobs, seeded and reproducible
- Precomputed cases with numpy-derived expected values, plus the reference-answer validation loop
- The environment package: `load_environment` per the signature above, graduated rubric, structured
  harness output
- A pinned Octave image, with the Octave version recorded
- Calibration results: baseline for the intended training model, per level
- One 10-step training run showing the reward curve is not flat
- Notes on anything in §6 of the Fork C handoff that turned out wrong when actually run

---

## 10. What this deliberately isn't

Not an Exercism track. Not a benchmark comparable to aider's numbers. Not a fork of anything.

If the Exercism Octave track is still appealing later — and it's a real gap, 83 tracks and no
Octave or MATLAB — it's a separate project with its own shape: ~100+ exercises, concept exercises,
a Docker test runner, a representer, an analyzer, and a community review process. Worth doing on
its own terms. Not on the path to training a model, and trying to serve both goals with one artifact
gets you neither.
