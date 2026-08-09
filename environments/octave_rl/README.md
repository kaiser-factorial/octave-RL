# octave-rl

A seeded GNU Octave coding curriculum for native `verifiers.v1`. Every task is
a function signature plus a natural-language spec; scoring runs the submitted
function against six hidden NumPy-derived cases on a pinned Octave interpreter
and awards the fraction that pass. No judge model, no partial credit for
looking plausible, and nothing the candidate prints can set its own score.

The task pool is generated rather than scraped, so it cannot leak from a public
benchmark and its hidden test cases are unbounded. It is **not** an unbounded
supply of distinct problems — see "What '500 tasks' means" below before
reporting a number.

## What this environment actually measures

Worth knowing before you train on it. A failure taxonomy over 222 baseline
rollouts put the competencies in this order:

1. **can the model emit runnable Octave at all** — ~36% of rollouts fail here;
2. **does it follow Octave's shape and orientation conventions** — ~5%;
3. **does it obey the prompt's stated constraints** (e.g. no loops) — ~4%;
4. **is the algorithm correct** — ~3%.

This is a language-fluency and convention-compliance benchmark far more than a
mathematical-reasoning one. That makes it a good RL substrate — the reward is
deterministic, cheap, and unhackable — but a result obtained here is a result
about learning a language's surface conventions. Read it that way.

## Quickstart

```bash
prime env install <owner>/octave-rl
```

```python
import verifiers as vf

env = vf.load_environment("octave-rl", level=1, num_tasks=200, seed=0)
```

Candidate code has to run somewhere. Pick a runtime before you evaluate — see
below; the default reaches for Prime Sandboxes.

## Choosing a runtime

| `octave_runtime` | Where candidate code runs | Use for |
| --- | --- | --- |
| `"prime"` (default) | one short-lived Prime Sandbox per execution | runs whose numbers you report |
| `"local"` | a bounded subprocess on the calling host | development, CI, pod-local training |

Both use the same input-only runner and the same host-side scorer, so hidden
values never enter the interpreter that runs model output either way. They
differ in containment and in speed: local scoring of the full 1,500-task pool
takes minutes, against roughly five minutes of cold provisioning *per
candidate* through a Sandbox.

The local runtime scores on whatever Octave it is pointed at, so pin it:

```bash
export OCTAVE_RL_OCTAVE_ROOTFS=/opt/octave-rootfs   # unpacked gnuoctave/octave:10.2.0
# or, if Octave is already installed:
export OCTAVE_RL_OCTAVE_BIN=/usr/local/bin/octave-cli
```

With a rootfs, candidates run under `unshare --net` → `chroot` → `ulimit`
bounds and see neither the host filesystem nor a network. If a network
namespace cannot be obtained the backend refuses to run unless
`OCTAVE_RL_ALLOW_UNISOLATED_LOCAL=1` is set, and records then report
`network_isolated = false`. `unshare`/`chroot` are Linux-only; on macOS run the
whole thing in the pinned image instead:

```bash
docker run --rm --platform linux/amd64 -v "$PWD":/w -w /w \
  -e OCTAVE_RL_OCTAVE_BIN=/usr/local/bin/octave-cli \
  -e OCTAVE_RL_ALLOW_UNISOLATED_LOCAL=1 \
  gnuoctave/octave:10.2.0 <your command>
```

## Families, levels, and difficulty

Ten families cover reductions, logical indexing, reshape/permutation, broadcast
arithmetic, sliding windows, linear solves, recurrences, matrix min/max
wrangling, string parsing, and signal identities.

- **Level 1** — diagnostic scalar/vector and simple matrix functions.
- **Level 2** — the working set, with more constraints and edge cases.
- **Level 3** — level 2 plus a no-loops constraint, or a harder variant.

**Family matters about 7x more than level.** Measured per-family pass rates
span a far wider range than the level ladder does, so a nominal "level 2 mix"
can contain anything from a near-unsolvable family to one the model passes
three times in four. If you are choosing a training mix, choose it on family
composition and target families near p = 0.5, where sampled groups carry the
most GRPO advantage. Do not choose it on level alone.

Pass rates at T=1.0, 72–96 rollouts per family, thinking off, single turn,
seed `20260808`. `0.1.0` is the pre-repair taskset, shown so the effect of the
0.2.0 fixes is visible rather than asserted. Treat these as a difficulty
ordering for two specific models, not as a property of the tasks.

| family | Nemotron 0.1.0 → **0.2.0** | Qwen3.5-4B 0.1.0 → **0.2.0** | note |
|---|---|---|---|
| `linsolve_tolerance` | 0.030 → **0.752** | 0.000 → **0.292** | was unrunnable as written |
| `struct_cell_wrangle` | 0.528 → **0.736** | 0.375 → **0.389** | matrix min/max; the name is historical |
| `logical_index` | 0.734 → **0.639** | 0.398 → **0.312** | the only family that fell |
| `sequence_recurrence` | 0.417 → **0.574** | 0.058 → **0.366** | |
| `reshape_permute` | 0.306 → **0.569** | 0.236 → **0.292** | prompt described a different signature |
| `broadcast_arith` | 0.236 → **0.458** | 0.014 → **0.083** | was unrunnable as written |
| `reduce_along_dim` | 0.351 → **0.389** | 0.137 → **0.326** | |
| `string_parse` | 0.264 → **0.366** | 0.032 → **0.056** | |
| `signal_identity` | 0.174 → **0.234** | 0.044 → **0.086** | genuinely hard: FFT circular autocorrelation |
| `sliding_window` | 0.148 → **0.213** | 0.014 → **0.190** | genuinely hard: stride semantics |

Paired on 96 tasks per model, the overall change is **+0.162** (SE 0.033) for
Nemotron and **+0.105** (SE 0.022) for Qwen. The families that moved most are
the ones that were defective; `signal_identity` and `sliding_window`, which
were diagnosed as *genuinely* hard rather than broken, barely moved for
Nemotron — the intended outcome.

`logical_index` fell in both models. Neither drop is individually significant
(|t| = 1.61 and 1.38) and the only change to that family was moving its
row-vector requirement onto the generated shape line, but the direction agrees
across two models and is recorded rather than explained away.

**Why this matters for training.** A family no model ever passes contributes
zero GRPO advantage at any group size. The lowest family pass rate rose from
0.030 to 0.213 (Nemotron) and 0.000 to 0.056 (Qwen), and the fraction of
unanimous — therefore gradient-free — groups at `group_size = 8` fell from
0.500 to **0.100** on Nemotron Level 2, and from 0.759 to **0.143** on Qwen
Level 2.

## What "500 tasks" means, and what a held-out pool holds out

Read this before reporting a number from this environment.

A prompt is built from the signature plus the family/level description. It
carries **nothing task-specific** — no sizes, no values. So all 50
`sequence_recurrence` level-1 tasks in a pool share a byte-identical prompt,
and a 1,500-task pool contains about **30 distinct problems**, one per
family/level cell. What varies between tasks in a cell is only the six hidden
test cases.

Pools are separated by seed, and by task the separation is exact — training
seed `0` and held-out seed `20260808` share 0 of 1,500 tasks. But they share
**30 of 30 prompts**. What is held out is the hidden test *inputs*, not the
question.

That makes this a genuine generalization test over inputs — the model never
sees a hidden case, so no answer can leak — but **not** a held-out-problem
benchmark. Two consequences:

- `num_tasks = 500` is 500 test-suite draws over 30 problems, not 500 problems.
  Every score is in-distribution on the problem.
- RL on this environment can drive toward memorising 30 function bodies, and a
  seed-disjoint evaluation will not detect that. It shows up as a high score
  with no transfer.

### Holding out a family gives you a real one

Set `families` to exclude some. Because prompts are determined by
(family, level), excluding a family is the only way to obtain a problem the
policy has genuinely never seen:

```toml
[taskset]
families = ["logical_index", "broadcast_arith", "sliding_window", "linsolve_tolerance",
            "sequence_recurrence", "struct_cell_wrangle", "string_parse", "signal_identity"]
```

`DEFAULT_HELDOUT_FAMILIES` is `["reduce_along_dim", "reshape_permute"]`, and
`training_families()` returns the complement. Both sit mid-difficulty for both
measured models, so neither is floored nor ceilinged; `reduce_along_dim` has a
near neighbour that stays in training (`struct_cell_wrangle` level 2+ is also a
column-wise reduction) so it tests transfer of a practiced idiom, while
`reshape_permute` has none. It is a default, not a recommendation — hold out
the two hardest families and a real gain hides against the floor.

Filtering *selects from* the full ten-family stream rather than cycling over
your selection, so a family's k-th task is byte-identical whichever others are
present. A train split and a holdout split drawn from one seed are therefore
disjoint **and** each individually comparable to a full-pool measurement.
Task ids come from the full stream, so they are stable but not contiguous
within a filtered pool.

**Use three splits, and do not blend them.**

| split | families | seed | used for |
|---|---|---|---|
| train | the trained ones | training | rollouts and gradient |
| validation | **the same trained ones** | held-out | level promotion, checkpoint selection |
| test | **the held-out ones** | held-out | generalization — read rarely, ideally once |

Two rules that are easy to get wrong:

- **Report the splits as separate numbers, never as one weighted score.**
  Averaging across families is what concealed a family stuck at 0.030 and a
  level stuck at 0.000 in this environment for weeks; a blended score rebuilds
  exactly that blindness.
- **Never gate promotion or select checkpoints on the held-out families.** That
  is selection leakage: you would be tuning against the thing you are claiming
  is untouched. Use the validation split for every decision, and keep the test
  split for the final read.

Also measure the held-out families *before* training. Base rates run from 0.21
to 0.75 across families, so an absolute post-training score is uninterpretable;
the quantity that means something is the change from base on the same config.

## Reward

`case_fraction` — the fraction of hidden cases passed — is the only reward.
There is no bonus for execution, for formatting, or for a
candidate-controlled result report. Formatting, vectorization, execution rate
and transposition are recorded as **metrics**, not folded into the score:

| metric | meaning |
| --- | --- |
| `execution_fraction` | cases that ran without raising |
| `correct_given_executed` | of those, the fraction correct |
| `transposed_fraction` | non-passing cases whose result is exactly the transpose of the expected value |
| `format_ok` | exactly one fenced block was emitted |
| `vectorized` | no `for`/`while` in the submission |

`execution_fraction` and `correct_given_executed` are worth logging separately
during training: they separate "cannot write Octave" from "wrote the wrong
algorithm", and those move independently.

## Output shape is graded

Scoring compares `size(actual)` against the expected value's shape exactly, so
orientation is part of every task — a correct answer returned as a column when
a row was asked for scores zero. Because of that, **every prompt states its
expected output shape**, in a sentence generated from the same expected values
the grader compares against:

```
Write this GNU Octave function:

    function out = linsolve_tolerance(A, b)

Solve the square linear system A*x=b.
Return a column vector (N-by-1).
Return exactly one fenced `octave` code block. Hidden tests include edge cases.
```

Arguments arrive as their signatures imply — a matrix as a matrix, a column
vector as a column — so the natural solution is conformant as written.
`scripts/validate_natural_solutions.py` in the source repository enforces this
by running a deliberately naive solution per family and level and requiring it
to pass.

## Multi-turn and the optional guide

`max_turns` maps to the persistent user's attempt budget. The user simulator
runs the submitted function after each attempt and reports back:

| Attempt | Feedback | Correctness multiplier |
| --- | --- | ---: |
| 1 | none | 1.00 |
| 2 | hidden pass count and Octave diagnostic | 0.85 |
| 3 | the same diagnostic plus one concise guide hint | 0.60 |

The guide defaults to `Qwen/Qwen3.5-35B-A3B` through Prime Inference. It sees
the public task prompt, the candidate source, and the first diagnostic only —
never hidden inputs, expected values, or the reference implementation.
Credentials are read at runtime from `PRIME_API_KEY` or the authenticated local
Prime config and are never written into traces or package configuration. Set
`guide_enabled = false` for a fully self-contained run with no external calls.

## Scoring boundary

The candidate sandbox receives only the candidate function, a generated
input-only runner, and the hidden *inputs*. It serialises each result's shape
and flattened values into a terminal `__OCTAVE_CANDIDATE_RESULT__<token>`
record carrying a token minted after generation. The trusted Python process
keeps expected outputs and pass counters entirely outside that sandbox,
validates the record, and applies the shape and tolerance comparison itself.

Candidate stdout therefore cannot set a score: printing a plausible result
record fails the token check, and the counters it would need to forge are never
in its process. Scoring does not use exit status. The in-Octave comparison
harness and its `__OCTAVE_HARNESS_RESULT__` protocol exist only in the separate
reference-pool validator, where the executed function is repository-owned.

`TaskData.image`, workdir, and container resources are deliberately unset: the
outer harness and user simulator run in their worker subprocesses, and only
candidate execution creates the pinned Octave Sandbox. This avoids a duplicate
container without weakening the boundary.

## Configuration

`level`, `num_tasks`, `seed`, and `require_vectorized` are typed Taskset config
fields; `octave_runtime`, `max_attempts` and `guide_enabled` are task and user
fields.

```toml
[taskset]
level = 2
num_tasks = 256
seed = 20260808

[taskset.task]
octave_runtime = "local"

[taskset.task.user]
octave_runtime = "local"
max_attempts = 1
guide_enabled = false
```

The package exports both `OctaveTaskset` and a `load_environment(level=1,
num_tasks=500, max_turns=2, require_vectorized=False, seed=0, **kwargs)` entry
point. It is native `verifiers.v1` throughout and does not mix in the legacy v0
`vf.Environment` form.

## Reproducibility

- Interpreter image: `gnuoctave/octave:10.2.0` (Docker Hub)
- Observed interpreter: GNU Octave 10.2.0
- Verifiers: `>=0.2.1,<0.3` · NumPy: `>=2.0,<3`
- Each task is fully determined by `(level, seed, task index)`.
- Six hidden cases per task, generated with NumPy and stored with the task.

## Changes in 0.3.0

- `families` is now a taskset config field, so train and eval pools can be
  disjoint by *problem* rather than only by hidden inputs. See "Holding out a
  family gives you a real one" above. Backward compatible: omitting it selects
  all ten families and reproduces 0.2.x exactly.

## Changes in 0.2.0

Task semantics changed; **0.2.0 scores are not comparable to 0.1.0 scores at
the task level.** Family names, task ids, level structure, hidden expected
values and reward multipliers are unchanged, so family-level comparison holds.

- `linsolve_tolerance` now receives `b` as a column, so `A\b` is conformant.
  Previously it arrived as a row and every hidden case failed with
  `nonconformant arguments` no matter what the model wrote — the family scored
  0.030 and 0.000 for two different models.
- `broadcast_arith` now receives `a` as a column, matching its own prompt, so
  `a + b` broadcasts as written.
- Every prompt states its graded output shape. The previous blanket "Preserve
  input orientation" line described neither the rule being enforced nor the
  output, and is gone.
- `reshape_permute` levels 2–3 describe their real contract. They previously
  said "For a 3-D array A" against a signature taking a flat vector and a size
  triple; both models scored `correct_given_executed = 0.000` across 96
  rollouts.
- Every level-3 description restates its own task. Two of them had been
  compressed to the point of dropping the specification —
  `struct_cell_wrangle` fell from 0.792 at level 2 to 0.000 at level 3 on the
  same computation.
- A generation truncated inside its code fence no longer reaches Octave with
  the opening fence attached, which had turned every such case into an
  uninformative line-1 syntax error.
