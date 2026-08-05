# Build an Octave RL environment for Prime Intellect

**Goal:** a `verifiers` environment that trains a language model to write correct, idiomatic
GNU Octave — packaged for the Prime Intellect Environments Hub and runnable through Hosted Training.

**Assumed knowledge:** the Prime Intellect platform skill (Lab, the Environments Hub, `load_environment`,
`vf.Rubric`, `prime eval run`, Hosted Training config). This document covers only what's specific to
Octave and to designing a *training* environment rather than a benchmark. Where the platform skill
already explains something, it isn't repeated here.

**Deliverable:** a published environment plus the evidence that its difficulty is calibrated for the
model you intend to train.

---

## 1. What you're building, concretely

Each task presents one fully-specified Octave function to write. The model returns a single code
block. The environment writes it to a filename *it* chose, runs it against generated hidden inputs
in a candidate sandbox, and has the trusted task process compare the reported values against
precomputed expected values. The score is the fraction of cases that match.

Everything about that sentence is deliberate; §5 and §6 explain why.

## 2. Why purpose-built, and not an adapted coding benchmark

The tempting shortcut is to take an existing multi-language coding benchmark and add Octave to it.
Don't. Public coding benchmarks are calibrated so that *frontier* models are separated by them —
problems are frequently selected precisely because strong models fail them. That's the right
property for ranking models and the wrong one for training a small one.

Point a small open model at such a set, in a language it has seen far less of than Python, and you
get a baseline near zero. Combined with the binary pass/fail reward such benchmarks use, that means
no variance, therefore no gradient — you pay for a flat line. The comparability you'd inherit is a
benchmark virtue you don't need when training one model rather than ranking many.

Generate tasks instead. It gives you volume (Hosted Training recipes commonly run `batch_size`
128–256; a few dozen hand-written tasks is an eval set, not a training set) and it makes difficulty
a parameter rather than a fixed property.

*(Incidental fact, so you don't go looking: Exercism has no Octave or MATLAB track. There is no
existing exercise set to port even if you wanted one. And MATLAB itself can't be containerised —
it's licensed — which is why the target is Octave.)*

## 3. The shape of a task

```python
{
  "prompt": [{"role": "user", "content": prompt_text}],
  "info": {
    "family": "col_kth_largest",
    "level": 1,
    "fn_name": "col_kth_largest",              # the model never chooses this
    "signature": "function out = col_kth_largest(A, k)",
    "cases": [
      {"args": [[[3, 1], [2, 9], [5, 4]], 2], "expected": [3, 4]},
      ...
    ],
    "tolerance": 1e-9,
    "require_vectorized": False,
  },
  "task": "octave-l1-col_kth_largest",
}
```

**Expected values are computed in Python (numpy) at dataset-build time**, not by running a reference
implementation at scoring time. Two consequences worth being deliberate about:

1. Neither reference outputs nor pass counters enter the sandbox that runs candidate code, so there
   is no score state for the model to read or overwrite.
2. Scoring is a pure host-side comparison — cheap, deterministic, no judge model, no second
   inference call.
   This is the single best property Octave gives you as an RL target: correctness is numerically
   checkable.

**The prompt must state the exact function signature.** The model should never be inferring the
interface; that measures mind-reading, not Octave. Where a task has error cases, state the exact
`error('id:reason', ...)` identifier the tests expect — a model cannot guess an identifier.

## 4. Reward design

Use a graduated primary reward and several observed-only metrics:

| Function | Weight | Captures |
|---|---|---|
| `case_fraction` | **1.0** | fraction of test cases passed |
| `vectorized` | 0.0 → 0.1 later | source free of `for`/`while` when the task requires it |
| `format_ok` | 0.0 | did the reply contain a parseable code block |

The graduation is the point. Binary pass/fail turns a ten-case task into one bit of signal; scoring
the fraction turns it into roughly ten, and it means a model producing valid-but-wrong Octave scores
above one producing garbage. That difference *is* the gradient you want it climbing. Apply the
retry multiplier to this correctness reward only; code executing in an untrusted sandbox receives
no separate execution or output-format bonus.

Register the weight-0.0 entries with `add_metric(func, weight=0.0)` — observed, not priced. Watch a
metric across a few hundred rollouts before deciding it deserves real weight. `verifiers` also ships
`Parser.get_format_reward_func()`, so `format_ok` may not need writing from scratch.

**Guard against the hack partial credit introduces.** Fractional scoring means a constant output
matching one common expected value earns free points. Two mitigations: make sure generated cases
have well-spread expected values (don't let half of them be `0`), and log the per-case pass pattern
— a model passing exactly the same subset every time is gaming, not learning.

## 5. Security by construction

Design the exploit out rather than validating against it. The rule worth internalising: **defences
that enumerate what is forbidden are always circumventable; defences that enumerate what is
permitted are not.**

- **The model never supplies a filename.** It returns one code block; you write it to
  `info["fn_name"] + ".m"`. There is no filename to validate because there is none in its output.
- **The candidate runner contains inputs only.** It calls the candidate function and returns its
  shape and flattened numeric values; expected values and pass counters remain in the trusted task
  process. A terminal report is transport, never a score.
- **Keep the candidate filesystem minimal.** Candidate code can modify every file in its own
  sandbox, so no scoring decision may depend on that filesystem or its output beyond reported values
  that are independently compared on the host.

If you later relax the single-block design (multi-function tasks, say), the allowlist comes back —
and note that hashing known files is *not* sufficient. A model can **add** a file that changes test
behaviour while every existing file's hash stays untouched. Catching that means diffing the whole
directory manifest, which means knowing what's legitimate: the allowlist, rebuilt as a detector.

## 6. The Octave execution layer

This is the part with no prior art, so it needs the most care.

### Image

```python
"octave": modal.Image.from_registry("debian:bookworm-slim", add_python="3.11").apt_install("octave")
```

Pin the base tag rather than using `:latest` — reproducible scoring needs reproducible toolchains.
Check whether Debian's packaged Octave is new enough for the language features your tasks use; if
not, pin a newer base or install from backports. **Record the Octave version the image actually
produces and state it in the environment README** — a benchmark whose interpreter version is unknown
isn't reproducible.

### Running the tests

Generate `run_candidate.m` per task with only the input literals, then invoke Octave over it. A
starting point:

```bash
octave --no-gui --quiet run_candidate.m
```

with the candidate runner responsible only for structured output:

```matlab
% run_candidate.m  (generated per task)
% ... for each case: call the function in a try/catch and save size(actual)
%     plus actual(:)' in a record ...
printf('__OCTAVE_CANDIDATE_RESULT__<fresh-token> %s\n', jsonencode(records));
```

**Have the runner print a machine-readable transport record, then score it outside the sandbox** —
don't rely on exit code or candidate-controlled pass counts. The trusted process checks shape and
tolerance against its private expected values, which retains partial credit without placing a score
oracle in the interpreter executing model code.

**Five things to verify empirically before trusting any of the above.** None of it has been run;
it's a reasoned starting point.

1. **Exit codes in both directions.** Octave does not reliably exit non-zero on an uncaught error.
   Confirm a passing run gives 0 *and* a failing run gives non-zero.
2. **`assert` failures are catchable.** Octave's `assert()` should throw and `try/catch` should
   catch it. Confirm with a deliberately failing assertion.
3. **Output reaches stdout.** The failure text is fed back to the model on its second attempt; if
   Octave writes diagnostics to stderr and you only capture stdout, the model gets an empty failure
   message and can't debug. Append `2>&1` if needed.
4. **Quoting survives the shell.** If you invoke via `sh -c "..."`, nested quotes are a hazard —
   writing a `.m` file and calling `octave file.m` is more robust than an inline `--eval`.
5. **Startup time.** Octave starts slower than Python. Confirm a typical task finishes well inside
   your sandbox timeout.

### Getting files into the sandbox

Prime Sandboxes have per-file `upload_file` rather than a directory mount (see the platform skill's
API reference for the SDK). For a task with a couple of small `.m` files that's fine; if you end up
with more, tar locally, upload one archive, and untar in the container.

### Octave gotchas that will bite

- **Function name must match filename.** `col_kth_largest.m` must define
  `function ... = col_kth_largest(...)`. This constrains task naming — use underscores, not hyphens.
- **1-based indexing.** If you port index arithmetic from a Python reference, re-derive it rather
  than translating it.
- **Strings:** single-quoted are char arrays, double-quoted support escapes. Cell arrays of char are
  the idiom for string lists (`strjoin`, `strsplit`, `regexprep` all expect them). There's no native
  string-array type as in modern MATLAB.
- **Errors:** `error('id:reason', 'message')`. Catch with `try ... catch err`, and check
  `err.identifier` (stable) rather than `err.message` (brittle). Identifiers need a colon and can't
  start with a digit.
- **`classdef` support is incomplete.** Design tasks around plain functions over vectors, cell arrays
  and structs. Avoid `containers.Map` too — its behaviour has diverged from MATLAB's across versions.
- **Integer division and type promotion** differ from MATLAB in edge cases. Prefer explicit `floor`
  or `idivide`.

## 7. Task families

Each family is a Python generator with difficulty knobs. Target **8–12 families**; that plus
parameter variation yields thousands of tasks.

| Family | Task shape | Difficulty knobs |
|---|---|---|
| `reduce_along_dim` | k-th largest / n-th moment / trimmed mean per column | matrix size, `k`, which dim, ties |
| `logical_index` | select or replace elements by predicate | predicate complexity, NaN presence |
| `reshape_permute` | reorder a tensor to a stated layout | ndims, ambiguity of dims |
| `broadcast_arith` | outer-product-ish quantity without loops | operand shapes, vectorisation required |
| `sliding_window` | moving statistic with edge handling | window size, edge policy, stride |
| `linsolve_tolerance` | solve / least-squares, report residual | conditioning, over- or under-determined |
| `sequence_recurrence` | generate a sequence from a stated recurrence | length, closed form available or not |
| `struct_cell_wrangle` | build or query a struct array / cell array | nesting depth, field count |
| `string_parse` | parse a formatted char array into numbers | format irregularity, error cases |
| `signal_identity` | apply or verify an FFT / filter identity | length, tolerance tightness |

Vectorisation tasks are worth special attention — they're distinctively Octave, and they're
verifiable *twice*: once on output, once by checking the submitted source contains no loop. That
second check is a natural home for the `vectorized` metric.

## 8. Curriculum

Three levels, generated as separate pools so you can weight them via `[[env]] ratio` and shift the
mix as the model improves.

**Level 1 — diagnostic.** Single function, scalar or vector in, deterministic out, no vectorisation
constraint, generous tolerance. Should be *easy*. Its job is to confirm the pipeline works: if
reward doesn't climb toward 1.0 within ~15 steps, you have a bug, not a hard problem. Don't skip it
because it looks trivial — it's the cheapest debugging tool you'll have.

**Level 2 — the working set.** Matrix operations, multiple constraints, edge cases (empty input,
NaN, singular matrices, ties). Most training time goes here.

**Level 3 — stretch.** Vectorisation required, tighter tolerance, or multiple cooperating functions.

## 9. Package layout

```
octave-rl/
├── octave_rl.py          # load_environment + the env class
├── generators/           # one module per task family
│   └── ...
├── harness.py            # builds input-only candidate runners and trusted reference harnesses
├── pyproject.toml
└── README.md
```

```python
def load_environment(
    level: int = 1,
    num_tasks: int = 500,
    max_turns: int = 2,
    require_vectorized: bool = False,
    seed: int = 0,
    **kwargs,
) -> vf.Environment:
```

`SingleTurnEnv` is right for Level 1 — cheapest rollouts, densest signal. `MultiTurnEnv` with
`max_turns=2` suits Levels 2–3, where a second attempt after seeing the failure output teaches
debugging, at roughly double the rollout cost. Make it a kwarg rather than hardcoding it.

**Pin `verifiers` with an upper bound in `pyproject.toml`.** An unbounded `verifiers>=X` is how
environments break silently on a fresh install months later — see the platform skill's versioning
reference. Give it a ceiling and note the version you developed against.

## 10. Calibrate before scaling

Do not build 12 families and 2,000 tasks against an unverified pipeline.

1. **One family, Level 1, ~20 tasks.** Get the image building and the harness reporting correctly.
2. **Evaluate with the model you actually intend to train** — not a frontier model. The only number
   that matters is that model's baseline, and the target is roughly **10–35%**.
3. **Interpret the result carefully.** Above ~70%: turn the knobs up or shift weight to Level 2.
   Near 0%: there are *three* candidate causes — tasks too hard, signature documentation
   insufficient, or the harness broken — and the third is common and looks exactly like the first.
   Distinguish them before touching difficulty.
4. **Then** generate at scale, and do one short calibration run (`--override max_steps=10`) before
   committing budget.

**Model selection:** training and inference are separately metered with separate model lists. Pick a
model present on **both** (`prime train models` ∩ `prime inference models`) so you can baseline it,
train it, and re-measure without switching models mid-experiment. Note that free-tier availability
does not overlap between the two lists.

## 11. Validation

- **Every generated task's reference answer passes a trusted reference harness.** Script this as a
  loop over the whole pool. This is the highest-value check in the document — a task with a wrong
  expected value trains the model to be wrong. That validator may use an expected-value harness
  because it executes repository-owned reference code, never model code.
- **A deliberately incorrect solution fails**, and `case_fraction` lands strictly between 0 and 1 for
  a partially-correct one. Verify the partial case explicitly; it's the entire point of the reward
  design.
- **The reward distribution isn't degenerate.** Plot it across a few hundred rollouts. All-0 or all-1
  means no gradient no matter how good the tasks look individually.
- **Check for the constant-output hack** (§4) before running anything long.
- **At least one end-to-end test** that drives the actual task runtime rather than calling only
  parser helpers. Direct-call tests never exercise runtime/provisioning behavior, so they can't
  catch API-contract mismatches — an environment can have a green test suite and be entirely
  unrunnable.

## 12. Deliverables

- Task-generation module: 8–12 families, seeded and reproducible, difficulty parameterised
- Precomputed cases with numpy-derived expected values, plus the reference-answer validation loop
- The environment package with the `load_environment` signature above and the graduated rubric
- A pinned Octave image, with the version it produces recorded in the README
- A bounded `verifiers` pin
- Calibration results: baseline for the intended training model, per level
- One 10-step training run showing the reward curve is not flat
- Notes on anything in §6 that turned out different when actually run — that section is reasoned,
  not executed
