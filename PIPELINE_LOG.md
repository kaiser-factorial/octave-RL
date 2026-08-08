# Pipeline log

A running record of defects found in the training pipeline, what each one
actually broke, and how it was caught. `OCTAVE_HANDOFF.md` says where the work
*is*; this file says what went wrong on the way and what that implies about
what to trust.

Entries are newest-first. Every entry answers "why did this survive until now",
because on this project the recurring failure has not been the bug itself — it
has been a green check that measured something adjacent to the thing that
mattered. A defect fixed without that field recorded will be reintroduced.

## How to add an entry

```markdown
## YYYY-MM-DD — one-line symptom

**Symptom.** What was observed, with the concrete numbers.
**Root cause.** The mechanism, precisely enough to re-derive it.
**Blast radius.** What data, runs, and published claims are affected.
**Why it survived.** Which check should have caught it and what it measured instead.
**Fix.** What changed, and what deliberately did not.
**Verification.** The command and its result.
**Residual risk.** What remains unproven.
```

Record the entry when the diagnosis lands, not at the end of the session. An
entry written from memory loses the numbers, and the numbers are the reason the
entry is worth having.

---

## 2026-08-08 — Orientation failures are literally transposes, and the judge over-counted them

**Question.** Is "orientation mismatch" a real, mechanically detectable
relation — and should it earn partial credit?

**It is exactly detectable, with no judge.** If a candidate returns the
transpose of the expected value, then because Octave flattens `actual(:)`
column by column, its reported values equal the **row-major** flatten of the
expected value, and its reported `shape` is reversed. That is a two-line test
against data the transport already sends.

**Result over the 19 rollouts that ran cleanly and got every answer wrong:**

| relation to expected | rollouts |
| --- | ---: |
| exact transpose on **every** case | 4 |
| genuinely different values | 15 |

No rollout was a reshape or permutation of the right values; it is transpose or
nothing.

**The judge over-counted by 2x.** It labelled 8 of 19 `orientation_or_shape`;
mechanically only 4 are transposes. The judge was reading the *code* ("initializes
the output as a column vector") rather than checking the *numbers*, and
inferring orientation from intent. This is a concrete instance of why the judge
stays out of the reward: on the one sub-question where a deterministic check
exists, the judge was wrong by a factor of two in the direction of its own
narrative.

**Detection added as a metric, not as reward.** `transposed_fraction` is now
reported per rollout, counted only for cases that did not already pass, so a
symmetric answer equal to its own transpose scores as correct rather than both.
**The reward is unchanged and a transposed answer still scores zero**, because
the prompts explicitly require preserving input orientation — it is a stated
part of the task, not an incidental convention.

**On rewarding it anyway.** Defensible, and cheap to add as a configurable
credit, but it is a real trade: partial credit for a transpose weakens a
requirement the prompt makes explicit, and the gradient it buys is small — 4 of
222 rollouts, about 1.8%. The stronger reason to keep it as a metric is that it
answers a *WS3* question rather than a training one. Arms can now be compared on
which competency each improved: whether the model learned to emit runnable
Octave (`execution_fraction`), learned the orientation convention
(`transposed_fraction`), or learned the algorithm (`correct_given_executed`).
SFT on reference solutions gets the convention for free from gold code; RL has
to discover it. If those arms differ in *which* of these they fix, that is a
more interesting routing readout than a single scalar reward — and it is now
measurable without changing the reward that makes the arms comparable.

## 2026-08-08 — What the octave environment actually measures (and it is mostly not reasoning)

**Question.** Can this substrate produce a reward for *algorithmic reasoning*,
and does it need a judge model to do so? Answered by splitting execution from
correctness in the retained data, then running a judge **offline** over the one
population where reasoning is the only thing that could have failed.

**Step 1: the environment already distinguished these and was discarding it.**
The candidate transport reports an `ok` flag per case, so "threw an exception"
and "ran and computed the wrong number" were always separable —
`raw_case_fraction` just collapsed both to zero. `score_candidate_output` now
also returns `executed` / `execution_fraction`, surfaced as the
`execution_fraction` and `correct_given_executed` metrics. **The reward is
unchanged**, so everything measured so far stays comparable.

**Step 2: the tier distribution** (222 baseline rollouts, thinking-on cell
excluded):

| tier | share |
| --- | ---: |
| doesn't run at all | 35.6% |
| runs, but wrong values | 12.2% |
| runs, correct | 48.2% |
| partially runs | 4.1% |

Adding just this one tier drops group-2 gradient waste from 50.1% to 37.5% —
about what raising `group_size` from 2 to 3 buys, and the two compose.

**Step 3: a judge over the 19 rollouts that ran cleanly and got everything
wrong.** Offline, `Qwen3.5-35B-A3B`, pre-registered categories, about $0.01:

| category | n | share |
| --- | ---: | ---: |
| orientation_or_shape | 8 | 42.1% |
| misread_spec | 4 | 21.1% |
| other (all: used loops despite a vectorization requirement) | 2 | 10.5% |
| wrong_algorithm | 2 | 10.5% |
| indexing_or_off_by_one | 2 | 10.5% |
| numerical_or_tolerance | 1 | 5.3% |

**The finding.** Only about a quarter of "algorithmic" failures are algorithmic.
42% are orientation/shape convention; another 32% are prompt-constraint
compliance (misread spec, or looping where the prompt required vectorization).
Genuine reasoning errors are 5 of 19 — roughly **3% of all rollouts**.

Ranked by how much of the reward signal each competency controls, this
environment measures:

1. can the model emit runnable Octave at all (~36% of rollouts fail here);
2. does it follow Octave's orientation/shape conventions (~5% of all rollouts);
3. does it obey the prompt's stated constraints (~4%);
4. is the algorithm correct (~3%).

**Why this matters for WS3.** The substrate is largely an Octave-fluency and
convention-compliance benchmark. That is a perfectly serviceable RL environment
— the reward is deterministic, cheap, unhackable and now Sandbox-free — but a
routing result obtained on it is a result about *learning a language's surface
conventions*, not about mathematical reasoning. That should be stated in the
write-up rather than discovered by a reader.

**On judge-in-the-reward: recommended against, for WS3 specifically.** It would
make the reward non-stationary and gameable inside the very quantity being
measured; it would add run-to-run variance to the matched-competence checkpoint
selection, which pre-registers a +/-2 point tolerance; and it would reintroduce
the external per-rollout dependency this project just removed. It also would not
address the dominant failure mode, since 59% of zeros never execute and a judge
asked whether unrun code is correct is measuring its own charity. Offline
taxonomy, as done here, has none of those problems.

**Residual risk.** n=19 for the taxonomy, single judge, no human agreement
check. Treat the category split as indicative. The direction of the result —
that convention and compliance dominate reasoning — is large enough to survive
considerable judge error, but the exact percentages are not load-bearing.

## 2026-08-08 — Correction: the guide DID fire; and most failures are execution errors, not wrong answers

**Two corrections and one substantive finding, from decoding the retained
rollouts rather than grepping logs.**

**Correction 1: the guide fired.** The smoke entry below says it never did.
That check was invalid: the hint is injected into the *user message content*,
which is stored tokenized in `rollouts/*/rank_0.bin`, never as text in any log,
so `grep "Guide hint"` could not have found it either way. Decoding the token
IDs with the model tokenizer finds hints in **3 of 15** unique retained
sequences, and they are substantive, e.g.

> *"The function fails for multi-dimensional inputs because `ndims(x)` returns
> the number of dimensions, not the actual dimension index; therefore, inputs
> with 3 or more dimensions are incorrectly rejected by the final `else`
> block"*

A per-rollout firing rate cannot be recovered from the retained artifacts,
because the rollout files hold the post-filter trainable batch rather than
every rollout. To measure it properly, log a counter in `OctaveUser`.

**Correction 2: check content, not logs.** The general lesson is the same one
this file keeps recording — the check has to touch the mechanism. A log grep
for a value that never enters the log will always return zero, and zero looks
like an answer.

**Finding: the zeros are mostly broken code, not wrong code.** Across the 138
zero-score rollouts in the 256-rollout baseline:

| what happened | count | share |
| --- | ---: | ---: |
| errored on **every** hidden case (`ok:false`, empty shape/values) | 82 | 59.4% |
| ran successfully but computed wrong values | 29 | 21.0% |
| never produced a parseable result at all | 27 | 19.6% |

So roughly **79% of failures are the model failing to produce executable
Octave**, not failing to produce correct Octave. Three consequences:

1. **This is why the reward is binary.** A function that throws scores 0/6 by
   construction. Partial credit needs code that runs and is right on some cases
   — rare. It explains the 4.3% graduated rate directly.
2. **This is why the tutor rarely rescues a rollout.** The guide receives the
   first diagnostic line and returns one hint of at most 96 tokens. When the
   candidate errors on all six cases the diagnostic is a runtime error, and one
   short hint seldom repairs a fundamentally broken function inside one more
   attempt at the same token budget.
3. **The environment currently measures Octave execution competence more than
   algorithmic reasoning.** For WS3 that is a substantive property of the
   substrate, not a detail: the behaviour RL will chase here is "emit runnable
   code" well before "emit correct code".

The 19.6% that never produced a parseable result are largely truncation, which
also means a hint cannot help — the model never finished its answer.

## 2026-08-08 — Retraction: the entropy "collapse" was a batch-composition artifact

**Claim retracted.** The 2026-08-08 smoke entry below flagged a 53% entropy
decline (0.7954 -> 0.4917 -> 0.3754) as the main risk to a longer run. Log
analysis says that reading was wrong, and the real findings are elsewhere.

**Why it cannot be policy sharpening.** The exported step-3 adapter gives the
effective weight change directly. Summed over all 128 adapted matrices,
`||dW_eff||_F = 0.0773`; the largest single element moved by `1.16e-5`; RMS
element-wise change is **1.53e-6** against base weights of order `1e-2`. That
is a relative perturbation of about **0.015%**. Three steps at lr `1e-5` with
grad norms 0.11-0.22 on rank-16 LoRA cannot halve a policy's entropy, and did
not.

**What the number actually tracked: batch length.** Entropy is a mean over the
tokens in each training batch, and the batches differed enormously.

| step | entropy | turns | truncation | peak trainer mem |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 0.7954 | 2.0 | 50% | 11.7 GiB |
| 2 | 0.4917 | 2.2 | 50% | 11.0 GiB |
| 3 | 0.3754 | 1.5 | 0% | 10.2 GiB |

Peak memory falls monotonically with batch token count. Steps 1-2 were
dominated by three-attempt failures that ran to the 1536-token cap — long,
uncertain, high-entropy text. Step 3's effective pair was short first-attempt
solves. Grad norm is also non-monotonic (0.109 -> 0.216 -> 0.113), which no
steady-sharpening story explains. With `batch_size 8 / group_size 2` a step
trains on **four distinct tasks**, so task-to-task variance dominates every
per-step statistic.

**Lesson.** Per-step scalars from a 3-step run on 4 tasks are not a trend.
Before treating one as a trend, check whether the batch composition changed —
`turns`, `truncation` and peak memory are the tells, and the adapter norm
settles it outright.

---

## 2026-08-08 — The graduated reward is effectively binary, and group_size 2 wastes 59% of rollouts

**Symptom.** Across **256 baseline rollouts**, `raw_case_fraction` was exactly
0.0 or exactly 1.0 in **95.7%** of cases: 138 zero, 107 full, 11 partial. The
only partial values observed anywhere were `1/6` and `3/6`. In the 29 logged
training rollouts there was no partial credit at all.

**Root cause.** Octave functions are all-or-nothing on these tasks. A wrong
algorithm fails all six hidden cases; a correct one passes all six. There is
little middle ground for a per-case fraction to express.

**Why this matters more than it looks.** The WS3 design doc justifies the
octave substrate partly on graduated reward giving "denser RL signal than
binary boxed-match". **That premise does not hold on this environment.** The
reward is Bernoulli in practice, so a GRPO group contributes nothing unless its
samples disagree.

That makes `group_size` the binding constraint on useful gradient. At the
measured Level-1 pass rate at rollout temperature (p = 0.2865):

| group_size | P(all samples agree) | wasted | useful rollouts per 8 |
| ---: | ---: | ---: | ---: |
| 2 | 0.5912 | 59.1% | 3.27 |
| 4 | 0.2659 | 26.6% | 5.87 |
| 8 | 0.0672 | 6.7% | 7.46 |

Observed trainable fractions at `group_size = 2` were 4/8, 4/8, 2/8 — **58.3%
wasted**, against 59.1% predicted. The model matches the measurement.

**Why the wrong value was chosen.** `group_size = 2` was inherited from the
curriculum controller's "stable envelope", but the CUDA failures that envelope
was built around occurred at *seven to eight simultaneous long generations* —
that is `max_inflight_rollouts`, a concurrency bound. `group_size` is samples
per task and does not by itself raise concurrency. The original 20-step run
used `group_size = 8` on the same hardware. The two knobs were conflated.

**Fix.** Raise `group_size` (4 or 8) while holding `max_inflight_rollouts = 2`.
This roughly doubles useful gradient per GPU-hour at no extra concurrency risk.
Increase `batch_size` alongside it to keep several distinct tasks per step —
`batch 16 / group 4` gives four tasks and 26% waste, against four tasks and 59%
waste today.

**Residual risk.** The pass rate p moves as the policy improves, and waste is
minimised near p = 0.5. A curriculum that holds p in band is therefore doing
double duty; re-check the group-size arithmetic whenever the mix changes.

## 2026-08-08 — 3-step training smoke passed; two bootstrap gaps closed

**Not a defect** — the loop works. Recorded for the two environment gaps it
exposed and the one trend worth watching. Full tables in
`artifacts/training/qwen-4b-3step-smoke-20260808/RESULTS.md`.

Three optimizer steps from base Qwen3.5-4B against the corrected reward, with
candidate scoring on the pinned Octave 10.2.0 rootfs and **zero Prime Sandbox
calls**. All eight pre-registered pass criteria met. (An earlier version of this entry
said the guide never fired in-training; see the correction entry above -- it
did, in 3 of 15 retained sequences.)
Trainer/inference mismatch KL was **0.0003-0.0004**, against the project's
0.015 monitoring line and the 2026-08-06 run's 0.0176. Zero rollout
infrastructure errors. $0.92 compute.

**Gap 1: `uv sync` alone does not install flash-attn.** The trainer imports
`ring_flash_attn`, which imports `flash_attn` at module load, so the run died
before step 0 with `ModuleNotFoundError`. prime-rl pins a prebuilt wheel behind
the `flash-attn` extra; the bootstrap now uses `uv sync --extra flash-attn`.

**Gap 2: any `uv sync` prunes the editable octave_rl install**, because it is
not in prime-rl's lock file. Re-running sync to add the extra silently removed
the environment package. The bootstrap now reinstalls it immediately after
every sync, with the ordering commented so it is not "tidied" back.

**Watch entropy on the real run.** It fell 0.7954 -> 0.4917 -> 0.3754 across
three steps, a 53% drop. Nothing is collapsed at this length, but that slope
sustained over 20+ steps is the most likely way a longer run ends badly. Treat
a continued decline as a stopping condition, not a curiosity.

**Do not read a trend from three steps.** Reward went 0.65 -> 0.46 -> 0.93 on
batches of 8. That is variance. Step 3's 0.9250 is a small-sample artifact and
must not be quoted as a capability number — the same mistake the 0.905 figure
invited.

---

## 2026-08-08 — First pod run on the corrected scorer; two measurement traps

**Not a defect** — the run worked. Recorded because two of its findings will
silently corrupt future numbers if forgotten. Full tables in
`artifacts/baseline-eval-20260808/RESULTS.md`.

**Trap 1: `reasoning_effort = "none"` does not disable thinking.** Only
`enable_thinking = false` at the renderer does. A control cell setting the
sampling knob alone produced 87.9% truncation, mean completion exactly at the
1024-token cap, format validity **0.12**, and `raw = 0.0938` — the model almost
never emits a fenced function. Those are structural zeros that look exactly
like a capability measurement. The tell is `finish_reason = "length"` plus
`format_ok`; check both before believing any low score.

**Trap 2: the G1 band must be measured at the rollout temperature.** GRPO's
advantage comes from reward spread within a group of samples, so what matters
is the pass rate at T=1.0, not at greedy. Base Qwen3.5-4B on Level 1 measures
**0.7031 greedy** but **0.2865 at T=1.0** — the greedy number sits far outside
the 10-35% target band while the sampled one sits inside it. Choosing a
curriculum mix from the greedy figure would have driven a pointless difficulty
increase. Do not raise sampling temperature to widen the band either: at
T != 1.0 the sampled distribution stops matching the distribution whose
log-probs the trainer scores, which is a systematic version of the
trainer/inference divergence the mismatch-KL threshold exists to catch.

**Incidental confirmation.** The T=1.0 / 1024-token / thinking-off cell scores
**0.2865** against the historical July 29 calibration's **0.2817** — a
replication, and independent evidence that the flattening defect never touched
the calibration numbers.

**Also carry forward.** Level 3 is not harder than Level 2 for this model
(0.406 vs 0.375 base; identical solve counts for step-20), so reweighting
toward L3 will not create headroom. And the 20-step policy shows no detectable
single-turn gain over base at n=32 — its measurable effect is output discipline
(format validity 1.00, zero truncation on L1/L2), not correctness.

**Cost.** Pod `092edcad9f6b4f7f96d5c89beb54945e`, 2x RTX 6000 Ada, 83 minutes,
$2.04 against a $12 ceiling. 256 rollouts, zero infrastructure errors, zero
Prime Sandbox calls. Terminated; final inventory zero pods and zero sandboxes.

---

## 2026-08-08 — Executor config field shadowed a verifiers base-class field

**Symptom.** Every rollout in the first pod evaluation failed instantly — 32
rollouts "finished" in 8 seconds — with empty `rewards`/`metrics` and

```text
AttributeError: 'str' object has no attribute 'model_dump'
  verifiers/v1/rollout.py:178 in run -> serve_user(...)
```

**Root cause.** `vf.UserConfig` already declares `runtime: RuntimeConfig =
SubprocessConfig()`, describing where the user-simulator process runs. The new
executor switch was also named `runtime`, so `OctaveUserConfig` replaced an
object field with the string `"local"`, and `serve_user` called `.model_dump()`
on it. The collision was invisible locally because no unit test constructed a
real rollout.

**Blast radius.** The first pod eval only; caught before it produced a number.
Had the field been named this way in a training run, every rollout would have
errored at the same point and produced a batch of structural zeros — the exact
"infrastructure zeros are not model zeros" trap already in this repo's lessons.

**Why it survived.** The local tests exercise `execute_candidate_locally` and
config construction directly, never `serve_user`. Pydantic silently accepts a
subclass narrowing a parent field's type, so nothing complained until the value
was used.

**Fix.** Renamed to `octave_runtime` on both `OctaveUserConfig` and
`OctaveTaskConfig`. `test_octave_runtime_does_not_shadow_the_verifiers_user_runtime`
now asserts the inherited `runtime` still equals `vf.UserConfig().runtime` and
still has `model_dump`.

**Residual risk.** Any future field added to these configs can collide the same
way. Check the verifiers base class before naming one — the repo's own
AGENTS.md rule ("go through the verifiers code as the source of truth") is the
guard here, and it was skipped.

---

## 2026-08-08 — Pod bring-up: three environment frictions

Recorded so the next pod costs minutes instead of a paid hour. All three are
fixed in `scripts/pod_bootstrap.sh` and `scripts/fetch_pinned_octave.py`.

1. **prime-rl submodules use `git@github.com:` URLs.** A fresh pod has no SSH
   key, so `git submodule update --init --recursive` fails on all four and
   aborts. Fix: `git config --global url."https://github.com/".insteadOf
   "git@github.com:"` before cloning. A partially-aborted attempt can also
   leave `deps/pydantic-config` empty while the other three populate, which
   surfaces later as `does not appear to be a Python project` during `uv sync`
   — re-run with `--force` for that submodule specifically.

2. **Python's `tarfile` cannot extract the Octave image.** `filter="tar"`
   rejects the distro's absolute symlinks (`etc/alternatives/awk` →
   `/usr/bin/mawk`) as escaping the destination. Relaxing to `fully_trusted`
   would take a downloaded archive's word on where its members may land, so the
   fetch script shells out to GNU `tar` instead, matching what a container
   runtime does.

3. **`chroot` lives in `/usr/sbin`, which scrubbed PATHs omit.** The executor
   builds a deliberately minimal child environment, and `execvpe` resolves
   argv[0] against *that* PATH — so an unqualified `chroot` failed to exec and
   every candidate returned empty output. `_host_tool` now resolves helper
   binaries against the parent PATH and falls back to the sbin directories.
   Worth noting that the "refuse rather than silently degrade" rule is what
   made this legible: the runtime raised `LocalExecutionUnavailable` naming the
   missing binary instead of quietly scoring zeros.

---

## 2026-08-08 — Reward path scored correct matrix answers as zero

**Symptom.** Reference solutions — code known to be correct, since the
generators produce it — scored `0/6` on every hidden case of
`octave-l1-broadcast_arith-00003`. The candidate transport was well-formed
(`structured_result = 1.0`) and reported the right *shape* (`[2, 4]`) and the
right *values*, in the wrong order:

```text
expected (row-major)  : [-4, -3, -4, -8,  3,  4,  3, -1]
reported (column-major): [-4,  3, -3,  4, -4,  3, -8, -1]
```

**Root cause.** `build_candidate_runner` reports `actual(:)'`, which Octave
flattens column by column. `candidate_record_matches` flattened the expected
JSON value row by row via `_flatten`, then compared the two sequences
element-wise. Scalars, row vectors, and column vectors flatten identically
under both orders, so the two agreed everywhere except on results with more
than one row *and* more than one column.

**Blast radius.** 250 of 1,500 tasks in the default pool — 1,500 of 9,000
hidden cases (16.7%). All six cases of every affected task are matrix-valued,
so an affected task is worth exactly `0.0` no matter what the model writes:

| Level | Affected families | Unsolvable tasks |
| ---: | --- | ---: |
| 1 | `broadcast_arith` | 50 / 500 (10%) |
| 2 | `broadcast_arith`, `struct_cell_wrangle` | 100 / 500 (20%) |
| 3 | `broadcast_arith`, `struct_cell_wrangle` | 100 / 500 (20%) |

This is worse than lost signal. Under GRPO, a correct matrix solution receives
the same zero advantage as a broken one, so the objective actively pushes the
policy away from correct matrix-producing code.

Affected runs: the 2026-08-06 two-step RTX 6000 Ada continuation (effective
rewards 0.8000 and 0.5000, trainable 4/8 and 2/8) was scored through this path.
Treat its `weights/step_2` policy as trained against a corrupted reward, not as
a clean continuation base.

Unaffected: the original 20-step run and its `0.905` held-out Level 1 result
predate the 2026-08-05 reward hardening and were scored through `build_harness`,
which compares inside Octave where both sides are Octave arrays.

**Why it survived.** `validate_reference_pool.py` — the source of the
"9,000/9,000 hidden cases passed" line repeated in `README.md` and
`OCTAVE_HANDOFF.md` — exercises `build_harness` + `parse_harness_result`. That
path is correct, and it is **not the path that computes rewards**. The reward
path (`build_candidate_runner` + `candidate_record_matches`) arrived with the
2026-08-05 reward hardening and was never validated against the reference pool;
the pre-existing 9,000/9,000 claim was carried across the rewrite unchanged.
Same shape as the WS2 "8/8 STABLE" false green: the check verified the
intention, not the mechanism.

**Fix.** `harness._octave_flatten` reorders a two-dimensional expected value
into Octave's column-major order before comparison, and
`candidate_record_matches` uses it. The candidate protocol is deliberately
unchanged — the trusted side holds the ground truth and is the correct side to
adapt. A transposed answer still fails, which
`test_matrix_results_are_compared_in_octave_column_major_order` pins.

**Verification.** `scripts/validate_local_runtime.py --num-tasks 500 --seed 0`
runs the full default pool through the *reward* path: **9,000/9,000 hidden
cases across 1,500 tasks, 138.9 s**, zero Prime calls. Pre-fix, the same command
fails 250 tasks.

**Residual risk.** None outstanding for comparison order. The first validation
ran on GNU Octave 8.4.0; it was subsequently rerun against the pinned 10.2.0
interpreter (see the rootfs entry below), which independently confirms the
column-major behaviour on the exact build the reference pool was validated with:

```text
c = [11 21 31; 12 22 32];  c(:)'  ->  [11 12 21 22 31 32]
```

---

## 2026-08-08 — Prime Sandbox is no longer required to score a candidate

**Change, not a defect.** Prime's managed Sandbox service has blocked this
project three times in ten days: `Payment required` mid-run on 2026-07-30,
`PROVISIONING` stalls on 2026-08-05, and four probes that never reached
`RUNNING` on 2026-08-07 — a plain-Python control reproducing the last one, which
ruled out the Octave image and this repository as causes. GPU compute was
healthy through all three.

`environments/octave_rl/executors.py` adds a `runtime = "local"` backend that
runs the same generated `run_candidate.m` on the calling host. `runtime`
defaults to `"prime"`; nothing changes unless it is set.

What is identical across backends: the input-only runner, so hidden expected
values and pass counters never enter the interpreter running model output; and
`harness.score_candidate_output`, so one scorer serves both. The property that
protects the reward signal does not depend on where the interpreter runs.

What is weaker locally: containment. A Sandbox is a separate container on
separate hardware. The local backend defends with a from-scratch child
environment (no `PRIME_API_KEY`, no Hugging Face or cloud credentials — built
rather than filtered, so a newly added variable cannot leak), a private 0700
working directory, `ulimit` bounds on CPU/address space/file size/process count,
a wall-clock kill of the whole process group, and a network namespace via
`unshare` when the host grants one. If a namespace cannot be obtained the
backend **raises** rather than quietly running with host network access;
`OCTAVE_RL_ALLOW_UNISOLATED_LOCAL=1` overrides that, and every record then
reports `network_isolated = false`. Note that the Prime CPU Sandbox also cannot
claim egress denial — its request model does not serialize a network policy — so
`runtime = "prime"` reports `network_isolated = false` as well.

Speed is the incidental result that matters for iteration: 1,500 tasks in 139 s
locally, against roughly five minutes of provisioning *per candidate* through a
cold Sandbox.

**Interpreter fidelity — resolved.** The first version of this backend used
whatever Octave the host had, which is a silent correctness risk: `apt install
octave` gives 8.4.0 on Ubuntu 24.04 while the reference pool was validated
against 10.2.0, and these tasks deliberately include tolerance and orientation
edge cases where interpreter versions can disagree.

`scripts/fetch_pinned_octave.py` removes the risk instead of documenting it. It
pulls the `gnuoctave/octave:10.2.0` layers straight from the registry over
HTTPS and extracts them into a directory — no Docker daemon, which matters
because Prime pods are themselves containers and cannot run one. Setting
`OCTAVE_RL_OCTAVE_ROOTFS` makes the backend `chroot` into that tree, so the
interpreter is the pinned build rather than a lookalike. Cost: one 1.6 GB pull,
about 4.3 GB unpacked.

This also strengthens containment rather than weakening it. The full stack is
`unshare --net --map-root-user` → `chroot <rootfs>` → `sh` applying the
`ulimit` envelope → `octave-cli`. The user namespace means it needs no
privileges, and candidate code sees neither the host filesystem nor a network.

**Verification.** Against the pinned 10.2.0 rootfs: 1,800/1,800 hidden cases at
60 tasks/level in 22.9 s, and the full 9,000-case pool clean. `runtime_description()`
records interpreter, source, and both isolation flags into every validation
report, so a number can always be traced to the interpreter that produced it.

**Residual risk.** `chroot` needs either root or a user namespace. Hosts with
unprivileged user namespaces disabled and no root will fall back to the host
interpreter — the report's `filesystem_isolated: false` and the `octave` version
string are the tell, so check them rather than assuming the rootfs took effect.

---

## 2026-08-08 — Prime API key rejected

**Symptom.** `prime pods list` and `prime sandbox list` both returned
`API key unauthorized` (CLI 0.6.21, key from `~/.prime/config.json`, no
`PRIME_API_KEY` shadowing it). Every Prime operation was blocked.

**Root cause.** Not established. Resolved by re-running `prime login`; the
account then showed zero pods and zero sandboxes.

**Blast radius.** Session-blocking only. No run was in flight.

**Why it survived.** Nothing to catch — but worth noting that a dead key and
the 2026-08-07 Sandbox stalls present very differently, and conflating them
would waste a debugging session. Those four probes *created* successfully,
returning IDs and two scheduler-audit UUIDs, then failed to reach `RUNNING`. A
revoked key fails at creation. Check which stage failed before assuming a
common cause.

**Residual risk.** Unknown whether the key expired, was rotated, or was
disabled by a billing state change. If it recurs, capture the exact failing
stage and the account's billing state before re-authenticating, since
re-authenticating destroys the evidence.
