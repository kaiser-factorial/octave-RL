# Octave RL handoff

Last updated: 2026-08-05

This is the shortest trustworthy orientation for continuing the Octave RL
work. Read `README.md` for the repository map, `REPORT.md` for the full
experiment narrative, and `CURRICULUM.md` for controller behavior and command
details.

## Current status

The requested environment, calibration, initial RL run, staged curriculum
controller, live level transition, timing visualization, and supporting
documentation are complete.

The most important live result is a verified transition at global step 15:

| Checkpoint | Held-out pool | Raw Level 1 | One-sided 95% Wilson lower bound | Errors |
| ---: | ---: | ---: | ---: | ---: |
| 10 | 24 | 0.75 | 0.5845 | 0 |
| 15 | 24 | 0.75 | 0.5845 | 0 |

These were distinct statically merged policies. Two consecutive passes cleared
the 0.55 gate, so the controller changed the training distribution from
100% Level 1 to 80% Level 1 / 20% Level 2 at exactly step 15.

Two post-transition optimizer updates then completed at global steps 16 and
17. Prime Sandbox began rejecting new provisioning with `Payment required`
during the following batch. That partial batch had errors in 324 of 326 trace
rows and is excluded from all plots, state, and capability claims. No
post-transition held-out evaluation was possible because it uses the same
Sandbox service.

The Prime GPU pod was terminated after artifact retrieval. The final CLI audit
reported zero active pods. Estimated compute was approximately $13.77 against
the authorized $20 ceiling.

## What was accomplished

### Environment

- Implemented a native typed `verifiers.v1` taskset and null harness for GNU
  Octave code generation.
- Added ten deterministic task families across three difficulty levels.
- Generated 500 tasks per level with six hidden cases per task.
- Preserved case-level partial credit, correctness-only reward, retry
  diagnostics, and raw correctness as a separate metric.
- Validated all 1,500 reference tasks: 9,000/9,000 hidden cases passed in the
  pinned GNU Octave 10.2.0 runtime.
- Audited constant-output shortcuts and important orientation/numerical edge
  cases.

### Model and interaction calibration

- Tested Qwen3.5 0.8B, 2B, and 4B.
- Selected `Qwen/Qwen3.5-4B`, the first tested size with a useful 10–35%
  one-turn baseline.
- Added a three-attempt debugging interaction:
  - attempt 1 multiplier: 1.00;
  - unguided attempt 2 multiplier: 0.85;
  - guided attempt 3 multiplier: 0.60.
- Added a concise third-turn guide using `Qwen/Qwen3.5-35B-A3B`; on the small
  matched calibration set, raw solved fraction increased from 20% to 40%.
- Increased the response cap from 1,024 to 1,536 tokens. A live 2,048-token
  comparison was slower and did not justify its cost.

### Initial training

- Completed the original 20-step Level 1 GRPO run on two RTX 6000 Ada GPUs.
- Completed 160/160 training rollouts with zero rollout infrastructure errors.
- Raised the small held-out Level 1 result from 0.010 at startup to 0.905 at
  step 20.
- Retained the deployable local model at:
  `artifacts/training/octave-qwen-4b-20step/weights/step_20/`.

### Staged curriculum and live trial

- Implemented five curriculum stages:
  - `level1_only`: 100/0/0;
  - `introduce_level2`: 80/20/0;
  - `level2_working_set`: 30/70/0;
  - `introduce_level3`: 20/60/20;
  - `advanced`: 10/40/50.
- Added confidence-bounded promotion, regression demotion, stage-local
  consecutive-pass accounting, durable state, checkpoint rebasing, and exact
  transition-step provenance.
- Added atomic multi-level trace ingestion so Level 1 and Level 2 results from
  one checkpoint count as one gate observation.
- Added train-only chunks followed by checkpoint-static evaluation.
- Added independent batch, GRPO-group, and inference-concurrency controls.
- Added fail-fast monitoring for fatal vLLM/CUDA errors and Prime Sandbox
  payment rejection.
- Added a guarded LoRA checkpoint merger and a rollout-dynamics plotter.
- Produced the final timing plot and CSV covering raw correctness, end-to-end
  latency, attempts, and truncation through global step 17.

## Artifact map

Rows for raw traces, models, and post-transition checkpoints refer to the
complete local workspace. Those large or hidden-case-bearing artifacts are
excluded from the public Git repository; the summary JSON/CSV/PNG files in
this table remain versioned.

| Purpose | Path |
| --- | --- |
| Repository orientation | `README.md` |
| Full experiment report | `REPORT.md` |
| Curriculum gates and commands | `CURRICULUM.md` |
| Native run procedure | `TRAINING_RUNBOOK.md` |
| Curriculum controller | `scripts/curriculum_controller.py` |
| LoRA merger | `scripts/merge_lora_checkpoint.py` |
| Timing visualization source | `scripts/plot_rollout_dynamics.py` |
| July 30 machine-readable summary | `artifacts/curriculum/live-2026-07-30/experiment-summary.json` |
| Final timing chart | `artifacts/curriculum/live-2026-07-30/rollout-dynamics.png` |
| Final timing table | `artifacts/curriculum/live-2026-07-30/rollout-dynamics.csv` |
| Conservative live controller state | `artifacts/curriculum/live-2026-07-30/controller/real-training-state.json` |
| Distinct static held-out traces | `artifacts/curriculum/live-2026-07-30/octave-staged-qwen/standalone-eval-staged-step10-c1-n24-v1/` and `standalone-eval-staged-step15-c1-n24-v1/` |
| Original deployable model | `artifacts/training/octave-qwen-4b-20step/weights/step_20/` |
| Post-transition evidence | `artifacts/curriculum/live-2026-07-30/octave-real-segment-15-25-c2/` |

## Important continuity caveat

The retrieved step-2 post-transition LoRA broadcast has a matching `STABLE`
marker and SHA-256:

```text
a797cd911c69a54590f20994d70c5b41b2247477b45274907e79d61ae687b78e
```

However, that adapter was trained relative to
`octave-staged-qwen/weights/step_15_merged`. That exact staged base was not
copied off the ephemeral pod before termination. Therefore:

- the adapter is valid evidence that optimizer steps 16 and 17 completed;
- it is **not deployable from the current local artifact set**;
- it must **not** be merged directly into the original local step-20 model;
- the historical controller transition is valid, but its state is not a
  resumable model checkpoint.

Unless the exact staged step-15 base can be recovered externally, the safest
continuation is a fresh, policy-matched run from the locally retained step-20
model.

## Common issues and lessons

### `verifiers` version boundaries

This project uses native `verifiers.v1` throughout. Do not port its taskset or
harness back to the legacy v0 `Environment` API, and do not interpret an API
signature failure as a model failure. Pin the known-compatible prime-rl and
verifiers revisions before changing environment code.

### Infrastructure zeros are not model zeros

Inspect `errors`, timing, and token usage before interpreting a zero reward.
Instant or repetitive zeros with missing usage usually mean the rollout never
reached the grader. The rejected partial step 18 is the canonical example and
must remain excluded.

### Qwen3.5/vLLM concurrency on RTX 6000 Ada

The Qwen3.5 GDN/hybrid-attention worker reproduced CUDA internal/illegal-memory
failures at seven to eight simultaneous long generations. The stable envelope
was:

```text
enforce_eager = true
attention_backend = "TRITON_ATTN"
batch_size = 8
group_size = 2
max_inflight_rollouts = 2
```

Do a short concurrency smoke test before committing to a long run if vLLM,
CUDA, PyTorch, the GPU type, or the model checkpoint changes.

### Integrated evaluation is unsafe on this stack

An eval environment's `multiplex = 1` does not limit prime-rl's global shared
dispatcher. Integrated evaluation still launched eight requests concurrently.
The controller now defaults to one train-only chunk per invocation. Stop at a
checkpoint, merge the adapter, and evaluate a static policy at concurrency
one; pass `--integrated-eval` only after a separate concurrency validation.

### Reward protocol is correctness-only and host-scored

The pre-2026-08-05 environment added a flat `0.1` reward whenever stdout
contained a `RESULT` marker, which let a fully correct first attempt reach
`1.1` and left its parser susceptible to marker spoofing. The current
environment rewards only case-level correctness, discounted for later
attempts. Its candidate sandbox receives no expected outputs or pass counters:
it reports values, while the trusted Python task process compares those values
with hidden expected outputs. Raw correctness remains the only
cross-run/curriculum comparison metric; historical shaped rewards are not
directly comparable to new ones.

### CPU Sandbox egress is not currently configurable

The installed `prime_sandboxes` CPU request model does not serialize a
`network_access` setting; its allow/deny-list fields are accepted only for
`vm=True`. Do not claim egress isolation for the current CPU Octave runner.
The scoring boundary remains sound because candidate Sandboxes receive no
credentials, expected outputs, reference source, or pass counters. If a future
deployment requires outbound-network denial as a separate policy, migrate this
runner to a supported VM configuration and verify that policy on the live API.

### Keep gates policy-consistent

Long integrated evals can straddle a weight broadcast. The controller can
select the largest single-policy cohort, but static evaluation is clearer.
Never combine two policies or count Level 1 and Level 2 files from one policy
as separate checkpoint observations.

### Training reward is not a promotion gate

Training batches are stochastic, filtered by GRPO, and can contain very small
per-level samples. For example, the first post-transition effective batch had
two Level 1 and four Level 2 traces. Treat those points as optimization
diagnostics, not held-out capability estimates.

### Token length has a steep latency cost

Three attempts at 1,536 tokens can take several minutes. The 2,048-token test
was approximately 52 minutes for one three-level held-out pass and did not
improve the score enough. Prefer better stopping and repair behavior before
raising the cap again.

### Sandbox billing is a separate run dependency

GPU compute can remain healthy while Prime Sandbox refuses new containers.
Check Sandbox access before provisioning an expensive GPU pod. The controller
now watches for payment rejection, but a preflight smoke test is cheaper.

### Preserve every continuation base

A LoRA adapter is only meaningful relative to its exact base. Before
terminating a pod, retrieve and hash:

1. the merged policy used for held-out evaluation;
2. its adapter and adapter config;
3. the durable controller state;
4. the resolved configs, logs, and effective traces.

Do not treat a downloaded adapter alone as a recoverable model.

### Full trainer checkpoints are large

The original optimizer state was roughly 19 GB and duplicated much of the
model. LoRA-only recovery is smaller, but exact optimizer continuation requires
the full checkpoint. Decide which form is required before starting the pod and
budget transfer time accordingly.

### Use unique output directories

prime-rl can clean an output directory at the start of a new segment. Never
point a retry at evidence that has not already been copied elsewhere.

### Repository state

At handoff, `git status` reports the repository contents as untracked. No
commit or push was requested or created. Establish a baseline commit before
the next experiment so source, configs, and evidence can be tied to a revision.

### Publication remains a user choice

The environment is ready for the Environments Hub, but it has not been pushed.
Public versus private visibility must be chosen explicitly before
`prime env push`.

## Recommended next steps

### 1. Preserve and version the current result

Create a repository baseline after reviewing secrets and large-artifact
policy. At minimum, commit source, configs, tests, summaries, plots, and docs.
Decide separately whether the 8.7 GB original model belongs in Git LFS,
external object storage, or a manifest-only artifact registry.

### 2. Restore and smoke-test Prime Sandbox access

Do this before provisioning GPUs:

```bash
prime config view
prime sandbox create python:3.11-slim --timeout-minutes 10
```

Run a trivial command in the sandbox, then terminate it. Also confirm:

```bash
prime pods list --output json --plain
```

The expected starting state is zero active pods.

### 3. Start a fresh resumable trajectory from the retained step-20 model

Do not reuse the historical step-15 controller state with a different policy.
Initialize a new state tied to:

```text
artifacts/training/octave-qwen-4b-20step/weights/step_20/
```

Use train-only chunks, the 1,536-token envelope, and the stable concurrency
settings. Because a throttled optimizer step took approximately 15 minutes,
prefer two-step chunks initially so a budget interruption cannot erase five
steps of resumable progress.

### 4. Re-establish the Level 1 gate on the new trajectory

After each sparse checkpoint:

1. merge the adapter into its exact retained base;
2. copy the merged checkpoint locally;
3. serve it statically at concurrency one;
4. evaluate at least 24 disjoint Level 1 tasks;
5. ingest the trace with the controller.

Require two consecutive, policy-distinct passes before shifting to Level 2.
The old transition remains scientific evidence but should not control a new
policy's sampling mix.

### 5. Evaluate Level 1 and Level 2 atomically after promotion

At `introduce_level2`, run both held-out pools against the same static
checkpoint and ingest them together with `ingest-traces`. Keep Level 3 out
until Level 2 has two confidence-bounded passes above its current threshold.

### 6. Improve durability before attempting 50 steps

- Save or export a resumable checkpoint every two optimizer steps at the
  reduced-concurrency setting.
- Add an artifact manifest containing model/base relationships and SHA-256
  hashes.
- Make pod cleanup conditional on successful local checksum verification.
- Retain a transfer-time reserve inside the dollar budget.
- Keep the controller's fatal-log and deadline guards enabled.

### 7. Improve sample efficiency and latency

Promising experiments, in priority order:

1. add deterministic per-batch level quotas so a nominal 80/20 mix cannot
   produce long stretches with no effective Level 2 samples;
2. stop generation once a complete fenced function is present;
3. allocate fewer tokens to later repair attempts rather than increasing the
   global cap;
4. make retry feedback identify only the first actionable Octave error;
5. compare guided attempt 3 against an unguided third attempt on a larger
   matched set;
6. explore safe Sandbox reuse or pooling while preserving task isolation;
7. test a newer or alternative Qwen3.5 serving stack with a short concurrency
   ladder before changing the validated runtime.

### 8. Strengthen evaluation

Use 32–40 tasks per required level once runtime and budget permit. Report raw
case fraction, eventual solve rate, first-attempt solve rate, mean attempts,
truncation, error rate, and the Wilson lower bound. Keep every evaluation
disjoint from training and record its seed, policy hash, and source trace.

## Verification at handoff

- Repository tests: 37 passed.
- Focused Ruff checks: passed.
- Python compilation: passed.
- Reference validation: 9,000/9,000 hidden cases passed.
- Retrieved adapter checksum: verified.
- Active Prime pods: zero.

If results in this handoff conflict with an older narrative, prefer
`artifacts/curriculum/live-2026-07-30/experiment-summary.json` for measured
values and the caveat above for model recoverability.
