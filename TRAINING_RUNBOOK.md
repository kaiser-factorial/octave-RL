# Native prime-rl 20-step run

This is the execution record for the training deliverable. It uses the native
`verifiers.v1` taskset/harness contract throughout and does not use the
alternative legacy v0 `Environment` form. Prime Hosted Training supports both
contracts; open-source prime-rl was chosen here for direct control over pinned
versions, the v1 user simulator, CUDA backend selection, diagnostics, logs,
and checkpoint retrieval.

## Resolved experiment intent

- prime-rl source: `PrimeIntellect-ai/prime-rl`, pinned for validation at
  commit `44539229436a23e624b0f39826014a4e58a703be`, whose verifiers submodule is
  compatible with released `verifiers==0.2.1`
- model: `Qwen/Qwen3.5-4B`
- topology: one trainer GPU plus one inference GPU
- training: LoRA rank 16, learning rate `1e-5`, 20 steps
- batches: 8 rollouts from one task per step (`group_size = 8`)
- taskset: Level 1, seed 314159, 500 generated tasks
- interaction: maximum three attempts; reward multipliers 1.00, 0.85, 0.60
- guide: `Qwen/Qwen3.5-35B-A3B` through Prime Inference before attempt three
- eval: 10 held-out examples at startup and every 5 steps, seed 271828
- execution: null harness with Prime Sandbox runtime and pinned Octave image
- monitoring: archived trainer/orchestrator/inference logs, resolved configs,
  and rollout records (W&B intentionally disabled because no project was
  requested)

The source config is
`configs/prime-rl/octave-qwen-4b-20step.toml`.

This is a historical execution record. The current environment uses a
correctness-only reward protocol and the curriculum controller defaults to
train-only, two-inflight operation; do not use this config as the continuation
launch configuration.

## Live compute envelope observed on 2026-07-29

The initially selected spot 2x A100 80GB pod was approximately `$1.253/hour`,
but it could not be reached before the account's SSH key was registered and
was terminated. Spot A100 capacity then disappeared. The valid experiment
uses an available non-spot 2x RTX 6000 Ada 48GB pod at `$1.50/hour`.
Alternative listings observed at launch included 2x L40S 48GB at `$1.64/hour`
and 2x A100 80GB at `$2.40/hour`.

The original authorized ceiling was two hours from pod creation; the user
extended it by one hour so the configured 20 steps, terminal evaluation, and
8.6 GB model transfer could finish. Stop and terminate the pod on:

- configuration or environment-server failure;
- repeated provider, sandbox, or tunnel errors above 5%;
- all-zero valid rewards for three consecutive steps;
- NaN/Inf loss, runaway KL, or out-of-memory failure;
- completion of step 20 and artifact retrieval;
- reaching the two-hour wall-clock ceiling.

## Observed completion

The valid run completed all 20 optimizer steps in 1h 53m of prime-rl runtime.
The terminal held-out Level 1 evaluation reached 0.905 reward with zero errors,
1.5 average turns, and 20% truncation. The final model and evidence transfer
completed within the extended authorization, after which pod
`4e7d4940bc924286a8375d22e88fc0a8` was terminated. `prime pods list` then
reported zero active pods.

Prime history records 2.6 hours for the valid pod and four minutes for the
failed spot pod. At the displayed rates, estimated total compute cost is about
$3.98 (`2.6 × $1.50` plus `4/60 × $1.253`). Provider/account rounding may
differ from this estimate.

The local artifact intentionally excludes the approximately 10 GB distributed
optimizer-state file. It retains the complete 8.6 GB deployable step-20 model,
LoRA adapter, all rollout traces, resolved configs, and logs. The omitted state
is useful only for resuming the exact optimizer trajectory and duplicates much
of the retained model data.

That complete archive is local-only. The public Git repository excludes model
weights, checkpoints, logs, and raw traces, and retains compact aggregate
evidence plus the source needed to reproduce the run.

The five-task evaluation cadence was appropriate for a short learning-signal
check but consumed substantial wall time. For a longer follow-on run, prefer a
startup baseline, evaluation every 10–20 steps, and a terminal evaluation.

## External dependencies

- two NVIDIA GPUs;
- Hugging Face access to `Qwen/Qwen3.5-4B`;
- authenticated Prime Compute/Sandbox access;
- `PRIME_API_KEY` in the pod process environment for the third-turn guide;
- optional `WANDB_API_KEY`, otherwise disable W&B and retain the file monitor.

Secrets must be injected in the pod shell or secret manager. They must never be
written to this repository, TOML config, trace files, or command output.

## Launch sequence

After explicit compute authorization:

1. Refresh `prime availability list` and create the chosen two-GPU pod.
2. Clone the pinned prime-rl commit and initialize its submodules.
3. Transfer this workspace, then install `environments/octave_rl` into the
   prime-rl environment.
4. Inject required credentials without printing them.
5. Run `uv run rl @ ... --dry-run` and archive the resolved configs.
6. Run the 20-step experiment, watch metrics, and enforce the stop criteria.
7. Copy logs, metrics, resolved configs, checkpoints, and adapter back here.
8. Terminate the pod and record its final billed duration.
9. Plot step reward and update the root README with observed results.

## RTX 6000 Ada compatibility settings

On this provider's SM89 host, vLLM's default CUDA-graph initialization hit an
illegal instruction and its bundled FlashAttention-2 path later hit a
misaligned-address error. The validated inference configuration therefore
uses eager execution and `TRITON_ATTN`:

```toml
[inference.model]
enforce_eager = true
max_model_len = 4096

[inference.vllm_extra]
attention_backend = "TRITON_ATTN"
```

The trainer's FlashAttention-2 implementation completed the first optimizer
step successfully; the fault was specific to vLLM's inference path.

## Staged follow-on path

The July 30 curriculum trial uses `scripts/curriculum_controller.py` and the
settings documented in `CURRICULUM.md`. Its stable operating envelope differs
from the original short run:

- 1,536 completion tokens, 4,096 sequence length, and 6,144 total
  conversation tokens;
- train-only chunks followed by statically merged held-out checkpoints;
- eight rollouts per update, GRPO groups of two, and at most two simultaneous
  generations;
- 80% Level 1 / 20% Level 2 after the verified step-15 transition;
- fatal-log stops for vLLM EngineCore/CUDA failures and Prime Sandbox payment
  rejection.

Use short durable chunks when operating at reduced concurrency. The live trial
completed optimizer steps 16 and 17 but reached its dollar deadline while
assembling the next batch, before the five-step checkpoint interval. Its
`STABLE` step-2 adapter is preserved, while the conservative controller state
correctly remains at step 15. The adapter depends on the staged step-15 merged
base, which was not copied off the ephemeral pod, so it is not deployable from
the local artifact set and must not be merged into the original step-20 base.
Unless that exact base can be recovered externally, restart from the retained
step-20 model with a fresh controller state and collect new, policy-matched
gate evaluations. For the next run, retrieve every merged gate checkpoint
before terminating its pod.

The July 30 pod cost is estimated at $13.77 against the authorized $20 ceiling.
It was terminated after artifacts were hash-checked; the final Prime CLI audit
reported zero active pods.
