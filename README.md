# Octave RL

Octave RL is a reproducible reinforcement-learning environment for teaching
language models to write GNU Octave functions. It uses the native
`verifiers.v1` taskset and harness APIs, generates seeded problems with hidden
NumPy-derived test cases, executes candidate `.m` files in isolated Prime
Sandboxes, and awards deterministic partial credit.

The repository includes the environment, evaluation and prime-rl
configurations, a staged curriculum controller, tests, analysis scripts, and
compact evidence from completed Qwen3.5-4B experiments. Raw traces,
checkpoints, and model weights are intentionally not committed.

## What is included

- Ten generator families spanning reductions, logical indexing,
  reshape/permutation, broadcasting, sliding windows, linear solves,
  recurrences, structs/cells, string parsing, and signal identities.
- Three difficulty levels with deterministic generation from
  `(level, seed, task index)`.
- Six hidden cases per task and fractional case-level correctness.
- Up to three attempts: an unassisted first answer, diagnostic retry, and an
  optional guide-model hint before the final retry.
- Declining correctness multipliers of `1.00`, `0.85`, and `0.60`, rewarding
  correct answers that require less assistance.
- A confidence-gated curriculum that introduces harder levels only after
  repeated held-out success and can demote after regression.

The environment uses `verifiers.v1` throughout. It does not use the legacy v0
`Environment` contract.

## Results at a glance

Small calibration runs selected `Qwen/Qwen3.5-4B`: it was the first tested
Qwen size with nonzero reward and landed in the desired 10–35% one-turn
baseline range on Levels 1–2. A broader 200-rollout evaluation produced:

| Level | Rollouts | Mean raw correctness | Fully solved |
| --- | ---: | ---: | ---: |
| 1 | 100 | 28.17% | 26 |
| 2 | 100 | 7.00% | 7 |
| Combined | 200 | 17.58% | 33 |

A native 20-step LoRA run on two RTX 6000 Ada GPUs completed all 160 training
rollouts without rollout errors. Held-out Level 1 reward rose from `0.010` at
startup to `0.905` at step 20. These are proof-of-learning results from small
held-out samples, not confidence-bounded benchmark claims.

![Training reward and loss](artifacts/training_curve.png)

The staged follow-on confirmed its first promotion gate twice at `0.75` raw
correctness over separate 24-task Level 1 checkpoints, then automatically
shifted from 100% Level 1 to an 80% Level 1 / 20% Level 2 mix.

![Rollout accuracy, latency, retries, and truncation](artifacts/curriculum/live-2026-07-30/rollout-dynamics.png)

See [REPORT.md](REPORT.md) for the full experiment analysis and
[OCTAVE_HANDOFF.md](OCTAVE_HANDOFF.md) for current limitations and next steps.

## Requirements

For local development and generation tests:

- Python 3.11–3.13
- [`uv`](https://docs.astral.sh/uv/)

For live evaluation:

- an authenticated [Prime Intellect](https://www.primeintellect.ai/) account
  with Prime Sandbox access;
- the `prime` CLI (`prime login`) or a runtime-injected `PRIME_API_KEY`;
- access to the configured inference models.

For native training, budget for two NVIDIA GPUs: one trainer GPU and one
inference GPU. The validated run used 2× RTX 6000 Ada 48 GB. Hardware pricing
and availability vary, so inspect current capacity before provisioning.

## Install and test

```bash
git clone git@github.com:kaiser-factorial/octave-RL.git
cd octave-RL
uv sync --dev
uv run pytest -q
```

The environment is also a standalone installable package:

```bash
uv pip install -e environments/octave_rl
```

Its public factory is:

```python
from octave_rl import load_environment

taskset = load_environment(
    level=1,
    num_tasks=500,
    max_turns=3,
    require_vectorized=False,
    seed=314159,
)
```

`max_turns` controls the environment's attempt budget; the evaluation or
training harness should use the same or a larger turn cap.

## Evaluate the base model

Log in to Prime, then run a supplied evaluation config:

```bash
prime login
uv run eval @ configs/eval/octave-qwen-4b-two-turn.toml
uv run eval @ configs/eval/octave-qwen-4b-guided-three-turn.toml
```

The third-turn guide defaults to `Qwen/Qwen3.5-35B-A3B` through Prime
Inference. It receives only the public prompt, candidate source, and first
diagnostic—not hidden inputs, expected values, or reference code. Disable the
guide in task configuration if you want a fully self-contained two-turn run.

To validate generated reference programs against the pinned Octave runtime:

```bash
uv run python scripts/validate_reference_pool.py
```

The full default pool previously passed 9,000/9,000 hidden cases: 500 tasks at
each of three levels, with six cases per task. Live validation creates Prime
Sandboxes and therefore incurs platform usage.

## Train with prime-rl

The completed experiment used open-source `prime-rl` commit
`44539229436a23e624b0f39826014a4e58a703be`, compatible with
`verifiers==0.2.1`. Pin this revision: newer `prime-rl` versions use a different
interleaving-agent API and are not source-compatible with this environment's
v1 `User` simulator.

On a two-GPU Prime pod (or equivalent CUDA host):

```bash
git clone https://github.com/PrimeIntellect-ai/prime-rl.git
cd prime-rl
git checkout 44539229436a23e624b0f39826014a4e58a703be
git submodule update --init --recursive
uv sync
uv pip install -e /path/to/octave-RL/environments/octave_rl

uv run rl @ /path/to/octave-RL/configs/prime-rl/octave-qwen-4b-20step.toml --dry-run
uv run rl @ /path/to/octave-RL/configs/prime-rl/octave-qwen-4b-20step.toml
```

Before the real launch, replace `/path/to/octave-RL` and confirm the resolved
model and output paths. The worker-subprocess User MCP runtime intentionally
strips `*_API_KEY` variables, so a remote Sandbox-backed run needs an approved
per-run secret mount/config path that the user subprocess can read. The
2026-08-05 pod smoke used a mode-0600 `.prime/config.json` inside an isolated
mode-0700 temporary `HOME`, then removed that directory on exit. Never put a
credential value in TOML files, shell history, traces, or the repository.

The checked-in 20-step TOML is historical evidence, not the continuation
command. For a fresh run, render the train-only controller configuration from
the retained step-20 base; it uses a worker-subprocess null harness and creates
only the explicit candidate Octave Sandboxes. The historical configuration
uses:

- `Qwen/Qwen3.5-4B` with LoRA rank 16 and learning rate `1e-5`;
- one trainer GPU plus one inference GPU;
- eight rollouts per optimizer step;
- three attempts with a stronger guide before attempt three;
- held-out evaluation at startup and every five steps;
- eager vLLM execution and `TRITON_ATTN`, required by the tested RTX 6000 Ada
  stack.

The 20-step config keeps the historically validated 1,024-token completion
cap. Longer curriculum runs should start from the tested 1,536-token envelope
described below; 2,048 tokens increased latency substantially without clearing
the next gate in the comparison run.

See [TRAINING_RUNBOOK.md](TRAINING_RUNBOOK.md) for compute assumptions,
failure stops, artifact retrieval, and the observed CUDA compatibility issues.

## Run the staged curriculum

Initialize durable controller state, render a config, and inspect it before
launching. The default starts directly with Level 1 only; use
`--start-stage introduce_level2` (or another named stage) to select a mix
deliberately. Alternatively, use `curriculum_controller.py assess` with
sequential static traces from all three levels to create an evidence-based
starting state without counting that baseline as a promotion.

```bash
uv run python scripts/curriculum_controller.py init \
  --state /path/to/run/state.json

uv run python scripts/curriculum_controller.py render \
  --state /path/to/run/state.json \
  --model-path /path/to/base-or-merged-checkpoint \
  --output-dir /path/to/run/output \
  --target-step 5 \
  --eval-interval 5 \
  --eval-examples 20 \
  --batch-size 8 \
  --group-size 2 \
  --max-inflight-rollouts 2 \
  --config /path/to/run/octave-staged.generated.toml
```

The stable Qwen path uses the controller's default one-chunk train-only run,
statically merges the LoRA adapter, evaluates the merged checkpoint on disjoint
held-out tasks, and then ingests those traces into the controller. This avoids
the shared inference concurrency failure observed during integrated evaluation.
Promotion depends only on held-out results, never training-batch reward.

For `Qwen/Qwen3.5-4B`, the generated renderer configuration explicitly sets
`enable_thinking = false`. The model otherwise opens a thinking block by
default, which consumed the fixed completion budget during a pod smoke before
it reached the required fenced Octave function. This is a renderer-only
generation setting; it does not alter tasks, hidden scoring, or rewards.

The reward protocol was hardened after the historical runs: it now rewards only
case-level correctness, so a correct first attempt is exactly `1.0`. Candidate
execution reports values only; the trusted task process retains expected values
and computes the score outside the candidate sandbox. Use
`raw_case_fraction` when comparing historical traces with future runs.

Read [CURRICULUM.md](CURRICULUM.md) before running this path. It documents the
promotion/demotion thresholds, Wilson-bound gate, sparse evaluation strategy,
checkpoint rebasing, budget deadline, stable concurrency settings, and exact
trace-ingestion commands.

## Repository map

| Path | Purpose |
| --- | --- |
| `environments/octave_rl/` | Native v1 taskset, generators, Octave harness, and package metadata |
| `configs/eval/` | Base-model and held-out evaluation configurations |
| `configs/prime-rl/` | Validated native training and inference configurations |
| `scripts/curriculum_controller.py` | Checkpointed difficulty scheduler and held-out gates |
| `scripts/validate_reference_pool.py` | Exhaustive reference validation in Prime Sandboxes |
| `scripts/plot_*.py` | Reproducible calibration, reward, training, and timing figures |
| `tests/` | Generation, parser, curriculum, checkpoint, and plotting regressions |
| `artifacts/` | Compact aggregate metrics and figures only |
| `REPORT.md` | Experiment methods, results, caveats, and failure analysis |

## Artifact policy

Model weights, LoRA broadcasts, trainer checkpoints, raw rollout traces, logs,
and generated configs are ignored. They are large, machine-specific, and raw
traces can expose hidden task cases. The public repository keeps only source,
reusable configs, aggregate metrics, controller state, and plots.

Consequently, the trained model described in the report is not downloadable
from this Git repository. Reproduce it with the pinned training recipe or
publish weights separately in an appropriate model artifact store. The local
post-transition adapter is also not independently deployable because its exact
staged base checkpoint was not retained; see the handoff before attempting to
continue it.

## Reproducibility notes

- Octave image: `ghcr.io/gnu-octave/octave:10.2.0`
- Observed interpreter: GNU Octave 10.2.0
- Environment dependencies: `verifiers>=0.2.1,<0.3`, `numpy>=2.0,<3`
- Training revision: `PrimeIntellect-ai/prime-rl@44539229436a23e624b0f39826014a4e58a703be`
- Training seed: `314159`
- Held-out seed: `271828`

Evaluation samples here are deliberately modest. When extending the work,
prefer larger disjoint held-out sets, sparser evaluation during training, and
confidence-gated level shifts over repeatedly testing a tiny fixed set.
