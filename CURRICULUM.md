# Staged curriculum controller

`scripts/curriculum_controller.py` runs prime-rl in checkpointed chunks and
changes task-level sampling ratios only between chunks. Each new chunk resumes
the preceding distributed trainer checkpoint, so optimizer state is preserved
while the environment mix changes.

## Stages

| Stage | Level 1 | Level 2 | Level 3 |
| --- | ---: | ---: | ---: |
| `level1_only` | 100% | 0% | 0% |
| `introduce_level2` | 80% | 20% | 0% |
| `level2_working_set` | 30% | 70% | 0% |
| `introduce_level3` | 20% | 60% | 20% |
| `advanced` | 10% | 40% | 50% |

Only disjoint held-out traces can trigger a transition. Training-batch reward
is recorded by prime-rl but is not a promotion input.

For this Qwen3.5/vLLM stack, held-out generation is run against a statically
merged checkpoint after the training process exits. The environment-level
`multiplex = 1` setting does not cap prime-rl's shared dispatcher: its global
`max_inflight_rollouts` remains tied to the GRPO group size of eight. A live
startup evaluation therefore launched eight long generations concurrently,
the same pattern that had reproduced a hybrid-attention worker fault. Static
serving lets each held-out worker use concurrency one without reducing the
training group size or allowing weight broadcasts during an evaluation.

The generated config evaluates only levels required by the current gate:
Level 1 alone initially, Levels 1–2 while Level 2 can trigger a transition,
and all three levels once Level 3 signal is required. Level 1 remains in every
later gate as the global regression guard. This avoids paying for harder
held-out pools before their scores can affect a decision.

Promotion requires the configured number of consecutive evaluations, a
minimum sample count, no more than 2% trace errors, and a one-sided 95% Wilson
lower bound above the stage threshold:

- introduce Level 2 after Level 1 exceeds 0.55;
- make Level 2 the working set after it exceeds 0.20;
- introduce Level 3 after Level 2 exceeds 0.45;
- enter the advanced mix only when Level 2 exceeds 0.60 and Level 3 exceeds
  0.10.

Level 1 falling below 0.55 demotes any later stage. Level 2 falling below 0.10
demotes stages in which it is the working set. Evaluations collected before a
transition cannot be reused to satisfy the next stage's consecutive-evaluation
requirement.

## Token envelope

The original run used 1,024 completion tokens. In its step-1, step-5, and
step-10 held-out evaluations, 29 of 30 model calls ended at that cap. An
additional live comparison at 2,048 tokens produced 100% Level 1 truncation,
a 0.25 raw case fraction over the 20-trace same-policy cohort, and a roughly
52-minute three-level held-out pass. Since the larger cap did not improve the
gate score enough to justify its latency, the staged configuration uses:

- 1,536 maximum tokens per response;
- 4,096 training sequence length;
- 6,144 maximum conversation and inference context.

This provides repair headroom while bounding a three-attempt interaction.

## Commands

### Choose how to start

The default manual path starts directly in the Level 1-only mix and does not
run an evaluation:

```bash
uv run python scripts/curriculum_controller.py init \
  --state artifacts/curriculum/state.json
```

To intentionally begin at another curriculum mix, select its stage explicitly.
The state records this as a manual initialization, not as a model promotion:

```bash
uv run python scripts/curriculum_controller.py init \
  --state artifacts/curriculum/medium-start-state.json \
  --start-stage introduce_level2
```

For an evidence-based placement, first evaluate one *static* policy
sequentially on disjoint Level 1, Level 2, and Level 3 pools. Then let
`assess` choose a conservative starting mix from the three trace files:

```bash
uv run python scripts/curriculum_controller.py assess \
  --trace 1:outputs/bootstrap-level1/traces.jsonl \
  --trace 2:outputs/bootstrap-level2/traces.jsonl \
  --trace 3:outputs/bootstrap-level3/traces.jsonl \
  --min-examples 24 \
  --state artifacts/curriculum/assessed-start-state.json
```

`assess` uses the curriculum thresholds and one-sided Wilson bound to recommend
a stage, and records the traces' aggregate metrics as initialization
provenance. It never adds those results to the promotion history: promotion
still requires two policy-distinct post-start checkpoints. The controller does
not launch this evaluation itself, because the Qwen path must be served
statically at concurrency one rather than through prime-rl's shared dispatcher.

### Render or run a selected state

Render one stage without launching:

```bash
uv run python scripts/curriculum_controller.py render \
  --state artifacts/curriculum/state.json \
  --model-path artifacts/training/octave-qwen-4b-20step/weights/step_20 \
  --output-dir artifacts/curriculum/live-run \
  --target-step 5 \
  --eval-interval 5 \
  --eval-examples 20 \
  --config configs/prime-rl/octave-staged.generated.toml
```

If evaluation must run from a statically merged checkpoint, ingest its trace
without fabricating prime-rl rollout directories:

```bash
uv run python scripts/curriculum_controller.py ingest-trace \
  --state artifacts/curriculum/verified-state.json \
  --trace-file outputs/heldout-step15/traces.jsonl \
  --step 15 \
  --level 1 \
  --consecutive 2 \
  --min-examples 20
```

For stages whose gate requires more than one level, record every level as one
atomic checkpoint observation:

```bash
uv run python scripts/curriculum_controller.py ingest-traces \
  --state artifacts/curriculum/real-training-state.json \
  --trace 1:outputs/heldout-step25-level1/traces.jsonl \
  --trace 2:outputs/heldout-step25-level2/traces.jsonl \
  --step 25 \
  --consecutive 2 \
  --min-examples 20
```

The controller records the absolute trace provenance and rejects duplicate
steps. Repeat `--trace-file` to combine precommitted, disjoint held-out shards
for one checkpoint; the controller aggregates them into one confidence gate
and records every source. When a runtime cannot hot-load the prior adapter
safely, merge that adapter into the base model, then rebase checkpoint
numbering while retaining global curriculum history:

```bash
uv run python scripts/curriculum_controller.py rebase \
  --source-state artifacts/curriculum/verified-state.json \
  --state artifacts/curriculum/real-training-state.json
```

The rebased state starts the new checkpoint segment at local step zero while
continuing global promotion steps from the source state.

If a budget interruption leaves a `STABLE` LoRA broadcast but no full trainer
checkpoint, first merge that broadcast into the segment's base model. Then
adopt exactly those verified optimizer steps while rebasing:

```bash
uv run python scripts/curriculum_controller.py rebase \
  --source-state artifacts/curriculum/real-training-state.json \
  --state artifacts/curriculum/continuation-step17.json \
  --advance-steps 2
```

This deliberately resets optimizer state; it is a recovery route for a
statically merged policy, not a claim that an interrupted full checkpoint
exists.

Run checkpointed chunks:

```bash
uv run python scripts/curriculum_controller.py run \
  --state artifacts/curriculum/state.json \
  --prime-rl-dir /path/to/prime-rl \
  --model-path /path/to/step_20 \
  --output-dir /path/to/staged-output \
  --config /path/to/octave-staged.generated.toml \
  --max-steps 50 \
  --chunk-steps 5 \
  --eval-examples 20 \
  --consecutive 2 \
  --min-examples 20 \
  --price-per-hour 1.50 \
  --budget-usd 20
```

On the tested Qwen stack, add `--disable-integrated-eval`, stop after one
chunk, merge its adapter, and run checkpoint-static held-out evaluation before
rebasing into the next output directory. This is the stable real-training
path; the integrated mode remains useful on runtimes that safely support its
shared inference concurrency.

If the Qwen GDN worker is unstable at eight simultaneous requests, preserve an
eight-rollout optimizer batch while using two-rollout GRPO groups and admitting
only two requests at once:

```bash
  --batch-size 8 \
  --group-size 2 \
  --max-inflight-rollouts 2
```

The controller validates that batch size is a multiple of group size and that
the inflight cap can contain at least one complete GRPO group. It also watches
the inference and orchestrator logs and aborts a chunk on a fatal
EngineCore/CUDA marker or a Prime Sandbox payment rejection instead of waiting
until the dollar deadline.

The dollar guard converts the supplied hourly rate into a monotonic wall-clock
deadline. Pod setup and artifact-transfer cost must be subtracted before
passing `--budget-usd`; the controller cannot observe billing that occurred
before it started.

## Code review

The preflight review covered state durability, held-out leakage, promotion and
demotion ordering, checkpoint continuation, generated TOML validity, token
limits, and cost enforcement. It found and fixed:

1. evaluation history from an earlier stage could satisfy a later gate;
2. the advanced stage did not initially require Level 3 signal;
3. Level 2 collapse lacked a demotion rule;
4. generated TOML paths were not escaped;
5. trace summarization unnecessarily wrote temporary files beside run data;
6. Python 3.13 dynamic-module test registration and lint/format issues.
7. long evaluations could mix policy versions, so promotion now uses the
   largest single-policy cohort and rejects undersized cohorts;
8. early stages evaluated Levels 2–3 even though they could not affect the
   current gate, so evaluation scope now expands with the curriculum.
9. a graceful budget-deadline interrupt could be mistaken for a completed
   chunk when the child exited zero, so interrupted chunks no longer advance
   durable progress;
10. concurrent held-out generations reproduced a Qwen3.5/vLLM worker fault,
    so generated eval environments now use one sequential worker;
11. statically served adapter checkpoints lacked a controller ingestion and
    continuation path, so trace ingestion records provenance and checkpoint
    rebasing preserves global stage history;
12. a process interruption between checkpoint persistence and gate ingestion
    could leave a completed eval unobserved, so every controller invocation
    now recovers pending trace sets before launching another chunk;
13. transitions discovered after a multi-step chunk were stamped with the
    chunk tip rather than the deciding held-out checkpoint, so transition
    provenance now records the exact evaluation step without rewinding trainer
    progress.
14. environment-level eval multiplexing did not limit prime-rl's shared
    dispatcher, so the stable Qwen path now omits integrated eval and evaluates
    a static merged checkpoint after each sparse training chunk;
15. separate Level 1 and Level 2 trace imports at one checkpoint would have
    collided on duplicate-step protection, so `ingest-traces` records all
    required levels as one atomic gate observation.
16. a dead inference worker left the outer launcher alive and could consume
    the remaining budget through connection retries, and a Sandbox payment
    rejection caused the same retry loop, so the controller now treats both
    conditions as a failed chunk;
17. the global inference cap was not configurable independently from batch
    size, so the controller now supports smaller GRPO groups and inflight
    concurrency while retaining eight rollouts per optimizer update.
18. an interrupted slow-concurrency chunk could leave a complete `STABLE`
    adapter broadcast without a full trainer checkpoint, so rebasing can now
    explicitly adopt a verified number of merged broadcast steps while
    documenting the optimizer-state reset.

The controller, checkpoint merger, timing visualization, and their tests pass
Ruff, Python compilation, TOML parsing, and the repository test suite. Live
prime-rl transition evidence is recorded in
`artifacts/curriculum/live-2026-07-30/experiment-summary.json`: distinct
step-10 and step-15 checkpoints each scored 0.75 on 24 held-out Level 1 tasks,
triggering `level1_only` → `introduce_level2` at step 15. Two optimizer steps
then completed with the new mix. Prime Sandbox billing blocked the next batch,
so the partial step and a post-transition held-out score are explicitly
excluded. The retrieved post-transition adapter is evidence only: its exact
staged step-15 merged base was not retained locally, so the adapter cannot be
used to reconstruct the policy from the local artifact set.
