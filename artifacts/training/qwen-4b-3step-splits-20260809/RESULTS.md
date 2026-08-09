# 3-step training smoke on the repaired taskset — 2026-08-09

First optimizer steps ever taken against the 0.2.x/0.3.0 taskset. The
2026-08-08 smoke that passed ran on the pre-repair pool, so nothing here had
been exercised since.

## Setup

Pod `c50a13b23bcb4694be7c7b0b6191d5f6`, 2x RTX 6000 Ada 48 GB, massedcompute,
$1.50/hr. **Created 13:50 UTC, terminated 14:29 UTC — 39 minutes, ~$0.98**
against a $4 cap. Config `configs/prime-rl/octave-qwen-4b-3step-smoke.toml`,
unchanged from the 2026-08-08 smoke except for the one variable under test:

```toml
families = ["logical_index", "broadcast_arith", "sliding_window",
            "linsolve_tolerance", "sequence_recurrence",
            "struct_cell_wrangle", "string_parse", "signal_identity"]
```

Level 1, three attempts, guide before attempt three, LoRA rank 16, lr 1e-5,
batch 8 / group 2 / max-inflight 2, candidate scoring on the pinned Octave
10.2.0 rootfs, no Prime Sandbox.

Bootstrap: prime-rl @ `44539229`, pinned rootfs verified at GNU Octave 10.2.0,
local runtime smoke 90/90 hidden cases before training started.

## The holdout held

| check | result |
|---|---|
| `families` present in the **resolved** orchestrator config | yes, all 8 |
| held-out families in resolved config | none |
| distinct tasks trained on | 19 |
| families those tasks belong to | 7 of the 8 trained |
| **held-out families touched** | **none** |

Verified by mapping the task indices in the environment log back through
`build_tasks(1, 500, 314159, families=training_families())`. Worth doing rather
than assuming: the retained rollout blobs are tokenized, so grepping them for
family names finds nothing whether or not the holdout leaked — the same trap
that produced a false "the guide never fired" claim on 2026-08-08.

## Results, and the 2026-08-08 comparison

| step | reward | trainable | turns | error | truncation |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 0.9250 | 8/10 (80.0%) | 1.8 | 20.0% | 25.0% |
| 2 | 0.9250 | 2/10 (20.0%) | 1.5 | 20.0% | 0.0% |
| 3 | 0.9250 | 2/12 (16.7%) | 1.5 | 33.3% | 0.0% |

Pre-repair, same config, 2026-08-08:

| step | reward | trainable | turns | error | truncation |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 0.6500 | 4/8 (50.0%) | 2.0 | 0.0% | 50.0% |
| 2 | 0.4625 | 4/8 (50.0%) | 2.2 | 0.0% | 50.0% |
| 3 | 0.9250 | 2/8 (25.0%) | 1.5 | 0.0% | 0.0% |

**The loop runs end to end on the repaired taskset with the train split.** That
was the question; the answer is yes. Truncation also fell (50/50/0 → 25/0/0).

## Two findings that change the plan

### 1. Level 1 is now too easy to train on

Reward sits at 0.9250 at every step and the trainable fraction collapses from
80% to 16.7%. With three attempts and the guide, Qwen3.5-4B now saturates
repaired Level 1 — and a group whose rollouts all score 1.0 is exactly as
gradient-free as one that all score 0. The repair moved this cell from "partly
broken" through "useful" and out the other side.

The platform's own guidance puts a useful starting baseline at 10–35% for the
model being trained. **The real run should not use Level 1.** Post-repair
single-turn T=1.0 rates put Qwen at 0.214 on Level 2 and 0.118 on Level 3, so
an L2/L3 mix is the right starting point — which is also what the handoff's
Option B specifies.

Caveat, and it is a large one: batches are 8–12 rollouts. Three identical
0.9250 readings on samples that size are not a trend, and the 2026-08-08 entry
warns against exactly this reading. What survives the small sample is the
*direction* (0.65/0.46 → 0.93 with turns falling 2.0 → 1.5) and the trainable
collapse, which is a structural consequence of a saturated cell rather than a
noisy point estimate.

### 2. A user-server error rate above the abort threshold

20%, 20%, 33% of rollouts hit
`UserError: user server at http://127.0.0.1:PORT/mcp respond failed:
JSONDecodeError('Expecting value: line 1 column 1 (char 0)')`.
The 2026-08-08 run reported zero infrastructure errors, so this is new.

What the environment log shows: the failure occurs **only on retry turns** —
the turns where the user simulator runs the submitted function and calls the
guide — and the body is *empty*, not malformed. Most rollouts retry and
complete (`reward=0.850 turns=3`); one of ten in step 1 was actually lost
(`stop=UserError`, reward 0.000).

The runbook's abort threshold is 5% for provider/sandbox/tunnel errors. This
sits well above it and **must be diagnosed before a 20-step run**, where it
would corrupt several percent of the gradient and, worse, do so
non-uniformly — it only touches multi-turn rollouts, which are the harder
tasks. Not diagnosed here: the smoke's budget was for the loop, not for this.

## Artifacts

- `evidence.tgz` — resolved configs and all logs (orchestrator, trainer,
  inference, environment).
- `step3-adapter.tgz` — the step-3 LoRA adapter,
  SHA-256 `a89568b50ca81f6d4e0c994086e7380db3294b9605c872bc5d75cd0d50f98f16`,
  verified remote-to-local before the pod was terminated.

The full trainer checkpoint was not copied, so any continuation from this
adapter needs an explicit optimizer-reset boundary.
