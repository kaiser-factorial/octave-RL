# Did tightening the `reshape_permute` prompt work?

**Yes for the defect it was aimed at, and it changes nothing else.** Runaway
generation fell on all six cells; solve rate did not move. The tightening
stays.

`reshape_permute` truncated 19-33% of Qwen3.5-4B rollouts and 58-63% of
Qwen3.5-2B's at the 2048-token cap — far above every other family — so its
measured solve rate was a floor rather than an estimate. The level-2 description
restated the same transform three ways in 927 characters. It now says it once,
in 506.

## Design

Two probes, one variable each. Both use `--families reshape_permute --levels 1 2 3
--num-tasks 48 --num-rollouts 4`, seed `20260808`, T=1.0, thinking off, and
generate through Prime Inference with local scoring against the pinned Octave
10.2.0. All 48 tasks are **shared between baseline and tightened in every cell**,
so the comparison is paired task-by-task, and the delta below is a paired
difference rather than two independent means.

1. **Budget** (`outputs/headroom-reshape`, run earlier): 927-char prompt, cap
   raised 2048 → 4096.
2. **Prompt** (`outputs/headroom-reshape-tightened`, this run): cap held at 2048,
   prompt 927 → 506 chars. Baseline is the definitive sweep's own
   `reshape_permute` cells, which ran at that cap.

Holding the cap at 2048 is what makes this a test of the prompt. Raising it
would have confounded the two.

## The budget was not the problem

Doubling the cap on the untightened prompt:

| cell | trunc @2048 | trunc @4096 | solve @2048 | solve @4096 |
|---|---:|---:|---:|---:|
| 2B L1 | 0.583 | 0.573 | 0.000 | 0.000 |
| 2B L2 | 0.630 | 0.682 | 0.000 | 0.000 |
| 2B L3 | 0.615 | 0.453 | 0.000 | 0.000 |
| 4B L1 | 0.193 | 0.146 | 0.156 | 0.250 |
| 4B L2 | 0.281 | 0.266 | 0.104 | 0.083 |
| 4B L3 | 0.328 | 0.240 | 0.078 | 0.062 |

Twice the room and 2B still truncates on more than half its rollouts and solves
nothing. The family induces runaway generation; more budget buys more runaway.

## The prompt was

Cap fixed at 2048. Truncation share, and median completion tokens per rollout:

| cell | trunc before | trunc after | Δ | p50 tokens before | p50 after |
|---|---:|---:|---:|---:|---:|
| 2B L1 | 0.583 | 0.422 | **−0.161** | 2048 | 842 |
| 2B L2 | 0.630 | 0.458 | **−0.172** | 2048 | 1061 |
| 2B L3 | 0.615 | 0.453 | **−0.162** | 2048 | 942 |
| 4B L1 | 0.193 | 0.156 | −0.037 | 510 | 457 |
| 4B L2 | 0.281 | 0.188 | −0.094 | 706 | 424 |
| 4B L3 | 0.328 | 0.297 | −0.031 | 586 | 604 |

Truncation falls in all six. The sharpest reading is 2B's median: **before, the
typical 2B rollout ran to the cap** — the median rollout was itself truncated —
and after, it finishes around 850-1050 tokens. Halving the prompt roughly halved
what the small model generates.

## Solve rate did not move

Paired difference per task, tightened minus baseline:

| cell | Δ solve | ± |
|---|---:|---:|
| 2B L1 | +0.000 | 0.000 |
| 2B L2 | +0.000 | 0.000 |
| 2B L3 | +0.000 | 0.000 |
| 4B L1 | −0.016 | 0.036 |
| 4B L2 | −0.031 | 0.032 |
| 4B L3 | −0.005 | 0.027 |

Every 4B delta is inside one standard error of zero, and 2B is at 0.000 on this
family before and after, on every level.

## What that means

Read together with the budget probe, the two say the same thing: **truncation was
never what gated the solve rate on this family.** Giving the models more room
didn't help, and using less of the room they had didn't help either. The models
fail `reshape_permute` because a 3-D reshape-reverse-transpose is genuinely hard
for them, not because they run out of tokens describing it.

So the tightening is a real but bounded win. It buys:

- **A cheaper failure.** Truncated rollouts are structural zeros that cost full
  budget and teach nothing. Cutting 2B's truncation by 16 points and its median
  generation by half makes the family cheaper to be wrong on — which matters
  during training, where every rollout is paid for.
- **A legible measurement.** With truncation this high the old solve figures were
  floors. They are still floors, but shallower ones, and the remaining gap is now
  attributable to capability rather than to a budget artefact.

It does not buy any capability, and nothing in the pool needed reverting.

## Consequence for the earlier artifacts

`artifacts/variant-sweep-20260811/` and `artifacts/threeturn-band-20260811/` both
carry a provenance note that their `reshape_permute` numbers measure the 927-char
prompt. That note stands, and this file quantifies it: on the current prompt,
expect those cells to show **4-17 points less truncation with the same solve
rate**. No other family is affected.
