# Group spread and the hosted/pod control — 2026-08-08

Three questions, all answered on Prime Inference for **$0.24 total** and no GPU:

1. How often does a sampled group actually carry any GRPO advantage? (The
   2026-08-08 G1 pre-read *modelled* this. The model was wrong.)
2. Is that a property of Nemotron or of the taskset?
3. Does a hosted eval reproduce a pod number at all — i.e. is any of this
   transferable to a training run?

## Setup

`num_rollouts = 8` at T=1.0, levels 1–3, 32 tasks per level, seed `20260808` —
the same 32 tasks per level as `baseline-eval-20260808/` and
`nemotron-eval-20260808/`. Single-turn, no guide, thinking off, 1536-token cap,
scored locally against the pinned Octave 10.2.0 rootfs.

| run | model | rollouts | cost |
|---|---|---:|---:|
| `nemo-g8` | `NVIDIA-Nemotron-3-Nano-30B-A3B-BF16` | 768 | $0.0896 |
| `qwen-g8` | `Qwen/Qwen3.5-4B` | 768 | $0.1328 |
| `qwen-greedy-control` | `Qwen/Qwen3.5-4B`, T=0 | 96 | ~$0.015 |

Zero infrastructure errors across all 1,632 rollouts.

## 1. The independence model understates waste by up to 14×

GRPO advantage is `rewards - rewards.mean()`, so a unanimous group contributes
exactly zero gradient. The G1 pre-read estimated the unanimous fraction as
`p**g + (1-p)**g` — rollouts as independent coin flips at the task's pass rate.
Measured against the run's own groups:

**Nemotron**

| level | p | dispersion | g=2 obs / model | g=4 obs / model | g=8 obs / model |
|---|---:|---:|---:|---:|---:|
| L1 | 0.515 | 3.20 | 0.639 / 0.500 | 0.351 / 0.126 | 0.148 / 0.008 |
| L2 | 0.340 | 4.60 | 0.792 / 0.551 | 0.621 / 0.203 | 0.500 / 0.036 |
| L3 | 0.143 | 2.01 | 0.829 / 0.755 | 0.693 / 0.539 | 0.593 / 0.290 |

**Qwen3.5-4B**

| level | p | dispersion | g=2 obs / model | g=4 obs / model | g=8 obs / model |
|---|---:|---:|---:|---:|---:|
| L1 | 0.240 | 3.00 | 0.720 / 0.636 | 0.494 / 0.338 | 0.333 / 0.112 |
| L2 | 0.100 | 3.16 | 0.903 / 0.821 | 0.824 / 0.657 | 0.759 / 0.432 |
| L3 | 0.079 | 3.18 | 0.896 / 0.855 | 0.807 / 0.720 | 0.710 / 0.519 |

*(obs at g<8 is exact, not resampled: a group with `s` successes out of 8 has
`[C(s,k) + C(8-s,k)] / C(8,k)` unanimous sub-groups of size `k`.)*

**Answer to Q2: it is the taskset, not the model.** Dispersion is 2.0–4.6 for
Nemotron and 3.0–3.2 for Qwen. Both models' rollouts within a task are far more
correlated than independent draws.

The mechanism is visible in the distribution of successes-per-group. Nemotron
L2, 32 tasks × 8 rollouts, marginal p = 0.340:

```
passes/8   observed   binomial
       0         11        1.2   ###########
       1          3        4.8   ###
       2          5        8.6   #####
       3          3        8.8   ###
       4          1        5.7   #
       5          2        2.3   ##
       6          2        0.6   ##
       7          1        0.1   #
       8          4        0.0   ####
```

Binomial predicts a hump at 2–3. Reality is **U-shaped**: piled at 0 and 8,
hollow in the middle. p = 0.34 is not a coin the model flips per rollout; it is
the average over a mixture of tasks it near-reliably solves and tasks it
near-reliably fails. T=1.0 varies the wording, not the capability.

This errs in one direction only. Positive within-task correlation can only make
groups *more* unanimous, so the independence model is a strict lower bound on
waste — it can only flatter a rollout budget.

## 2. Corrected G1 mix table

Pooled pass rate with **clustered** standard errors (across the 32 task means,
not the 256 rollouts), and the share of rollouts landing in a non-degenerate
group — the ones that actually receive gradient:

| mix | pooled p | ±SE | in band | useful g=2 | useful g=4 | useful g=8 |
|---|---:|---:|:---:|---:|---:|---:|
| L2 only | 0.340 | 0.063 | edge | 20.8% | 37.9% | 50.0% |
| L1/L2/L3 equal | 0.333 | 0.030 | edge | 24.7% | 44.5% | **58.6%** |
| L2/L3 50–50 | 0.242 | 0.035 | yes | 18.9% | 34.3% | 45.4% |
| L2/L3 25–75 | 0.192 | 0.028 | yes | 18.0% | 32.5% | 43.1% |
| L3 only | 0.143 | 0.031 | yes | 17.1% | 30.7% | 40.7% |

Raising `group_size` still pays — L2 goes 21% → 50% useful from g=2 to g=8 —
but nothing like the modelled 47% → 96%. Budget the arm off the left column.

## 3. Hosted eval reproduces the pod (the control that licenses all of this)

Every earlier number came from a pod-local vLLM through the **train** client;
everything here comes from a hosted provider through the **eval** client.
Different rendering path, different transport. Paired on the identical 32 tasks
per level, base Qwen3.5-4B greedy:

| level | pod (vLLM / train) | hosted (eval) | pod-only wins | hosted-only wins | McNemar p |
|---|---:|---:|---:|---:|---:|
| L1 | 0.7031 | 0.6406 | 3 | 1 | 0.625 |
| L2 | 0.3750 | 0.2865 | 4 | 1 | 0.375 |
| L3 | 0.4062 | 0.4062 | 1 | 1 | 1.000 |

Indistinguishable at every level, with only 4–5 discordant tasks out of 32.
Mean completion tokens track too (178/273/444 hosted against 176/276/409 pod),
as do truncation and format validity on L3 (0.156 / 0.84, both paths).

**This is the result that makes a 2-cent pre-read worth trusting.** It is also a
prerequisite for reading anything above as advice about a training run.

## 4. At the training temperature Nemotron beats Qwen on every level

The 2026-08-08 pre-read compared the two at **greedy, n=32** and concluded they
were indistinguishable on L1/L2 with Qwen significantly better on L3. At n=256
and at T=1.0 — the temperature GRPO actually samples at — that reverses:

| level | Nemotron | Qwen | diff | SE | p |
|---|---:|---:|---:|---:|---:|
| L1 | 0.515 | 0.240 | +0.275 | 0.040 | <0.0001 |
| L2 | 0.340 | 0.100 | +0.240 | 0.047 | <0.0001 |
| L3 | 0.143 | 0.079 | +0.064 | 0.027 | 0.015 |

Paired on group means over the same 32 tasks. **The earlier "Qwen is better on
L3" finding holds only at greedy**; Qwen's L3 collapses from 0.406 greedy to
0.079 at T=1.0 while Nemotron's holds (0.125 → 0.143). Section 3's control rules
out serving path as the explanation.

The mechanism is output discipline under sampling, not reasoning:

| | Nemotron fmt | Qwen fmt | Nemotron trunc | Qwen trunc |
|---|---:|---:|---:|---:|
| L1 | 1.00 | 0.79 | 0.000 | 0.070 |
| L2 | 0.99 | 0.70 | 0.004 | 0.102 |
| L3 | 0.98 | 0.55 | 0.019 | 0.180 |

At T=1.0 Qwen fails to emit a parseable function on **45% of L3 rollouts**.
Nemotron is at 2%. Greedy hides this entirely — both models are at 0.84–1.00
format validity at T=0.

For WS3 this matters beyond model choice: an objective whose gradient is
dominated by "did you emit a fenced function" would move routing for reasons
unrelated to reasoning content. Nemotron's signal is much cleaner on that axis.

## Caveats

- 32 tasks per level, one seed. The clustered SEs are the honest ones; the
  naive per-rollout SEs are ~2× too small and the summarizer now reports both.
- The hosted/pod control is on Qwen only, greedy only. It does not prove that a
  hosted *Nemotron* number would match a pod Nemotron number, only that the
  client-path difference is not by itself a large effect.
- `dispersion` and the sub-group figures assume the 8 rollouts of a task are
  exchangeable, which they are by construction here (same prompt, same
  sampling config, no shared state).
- Provider-side serving config for a hosted endpoint is not pinned.
