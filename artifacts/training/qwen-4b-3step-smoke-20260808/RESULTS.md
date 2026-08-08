# 3-step training smoke — 2026-08-08

First optimizer steps taken against the corrected reward, and the first
training run of any length with **no Prime Sandbox in the path**.

Pod `5d52542042524de787021b492e2e6e95`, 2x RTX 6000 Ada 48 GB, 38 minutes,
**$0.92** compute + **$0.0048** Prime Inference (the guide preflight). prime-rl
@ `44539229436a23e624b0f39826014a4e58a703be`, config
`configs/prime-rl/octave-qwen-4b-3step-smoke.toml`. Trained from base
`Qwen/Qwen3.5-4B`, Level 1 only, three attempts with the 35B guide, LoRA rank
16, lr 1e-5, batch 8 / group 2 / max-inflight 2.

## Steps

| step | wall | reward | trainable | turns | truncation | errors | loss | entropy | KL |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 6m24s | 0.6500 | 4/8 | 2.0 | 50.0% | 0.0% | 0.0901 | 0.7954 | 0.0004 |
| 2 | 6m43s | 0.4625 | 4/8 | 2.2 | 50.0% | 0.0% | 0.0875 | 0.4917 | 0.0003 |
| 3 | 4m12s | 0.9250 | 2/8 | 1.5 | 0.0% | 0.0% | -0.0037 | 0.3754 | 0.0003 |

## Pre-registered criteria

Registered before the run, not after seeing the numbers.

| # | criterion | result |
| --- | --- | --- |
| 1 | 3 optimizer steps complete | **pass** |
| 2 | zero rollout infrastructure errors | **pass** — 0.0% every step |
| 3 | reward non-degenerate, >=1 effective batch/step | **pass** — 4/8, 4/8, 2/8 trainable |
| 4 | trainer/inference mismatch KL < 0.015 | **pass** — 0.0003-0.0004 |
| 5 | finite loss, no entropy collapse | **pass**, with a caveat below |
| 6 | STABLE checkpoint + adapter round-trip | **pass** — SHA-256 verified remote to local |
| 7 | guide hints actually retrieved | **pass** — see correction below |
| 8 | zero Prime Sandbox calls | **pass** — zero references in any log, zero in inventory |

**On criterion 7.** The guide's credential path was proven in a preflight before
any GPU spend: with `PRIME_API_KEY` removed from the environment and only an
isolated mode-0700 `HOME` holding a mode-0600 `.prime/config.json`, the 35B
returned a correct hint (`"...computes the mean along dimension 2 (rows), but
'column means' requires dimension 1"`), and the $0.0048 inference charge
confirms the call was billed. It **did** fire during training: decoding the
retained rollout token IDs finds hints in 3 of 15 unique sequences. An earlier
version of this file said otherwise, based on grepping the env log — but the
hint is injected into the user message content and is stored tokenized in the
rollout blobs, never as log text, so that check could only ever return zero.

## Read this before scaling up

**Do not read a learning trend from three steps.** Reward went 0.65 -> 0.46 ->
0.93 on batches of 8. That is variance, not progress. Step 3's 0.9250 is a
small-sample artifact and should not be quoted as a capability number.

**The entropy decline was a measurement artifact, not sharpening.** The
exported step-3 adapter moved the weights by an RMS of 1.53e-6 against a base
scale of ~1e-2 — about 0.015% relative — which cannot halve policy entropy.
Entropy tracked batch length instead: peak trainer memory fell monotonically
(11.7 -> 11.0 -> 10.2 GiB) as steps 1-2's three-attempt truncated failures gave
way to step 3's short first-attempt solves.

**The real constraint is reward density.** 95.7% of 256 baseline rollouts
scored exactly 0.0 or 1.0, so a GRPO group teaches nothing unless its samples
disagree. At the measured Level-1 pass rate of 0.2865, `group_size = 2` wastes
a predicted 59.1% of rollouts; observed waste was 58.3%. Raise `group_size` to
4 (26.6% waste) or 8 (6.7%), holding `max_inflight_rollouts = 2` — the CUDA
limit was on simultaneous generations, which is a different knob.

**Truncation is binding.** 50% of calls hit the cap on steps 1 and 2. The
1536-token budget is tight for three attempts; the project has already measured
that raising it to 2048 costs a lot of latency for little gain, so better
stopping behaviour is the cheaper lever.

**Cost model for a real run.** 4-7 minutes per step at this envelope, so 20
steps is roughly 2 hours and about $3.20 of GPU, plus a small inference line
once the guide starts firing regularly.
