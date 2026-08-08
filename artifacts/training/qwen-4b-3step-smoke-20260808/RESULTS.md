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
| 7 | guide hints actually retrieved | **partial** — see below |
| 8 | zero Prime Sandbox calls | **pass** — zero references in any log, zero in inventory |

**On criterion 7.** The guide's credential path was proven in a preflight before
any GPU spend: with `PRIME_API_KEY` removed from the environment and only an
isolated mode-0700 `HOME` holding a mode-0600 `.prime/config.json`, the 35B
returned a correct hint (`"...computes the mean along dimension 2 (rows), but
'column means' requires dimension 1"`), and the $0.0048 inference charge
confirms the call was billed. It did **not** fire during the three training
steps: the guide only triggers when attempt 2 fails, and turns averaged 1.5-2.2,
so few rollouts reached a third attempt. The mechanism is verified; its
in-training behaviour is not yet.

## Read this before scaling up

**Do not read a learning trend from three steps.** Reward went 0.65 -> 0.46 ->
0.93 on batches of 8. That is variance, not progress. Step 3's 0.9250 is a
small-sample artifact and should not be quoted as a capability number.

**Entropy fell 53% in three steps** (0.7954 -> 0.4917 -> 0.3754). Nothing here
is collapsed, but that slope over a 20+ step run is the thing most likely to
end it badly. Watch entropy per step, and treat a continued decline as a
stopping condition rather than a curiosity.

**Truncation is binding.** 50% of calls hit the cap on steps 1 and 2. The
1536-token budget is tight for three attempts; the project has already measured
that raising it to 2048 costs a lot of latency for little gain, so better
stopping behaviour is the cheaper lever.

**Cost model for a real run.** 4-7 minutes per step at this envelope, so 20
steps is roughly 2 hours and about $3.20 of GPU, plus a small inference line
once the guide starts firing regularly.
