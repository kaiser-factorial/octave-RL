# Qwen3.5-0.8B, 3-step smoke at group_size 8 — 2026-08-09

Does the smallest available Qwen train on the repaired taskset, with the family
holdout in place and the guide credential fixed?

## Setup

Pod `c72a4013eb1c46959b96b61c1c5551c7`, 2x RTX 6000 Ada, $1.50/hr, **64 minutes
(~$1.60)** covering three configurations. Level 1, three attempts with the
guide, LoRA rank 16, lr 1e-5, the 8-family train split, candidate scoring on the
pinned Octave 10.2.0, no Prime Sandbox.

`batch_size = 16`, **`group_size = 8`**, `max_inflight_rollouts = 8`.

## Result

| step | reward | trainable | turns | **errors** | truncation |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 0.1778 | 8/16 (50.0%) | 2.8 | **0.0%** | 88.9% |
| 2 | 0.1250 | 8/16 (50.0%) | 2.8 | **0.0%** | 87.5% |
| 3 | 0.2250 | **16/16 (100.0%)** | 2.6 | **0.0%** | 81.2% |

Against 4B on the same scaffold: reward 0.9250 flat with trainable collapsing
80% → 20% → 16.7%.

**Three things this establishes.**

1. **The guide fix holds under training.** Zero `UserError` across all three
   runs and every step, where the 4B smoke lost 20-33% of rollouts to it.
2. **`group_size = 8` delivers what it promised.** 50-100% of groups carry
   gradient, against 4B's collapse to 16.7%. 0.8B's near-independent rollouts
   (dispersion 1.44) are what make the larger group pay.
3. **The reward sits in the band and moves.** 0.18 / 0.13 / 0.23 on batches of
   16 — small samples, so read the range, not a trend. What matters is that it
   is neither pinned at zero nor saturated.

No CUDA fault at 8 simultaneous generations. That was a real risk taken
knowingly: the SM89 illegal-memory failures came from 7-8 concurrent **4B**
generations, and 0.8B is a fifth the size.

## `max_inflight_rollouts >= group_size` is enforced

prime-rl refuses to start otherwise:
`Value error, max_inflight_rollouts must be at least the number of rollouts per
example`. The 2026-08-08 handoff advice to "raise `group_size` to 4-8 while
holding `max_inflight_rollouts = 2`" **is not runnable**. They are distinct
knobs but not independent ones, and raising the group necessarily raises
concurrent generation — which is the quantity the CUDA hazard is about.

## The truncation number is not what it looks like

81-89% truncation against 2.5% for the same model in hosted eval. Two fixes were
tried and both were wrong:

| change | truncation |
|---|---|
| baseline (`seq_len` 4096, `max_total_tokens` 6144) | 88.9 / 87.5 / 81.2 |
| `seq_len` → 8192 | 87.5 / 81.2 / 87.5 — no effect |
| `max_total_tokens` → 10240 | 75.0 / 93.8 / **100.0** — *worse* |

`Trace.is_truncated` counts `stop_condition == "max_turns"` as truncated. At a
0.117 solve rate almost every rollout exhausts its three attempts, so the figure
is a restatement of the solve rate. A larger budget lets *more* rollouts reach
attempt 3 rather than stopping early on length, which is why raising it made the
number go up.

**Do not raise the token budget for 0.8B.** By the token measure — per-call
`finish_reason == "length"` — it truncates 2.5% against 2B's 10.0%. Full entry
in `PIPELINE_LOG.md`.

## Recommendation

**0.8B at `group_size = 8` is a viable training target.** It is the cheapest by
5x, the loop is clean, and the group economics work. Before a 20-step run:

- keep `max_total_tokens = 6144`; the 10240 variant showed no benefit;
- report **solve rate** beside truncation, and say which truncation is meant;
- watch whether `format_ok` (0.64 at baseline) moves — if RL lifts it quickly,
  this is a good curriculum; if it plateaus, the run is measuring fence
  discipline rather than Octave.

`seq_len = 4096` against `max_total_tokens = 6144` remains a real inconsistency
in every config here. It did not bite at these lengths; raise `seq_len` to 8192
for a longer run, which was tested and costs nothing.

## Files

- `evidence.tgz` — logs for all three configurations, plus resolved configs.
