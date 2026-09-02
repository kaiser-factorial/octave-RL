# The level ladder under the training scaffold, and what it does to the gates

**There is no ladder left to gate on.** Qwen3.5-4B, measured under the scaffold
the curriculum controller actually evaluates with — three attempts with the
guide — scores 0.683, 0.634 and 0.441 across Levels 1, 2 and 3, and all three
levels teach roughly equally well. Two of the five promotion gates are already
passed by the untrained policy, and the policy sits closer to the gate that
opens stage 3 than to the one that opens stage 1.

Qwen3.5-4B, 160 tasks × 4 rollouts per level, T=1.0, thinking off, seed
`20260808`, 2048-token cap, three attempts with the guide. 1,920 rollouts.
Level 1 is the cell from `artifacts/threeturn-band-20260811/`; Levels 2 and 3
were measured for this derivation. Generation through Prime Inference, scoring
local against the pinned Octave 10.2.0.

Reproduce with `scripts/derive_curriculum_gates.py`; `gates.json` is its output.

## Why this had to be measured under the scaffold

The gates threshold `raw_case_fraction` from held-out evaluations that run three
attempts with the guide. Every per-level number previously available on the
0.5.0 pool was **single-turn**, and the scaffold was worth 3-4x on the old pool.
Deriving thresholds from single-turn rates would have calibrated every gate
against an instrument the controller never uses.

`raw_case_fraction` is undiscounted, so a success on attempt 2 or 3 counts. On
this pool it tracks `solved` to within 0.002 — only about 0.6% of rollouts earn
partial credit — so the gates are effectively thresholding a binary solve rate.

## The ladder

| level | rate | solved | execution | format_ok | truncation | attempts |
|---|---:|---:|---:|---:|---:|---:|
| 1 | 0.683 ± 0.023 | 0.681 | 0.752 | 0.906 | 0.097 | 2.13 |
| 2 | 0.634 ± 0.024 | 0.631 | 0.726 | 0.881 | 0.098 | 2.24 |
| 3 | 0.441 ± 0.027 | 0.439 | 0.593 | 0.816 | 0.217 | 2.53 |

**Levels 1 and 2 are 0.049 apart — about two standard errors.** Under the
scaffold they are very nearly the same task. Level 3 is genuinely harder, and
still comfortably inside the productive band.

Level 3's truncation is 0.217 against 0.098 for the other two: a fifth of its
rollouts are structural zeros at the 2048-token cap, so 0.441 is a floor.

## Every level teaches, and Level 2 teaches most

GRPO computes advantage within a group, so a group whose rollouts all score
alike contributes nothing. At the training group size of 2, measured over every
pair each task's rollouts can form:

| level | signal@g2 | all-zero groups | all-one groups | if independent |
|---|---:|---:|---:|---:|
| 1 | 0.357 | 0.050 | 0.306 | 0.433 |
| 2 | **0.390** | 0.075 | 0.250 | 0.464 |
| 3 | 0.349 | 0.231 | 0.138 | 0.493 |

Measured signal runs below the independence estimate `2p(1-p)` at every level,
because per-task rates cluster — some tasks are dead for every rollout, and
those groups are unanimous however favourable the mean looks.

The spread across levels is 0.041. **No level is close to being wasted**, and
the one the curriculum treats as the endpoint is not the weakest teacher. If
anything Level 1 is the most exhausted: 30.6% of its groups are already
unanimous successes, against 13.8% at Level 3.

## The gates, against that ladder

Two separate problems.

### The stated thresholds are not the thresholds

A gate passes on a one-sided 95% Wilson lower bound, and the controller
evaluates 20 rollouts per level by default. That turns every nominal threshold
into a substantially higher observed requirement:

| gate | level | nominal | observed required | base policy | |
|---|---:|---:|---:|---:|---|
| `level1_mastery` | 1 | 0.55 | 0.750 | 0.683 | |
| `level2_signal` | 2 | 0.20 | 0.350 | 0.634 | **already passes** |
| `level2_mastery` | 2 | 0.45 | 0.650 | 0.634 | |
| `level2_advanced` | 2 | 0.60 | 0.800 | 0.634 | |
| `level3_signal` | 3 | 0.10 | 0.250 | 0.441 | **already passes** |

CURRICULUM.md says *"introduce Level 2 after Level 1 exceeds 0.55."* It is 0.75.
The +0.20 gap is a property of the evaluation size, not of the curriculum, and
it is **four times the spacing between Levels 1 and 2**. At this evaluation size
the gates separate sampling noise rather than difficulty. The penalty falls to
+0.09 at 100 rollouts and +0.06 at 200; putting it under the L1-L2 gap of 0.049
takes 310.

### The staging is inverted

Measured against the base policy, in observed-rate space:

| gate | opens stage | distance from base |
|---|---:|---:|
| `level1_mastery` | 1 | +0.067 |
| `level2_signal` | 2 | **−0.284 (passed)** |
| `level2_mastery` | 3 | **+0.016** |
| `level2_advanced` | 4 | +0.166 |
| `level3_signal` | 4 | **−0.191 (passed)** |

**The policy is four times closer to the gate that opens stage 3 than to the one
that opens stage 1.** Two gates are passed before training starts. A promotion
sequence through this ladder would report that difficulty was approached in
order when what actually happened is that Level 1 was the last thing to clear.

## What this means

The curriculum's premise is that hard levels are wasted early and easy levels
are exhausted late. On the parameterised pool, under the scaffold, for the model
we are training, **neither half holds**: every level is productive at step 0,
and the easiest level is the closest to exhaustion.

Re-numbering the five gates cannot fix that. `level2_signal` means "Level 2 now
produces enough signal to become the working set", and Level 2 produces more
signal than Level 1 before any training happens — there is no threshold that
makes that sentence true.

## Limit of this reading

Every measured policy on this pool tops out at 0.683, so **the high-rate side of
the signal curve is unobserved**. Where signal decays as a level approaches
mastery is an extrapolation, and any "stop training on this level" threshold set
from these numbers is provisional until a training run produces rates above 0.7.
Nothing here says a staged curriculum could not help a stronger policy; it says
this one has nothing to stage at step 0.

This also measures one model. Qwen3.5-2B and 0.8B were ruled out as training
targets in `artifacts/threeturn-band-20260811/`, so the ladder was not
re-measured for them.
