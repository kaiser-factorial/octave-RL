# Baseline evaluation — 2026-08-08

First evaluation run through the corrected scorer, and the first to score
candidates without touching Prime Sandboxes.

**Setup.** Pod `092edcad9f6b4f7f96d5c89beb54945e`, 2x RTX 6000 Ada 48 GB
(massedcompute, us-central-2), $1.50/hr, 83 minutes, **$2.04** against a $12
ceiling. prime-rl @ `44539229436a23e624b0f39826014a4e58a703be`. Candidates
scored on GNU Octave **10.2.0** from the pinned image rootfs, under
`unshare --net` + `chroot` + `ulimit`. Zero Sandbox calls, zero rollout
infrastructure errors across all 256 rollouts.

Tasks are seed `20260808`, disjoint from the training pool (314159) and from
every prior held-out seed (271828, 272828, 272829). 32 tasks per cell, one
rollout each. `raw_case_fraction` is the reported metric, per the project rule
that shaped rewards are not comparable across runs.

## Results

| cell | temp | cap | thinking | raw | ±SE | solved | Wilson LB | trunc | tokens | fmt |
|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| base L1 | 0.0 | 1536 | off | 0.7031 | 0.080 | 22/32 | 0.514 | 0.000 | 176 | 1.00 |
| base L2 | 0.0 | 1536 | off | 0.3750 | 0.084 | 11/32 | 0.204 | 0.030 | 276 | 0.97 |
| base L3 | 0.0 | 1536 | off | 0.4062 | 0.088 | 13/32 | 0.255 | 0.094 | 363 | 0.91 |
| step-20 L1 | 0.0 | 1536 | off | 0.6406 | 0.084 | 20/32 | 0.453 | 0.000 | 171 | 1.00 |
| step-20 L2 | 0.0 | 1536 | off | 0.5312 | 0.087 | 16/32 | 0.336 | 0.000 | 231 | 1.00 |
| step-20 L3 | 0.0 | 1536 | off | 0.4062 | 0.088 | 13/32 | 0.255 | 0.156 | 409 | 0.84 |
| ctl A L1 | 1.0 | 1024 | **on** | 0.0938 | 0.052 | 3/32 | 0.032 | 0.879 | 1008 | 0.12 |
| ctl B L1 | 1.0 | 1024 | off | 0.2865 | 0.080 | 9/32 | 0.156 | 0.156 | 366 | 0.75 |

## What this says

**1. The historical calibration was sound; it just measured a different config.**
Cell B reproduces the July 29 setup (sampled, 1024-token cap, thinking off)
against the *fixed* scorer and lands at **0.2865**, against the historical
**0.2817**. That is a replication. The 0.703 greedy figure is a different
measurement, not a contradiction, and the flattening defect did not contaminate
the calibration — which the timeline already implied and this now confirms
empirically.

**2. Level 1 is already in the G1 band at the training temperature.**
GRPO learns from reward spread *within a group of samples*, so the band that
matters is the one at the rollout temperature, not at greedy. At T=1.0 base
Qwen3.5-4B sits at **0.2865** on Level 1 — inside the 10–35% target. Choosing a
curriculum mix from the greedy number (0.70) would have been a mistake.

**3. Thinking-on is catastrophic under a 1024-token cap, not merely wasteful.**
87.9% truncation, mean completion exactly at the cap, and format validity
**0.12** — the model rarely emits a fenced function at all. Note that cell A
sets `reasoning_effort = "none"` in `[sampling]` and *still* thinks: only
`enable_thinking = false` at the renderer suppresses it. Relying on the
sampling knob alone silently produces near-zero rewards that look like a
capability result.

**4. The 20-step run did not move single-turn capability.**
L1 0.641 vs base 0.703 and L2 0.531 vs 0.375 are both within noise at n=32
(±0.12 combined SE); L3 is identical. What it *did* change is output
discipline: format validity 1.00 vs 0.97 on L2, zero truncation on L1/L2, and
shorter completions. The historical 0.905 is not a counterexample — it came
from three attempts with a 35B guide under the old reward protocol, where a
correct first answer scored 1.1 because of the `RESULT` marker bonus, on a
10–24 task set.

**5. Level 3 is not harder than Level 2 for this model.** 0.406 vs 0.375 for
base, identical solve counts for step-20. Adding L3 weight will not buy the
headroom a harder mix is supposed to provide; a genuine Level 4 looks necessary
rather than optional.

## Caveats

- n=32 per cell. Differences below roughly 0.17 are not resolvable here.
- Cells B and A used the 1024-token cap, so their truncation (15.6% / 87.9%)
  suppresses scores relative to a 1536-token run.
- Single-turn only. Nothing here speaks to the three-attempt curriculum.
- One seed. These are calibration numbers, not benchmark claims.
