# Nemotron smoke evaluation — 2026-08-08

First run of a **hosted** model through the octave pipeline, and the first use
of the verifiers **eval client** rather than the train client. Nothing in this
run touched a GPU, a pod, or a Prime Sandbox.

**Setup.** `nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16` on Prime Inference
(`https://api.pinference.ai/api/v1`), single-turn (`max_turns = 1`), no guide,
thinking off, 1536-token cap. Candidates scored in the session container against
the pinned GNU Octave **10.2.0** rootfs under `unshare --net --map-root-user` →
`chroot` → `ulimit`; every rollout reports `network_isolated = true` and
`filesystem_isolated = true`.

Tasks are seed `20260808`, pool 500, 32 tasks per level — **the same seed, pool
and count as `artifacts/baseline-eval-20260808/`**, so the Qwen and Nemotron
cells drew the *identical 32 tasks per level* (verified: 32/32 name overlap on
every level). Comparisons below are therefore paired, and use McNemar's exact
test on discordant pairs rather than comparing two independent proportions.

**Cost.** 192 rollouts, 22,858 prompt + 38,699 completion tokens, **$0.0213**
reported by the provider, 7.5 minutes wall clock. Zero infrastructure errors.

## Results

| cell | temp | raw | ±SE | solved | Wilson LB | trunc | tokens | fmt | err |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| L1 | 0.0 | 0.5938 | 0.088 | 19/32 | 0.423 | 0.000 | 134 | 1.00 | 0 |
| L2 | 0.0 | 0.3438 | 0.085 | 11/32 | 0.204 | 0.000 | 241 | 1.00 | 0 |
| L3 | 0.0 | 0.1250 | 0.059 | 4/32 | 0.050 | 0.000 | 194 | 1.00 | 0 |
| L1 | 1.0 | 0.4792 | 0.088 | 15/32 | 0.309 | 0.000 | 199 | 1.00 | 0 |
| L2 | 1.0 | 0.3750 | 0.087 | 12/32 | 0.229 | 0.031 | 269 | 0.97 | 0 |
| L3 | 1.0 | 0.1250 | 0.059 | 4/32 | 0.050 | 0.000 | 172 | 1.00 | 0 |

## Paired against base Qwen3.5-4B (greedy, identical tasks)

| cell | Qwen | Nemotron | Qwen-only wins | Nemo-only wins | McNemar p |
|---|---:|---:|---:|---:|---:|
| L1 greedy | 0.7031 | 0.5938 | 7 | 4 | 0.549 |
| L2 greedy | 0.3750 | 0.3438 | 4 | 4 | 1.000 |
| L3 greedy | 0.4062 | 0.1250 | 10 | 1 | **0.012** |
| L1 T=1.0 † | 0.2865 | 0.4792 | 2 | 8 | 0.109 |

† Confounded: Qwen's ctl B ran at a 1024-token cap with 15.6% truncation,
Nemotron at 1536 with 0.0%. The row is reported for completeness, not as a
capability comparison.

## What this says

**1. The pipeline runs a second model, through a second client type, with no
infrastructure of its own.** This is the result that matters for WS3. Until now
every number came from Qwen on a pod-local vLLM through the *train* client. A
hosted model through the *eval* client exercises a different rendering path, a
different transport, and different auth, and it produced 192 clean rollouts at
2 cents. Pipeline confidence is no longer conditional on one model or one
serving mode.

**2. Thinking is suppressed by a different knob on each client, and the
existing warning is path-specific.** `artifacts/baseline-eval-20260808/RESULTS.md`
records that `reasoning_effort = "none"` did *not* suppress thinking and only
the renderer's `enable_thinking = false` did. That is true **on the train
client**, where the renderer templates client-side and sampling args go to a
vLLM generate endpoint. On the eval client the server applies the chat
template, there is no renderer to configure, and `reasoning_effort = "none"` is
what works — measured directly before the run (17×23, 200-token cap):

| setting | completion tokens | reasoning chars |
|---|---:|---:|
| none given | 60 | 152 |
| `reasoning_effort = "low"` | 60 | 152 |
| `reasoning_effort = "none"` | 4 | 0 |
| `chat_template_kwargs.enable_thinking = false` | 4 | 0 |

Confirmed across the run: **0 of 192** responses carried `reasoning_content`,
0 carried a `<think>` tag, truncation was 0.000, and format validity 1.00.
Nemotron defaults thinking **on** (Qwen's renderer defaults it to `None`), so
omitting the knob does not fail loudly — it spends the budget on reasoning and
returns a capability-shaped number that is really a configuration artifact.

**3. Nemotron and Qwen are indistinguishable on L1 and L2; Qwen is genuinely
better on L3.** Paired, the L1 and L2 differences are noise (p = 0.55, 1.00).
L3 is not: 10 tasks Qwen solved and Nemotron did not, against 1 the other way
(p = 0.012). A 30B-A3B model losing to a 4B on the hardest level is worth
treating as a finding about the *tasks* rather than the models — L3's families
may reward a specific convention Qwen picked up rather than deeper reasoning.

**4. The level ladder is monotone for Nemotron, which it was not for Qwen.**
0.594 / 0.344 / 0.125 greedy, 0.479 / 0.375 / 0.125 sampled. The baseline run
concluded "Level 3 is not harder than Level 2 for this model" and inferred that
a genuine Level 4 was necessary. That conclusion was Qwen-specific: the ladder
does separate, and the earlier flatness was a property of the measuring
instrument, not of the curriculum.

**5. For WS3's 10–35% G1 band, a Nemotron curriculum is weighted opposite to a
Qwen one.** At the rollout temperature Nemotron sits at L1 **0.479** (above
band), L2 **0.375** (at the upper edge), L3 **0.125** (inside, low end). For
Qwen the in-band level was L1. Routing a curriculum mix off the Qwen numbers
would put Nemotron mostly above the band.

**6. …but weighting toward L3 collides with `group_size = 2`.** Reward is
**98.9% binary** here (190 of 192 rollouts scored exactly 0.0 or 1.0), even more
so than Qwen's 95.7%, so a group carries advantage only when it is not
unanimous. Under independent rollouts:

| level | p(T=1.0) | g=2 | g=4 | g=8 | g=16 |
|---|---:|---:|---:|---:|---:|
| L1 | 0.479 | 0.501 | 0.126 | 0.008 | 0.000 |
| L2 | 0.375 | 0.531 | 0.172 | 0.024 | 0.001 |
| L3 | 0.125 | **0.781** | 0.586 | 0.344 | 0.118 |

(fraction of groups with zero advantage; independence makes these a *lower*
bound on waste, since correlated rollouts are more unanimous, not less). The
level that puts Nemotron in band is the level that wastes the most rollouts at
`group_size = 2`. These two decisions have to be made together.

**7. The failure profile is better shaped for RL than Qwen's.** Of 125
zero-scoring rollouts: 66 (52.8%) errored on every hidden case, 50 (40.0%) ran
cleanly and got the algorithm wrong, 8 (6.4%) returned an exact transpose, and
**1** (0.8%) failed to emit a parseable function. Qwen's comparable split had
19.6% unparseable and 21.0% ran-but-wrong. Nemotron spends far less of the
reward signal on "can you emit a fenced function" and more of it on the thing
the environment is supposed to measure.

**8. The transpose detector fires on Nemotron too.** 8 exact transposes, all on
L1/L2 sampled and L2 greedy. This run had `guide_enabled = false` and
`max_attempts = 1`, so the orientation hint could not fire — but the population
it targets exists in this model, so the feature is not Qwen-specific scaffolding.

## Caveats

- n=32 per cell, one rollout per task, one seed. Only the L3 paired difference
  survives a significance test; treat everything else as "not resolvable here".
- Single-turn, no guide. Nothing here speaks to the three-attempt curriculum or
  to the orientation hint's effect.
- The G1 band figures are *marginal* pass rates at T=1.0. They predict expected
  group pass rate but were not measured with `group_size > 1`, so the degenerate
  -group table above is a model, not an observation.
- Provider-side serving config for the BF16 endpoint is not under our control
  and is not pinned the way the Octave interpreter is.
