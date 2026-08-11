# Per-variant pass rates on the 0.5.0 pool

Four models, ten families, eighty variants, three levels. **Exactly 6 tasks per
variant** (480 tasks per level), 4 rollouts each, single-turn, no guide, T=1.0,
thinking off, seed `20260808`, 2048-token completion cap. Generation through
Prime Inference; scoring local against the pinned Octave 10.2.0. **23,040
rollouts, $3.03, no GPU and no Sandbox.**

`per_variant.json` holds one record per (model, family, variant, level): solve
rate with standard error across tasks, execution fraction, correct-given-executed,
format validity, and truncation share.

Solve rate comes from the `solved` metric — undiscounted — never from
thresholding the reward, which is discounted by attempt.

## Provenance: this measures the pre-tightening `reshape_permute` prompt

The working tree that produced this run predates the `reshape_permute` prompt
tightening now on `main` (927-character level-2 description here against 534
there). **Every `reshape_permute` number below describes a prompt that no longer
exists**, and must not be compared against a post-tightening figure. Nothing else
in the pool differs.

## Level rates

Solve rate ± standard error across tasks, n = 480 tasks per cell.

| model | Level 1 | Level 2 | Level 3 |
|---|---:|---:|---:|
| Nemotron-3-Nano-30B-A3B | 0.574 ± 0.015 | 0.567 ± 0.015 | 0.346 ± 0.017 |
| Qwen3.5-4B | 0.327 ± 0.013 | 0.289 ± 0.013 | 0.167 ± 0.011 |
| Qwen3.5-2B | 0.052 ± 0.006 | 0.033 ± 0.005 | 0.026 ± 0.004 |
| Qwen3.5-0.8B | 0.010 ± 0.002 | 0.005 ± 0.002 | 0.003 ± 0.001 |

Against the 0.4.x pool at the same single-turn design: Nemotron L1 was 0.570,
Qwen3.5-4B 0.400, 2B 0.068, 0.8B 0.030. **The parameterised pool costs a strong
model nothing and a weak one a great deal**, which sharpens the ladder rather
than flattening it.

**Level 2 is not meaningfully harder than Level 1 for Nemotron** (0.567 against
0.574). The same flatness appeared between Levels 2 and 3 on the 0.4.x pool; it
has moved one rung down.

## No variant is illegible

**Nine of 240 variant-levels sit at or below 0.02 for both Qwen3.5-4B and
Nemotron, and all nine are Level 3.** None has the shape of a broken prompt.

An undisclosed convention reads as **high execution with zero solve** — the model
writes running Octave and still disagrees with the grader. Execution on these
nine runs 0.16 to 0.46, so the models are mostly failing to produce runnable code
at all, which is difficulty rather than ambiguity.

| level-3 variant | 4B | Nemotron | exec |
|---|---:|---:|---:|
| `sequence_recurrence:order1-cumulative` | 0.00 | 0.00 | 0.46 |
| `string_parse:semicolon-integers-column` | 0.00 | 0.00 | 0.25 |
| `sliding_window:mean-stride1` | 0.00 | 0.00 | 0.22 |
| `sliding_window:max-strided` | 0.00 | 0.00 | 0.19 |
| `sliding_window:median-strided` | 0.00 | 0.00 | 0.18 |
| `sliding_window:mean-strided` | 0.00 | 0.00 | 0.17 |
| `sliding_window:range-strided` | 0.00 | 0.00 | 0.17 |
| `string_parse:mixed-decimals-row` | 0.00 | 0.00 | 0.17 |
| `sliding_window:min-stride1` | 0.00 | 0.00 | 0.16 |

Six of nine are `sliding_window` level 3, where a loop-free windowed statistic
needs the index-matrix idiom. The highest-execution case,
`sequence_recurrence:order1-cumulative`, was read by hand: models write
`x(i) = p*x(i-1) + d + i` over an index vector — a plausible-looking vectorised
recurrence that runs and computes the wrong thing, because a recurrence needs
`filter`. `transposed_fraction` is 0.0, so it is not an orientation convention.
Genuine difficulty.

**Worth knowing about Level 3: the loop ban is not enforced by the reward.**
`require_vectorized` drives a metric, not a penalty, so a model that ignores the
ban and writes a loop still earns a full solve if the answer is right. These
cells fail because the models obey a constraint that costs them nothing to
break. Level 3 measures instruction-following as much as vectorisation.

## Truncation

Share of rollouts hitting the 2048-token completion cap. A low solve rate beside
a high truncation share is a budget result, not a capability one.

| model | L1 | L2 | L3 |
|---|---:|---:|---:|
| Nemotron | 0.001 | 0.001 | 0.005 |
| Qwen3.5-4B | 0.045 | 0.046 | 0.147 |
| Qwen3.5-2B | 0.098 | 0.115 | 0.144 |
| Qwen3.5-0.8B | 0.057 | 0.081 | 0.108 |

Nemotron's output discipline is the same effect measured on the 0.4.x pool,
where it emitted a parseable function on 98% of Level 3 rollouts against Qwen's
55%. `reshape_permute` alone drove 25-61% truncation at this cap, which is what
prompted the prompt tightening.

## What this does not answer

**Which model to train.** Training runs three attempts with a guide, worth 3-4x
on the old pool. A single-turn number is the wrong instrument for the band
judgement; that needs its own three-turn cell.

**Whether a prompt is legible to a *trained* policy.** These are base models. A
variant that is hard for all four here may still be learnable.
