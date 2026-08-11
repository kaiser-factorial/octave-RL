"""The spec form: one drawn choice produces prompt, reference, and naive solution.

Before 0.5.0 a task's prompt was determined entirely by ``(family, level)``, so a
1,500-task pool contained **30 distinct prompts** and two pools drawn with
different seeds shared all 30. Only the hidden inputs varied. A seed split held
out test data, never a question, and with 30 problems the retry scaffold behaved
mostly like best-of-N resampling.

A **variant** is what fixes that: a named choice -- which statistic, which axis,
which operator -- that changes what the function must compute. Eight variants per
family, rendered at three levels, gives 24 distinct prompts per family and 240
across the pool.

## The one rule that matters

Every variant produces **three artefacts from one definition**:

- ``description``  -- what the model is told;
- ``reference``    -- known-correct, may use defensive coercion;
- ``natural``      -- what a competent Octave programmer writes from the
                      description alone, with no coercion, no ``(:)``, and no
                      transpose the prompt did not ask for.

They cannot drift, because they are written together and validated together.
This exists because the project has three times shipped a task solvable only
through a convention the prompt never stated, and both reference-based
validators stayed green every time -- a reference passes precisely *because* it
contains the convention. ``scripts/validate_natural_solutions.py`` scores the
``natural`` field, so a variant whose natural solution fails is not shippable.

## Determinism, and why the variant is not drawn from the rng

``build_tasks`` cycles families and filters, so a family's k-th task is identical
whichever other families were selected -- that is what makes the holdout splits
trustworthy, and ``test_a_family_generates_the_same_tasks_whichever_others_are_present``
guards it. Drawing the variant from the shared rng stream would consume draws
and break that property the moment a variant selection changed.

The variant is therefore a **pure function of the task's position in its
family's stream**: task k of a family gets variant ``k % len(variants)``. That
keeps rng consumption identical, makes per-variant counts exact rather than
multinomial, and lets a variant holdout be expressed without perturbing the
tasks that remain.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True)
class Variant:
    """One distinct problem within a family, at one level.

    ``key`` is stable and is what a variant holdout names, so renaming one
    silently changes which problems a split holds out. Treat keys as a public
    interface: add and deprecate, do not rename.
    """

    key: str
    description: str
    signature: str
    cases: list[dict[str, Any]]
    reference: str
    natural: str
    tolerance: float = 1e-9
    vectorized: bool = False


# A family module exposes ``VARIANT_KEYS: list[str]`` and one builder,
# ``build(rng, level, key) -> Variant``. The rng is consumed for hidden case
# inputs only; the variant arrives as an argument, never as a draw.
VariantBuilder = Callable[[np.random.Generator, int, str], Variant]


def variant_for_index(keys: Sequence[str], index: int) -> str:
    """Select the variant for task ``index`` within its family's stream.

    Round-robin rather than random: exact per-variant counts, no rng consumed,
    and stable under a change of variant selection.
    """
    if not keys:
        raise ValueError("a family must define at least one variant")
    return keys[index % len(keys)]


def resolve_variants(
    declared: dict[str, list[str]],
    selected: list[str] | None,
) -> dict[str, list[str]]:
    """Validate a variant selection against what the families declare.

    ``selected`` names variants as ``"family:key"`` so one flat list can express
    a holdout that spans families, which is what a config field can carry.
    ``None`` selects everything.
    """
    if selected is None:
        return {family: list(keys) for family, keys in declared.items()}
    chosen: dict[str, list[str]] = {}
    unknown: list[str] = []
    for name in selected:
        family, _, key = name.partition(":")
        if not key:
            raise ValueError(
                f"variant {name!r} must be written 'family:key', for example "
                "'reduce_along_dim:mean-columns'"
            )
        if family not in declared or key not in declared[family]:
            unknown.append(name)
            continue
        if key not in chosen.setdefault(family, []):
            chosen[family].append(key)
    if unknown:
        raise ValueError(
            f"unknown task variants {sorted(unknown)}; valid names are "
            f"{sorted(f'{fam}:{key}' for fam, keys in declared.items() for key in keys)}"
        )
    if not chosen:
        raise ValueError("variants must select at least one variant")
    return chosen


def complement(
    declared: dict[str, list[str]],
    heldout: list[str],
) -> list[str]:
    """The ``"family:key"`` names that a holdout leaves behind, in declared order.

    A variant holdout is the split parameterisation makes possible for the first
    time. The family holdout it complements costs a fifth of training coverage
    to buy a held-out problem; this one holds out problems inside every family
    the model still trains on, which is a strictly harder generalization test and
    a cheaper one.
    """
    excluded = set(heldout)
    for name in excluded:
        family, _, key = name.partition(":")
        if family not in declared or key not in declared.get(family, []):
            raise ValueError(f"unknown held-out variant {name!r}")
    kept = [
        f"{family}:{key}"
        for family, keys in declared.items()
        for key in keys
        if f"{family}:{key}" not in excluded
    ]
    if not kept:
        raise ValueError("holding out every variant leaves nothing to train on")
    return kept
