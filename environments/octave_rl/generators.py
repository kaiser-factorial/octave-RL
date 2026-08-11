"""Seeded task generation: ten families, eight variants each, three levels.

Each family lives in `families/` and renders its own problems; this module
cycles them, assigns variants by stream position, and applies the two holdout
filters. See `specs.py` for the variant contract.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from families import broadcast_arith as _broadcast_arith_variants
from families import linsolve_tolerance as _linsolve_tolerance_variants
from families import logical_index as _logical_index_variants
from families import reduce_along_dim as _reduce_along_dim_variants
from families import reshape_permute as _reshape_permute_variants
from families import sequence_recurrence as _sequence_recurrence_variants
from families import signal_identity as _signal_identity_variants
from families import sliding_window as _sliding_window_variants
from families import string_parse as _string_parse_variants
from families import struct_cell_wrangle as _struct_cell_wrangle_variants
from specs import Variant, resolve_variants, variant_for_index

Task = dict[str, Any]

# Every family, each rendering eight distinct problems per level. The mapping is
# deliberately explicit rather than an import that fails soft: a family missing
# from here used to fall back to a single fixed prompt, which is the defect this
# change removes, and it would have done it invisibly.
#
# The staged conversion is finished, so this must stay exhaustive over
# `FAMILY_NAMES`; `test_every_family_is_on_the_variant_form` holds it to that.
VARIANT_MODULES: dict[str, Any] = {
    "reduce_along_dim": _reduce_along_dim_variants,
    "broadcast_arith": _broadcast_arith_variants,
    "sliding_window": _sliding_window_variants,
    "logical_index": _logical_index_variants,
    "struct_cell_wrangle": _struct_cell_wrangle_variants,
    "string_parse": _string_parse_variants,
    "reshape_permute": _reshape_permute_variants,
    "signal_identity": _signal_identity_variants,
    "linsolve_tolerance": _linsolve_tolerance_variants,
    "sequence_recurrence": _sequence_recurrence_variants,
}


def _shape_sentence(cases: list[dict[str, Any]]) -> str:
    """State the graded output shape, derived from the values the grader uses.

    Scoring compares ``size(actual)`` against the expected value's Octave shape
    exactly, so output orientation is part of the task. Deriving the sentence
    from ``expected`` rather than writing it by hand keeps the prompt's claim
    and the grader's comparison from drifting apart -- the failure mode that
    made three families near-unsolvable before 2026-08-09.

    A row count is stated only when every hidden case agrees on it; families
    whose output width follows the inputs get the unquantified form.
    """
    expected = cases[0]["expected"]
    if not isinstance(expected, list):
        return "Return a scalar."
    if expected and all(isinstance(row, list) for row in expected):
        if len(expected[0]) == 1:
            return "Return a column vector (N-by-1)."
        heights = {len(case["expected"]) for case in cases}
        if len(heights) == 1:
            return f"Return a matrix with {heights.pop()} rows."
        return "Return a 2-D matrix."
    return "Return a row vector (1-by-N)."


def _row_from_variant(variant: Variant, family: str, level: int) -> Task:
    """Wrap one rendered variant in the row shape the taskset consumes.

    The prompt is assembled exactly as `_row` assembles it, including
    `_shape_sentence`, so a converted family's prompts differ from an
    unconverted one's only in the sentence that states the task. That keeps the
    staged conversion from confounding a measurement with a formatting change.

    `info` gains two fields. `variant` is what a per-variant breakdown and a
    variant holdout key on. `natural` carries the naive solution to
    `validate_natural_solutions.py`, which is the only check that has ever
    caught a task solvable solely through an undisclosed convention -- it has to
    travel with the task, because a lookup table beside the generator is what
    silently covered one variant per family and passed the other seven.
    """
    prompt = (
        f"Write this GNU Octave function:\n\n    {variant.signature}\n\n"
        f"{variant.description}\n"
        f"{_shape_sentence(variant.cases)}\n"
        f"Return exactly one fenced `octave` code block. "
        f"Hidden tests include edge cases."
    )
    fn = variant.signature.split("=")[1].split("(")[0].strip()
    return {
        "prompt": [{"role": "user", "content": prompt}],
        "info": {
            "family": family, "level": level, "fn_name": fn,
            "signature": variant.signature, "cases": variant.cases,
            "tolerance": variant.tolerance,
            "require_vectorized": variant.vectorized,
            "variant": variant.key, "natural": variant.natural,
        },
        "task": f"octave-l{level}-{family}-{variant.key}",
        "_reference": variant.reference,
    }


# The canonical family order. It is the round-robin order `build_tasks` cycles,
# so it is part of the split contract: reordering silently changes which task
# index carries which family, and both holdouts rest on that being stable.
#
# Until 0.5.0 this was a list of generator *functions* and the names were derived
# from them. Those functions, the `DESCRIPTIONS` table they read, and the `_row`
# helper that assembled their prompts are gone -- every family now renders from
# its module in `VARIANT_MODULES`, and leaving a second, stale definition of
# what each family asks would be a standing invitation to read the wrong one.
FAMILY_NAMES: list[str] = [
    "reduce_along_dim", "logical_index", "reshape_permute", "broadcast_arith",
    "sliding_window", "linsolve_tolerance", "sequence_recurrence",
    "struct_cell_wrangle", "string_parse", "signal_identity",
]

# The default generalization split. Holding out a *family* is the only way this
# taskset can support a held-out-problem claim: a pool's prompts are determined
# by (family, level), so two pools drawn with different seeds share every
# prompt and differ only in hidden test inputs.
#
# These two were chosen for dynamic range and for what they test. Both sit
# mid-difficulty for both measured models (2026-08-09: reduce_along_dim
# 0.389/0.326, reshape_permute 0.569/0.292), so neither is floored nor
# ceilinged. `reduce_along_dim` has a near neighbour that stays in training --
# `struct_cell_wrangle` level 2+ is also a column-wise reduction -- so it tests
# transfer of a practiced idiom. `reshape_permute` has none, so it tests
# whether general Octave fluency reaches an unpracticed one.
#
# This is a default, not a recommendation for every experiment. Hold out the
# two hardest families and a real improvement will be invisible against the
# floor; hold out the two easiest and it vanishes into the ceiling.
DEFAULT_HELDOUT_FAMILIES: list[str] = ["reduce_along_dim", "reshape_permute"]


def resolve_families(families: list[str] | None) -> list[str]:
    """Validate a family selection, or return all ten."""
    if families is None:
        return list(FAMILY_NAMES)
    unknown = [name for name in families if name not in FAMILY_NAMES]
    if unknown:
        raise ValueError(
            f"unknown task families {unknown}; valid names are {FAMILY_NAMES}"
        )
    if not families:
        raise ValueError("families must select at least one family")
    return list(dict.fromkeys(families))


def declared_variants() -> dict[str, list[str]]:
    """Every ``family -> [variant key]`` a converted family offers.

    Unconverted families are absent rather than present-and-empty, so a caller
    can tell "this family has one problem per level" from "this family has no
    problems", and a variant selection naming an unconverted family fails loudly
    instead of silently selecting nothing.
    """
    return {
        name: list(module.VARIANT_KEYS)
        for name, module in VARIANT_MODULES.items()
    }


# The default variant holdout: the last two variants of each converted family.
#
# **Provisional, and chosen without measurement.** The family holdout beside it
# was picked from measured per-family pass rates, so that neither held-out family
# sits on the floor or the ceiling. No per-variant pass rates exist yet -- that
# sweep is step 4 of the measurement plan in `PARAMETERIZATION_DESIGN.md` -- so
# this is a positional default, not a recommendation. Re-choose it once the
# sweep lands, and do not quote a generalization number that rests on it before
# then.
#
# Two of eight holds out a quarter of the problems while leaving every family in
# training, where the family holdout costs a fifth of training coverage to hold
# out any problem at all.
DEFAULT_HELDOUT_VARIANTS: list[str] = [
    f"{family}:{key}"
    for family, keys in {
        name: list(module.VARIANT_KEYS) for name, module in VARIANT_MODULES.items()
    }.items()
    for key in keys[-2:]
]


def training_families(heldout: list[str] | None = None) -> list[str]:
    """The complement of a holdout, in canonical order."""
    excluded = set(resolve_families(heldout if heldout is not None else DEFAULT_HELDOUT_FAMILIES))
    kept = [name for name in FAMILY_NAMES if name not in excluded]
    if not kept:
        raise ValueError("holding out every family leaves nothing to train on")
    return kept


def build_tasks(
    level=1,
    num_tasks=500,
    seed=0,
    require_vectorized=False,
    include_reference=False,
    families=None,
    variants=None,
):
    """Generate ``num_tasks`` tasks, optionally restricted to families or variants.

    Restricting *filters* the full ten-family stream rather than cycling over the
    selection, so a given family's k-th task is byte-identical whichever other
    families are present, and whichever variants are selected. That is what makes
    a train split and a holdout split drawn from one seed genuinely disjoint *and*
    individually comparable to a full-pool measurement. Task indices come from the
    full stream, so ids stay stable and are not contiguous within a filtered pool.

    Both filters run *after* generation for the same reason: every task is drawn
    from the shared rng whether or not it is kept, so a selection cannot shift the
    stream for the tasks that remain. Generating only the selected tasks would be
    faster and would silently make every split incomparable with every other.

    ``variants`` names ``"family:key"`` pairs and applies only to converted
    families; an unconverted family contributes its single problem per level
    regardless. ``None`` selects everything.
    """
    selected = set(resolve_families(families))
    selected_variants = resolve_variants(declared_variants(), variants)
    rng = np.random.default_rng(seed)
    rows = []
    index = 0
    while len(rows) < num_tasks:
        name = FAMILY_NAMES[index % len(FAMILY_NAMES)]
        module = VARIANT_MODULES[name]
        # The variant follows the task's ordinal within its own family's
        # stream, not an rng draw. See `specs.variant_for_index`.
        key = variant_for_index(module.VARIANT_KEYS, index // len(FAMILY_NAMES))
        task = _row_from_variant(module.build(rng, level, key), name, level)
        task["task"] += f"-{index:05d}"
        index += 1
        if name not in selected:
            continue
        if key not in selected_variants.get(name, []):
            continue
        if require_vectorized:
            task["info"]["require_vectorized"] = True
        if not include_reference:
            task.pop("_reference")
        rows.append(task)
    return rows
