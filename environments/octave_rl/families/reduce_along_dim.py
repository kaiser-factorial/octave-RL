"""``reduce_along_dim``: a statistic along one axis, over a trimmed slice.

**This module is the worked exemplar for the 0.5.0 variant form.** A family
module exposes exactly two names:

- ``VARIANT_KEYS: list[str]`` -- stable, ordered, one per distinct problem;
- ``build(rng, level, key) -> Variant`` -- renders that problem at that level.

Read ``specs.py`` first for why the variant is an argument rather than an rng
draw, and for the rule that ``description``/``reference``/``natural`` are written
together so they cannot drift.

## The spec

Two dimensions, per ``PARAMETERIZATION_DESIGN.md``: which statistic, and which
axis. Eight of the twelve combinations ship, chosen to cover both axes for the
statistics whose axis convention is easy to get wrong (``mean``, ``median``) and
to include the three that fail differently -- ``sum`` is order-insensitive,
``max``/``min`` are selections rather than aggregates, and ``range`` composes
two reductions.

## The level ladder, and the trim that had to be abandoned

The ladder is retained from 0.4.x: level 3 is level 2 plus a vectorization
constraint. Level 2 replaces each slice by its **running total** and then
reduces, which moves all six statistics away from their level-1 values.

**This is the second ladder here; the first one shipped broken.** It trimmed the
``k`` largest and ``k`` smallest values of each slice. That was chosen because a
*one-sided* trim leaves ``min`` (or ``max``) unchanged -- but a symmetric trim
leaves the **median** unchanged, exactly and by construction, so
``median-columns`` and ``median-rows`` rendered a level-2 prompt whose answer
equalled its level 1 on 240 of 240 measured cases. Two of eight variants were a
distinct prompt that was not a distinct problem: the precise failure this whole
change exists to remove, committed in the file whose docstring warns about it.

Nothing could have caught it downstream. The reference and the naive solution
both pass a degenerate level 2, because both compute the thing the description
asks for -- it is the *description* that fails to ask for something new. See
``PIPELINE_LOG.md`` for the entry.

A running total moves every statistic because it changes the values being
reduced rather than which of them are kept. It also stays in exact integer
arithmetic, so no variant here rests on a floating-point tolerance.
"""

from __future__ import annotations

import numpy as np
from specs import Variant

# (statistic, axis). Order is the round-robin order and is part of the split
# contract -- appending is safe, reordering silently changes which task gets
# which problem.
VARIANT_KEYS: list[str] = [
    "mean-columns",
    "mean-rows",
    "median-columns",
    "median-rows",
    "sum-columns",
    "max-columns",
    "min-rows",
    "range-columns",
]

_STATISTICS = {
    # name -> (English, NumPy reducer, Octave expression over a trimmed matrix T
    # reduced along `dim`)
    "mean": ("arithmetic mean", np.mean, "mean({T}, {dim})"),
    "median": ("median", np.median, "median({T}, {dim})"),
    "sum": ("sum", np.sum, "sum({T}, {dim})"),
    "max": ("maximum", np.max, "max({T}, [], {dim})"),
    "min": ("minimum", np.min, "min({T}, [], {dim})"),
    "range": (
        "range (maximum minus minimum)",
        lambda a, axis: np.max(a, axis=axis) - np.min(a, axis=axis),
        "max({T}, [], {dim}) - min({T}, [], {dim})",
    ),
}


def _parse(key: str) -> tuple[str, str]:
    statistic, _, axis = key.partition("-")
    if statistic not in _STATISTICS or axis not in ("columns", "rows"):
        raise ValueError(f"unknown reduce_along_dim variant {key!r}")
    return statistic, axis


def _describe(key: str, level: int) -> str:
    """The prompt text. Every level restates its own task in full.

    Level 3 saying only "...without loops" is what let `struct_cell_wrangle`
    level 3 fall from 0.792 to 0.000 while models guessed at the task from the
    family name alone. Guarded by
    `test_level_three_restates_its_own_task_for_every_problem`, which reads the
    generated prompt rather than a table, so it sees all eight variants.
    """
    statistic, axis = _parse(key)
    english = _STATISTICS[statistic][0]
    slice_word = "column" if axis == "columns" else "row"
    if level == 1:
        return f"Return the {english} of each {slice_word} of A."
    direction = "top to bottom" if axis == "columns" else "left to right"
    running = (
        f"Replace each {slice_word} of A by its running total from {direction}, "
        f"so that entry i of a {slice_word} becomes the sum of entries 1 through "
        f"i of that {slice_word}. Then return the {english} of each "
        f"{slice_word} of the result."
    )
    if level == 2:
        return running
    return running + " Do not use for/while loops."


def build(rng: np.random.Generator, level: int, key: str) -> Variant:
    statistic, axis = _parse(key)
    reducer = _STATISTICS[statistic][1]
    # NumPy axis of the reduction: "each column" reduces down rows (axis 0).
    np_axis = 0 if axis == "columns" else 1
    # Octave dim argument is 1-based and matches: dim 1 reduces columns.
    dim = np_axis + 1

    cases: list[dict] = []
    for _ in range(6):
        # One draw pattern for every variant and level, so a variant selection
        # cannot shift the shared rng stream. At least three entries along the
        # reduced axis, so a running total is not a near-no-op.
        rows = int(rng.integers(3, 7)) if np_axis == 0 else int(rng.integers(2, 5))
        columns = int(rng.integers(2, 5)) if np_axis == 0 else int(rng.integers(3, 7))
        A = rng.integers(-9, 15, (rows, columns))

        values = A.astype(float)
        if level > 1:
            values = np.cumsum(values, axis=np_axis)
        out = reducer(values, axis=np_axis)
        # "Each row" reduces to one value per row, which Octave returns as a
        # column. Match that, or the natural solution fails on orientation
        # alone -- the defect this project has shipped three times.
        expected = out.reshape(-1, 1) if np_axis == 1 else out.reshape(1, -1)
        cases.append({"args": [A.tolist()], "expected": expected.tolist()})

    signature = "function out = reduce_along_dim(A)"
    expression = _STATISTICS[statistic][2]

    # The naive reading of "the mean of each column" is `mean(A)`, whose default
    # dim is 1. Writing the naive solution the lazy way is the point: if the
    # graded orientation disagrees with it, this fails and the variant does not
    # ship.
    def _lazy(text: str) -> str:
        if dim != 1:
            return text
        return text.replace("(A, 1)", "(A)").replace("(A, [], 1)", "(A)")

    if level == 1:
        body = " out = " + expression.format(T="A", dim=dim) + ";"
        natural = " out = " + _lazy(expression.format(T="A", dim=dim)) + ";"
    else:
        running = f"cumsum(A, {dim})"
        body = " C = " + running + ";\n out = " + expression.format(T="C", dim=dim) + ";"
        lazy_running = "cumsum(A)" if dim == 1 else running
        natural = (
            " C = " + lazy_running + ";\n out = "
            + _lazy(expression.format(T="C", dim=dim)).replace("C, 1)", "C)")
            .replace("C, [], 1)", "C)") + ";"
        )

    reference = f"{signature}\n{body}\nendfunction"
    natural_source = f"{signature}\n{natural}\nendfunction"

    return Variant(
        key=key,
        description=_describe(key, level),
        signature=signature,
        cases=cases,
        reference=reference,
        natural=natural_source,
        vectorized=level == 3,
    )
