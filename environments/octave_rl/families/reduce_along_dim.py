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

## The level ladder, and why it is trimming

The ladder is retained from 0.4.x: level 3 is level 2 plus a vectorization
constraint, and level 2 is level 1 plus a parameter. Trimming the ``k`` largest
**and** ``k`` smallest values is what generalises across all eight variants
without going degenerate. A one-sided trim would have: drop the k largest and
``min`` is unchanged, drop the k smallest and ``max`` is unchanged. Either way
two of the eight variants would render a level-2 prompt whose answer is
identical to its level 1 -- a distinct prompt that is not a distinct problem,
which is the exact failure this whole change exists to remove.

Hidden cases guarantee the slice is long enough that trimming leaves at least
two values, so no case is decided by an empty-reduction edge case that the
prompt does not describe.
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
    `test_level_three_descriptions_restate_their_own_task`.
    """
    statistic, axis = _parse(key)
    english = _STATISTICS[statistic][0]
    slice_word = "column" if axis == "columns" else "row"
    if level == 1:
        return f"Return the {english} of each {slice_word} of A."
    trimmed = (
        f"Return the {english} of each {slice_word} of A, after discarding the "
        f"k largest and the k smallest values of that {slice_word}. Ties count "
        f"separately, so a {slice_word} always loses exactly 2*k values."
    )
    if level == 2:
        return trimmed
    return trimmed + " Do not use for/while loops."


def _trim(slice_values: np.ndarray, k: int) -> np.ndarray:
    """Drop the k largest and k smallest, counting ties separately.

    Sorting and slicing by position is what "ties count separately" means, and
    it is why the description says so: dropping *values* equal to the extremes
    would remove a variable number of entries and the answer would depend on a
    convention the prompt never stated.
    """
    ordered = np.sort(slice_values)
    return ordered[k : len(ordered) - k]


def build(rng: np.random.Generator, level: int, key: str) -> Variant:
    statistic, axis = _parse(key)
    reducer = _STATISTICS[statistic][1]
    # NumPy axis of the reduction: "each column" reduces down rows (axis 0).
    np_axis = 0 if axis == "columns" else 1
    # Octave dim argument is 1-based and matches: dim 1 reduces columns.
    dim = np_axis + 1

    cases: list[dict] = []
    for _ in range(6):
        if level == 1:
            k = 0
            rows = int(rng.integers(3, 7))
            columns = int(rng.integers(2, 5))
        else:
            k = int(rng.integers(1, 3))
            # Long enough along the reduced axis that trimming leaves >= 2
            # values, so no hidden case turns on an empty reduction.
            minimum = 2 * k + 2
            rows = int(rng.integers(minimum, minimum + 4)) if np_axis == 0 else int(rng.integers(2, 5))
            columns = int(rng.integers(2, 5)) if np_axis == 0 else int(rng.integers(minimum, minimum + 4))
        A = rng.integers(-9, 15, (rows, columns))

        if k == 0:
            out = reducer(A.astype(float), axis=np_axis)
        else:
            trimmed = np.apply_along_axis(_trim, np_axis, A.astype(float), k)
            out = reducer(trimmed, axis=np_axis)
        # "Each row" reduces to one value per row, which Octave returns as a
        # column. Match that, or the natural solution fails on orientation
        # alone -- the defect this project has shipped three times.
        expected = out.reshape(-1, 1) if np_axis == 1 else out.reshape(1, -1)
        args = [A.tolist()] if level == 1 else [A.tolist(), k]
        cases.append({"args": args, "expected": expected.tolist()})

    signature = (
        "function out = reduce_along_dim(A)"
        if level == 1
        else "function out = reduce_along_dim(A, k)"
    )
    expression = _STATISTICS[statistic][2]

    if level == 1:
        body = " out = " + expression.format(T="A", dim=dim) + ";"
        # The natural reading of "the mean of each column" is `mean(A)`, whose
        # default dim is 1. Writing the naive solution the lazy way is the
        # point: if the graded orientation disagrees with it, this fails and
        # the variant does not ship.
        natural_expr = expression.format(T="A", dim=dim)
        if dim == 1:
            natural_expr = (
                expression.format(T="A", dim=dim)
                .replace("(A, 1)", "(A)")
                .replace("(A, [], 1)", "(A)")
            )
        natural = " out = " + natural_expr + ";"
    else:
        # Sort along the reduced axis, drop k from each end, then reduce.
        sort_dim = dim
        trim_index = (
            "S(k+1:end-k, :)" if dim == 1 else "S(:, k+1:end-k)"
        )
        body = (
            f" S = sort(A, {sort_dim});\n"
            f" T = {trim_index};\n"
            " out = " + expression.format(T="T", dim=dim) + ";"
        )
        natural = body

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
