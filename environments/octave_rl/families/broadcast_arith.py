"""``broadcast_arith``: an outer grid of one binary operation, then running totals.

Written in the 0.5.0 variant form described in ``specs.py`` and modelled on
``families/reduce_along_dim.py``, the worked exemplar. The module exposes
exactly two names: ``VARIANT_KEYS`` and ``build(rng, level, key)``.

## The spec

One dimension, per ``PARAMETERIZATION_DESIGN.md``: which binary operation
combines ``a(i)`` with ``b(j)``. The design lists six -- sum, difference,
product, squared difference, absolute difference, max -- and eight are wanted,
so ``min`` and ``sum of squares`` join them. Both were picked to stay inside
exact integer arithmetic: every hidden expected value is an integer that Octave
and NumPy compute identically, so no case can be decided by a last-ulp
disagreement between the two. ``hypot`` -- ``sqrt(a(i)^2 + b(j)^2)`` -- would
have been the more obvious eighth and was passed over for exactly that reason;
it is shippable, it is merely not free.

**Rejected: a quotient variant.** ``a(i) / b(j)`` is the natural ninth
operation and cannot ship as the others are drawn. ``b`` may contain a zero, so
a hidden case would return ``Inf``, which ``candidate_record_matches`` scores as
a failure however the model writes it -- a case decided by a degenerate edge the
prompt never mentions. Excluding zero from ``b`` would fix it only by imposing a
draw restriction on all eight variants for the benefit of one.

## The level ladder, and why it is a running total down each column

Level 1 is the outer grid itself. Level 2 keeps that grid and replaces each
column by its running total from top to bottom; level 3 is level 2 plus the
vectorization constraint, restating its own task in full.

The ladder has to compose across all eight operations without any of them going
degenerate at level 2, and that is a real constraint here, because five of the
eight are **separable**: for ``sum`` and ``difference``, ``f(a(i), b(j))`` splits
into a term in ``i`` plus a term in ``j``.

**Rejected ladder: subtract each row's mean.** It is the prettiest option -- a
second broadcast, which is this family's whole subject -- and it is degenerate
for the separable operations. Centring row ``i`` of the sum grid gives
``b(j) - mean(b)``: the ``a`` dependence cancels exactly, every row of the answer
is identical, and a solution that ignored ``a`` entirely would score 6/6. That is
a distinct prompt which is not a distinct problem, the same failure mode the
exemplar rejects a one-sided trim for.

A running total down each column survives the separable case: the sum grid
accumulates to ``cumsum(a)(i) + i*b(j)``, which depends on both inputs and is
not the level-1 answer for any operation. It is also exact in integers, and it
is orientation-sensitive in a family whose orientation is graded -- accumulating
along the wrong dimension gives a same-sized, wrong-valued answer.

Measured over 2,000 tasks per variant: no task has a level-2 answer equal to
its level-1 answer on more than 2 of its 6 cases, so a level-1 solution never
scores full marks at level 2 for any of the eight. The residual coincidences are
draws where the grid is all zeros (``b`` all zeros makes every running total
zero too), which is an ordinary draw rather than an undescribed edge -- correct
code passes it either way.

The same census flags one honest weakness: for ``max`` and ``min``, 12% of cases
draw an ``a`` and a ``b`` that do not interleave, and on those the answer does
not depend on the dominated vector's individual values. Worst observed is 4 of 6
such cases in a task, so no task is decided by it, and it is inherent to the two
operations rather than to how they are drawn.

Hidden cases always draw at least two elements for each of ``a`` and ``b``, so
the graded result is genuinely 2-D (which is where column-major flattening
bites, see PIPELINE_LOG 2026-08-08) and the bare ``cumsum(M)`` a competent
reader writes -- default dimension 1 for a matrix -- is the right one. No case
turns on a length-1 or empty vector, neither of which the prompt describes.

## Two defects this family has already shipped

Both are recorded in ``PIPELINE_LOG.md`` and both are constraints on this file.

1. **The expected value was flattened row by row while Octave reports
   ``actual(:)'`` column by column** (2026-08-08). Every matrix-valued task
   scored zero however correct the answer. Fixed in ``harness._octave_flatten``,
   but it is why this family's output must remain a real matrix and why
   ``expected`` is nested row-by-row exactly as the grader's ``_octave_shape``
   reads it.
2. **``a`` and ``b`` were both serialised as rows** while the prompt called ``a``
   a column vector (2026-08-09), so bare ``a + b`` was nonconformant and only the
   reference's undisclosed ``a(:) + b(:)'`` ran. ``a`` is therefore serialised as
   a column -- a list of one-element lists -- and ``b`` as a flat list, which is
   a row. Preserve this: it is what makes the ``natural`` field below run at all.
"""

from __future__ import annotations

import numpy as np
from specs import Variant

# The binary operation applied to every (a(i), b(j)) pair. Order is the
# round-robin order and part of the split contract -- appending is safe,
# reordering silently changes which task gets which problem. The first six are
# the operations `PARAMETERIZATION_DESIGN.md` proposes, in its order.
VARIANT_KEYS: list[str] = [
    "sum",
    "difference",
    "product",
    "squared-difference",
    "absolute-difference",
    "max",
    "min",
    "sum-of-squares",
]

_OPERATIONS = {
    # key -> (English name of the grid, the (i,j) entry as the prompt states it,
    # NumPy operation on a column and a row, Octave expression over operands
    # {a} and {b}).
    #
    # The Octave column is written once and used for both the reference and the
    # natural solution, which differ only in what is substituted for {a}/{b}:
    # the reference gets coerced operands, the natural solution gets the bare
    # arguments. Both broadcast in GNU Octave 10.2.0 -- including `max`/`min`,
    # which the manual lists among the broadcasting functions and which was
    # confirmed on the pinned interpreter rather than assumed.
    "sum": (
        "pairwise sums",
        "a(i) + b(j)",
        lambda a, b: a + b,
        "{a} + {b}",
    ),
    "difference": (
        "pairwise differences",
        "a(i) - b(j)",
        lambda a, b: a - b,
        "{a} - {b}",
    ),
    "product": (
        "pairwise products",
        "a(i) * b(j)",
        lambda a, b: a * b,
        "{a} .* {b}",
    ),
    "squared-difference": (
        "squared pairwise differences",
        "(a(i) - b(j))^2",
        lambda a, b: (a - b) ** 2,
        "({a} - {b}) .^ 2",
    ),
    "absolute-difference": (
        "absolute pairwise differences",
        "abs(a(i) - b(j))",
        lambda a, b: np.abs(a - b),
        "abs({a} - {b})",
    ),
    "max": (
        "pairwise maxima",
        "max(a(i), b(j))",
        np.maximum,
        "max({a}, {b})",
    ),
    "min": (
        "pairwise minima",
        "min(a(i), b(j))",
        np.minimum,
        "min({a}, {b})",
    ),
    "sum-of-squares": (
        "pairwise sums of squares",
        "a(i)^2 + b(j)^2",
        lambda a, b: a**2 + b**2,
        "{a} .^ 2 + {b} .^ 2",
    ),
}


def _parse(key: str) -> tuple[str, str, object, str]:
    if key not in _OPERATIONS:
        raise ValueError(f"unknown broadcast_arith variant {key!r}")
    return _OPERATIONS[key]


def _describe(key: str, level: int) -> str:
    """The prompt text. Every level restates its own task in full.

    Level 3 saying only "...without loops" is what let `struct_cell_wrangle`
    level 3 fall from 0.792 to 0.000 while models guessed at the task from the
    family name alone. Guarded by
    `test_level_three_descriptions_restate_their_own_task`.

    Each description also names the orientation of the result outright -- one
    row per element of `a`, one column per element of `b`. `size(actual)` is
    compared exactly, so orientation is graded, and the generated shape sentence
    can only say "Return a 2-D matrix." for this family because the row count
    varies from case to case. Stating it here is the difference between a graded
    convention and an undisclosed one.
    """
    english, formula, _, _ = _parse(key)
    grid = (
        f"the matrix of {english} of the column vector a and the row vector b, "
        f"which has one row per element of a and one column per element of b, "
        f"and whose (i,j) entry is {formula}"
    )
    if level == 1:
        return f"Return {grid}."
    accumulated = (
        f"Let M be {grid}. Return M with each column replaced by its running "
        f"total from top to bottom, so that entry (i,j) of the result is the "
        f"sum of M(1,j) through M(i,j). The result has the same size as M."
    )
    if level == 2:
        return accumulated
    return accumulated + " Do not use for/while loops."


def build(rng: np.random.Generator, level: int, key: str) -> Variant:
    _, _, operation, expression = _parse(key)

    cases: list[dict] = []
    for _ in range(6):
        # Drawn before the vectors so the rng is consumed in one fixed pattern:
        # two lengths, then that many values each. Nothing here reads `key`, so
        # changing the variant selection cannot shift the stream. Both lengths
        # are at least 2, which keeps every graded result genuinely 2-D.
        n = int(rng.integers(2, 6))
        m = int(rng.integers(2, 6))
        a = rng.integers(-8, 9, n)
        b = rng.integers(-8, 9, m)

        out = operation(a.reshape(-1, 1), b.reshape(1, -1))
        if level > 1:
            out = np.cumsum(out, axis=0)
        # `a` as a column (a list of one-element lists) and `b` as a flat list,
        # which `octave_literal` renders as a row. This is what makes the
        # prompt's "column vector a and row vector b" literally true and lets
        # bare `a + b` broadcast; sending both as rows is the 2026-08-09 defect.
        cases.append(
            {
                "args": [a.reshape(-1, 1).tolist(), b.tolist()],
                "expected": out.tolist(),
            }
        )

    signature = "function out = broadcast_arith(a, b)"

    # The reference may coerce defensively; it does so once, up front, so the
    # operation itself is written identically in both solutions below.
    grid_reference = expression.format(a="A", b="B")
    if level == 1:
        body = f" A = a(:); B = b(:)';\n out = {grid_reference};"
    else:
        body = f" A = a(:); B = b(:)';\n out = cumsum({grid_reference}, 1);"

    # What a competent Octave programmer writes from the description alone: no
    # `(:)`, no reshape, no transpose the prompt did not ask for, and the lazy
    # `cumsum(M)` rather than `cumsum(M, 1)` -- the default dimension for a
    # matrix, and correct because every hidden case has at least two rows. If
    # this cannot pass, the variant is not shippable.
    grid_natural = expression.format(a="a", b="b")
    if level == 1:
        natural = f" out = {grid_natural};"
    else:
        natural = f" M = {grid_natural};\n out = cumsum(M);"

    return Variant(
        key=key,
        description=_describe(key, level),
        signature=signature,
        cases=cases,
        reference=f"{signature}\n{body}\nendfunction",
        natural=f"{signature}\n{natural}\nendfunction",
        vectorized=level == 3,
    )
