"""``struct_cell_wrangle``: a stack of several statistics, one slice at a time.

Written against the worked exemplar in ``reduce_along_dim.py`` -- same two
public names, same level-ladder discipline. Read ``specs.py`` first for why the
variant is an argument rather than an rng draw, and for the rule that
``description``/``reference``/``natural`` are written together so they cannot
drift.

**The family name is a lie and no description here may lean on it.** Nothing in
this family involves a struct or a cell array; every task is a plain numeric
matrix. The name is retained only because task ids and earlier measurements are
keyed on it. It is also the direct cause of this family's worst shipped defect:
level 3 read "Return [column minima; column maxima], without for/while loops"
with the sentence defining ``A`` dropped, so the model's only remaining clue was
the function name, and models duly wrote cell-array code -- 0.792 at level 2 to
**0.000** at level 3 (PIPELINE_LOG, 2026-08-09). Every description below states
what the inputs are and what the output layout is, at every level, without
reference to the family name.

## The spec

Two dimensions, per ``PARAMETERIZATION_DESIGN.md``: which set of statistics is
stacked, and which axis the slices run along. Four sets times two axes is
exactly the eight wanted, so nothing is dropped for count -- the choices below
are about *which* four sets.

The axis is not decoration here, because the stack's layout follows it:

- ``columns``: one column of the answer per column of the input, statistics
  down the rows -- a ``k``-by-N matrix, ``[min; max]``;
- ``rows``: one row of the answer per row of the input, statistics across the
  columns -- an M-by-``k`` matrix, ``[min, max]``.

Both are genuinely 2-D, which is where column-major flattening bites (see
PIPELINE_LOG 2026-08-08), and ``size(actual)`` is compared exactly. The layout
is therefore stated outright in every description, the way ``broadcast_arith``
states its grid orientation: an unstated stacking convention is an undisclosed
convention, and the transposed answer is the single most likely wrong answer a
competent reader produces.

## What ``count`` had to become

``PARAMETERIZATION_DESIGN.md`` proposes ``[sum; count]``. Taken literally,
"count" means the number of entries in the slice, which is the same number for
every slice, the same number at every level, and equal to ``size(A, 1)`` -- a
row of the answer that no computation can get wrong, that the level-2 step
cannot move, and that a solver can read off the input's shape. It ships here as
**the number of entries strictly greater than zero**, which varies by slice,
moves under the level-2 step, and stays in exact integer arithmetic.

## ``std``: shipped, with the divisor disclosed

Octave's ``std`` normalises by ``N-1``; NumPy's default normalises by ``N``.
``sliding_window`` rejected ``std`` outright for that reason and was right to,
given how many words its windowing description already spends. Here the
description is short enough to afford the disclosure, so ``meanstd`` ships and
every rendering of it says *in the prompt* that the divisor is the number of
entries minus one. The NumPy side uses ``ddof=1`` to match. Undisclosed
conventions are the defect; a *disclosed* convention is just a specification.

The match was confirmed by execution rather than by reading either manual.
Running the reference through the local executor and comparing the values Octave
reported with the values NumPy computed, over 720 hidden cases (both ``meanstd``
variants, all three levels, 20 seeds), the worst relative difference is
**exactly 0.0** -- Octave 10.2.0 and NumPy return bit-identical doubles for both
``mean`` and ``std`` on these inputs, which are sums of integers below 20 in
magnitude divided by a count of 3 to 6. The inherited 1e-9 relative tolerance is
therefore doing no work at all here, and the other six variants are integer-exact
end to end. Had the divisor gone undisclosed, an ``N``-normalised answer would
have missed by 10-20% relative, which no tolerance forgives -- the failure would
have looked like arithmetic, not like a convention.

## The level ladder, and why it is deliberately not a running total

Level 3 is level 2 plus a vectorization constraint, restating its own task in
full. Level 2 introduces a **second matrix ``B`` of the same size and asks for
the statistics of the elementwise difference ``D = A - B``.**

**This family overlaps ``reduce_along_dim`` and the ladder is where that had to
be resolved.** Both reduce a matrix along an axis; if this one also stepped from
level 1 to level 2 by transforming each slice with a running total and reducing
again, the two families would differ only by how many statistics get stacked,
and the family holdout -- which costs a fifth of training coverage to buy one
held-out problem -- would be buying a problem the model had already trained on
in another file. So the running total is left to ``reduce_along_dim`` and to
``broadcast_arith``, and dilation to ``sliding_window``.

Aligning two inputs elementwise is a different *shape* of problem rather than a
different transform of one input: the arity changes, the reader has to notice
the two matrices are conformant, and the derived matrix is not a scan along the
axis being reduced. It also keeps every statistic in play -- it changes the
values being summarised rather than which of them survive, which is the property
the exemplar's abandoned trim lacked.

**Three level-2 steps rejected, each degenerate for a specific row of the stack:**

- *Subtract each slice's mean.* The prettiest option and fatal: ``std`` is
  shift-invariant, so the ``std`` row of ``meanstd`` would be identical to its
  level 1 by construction, and the ``mean`` row would be the constant zero. Two
  variants would have had a level 2 that was strictly *less* informative than
  their level 1.
- *Keep only the rows a mask selects.* Changes which values survive rather than
  the values, so it is the exemplar's trim wearing a different hat: the minimum
  of a subset equals the minimum of the whole slice whenever the extreme entry
  is selected, which for a 5-row matrix keeping 3 rows is 60% of entries of the
  ``min`` row.
- *Successive differences along each slice.* Moves every statistic and is exact,
  but it is ``cumsum`` read backwards -- transform each slice, then reduce -- and
  so lands back on ``reduce_along_dim``'s ladder, which is the thing this family
  most needs to not be.

## Measuring that level 2 is a different problem, since no test can

The repository's guard, ``test_no_variant_has_a_level_two_its_level_one_solution
_already_solves``, runs the level-1 *natural solution* against the level-2
cases. **For this family that guard is vacuous**: level 1 takes one argument and
level 2 passes two, so Octave raises "called with too many inputs" and the
fraction is 0.0 for reasons that have nothing to do with the mathematics. (It is
already vacuous for ``sliding_window``, which likewise gains an argument at
level 2.) A green check there is not evidence, and it must not be read as any.

The check that is evidence is a solver that *takes both arguments and ignores
``B``* -- exactly what a model that learned level 1 and skimmed level 2 would
write. Measured on the pinned Octave 10.2.0, all eight variants, 40 seeds per
variant, six hidden cases per seed:

    ignore-B solver:  0 / 1920 hidden cases passed, 0 / 240 for every variant

Beside it, an entrywise census of the same 1,920 level-2 answers against the
level-1 statistic of the same ``A``. No variant's whole answer coincides on a
single case (0/240 each), and no variant's summary of a single slice coincides
on more than 2.3% of slices. Per row of the stack:

    min 6.2-6.9%   max 6.2-6.9%   median 6.0-8.1%
    mean 2.8-4.0%  std 0.5%       count 39.6-43.7%

**The ``count`` row is the weak one and is called out rather than buried.** A
count of strictly positive entries among 3 to 6 entries takes about seven
values, so two *unrelated* matrices already agree on it 26.4% of the time (that
floor is measured, over 20,000 independent pairs); ``D = A - B`` is correlated
with ``A``, which lifts it to 40.7%. It is a low-entropy statistic, not a
ladder that leaves the row fixed -- the step moves every row, and the whole
answer is never unchanged -- but a model that ignored ``B`` would get that one
row right about two cases in five, and no ladder can push a small count much
below its own chance floor.
"""

from __future__ import annotations

import numpy as np
from specs import Variant

# (statistic set, axis). Order is the round-robin order and is part of the split
# contract -- appending is safe, reordering silently changes which task gets
# which problem.
#
# The order groups by axis rather than by statistic set so that
# `DEFAULT_HELDOUT_VARIANTS` -- the last two keys of each family -- holds out two
# *different* statistic sets on one axis. Held out that way, both sets are still
# trained on (via their `columns` variants) and the `rows` layout is still
# trained on (via `minmax-rows` and `meanstd-rows`), so the holdout tests
# recombination rather than asking for a statistic the model has never seen.
# Grouping by set instead would have made the default holdout delete
# `sumcount` from training entirely.
VARIANT_KEYS: list[str] = [
    "minmax-columns",
    "meanstd-columns",
    "minmedmax-columns",
    "sumcount-columns",
    "minmax-rows",
    "meanstd-rows",
    "minmedmax-rows",
    "sumcount-rows",
]

# name -> (English phrase completed by the slice word, NumPy reducer over an
# axis, Octave form with the dimension left implicit, Octave form over dim 1,
# Octave form over dim 2).
#
# The implicit form is what the `natural` solution uses on the `columns` axis --
# `min(A)` reduces dim 1 for a matrix, and every hidden case has at least two
# rows, so the vector special case never arises. There is no lazier way to write
# a row-wise reduction than naming dim 2, so the `rows` axis uses the same text
# in both solutions.
_STATISTICS = {
    "min": (
        "the minimum of that {slice}",
        np.min,
        "min({M})",
        "min({M}, [], 1)",
        "min({M}, [], 2)",
    ),
    "max": (
        "the maximum of that {slice}",
        np.max,
        "max({M})",
        "max({M}, [], 1)",
        "max({M}, [], 2)",
    ),
    "mean": (
        "the arithmetic mean of that {slice}",
        np.mean,
        "mean({M})",
        "mean({M}, 1)",
        "mean({M}, 2)",
    ),
    # The divisor is named in the prompt, not assumed. NumPy's default is `ddof=0`
    # and Octave's is `N-1`; a description that said only "standard deviation"
    # would be decided by which of the two the reader happened to have in mind.
    # `std(M, 0, 2)` is the row-wise spelling -- the middle argument is the
    # normalisation selector, so `std(M, 2)` is an error rather than a dimension.
    "std": (
        (
            "the standard deviation of that {slice}, using the divisor "
            "(number of entries) - 1 rather than the number of entries, as "
            "Octave's std does by default"
        ),
        lambda values, axis: np.std(values, axis=axis, ddof=1),
        "std({M})",
        "std({M}, 0, 1)",
        "std({M}, 0, 2)",
    ),
    "median": (
        "the median of that {slice}",
        np.median,
        "median({M})",
        "median({M}, 1)",
        "median({M}, 2)",
    ),
    "sum": (
        "the sum of that {slice}",
        np.sum,
        "sum({M})",
        "sum({M}, 1)",
        "sum({M}, 2)",
    ),
    # Not "the number of entries", which is `size(A, 1)` for every column of
    # every case at every level. See the module docstring.
    "count": (
        "the number of entries of that {slice} that are strictly greater than zero",
        lambda values, axis: np.sum(values > 0, axis=axis),
        "sum({M} > 0)",
        "sum({M} > 0, 1)",
        "sum({M} > 0, 2)",
    ),
}

# Which statistics each set stacks, in the order they occupy the stack. The
# order is graded -- it is the row order of the answer on the `columns` axis and
# the column order on the `rows` axis -- so it is stated in the description
# position by position rather than left to the reader.
_SETS: dict[str, tuple[str, ...]] = {
    "minmax": ("min", "max"),
    "meanstd": ("mean", "std"),
    "minmedmax": ("min", "median", "max"),
    "sumcount": ("sum", "count"),
}


def _parse(key: str) -> tuple[tuple[str, ...], str]:
    names, _, axis = key.partition("-")
    if names not in _SETS or axis not in ("columns", "rows"):
        raise ValueError(f"unknown struct_cell_wrangle variant {key!r}")
    return _SETS[names], axis


def _describe(key: str, level: int) -> str:
    """The prompt text. Every level restates its own task in full.

    Level 3 saying only "...without loops" is what let this very family's level 3
    fall from 0.792 to 0.000 while models guessed at cell arrays from the family
    name. Level 3 here is level 2 verbatim plus the loop clause, so it cannot
    drop a term. Guarded by
    `test_level_three_restates_its_own_task_for_every_problem`, which reads the
    generated prompt rather than a table, so it sees all eight variants.

    The output *shape* sentence is appended by the prompt builder from the
    expected values, so it cannot drift from the grader. The output *layout* --
    which statistic lands in which row or column -- cannot be derived from a
    shape, so it is spelled out here, position by position.
    """
    statistics, axis = _parse(key)
    slice_word = "column" if axis == "columns" else "row"
    if level == 1:
        preamble = (
            "A is a numeric matrix with at least two rows and at least two "
            "columns."
        )
        source = "A"
    else:
        preamble = (
            "A and B are numeric matrices of the same size, with at least two "
            "rows and at least two columns. Let D be their elementwise "
            "difference, D = A - B."
        )
        source = "D"
    entries = [
        _STATISTICS[name][0].format(slice=slice_word) for name in statistics
    ]
    if axis == "columns":
        layout = (
            f"Return a matrix with {len(statistics)} rows and one column per "
            f"column of {source}, in which column j summarises column j of "
            f"{source}: "
            + "; ".join(f"row {i} is {text}" for i, text in enumerate(entries, 1))
            + "."
        )
    else:
        layout = (
            f"Return a matrix with one row per row of {source} and "
            f"{len(statistics)} columns, in which row i summarises row i of "
            f"{source}: "
            + "; ".join(
                f"column {i} is {text}" for i, text in enumerate(entries, 1)
            )
            + "."
        )
    task = f"{preamble} {layout}"
    if level == 3:
        return task + " Do not use for/while loops."
    return task


def build(rng: np.random.Generator, level: int, key: str) -> Variant:
    statistics, axis = _parse(key)
    # NumPy axis of the reduction: "each column" reduces down rows (axis 0), and
    # the stack then grows along that same axis -- statistics down the rows for
    # the column layout, across the columns for the row layout.
    np_axis = 0 if axis == "columns" else 1

    cases: list[dict] = []
    for _ in range(6):
        # One draw pattern for every variant: two shape draws, one matrix, and
        # at level 2 or 3 a second matrix of the same shape. Nothing here reads
        # `key`, so a variant selection cannot shift the shared rng stream --
        # `build_tasks` draws every family's every task from one generator, so a
        # key-dependent draw count would silently reshuffle every later task.
        #
        # Both dimensions are at least 3: at least two so every graded answer is
        # a genuine 2-D matrix and the implicit-dimension `min(A)` in the natural
        # solution means dim 1, and at least three so the reduced slice is never
        # so short that a median or a standard deviation is decided by two
        # numbers.
        rows = int(rng.integers(3, 7))
        columns = int(rng.integers(3, 7))
        A = rng.integers(-9, 10, (rows, columns))

        args: list = [A.tolist()]
        values = A.astype(float)
        if level > 1:
            B = rng.integers(-9, 10, (rows, columns))
            args.append(B.tolist())
            values = values - B

        stacked = [_STATISTICS[name][1](values, np_axis) for name in statistics]
        # Each reducer returns one value per slice; stacking along the reduced
        # axis puts the statistics down the rows for `columns` (k-by-N) and
        # across the columns for `rows` (M-by-k), which is the layout the
        # description states. `expected` nests row by row, which is what the
        # grader's `_octave_shape`/`_octave_flatten` read.
        out = np.stack(stacked, axis=np_axis)
        cases.append({"args": args, "expected": out.tolist()})

    parameters = "A" if level == 1 else "A, B"
    signature = f"function out = struct_cell_wrangle({parameters})"

    source = "A" if level == 1 else "D"

    def _stack(forms: list[str]) -> str:
        # `[a; b]` stacks 1-by-N rows into k-by-N; `[a, b]` stacks M-by-1
        # columns into M-by-k. Getting this the wrong way round produces a
        # same-valued, wrong-shaped answer, which the grader rejects on
        # `size(actual)` -- which is why the description names the layout.
        separator = "; " if axis == "columns" else ", "
        return "[" + separator.join(form.format(M=source) for form in forms) + "]"

    explicit_index = 3 if np_axis == 0 else 4
    implicit_index = 2 if np_axis == 0 else 4
    reference_stack = _stack([_STATISTICS[n][explicit_index] for n in statistics])
    natural_stack = _stack([_STATISTICS[n][implicit_index] for n in statistics])

    # There is nothing to coerce: both arguments arrive as matrices of the shape
    # the description states, and every expression below is written against that
    # shape. The reference differs from the natural solution only by naming the
    # reduction dimension explicitly where the natural solution leaves Octave's
    # default to do the work.
    prologue = "" if level == 1 else " D = A - B;\n"
    reference = f"{signature}\n{prologue} out = {reference_stack};\nendfunction"
    # What a competent Octave programmer writes from the description alone: no
    # `(:)`, no reshape, no transpose the prompt did not ask for. If this cannot
    # pass, the variant is not shippable.
    natural = f"{signature}\n{prologue} out = {natural_stack};\nendfunction"

    return Variant(
        key=key,
        description=_describe(key, level),
        signature=signature,
        cases=cases,
        reference=reference,
        natural=natural,
        # 1e-9 relative, inherited. Six of the eight variants are integer-exact;
        # `meanstd` is the only one where Octave and NumPy can disagree at all,
        # and they disagree at ~1e-16 relative. See the module docstring.
        vectorized=level == 3,
    )
