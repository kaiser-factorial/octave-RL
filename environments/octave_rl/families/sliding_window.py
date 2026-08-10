"""``sliding_window``: a statistic over the windows of a vector.

Written against the worked exemplar in ``reduce_along_dim.py`` -- same two
public names, same level-ladder discipline. Read ``specs.py`` first for why the
variant is an argument rather than an rng draw, and for the rule that
``description``/``reference``/``natural`` are written together so they cannot
drift.

## The spec

Two dimensions, per ``PARAMETERIZATION_DESIGN.md``: which statistic, and whether
the stride is pinned at 1 or arrives as an argument. Eight of the twelve
combinations ship, chosen the way the exemplar chose its eight -- both stride
forms for ``mean`` and ``median``, the two statistics whose value depends on
*which* elements a window holds rather than only on how many, and the other four
once each, split evenly between the two stride forms so neither form is a
minority dialect.

``std`` and ``mode`` were considered as further statistics and both were
**rejected as un-shippable**. Octave's ``std`` normalises by ``N-1`` and NumPy's
by ``N``, so "the standard deviation of each window" is exactly the kind of
undisclosed convention this rewrite exists to delete; a prompt that stated the
normalisation would spend more words on the divisor than on the windowing.
``mode`` is worse: short integer windows tie constantly, and which of the tied
values wins is a convention rather than a computation, so its expected values
would be decided by a rule the prompt does not carry.

## Windowing semantics are the whole risk here

The window width, the stride, where a window starts and *how many windows there
are* are all conventions, and each is an off-by-one waiting to happen. Every
description therefore states four things in the same order:

1. what the window starting at index ``p`` contains, element by element;
2. which ``p`` a window may start at, written as an explicit list ``1, 1+s, ...``;
3. the condition that keeps a window, written as an inequality on its last
   index rather than as the phrase "valid windows only";
4. the resulting number of windows, as a closed-form expression.

A reader who lands on a different count than the grader is disagreeing with an
arithmetic formula printed in the prompt, not guessing at an unstated intent.
Point 4 is the one the 0.4.x description omitted -- "valid windows only" leaves
a reader to derive ``numel(x)-w+1`` and gives them nothing to check it against.

## The level ladder, and why it is dilation

Level 3 is level 2 plus a vectorization constraint (and restates its own task in
full, since a bare "...without loops" is what dropped ``struct_cell_wrangle``
level 3 from 0.792 to 0.000). Level 2 is level 1 plus a parameter ``d``: the
window still holds ``w`` elements, but they are spaced ``d`` apart rather than
adjacent, so window ``p`` covers ``x(p), x(p+d), ..., x(p+(w-1)*d)``.

Dilation is what generalises across all eight variants without going degenerate.
The obvious alternative -- reusing the exemplar's trim, "discard the k largest
and k smallest values of each window" -- is **not shippable in this family**:
a symmetric trim of a sorted window leaves the median unchanged, so
``median-stride1`` and ``median-strided`` would render a level-2 prompt whose
answer is identical to their level 1. A distinct prompt that is not a distinct
problem is the precise defect this whole change exists to remove. Dilation moves
the answer for every one of the six statistics, because it changes *which*
elements the window holds rather than how they are summarised.

It also lands level 3 on the idiom this family already uses: the index matrix
``(1:s:(numel(x)-(w-1)*d))' + (0:w-1)*d``, of which the 0.4.x reference's
``(1:s:(numel(x)-w+1))' + (0:w-1)`` is the ``d == 1`` case.

Hidden cases guarantee at least two valid windows, so no case is decided by an
empty or single-window edge that the prompt does not describe.
"""

from __future__ import annotations

import numpy as np
from specs import Variant

# (statistic, stride form). Order is the round-robin order and is part of the
# split contract -- appending is safe, reordering silently changes which task
# gets which problem.
VARIANT_KEYS: list[str] = [
    "mean-stride1",
    "mean-strided",
    "median-stride1",
    "median-strided",
    "sum-stride1",
    "max-strided",
    "min-stride1",
    "range-strided",
]

_STATISTICS = {
    # name -> (English, NumPy reducer over axis 1, Octave form over one window
    # vector `W`, Octave form over a window matrix `M` reduced along dim 2)
    "sum": ("sum", np.sum, "sum({W})", "sum({M}, 2)"),
    "mean": ("arithmetic mean", np.mean, "mean({W})", "mean({M}, 2)"),
    # Even widths are safe: an even-length median is the mean of the two middle
    # values in both NumPy and Octave, so no window width needs avoiding.
    "median": ("median", np.median, "median({W})", "median({M}, 2)"),
    "max": ("maximum", np.max, "max({W})", "max({M}, [], 2)"),
    "min": ("minimum", np.min, "min({W})", "min({M}, [], 2)"),
    "range": (
        "range (maximum minus minimum)",
        lambda a, axis: np.max(a, axis=axis) - np.min(a, axis=axis),
        "max({W}) - min({W})",
        "max({M}, [], 2) - min({M}, [], 2)",
    ),
}

# The largest stride a strided case can draw. Hidden inputs are sized against
# this constant rather than against the case's own `s`, so `numel(x)` -- and
# therefore how many values the case takes from the rng -- is identical for
# every variant. Sizing against `s` itself would make a variant selection shift
# the shared stream, which is what
# `test_a_family_generates_the_same_tasks_whichever_others_are_present` forbids.
_MAX_STRIDE = 3


def _parse(key: str) -> tuple[str, bool]:
    statistic, _, stride = key.partition("-")
    if statistic not in _STATISTICS or stride not in ("stride1", "strided"):
        raise ValueError(f"unknown sliding_window variant {key!r}")
    return statistic, stride == "strided"


def _describe(key: str, level: int) -> str:
    """The prompt text. Every level restates its own task in full.

    Level 3 saying only "...without loops" is what let `struct_cell_wrangle`
    level 3 fall from 0.792 to 0.000 while models guessed at the task from the
    family name alone. Guarded by
    `test_level_three_descriptions_restate_their_own_task`.

    The output shape is deliberately absent: the prompt builder appends the
    shape sentence it derives from the expected values, so the prompt's claim
    and the grader's comparison cannot drift apart.
    """
    statistic, strided = _parse(key)
    english = _STATISTICS[statistic][0]
    if level == 1:
        contents = (
            "The window starting at index p holds the w consecutive elements "
            "x(p), x(p+1), ..., x(p+w-1)."
        )
        last_index = "p+w-1"
        count = "floor((numel(x)-w)/s)+1" if strided else "numel(x)-w+1"
    else:
        contents = (
            "The window starting at index p holds w elements of x spaced d "
            "apart: x(p), x(p+d), x(p+2*d), ..., x(p+(w-1)*d)."
        )
        last_index = "p+(w-1)*d"
        count = (
            "floor((numel(x)-(w-1)*d-1)/s)+1" if strided else "numel(x)-(w-1)*d"
        )
    starts = (
        "Windows start at p = 1, 1+s, 1+2*s, and so on"
        if strided
        else "Windows start at p = 1, 2, 3, and so on"
    )
    # Stating the window count as an expression is what makes an off-by-one a
    # disagreement with the prompt rather than with an unstated intent.
    task = (
        f"x is a row vector. {contents} {starts}, and a window is used only if "
        f"it lies entirely inside x, that is only if {last_index} <= numel(x); "
        f"there are {count} such windows. Return the {english} of each of them, "
        "ordered by increasing p."
    )
    if level == 3:
        return task + " Do not use for/while loops."
    return task


def build(rng: np.random.Generator, level: int, key: str) -> Variant:
    statistic, strided = _parse(key)
    _, reducer, window_form, matrix_form = _STATISTICS[statistic]

    cases: list[dict] = []
    for _ in range(6):
        # Draw order and draw count are identical for every key: `s` is drawn
        # even by the stride-1 variants, which then discard it. A variant is
        # never allowed to move the stream.
        w = int(rng.integers(2, 6)) if level == 1 else int(rng.integers(2, 5))
        d = 1 if level == 1 else int(rng.integers(2, 4))
        drawn_stride = int(rng.integers(1, _MAX_STRIDE + 1))
        s = drawn_stride if strided else 1
        # First to last index of a single window, inclusive.
        span = (w - 1) * d + 1
        # `n - span >= _MAX_STRIDE >= s` guarantees a second start position, so
        # no hidden case turns on an empty or single-window result -- an edge
        # the description says nothing about.
        n = span + _MAX_STRIDE + int(rng.integers(0, 6))
        x = rng.integers(-9, 10, n)

        # Zero-based start positions: p - 1 for each window the prompt keeps.
        starts = np.arange(0, n - span + 1, s)
        assert len(starts) >= 2, "every hidden case must have two valid windows"
        windows = x[starts[:, None] + np.arange(w) * d].astype(float)
        out = reducer(windows, axis=1)

        args: list = [x.tolist(), w]
        if strided:
            args.append(s)
        if level >= 2:
            args.append(d)
        # One value per window, in start order: a JSON list of scalars, which
        # the harness renders as the 1-by-N row the shape sentence promises.
        cases.append({"args": args, "expected": out.tolist()})

    parameters = ["x", "w"] + (["s"] if strided else []) + (["d"] if level >= 2 else [])
    signature = f"function out = sliding_window({', '.join(parameters)})"

    # The last start position the keep-condition allows, written the way the
    # description writes it so a reader can transcribe it directly.
    last_start = "numel(x) - w + 1" if level == 1 else "numel(x) - (w-1)*d"
    starts_expr = f"1:s:({last_start})" if strided else f"1:({last_start})"
    window_expr = "x(p:p+w-1)" if level == 1 else "x(p:d:p+(w-1)*d)"
    offsets = "(0:w-1)" if level == 1 else "(0:w-1)*d"
    # Rows are windows, columns are positions within a window; reducing along
    # dim 2 therefore reduces each window. `x(idx)` takes the shape of `idx`
    # for a vector `x`, whichever orientation `x` arrives in.
    index_matrix = f"({starts_expr})' + {offsets}"

    loop_body = (
        f" starts = {starts_expr};\n"
        " out = zeros(1, numel(starts));\n"
        " for j = 1:numel(starts)\n"
        "   p = starts(j);\n"
        f"   out(j) = {window_form.format(W=window_expr)};\n"
        " endfor"
    )
    vector_body = (
        f" idx = {index_matrix};\n"
        f" out = ({matrix_form.format(M='x(idx)')})';"
    )

    # The reference may coerce; `x(:)'` is defensive only, since every
    # expression below is already orientation-free.
    reference = f"{signature}\n x = x(:)';\n{vector_body}\nendfunction"
    # The natural solution is the direct transcription of the description, with
    # no coercion and no transpose the prompt did not ask for: a loop over the
    # start positions while loops are allowed, and the index matrix once they
    # are not. The trailing `'` is asked for -- the generated shape sentence
    # states a 1-by-N row.
    natural_body = vector_body if level == 3 else loop_body
    natural = f"{signature}\n{natural_body}\nendfunction"

    return Variant(
        key=key,
        description=_describe(key, level),
        signature=signature,
        cases=cases,
        reference=reference,
        natural=natural,
        vectorized=level == 3,
    )
