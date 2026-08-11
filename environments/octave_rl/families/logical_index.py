"""``logical_index``: select or replace the elements a predicate picks out.

Written against the worked exemplar in ``families/reduce_along_dim.py`` -- same
two public names, same level-ladder discipline -- and alongside
``families/broadcast_arith.py`` and ``families/sliding_window.py``. Read
``specs.py`` first for why the variant is an argument rather than an rng draw,
and for the rule that ``description``/``reference``/``natural`` are written
together so they cannot drift.

## The spec

Two dimensions, per ``PARAMETERIZATION_DESIGN.md``: which predicate selects the
elements, and what is done with the ones it selects. The design lists five
predicates (``> 0``, ``< 0``, ``|x| > t``, outside ``[lo, hi]``, even) and four
actions (extract, set NaN, set 0, clamp); eight of the twenty combinations ship,
chosen so that **every predicate and every action appears at least once**, the
two 0.4.x problems survive as variants (``positive-extract`` was the old level 1,
``outside-nan`` the old level 2), and the three predicates whose Octave spelling
is easiest to get wrong -- ``even``, ``|x| > t``, outside ``[lo, hi]`` -- each
appear with two different actions.

``clamp`` ships only with the ``outside`` predicate, and that is not an
oversight: clamping means "replace by the nearer end of the range", so it is only
defined when the predicate carries a range. A ``positive-clamp`` would have to
invent a bound the prompt never gave.

The twelve combinations left on the shelf are left there for room, not for
danger; the one that is genuinely un-shippable is any ``clamp`` on a predicate
without a range, for the reason above.

## Every hidden case is balanced for every predicate, by construction

The sharp edge in this family is a predicate that matches nothing or everything.
An extraction that matches nothing returns an empty result -- ``zeros(1, 0)``,
not ``[]`` (see ``harness.octave_literal``) -- and an action that changes no
element makes the answer equal the input, so code that ignores the task scores
full marks. Neither edge is described by any prompt here, so both are **excluded
from the draws** rather than described.

The exclusion cannot be rejection sampling: redrawing until a *particular*
predicate is balanced would consume a different number of rng values per
variant, and
``test_a_family_generates_the_same_tasks_whichever_variants_are_present`` forbids
a variant selection shifting the shared stream. Instead every case overwrites
eight randomly chosen positions of the drawn vector with the fixed set

    t+2, -(t+2), lo, hi, 2, -2, 1, -1

which guarantees, simultaneously and for all five predicates, **at least two
matching and at least two non-matching elements**: ``±(t+2)`` are positive and
negative, exceed ``t`` in magnitude and lie outside ``[lo, hi]``; ``lo`` and
``hi`` lie inside it; ``±2`` are even and ``±1`` odd, and all four are within
``t`` of zero. The parameter ranges (``t`` in 2..3 and ``-3 <= lo < hi <= 3``)
are what make those claims hold for every draw. The overwrite is the same eight
values whatever the variant, so the rng is consumed identically for every key --
the hidden inputs of a given task index do not depend on which variant it is,
which was verified directly rather than assumed (all eight keys draw byte-
identical hidden vectors at all three levels).

Measured over 2,000 tasks per variant at each level, the weakest guarantee any
variant gets is **2 matching and 2 non-matching elements** out of 10 to 13, so no
hidden case anywhere is an empty extraction or an action that changes nothing.

``lo < hi`` is strict, and that is a defect this file already had once: drawing
two bounds and sorting them collides on about one case in seven, and ``lo == hi``
makes ``outside-clamp`` return the constant vector ``lo`` whatever the input --
an answer with no information in it, and one a level-1 solution reproduces
exactly. It showed up as 69 of 360 level-2 cases (19%) matching their level 1 in
the first census run; with ``lo < hi`` enforced the same variant sits at 190 of
12,000 (1.6%), and no task has more than 2 of its 6 cases coinciding.

## The level ladder, and the three ladders rejected before it

Level 3 is level 2 plus a vectorization constraint, restating its own task in
full (a bare "...without loops" is what dropped ``struct_cell_wrangle`` level 3
from 0.792 to 0.000). Level 2 applies **the same predicate and the same action to
the running totals of x** rather than to x itself: ``c(i) = x(1) + ... + x(i)``.

The trap this family had to be walked around is stated in ``PIPELINE_LOG.md``
(2026-08-10) and it is sharper here than anywhere else, because a
predicate/action pair can make a level-2 *step* a no-op:

- **Rejected ladder: "do level 1, then set the still-matching elements to 0".**
  An exact identity for all four replacement variants, because after the level-1
  action nothing matches any more: the zeroed entries of ``magnitude-zero`` are
  within ``t`` of zero, the clamped entries of ``outside-clamp`` are inside
  ``[lo, hi]``, and -- the trap the brief names -- ``NaN < lo`` and ``NaN > hi``
  are both **false**, so the NaN entries of ``even-nan`` and ``outside-nan``
  match nothing either. Measured: 12,000 of 12,000 level-2 answers identical to
  their level 1 for each of those four. For the extract variants it is not an
  identity but it is worse than one -- every selected element matches, so the
  answer degenerates to a vector of zeros whose only content is its length.
- **Rejected ladder: "do level 1, then clamp the result into [lo, hi]".** An
  exact identity for ``outside-nan`` and ``outside-clamp`` by construction --
  after either action no element lies outside the range, and ``NaN`` survives a
  clamp unchanged. Measured: 12,000 of 12,000 for both.
- **Rejected ladder: mask from x, values from a second argument y** --
  ``out = y(x > 0)``, the classic two-vector logical-index exercise. It is a
  genuinely new problem for all eight variants and it was still passed over: it
  adds an argument, so a level-1 solution called against level-2 cases dies on
  arity rather than on being wrong, and
  ``test_no_variant_has_a_level_two_its_level_one_solution_already_solves``
  would pass this family for free. A ladder whose safety cannot be *measured* is
  a worse deal than one whose safety must be.

Running totals avoid all of that because they change *the vector the predicate
reads*, rather than post-processing an answer that already satisfies the
predicate, and they do it without adding an argument. They also keep the
family's own subject -- a mask and an indexed assignment -- intact at every
level, and stay in exact integer arithmetic, so no variant rests on a
floating-point tolerance.

Two consequences worth stating, because they are what make the ladder checkable:

1. **The signature does not change between levels.** Level 2 introduces no new
   argument, so a level-1 solution *runs* against level-2 cases instead of dying
   on arity -- which is what makes
   ``test_no_variant_has_a_level_two_its_level_one_solution_already_solves``
   non-vacuous for this family. (A family whose level 2 adds a parameter passes
   that test for free, on an error rather than on a measurement.)
2. **The guarantees above have to hold for ``c``, not for ``x``,** at level 2.
   So a level-2 case draws the *running totals* first -- the balanced vector
   built above -- and serialises ``x = diff([0, c])``, whose cumulative sum is
   that vector exactly. Same number of rng draws, same guarantees, and no case
   can turn on an empty extraction or an action that changes nothing.

Measured for the shipped ladder, 2,000 tasks x 6 cases per variant (12,000 cases
each), counting level-2 cases whose answer equals what the level-1 rule returns
on the same input:

    positive-extract     2/12000    worst task 1 of 6
    negative-extract     0/12000    worst task 0 of 6
    even-extract         0/12000    worst task 0 of 6
    magnitude-extract    1/12000    worst task 1 of 6
    magnitude-zero       0/12000    worst task 0 of 6
    even-nan             0/12000    worst task 0 of 6
    outside-nan          0/12000    worst task 0 of 6
    outside-clamp      190/12000    worst task 2 of 6

The right-hand column is the one that decides shippability: a level-1 solution
needs all six cases of a task to score full marks at level 2, and **no task of
any variant gives it more than two**. The residual coincidences are ordinary
small-integer collisions -- ``outside-clamp`` saturates most entries to ``lo`` or
``hi``, so two different vectors can clamp alike -- rather than a structural
identity, which is exactly what the two rejected ladders above look like when
put through the same census (12,000/12,000). That contrast is the evidence that
this census can see the defect it is looking for.
"""

from __future__ import annotations

import numpy as np
from specs import Variant

# (predicate, action). Order is the round-robin order and is part of the split
# contract -- appending is safe, reordering silently changes which task gets
# which problem. `positive-extract` and `outside-nan` are the two 0.4.x problems,
# kept first and next-to-last so the pre-0.5.0 pool remains a subset of this one.
VARIANT_KEYS: list[str] = [
    "positive-extract",
    "negative-extract",
    "even-extract",
    "magnitude-extract",
    "magnitude-zero",
    "even-nan",
    "outside-nan",
    "outside-clamp",
]

_PREDICATES = {
    # name -> (English, NumPy mask over a vector and the drawn parameters,
    # Octave mask over the operand {V}, extra arguments in signature order)
    #
    # The English is written to slot into "the elements of c that are ..." and
    # "every element that is ... has been replaced by ...", so one phrasing
    # serves both the extraction and the replacement actions.
    "positive": (
        "greater than 0",
        lambda v, p: v > 0,
        "{V} > 0",
        (),
    ),
    "negative": (
        "less than 0",
        lambda v, p: v < 0,
        "{V} < 0",
        (),
    ),
    "even": (
        # Spelled out because "even" alone leaves the negative entries to
        # convention, and every value drawn here can be negative. Octave's
        # `mod(-4, 2)` is 0 and `mod(-3, 2)` is 1, which is what "divisible by
        # 2" says.
        "even (divisible by 2, which includes 0 and the negative even numbers)",
        lambda v, p: v % 2 == 0,
        "mod({V}, 2) == 0",
        (),
    ),
    "magnitude": (
        "greater than t in absolute value",
        lambda v, p: np.abs(v) > p["t"],
        "abs({V}) > t",
        ("t",),
    ),
    "outside": (
        "outside the inclusive range [lo, hi] (less than lo or greater than hi)",
        lambda v, p: (v < p["lo"]) | (v > p["hi"]),
        "{V} < lo | {V} > hi",
        ("lo", "hi"),
    ),
}

_ACTIONS = {
    # name -> (NumPy action over the operand, its mask and the parameters;
    # Octave body over the operand {V} and the mask expression {MASK})
    #
    # `clamp` is the one action that does not consume {MASK}: "replace by the
    # nearer end of the range" is stated directly as two guarded assignments,
    # which is also how the description phrases it. It therefore only pairs with
    # the `outside` predicate, whose parameters it needs.
    "extract": (
        lambda v, mask, p: v[mask].tolist(),
        "out = {V}({MASK});",
    ),
    "nan": (
        lambda v, mask, p: np.where(mask, np.nan, v).tolist(),
        "out = {V}; out({MASK}) = NaN;",
    ),
    "zero": (
        lambda v, mask, p: np.where(mask, 0, v).tolist(),
        "out = {V}; out({MASK}) = 0;",
    ),
    "clamp": (
        lambda v, mask, p: np.clip(v, p["lo"], p["hi"]).tolist(),
        "out = {V}; out({V} < lo) = lo; out({V} > hi) = hi;",
    ),
}

# How many positions of each drawn vector are overwritten with the guaranteed
# values. Every vector is longer than this, so each case still carries free
# random entries.
_GUARANTEED_SLOTS = 8


def _parse(key: str) -> tuple[str, str]:
    predicate, _, action = key.partition("-")
    if predicate not in _PREDICATES or action not in _ACTIONS:
        raise ValueError(f"unknown logical_index variant {key!r}")
    if action == "clamp" and predicate != "outside":
        raise ValueError(
            f"logical_index variant {key!r} clamps without a range to clamp into"
        )
    return predicate, action


def _guaranteed_values(t: int, lo: int, hi: int) -> list[int]:
    """The eight values that make every predicate non-trivial on every case.

    Two elements match and two do not, for all five predicates at once:

    - ``> 0``:            ``t+2, 2, 1`` match; ``-(t+2), -2, -1`` do not.
    - ``< 0``:            mirror image.
    - even:               ``2, -2`` match; ``1, -1`` do not.
    - ``|x| > t``:        ``±(t+2)`` match; ``±1, ±2`` do not, since ``t >= 2``.
    - outside ``[lo,hi]``: ``±(t+2)`` are outside, since ``t+2 >= 4 > hi`` and
      ``-(t+2) <= -4 < lo`` for the drawn parameter ranges; ``lo`` and ``hi``
      are inside it, and are two distinct elements because ``lo < hi``.
    """
    return [t + 2, -(t + 2), lo, hi, 2, -2, 1, -1]


def _describe(key: str, level: int) -> str:
    """The prompt text. Every level restates its own task in full.

    Level 3 saying only "...without loops" is what let `struct_cell_wrangle`
    level 3 fall from 0.792 to 0.000 while models guessed at the task from the
    family name alone. Guarded by
    `test_level_three_restates_its_own_task_for_every_problem`, which reads the
    generated prompt rather than a table, so it sees all eight variants. Level 3
    here is level 2 verbatim plus the loop clause, so it cannot drop a term.

    The output shape is deliberately absent: the prompt builder appends the
    shape sentence it derives from the expected values, so the prompt's claim
    and the grader's comparison cannot drift apart.
    """
    predicate, action = _parse(key)
    english, _, _, parameters = _PREDICATES[predicate]

    lead = "x is a row vector of integers."
    if "t" in parameters:
        lead += " t is a positive integer scalar."
    if "lo" in parameters:
        lead += " lo and hi are integer scalars with lo < hi."
    if level == 1:
        operand = "x"
    else:
        operand = "c"
        lead += (
            " Let c be the running totals of x, so that c(i) is the sum of "
            "x(1) through x(i); c has the same length as x."
        )

    if action == "extract":
        task = (
            f"Return the elements of {operand} that are {english}, in their "
            f"original order."
        )
    elif action == "clamp":
        task = (
            f"Return a copy of {operand} in which every element less than lo "
            f"has been replaced by lo and every element greater than hi has "
            f"been replaced by hi; elements inside the inclusive range [lo, hi] "
            f"are left unchanged. The result has the same length as {operand}."
        )
    else:
        replacement = "NaN" if action == "nan" else "0"
        task = (
            f"Return a copy of {operand} in which every element that is "
            f"{english} has been replaced by {replacement}, and every other "
            f"element is left unchanged. The result has the same length as "
            f"{operand}."
        )

    description = f"{lead} {task}"
    if level == 3:
        return description + " Do not use for/while loops."
    return description


def build(rng: np.random.Generator, level: int, key: str) -> Variant:
    predicate, action = _parse(key)
    _, matches, mask_form, parameters = _PREDICATES[predicate]
    apply_action, body_form = _ACTIONS[action]

    cases: list[dict] = []
    for _ in range(6):
        # One draw pattern for every variant and every level: a length, that
        # many values, a threshold, two bounds, and a permutation. Nothing here
        # reads `key`, so a variant selection cannot shift the shared stream --
        # `t`, `lo` and `hi` are drawn even by the variants that never use them.
        n = int(rng.integers(10, 14))
        values = rng.integers(-9, 10, n)
        t = int(rng.integers(2, 4))
        # `lo < hi` strictly, and the width is drawn rather than the upper
        # bound. A drawn pair sorted into (lo, hi) collides on about one case in
        # seven, and `lo == hi` makes the clamp variant's answer the constant
        # vector `lo` whatever the input -- measured at 69 of 360 level-2 cases
        # identical to their level 1 before this was fixed, 190 of 12,000 after.
        lo = int(rng.integers(-3, 1))
        hi = lo + int(rng.integers(1, 4))
        slots = rng.permutation(n)[:_GUARANTEED_SLOTS]
        values[slots] = _guaranteed_values(t, lo, hi)

        # `values` is the vector the task actually reads: x itself at level 1,
        # and the running totals of x from level 2 on. Serialising
        # `diff([0, values])` at level 2 makes `cumsum(x)` equal the balanced
        # vector exactly, so the match/non-match guarantees apply where the
        # predicate is evaluated rather than where the input happens to be.
        x = values if level == 1 else np.diff(np.concatenate(([0], values)))
        drawn = {"t": t, "lo": lo, "hi": hi}
        expected = apply_action(values, matches(values, drawn), drawn)
        assert expected, "an extraction must never render an empty hidden case"
        cases.append(
            {
                "args": [x.tolist(), *(drawn[name] for name in parameters)],
                "expected": expected,
            }
        )

    signature = f"function out = logical_index({', '.join(('x', *parameters))})"

    # The reference may coerce defensively; it does so once, up front, so the
    # masked assignment below is written identically in both solutions.
    # `double` is what would let NaN be stored if a case ever arrived as an
    # integer type, and `(:)'` fixes the orientation the prompt states.
    if level == 1:
        reference_body = " v = double(x(:)');\n"
        reference_operand = "v"
    else:
        reference_body = " v = double(x(:)');\n c = cumsum(v);\n"
        reference_operand = "c"
    reference_body += " " + body_form.format(
        V=reference_operand, MASK=mask_form.format(V=reference_operand)
    )

    # What a competent Octave programmer writes from the description alone: no
    # `(:)`, no reshape, no transpose the prompt did not ask for, and no
    # coercion -- `x` arrives as a row of doubles, `x(mask)` keeps that
    # orientation, and `cumsum` of a row is a row. It is already loop-free, so
    # level 3 needs nothing extra. If this cannot pass, the variant is not
    # shippable.
    natural_operand = "x" if level == 1 else "c"
    natural_body = "" if level == 1 else " c = cumsum(x);\n"
    natural_body += " " + body_form.format(
        V=natural_operand, MASK=mask_form.format(V=natural_operand)
    )

    return Variant(
        key=key,
        description=_describe(key, level),
        signature=signature,
        cases=cases,
        reference=f"{signature}\n{reference_body}\nendfunction",
        natural=f"{signature}\n{natural_body}\nendfunction",
        vectorized=level == 3,
    )
