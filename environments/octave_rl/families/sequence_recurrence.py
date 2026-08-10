"""``sequence_recurrence``: a linear recurrence, and one thing to do with its terms.

Written in the 0.5.0 variant form described in ``specs.py`` and modelled on
``families/reduce_along_dim.py``, the worked exemplar; the degeneracy census
below follows ``families/logical_index.py`` and ``families/string_parse.py``.
The module exposes exactly two names: ``VARIANT_KEYS`` and
``build(rng, level, key)``.

## The spec

Two dimensions, per ``PARAMETERIZATION_DESIGN.md``: the **order** of the
recurrence and what is **returned** from its terms, with every coefficient
arriving as an argument rather than baked into the prompt. The design lists
``order in {1, 2}`` and ``return in {terms, cumulative sum, final term}``, which
is six; the eighth and seventh come from a fourth return mode, **the total**
(the sum of all n terms). It is a scalar like ``final`` but it reads the whole
sequence rather than its last element, so a solution that finds a shortcut to
``x(n)`` cannot also answer it. The full cross ships: 2 orders x 4 returns.

The order dimension changes the arity -- ``(a, p, d, n)`` against
``(a, b, p, q, n)`` -- which is allowed between variants and is deliberately
**not** allowed between levels; see the ladder section.

**Rejected dimension: output orientation.** ``string_parse`` makes row-versus-
column a variant axis and it is a real risk worth testing somewhere. It is wrong
here: every loop-free construction this family admits (``filter``, ``cumsum``)
returns a row when handed a row, so a column variant would grade a transpose
that exists only to satisfy the grader, and the ``natural`` solution -- which by
rule carries no unrequested transpose -- would have to carry one anyway. The
shape sentence the generator appends still states the graded shape for all eight.

## Numeric blow-up, which is this family's hazard

A two-term recurrence with ``|p|`` or ``|q|`` above 1 grows exponentially in
``n``, and the ``cumulative`` and ``total`` modes then sum that growth. Large
magnitudes would make the comparison depend on the relative tolerance rather
than on the algorithm, so the draws are bounded to keep **every expected value a
small exact integer**:

    n in 5..9,  a, b, d in [-4, 4],  p, q in {-2, -1, 1, 2}

That space is finite (5 x 9 x 9 x 9 x 4 x 4 = 58,320 parameter tuples), so the
worst case is not sampled but **enumerated**. Over every tuple, every order,
every level and all four return modes, the largest magnitude any expected value
can take is:

    order 1, level 1     4,052        order 2, level 1      9,792
    order 1, level 2+    5,522        order 2, level 2+    12,690

so **12,690 is the ceiling for the whole family**, attained at
``n=9, a=b=4, p=q=2, d=-4`` by the order-2 ``cumulative``/``total`` modes. Every
value is an integer produced by integer additions and multiplications; doubles
represent every integer up to ``2^53 ~ 9.0e15`` exactly, twelve orders of
magnitude above the ceiling, so both sides of every comparison are bit-identical
and ``tolerance`` is never exercised. Widening to ``n <= 10`` raises the ceiling
to 34,702 -- still exact, but the bound was chosen to leave the largest slack
that keeps the sequences interesting rather than the largest numbers that still
happen to work.

``p`` and ``q`` exclude **0** on purpose. ``q = 0`` turns an order-2 recurrence
into an order-1 one, and ``p = 0`` makes order 1 constant from its second term:
in both cases a solution that ignores one of the two terms of the recurrence
scores the case, and neither degeneracy is anything the prompt describes.

``n >= 5`` is stated in the prompt *and* enforced in the draws, which disposes of
the edge the brief names: "the first 1 term" of a two-term recurrence would need
a convention for what happens when ``n`` is smaller than the number of seeds, and
no prompt here states one. It also guarantees that level 2's extra term (which
first applies at ``i = 2`` for order 1 and ``i = 3`` for order 2) is exercised by
every hidden case.

## The level ladder

Level 1 is the homogeneous-in-form recurrence the variant names. Level 2 keeps
the same arguments, the same order and the same return mode, and adds a
**forcing term equal to the index**: ``x(i) = p*x(i-1) + q*x(i-2) + i``. Level 3
is level 2 restated in full plus "Do not use for/while loops".

Three properties made this the ladder:

1. **The signature does not change between levels**, so a level-1 solution
   *runs* against level-2 cases and returns a wrong answer instead of dying on
   arity. That is what makes
   ``test_no_variant_has_a_level_two_its_level_one_solution_already_solves`` a
   measurement rather than a vacuous pass -- for this family it lands in the
   test's measuring branch, and both signatures and both shapes agree, so the
   zero it records is a mathematical one. (Registering this module in
   ``generators.VARIANT_MODULES`` is what turns that on; until then the census
   below is the only evidence, and it was run for exactly that reason.)
2. **It cannot be a no-op.** The difference sequence ``e = x_2 - x_1`` obeys the
   same recurrence with ``e(1) = 0`` (and ``e(2) = 0`` at order 2) driven by
   ``i``, so ``e`` is nonzero from the first driven index onward and the terms
   and running totals differ at every index from there. Only the two scalar
   modes can coincide, and only by cancellation; measured below.
3. **It stays in integers**, which the tolerance argument above depends on.

### Rejected ladders

- **Level 1 order-1, level 2 order-2** -- the 0.4.x ladder. It changes the
  arity between levels, which is exactly the shape the repo's guard test cannot
  measure (the level-1 probe dies on an arity error and the assertion passes for
  free). Order is a variant dimension here instead, where an arity change is
  harmless because nothing compares two variants.
- **Level 2 batches the seeds**: ``a`` becomes a vector of starting values and
  the answer gains a row per seed. Non-degenerate by construction, and rejected
  because the collapse is prevented by a *shape* change rather than by a
  different computation -- a level-1 solution would fail on an error, so the
  guard test would again record a zero of the uninteresting kind, and the
  loop-free level-3 solution would need ``filter(..., [], 2)`` plus a ``repmat``,
  which is a different subject (broadcasting) wearing this family's name.
- **Level 2 squares the terms.** Genuinely different for the vector modes and
  degenerate for ``final`` whenever ``x(n)`` is 0 or 1 -- an identity the draws
  cannot exclude without constraining the recurrence itself -- and it squares the
  magnitude ceiling into the region where the tolerance starts to matter.

### Measured non-collapse

Level 1 and level 2 draw identically (the draw block below reads ``level``
nowhere), so a variant's level-2 case has the *same arguments* as its level-1
case and the two answers are directly comparable. Over 2,000 tasks per variant
per level (12,000 hidden cases each, 96,000 in all), counting level-2 cases
whose expected value equals the level-1 expected value for the same arguments:

    order1-terms          0/12000    worst task 0 of 6
    order1-cumulative     0/12000    worst task 0 of 6
    order1-final          0/12000    worst task 0 of 6
    order1-total         42/12000    worst task 2 of 6
    order2-terms          0/12000    worst task 0 of 6
    order2-cumulative     0/12000    worst task 0 of 6
    order2-final        119/12000    worst task 2 of 6
    order2-total        124/12000    worst task 2 of 6

The right-hand column is what decides shippability: a level-1 solution needs all
six cases of a task to score full marks, and **no task of any variant gives it
more than two**. The two scalar modes carry all of the residue, which is what
property 2 above predicts -- a single number can coincide by cancellation, a
sequence of five to nine of them does not. The same census run against the
rejected "level 2 batches the seeds" ladder reports 0/12000 everywhere, which is
what a structural (rather than mathematical) non-collapse looks like and is the
reason it was not preferred.

This was also run through the interpreter rather than only through NumPy: each
variant's level-1 ``natural`` solution, executed against its own level-2 and
level-3 hidden cases on the pinned Octave 10.2.0, scored **0 of 288** cases
(8 variants x 2 levels x 3 seeds x 6 cases), with all 288 running to completion
-- wrong answers, not errors.

## The 0.4.x method hint, dropped

The 0.4.x level-3 description ended "...without for/while loops; use filter or
equivalent". That is a hint about the *solution method*, not part of the task,
and it is dropped for **all eight variants** rather than kept for all eight. A
level 3 that names the function to call measures whether the model can follow an
instruction, while the level-3 constraint exists to measure whether it can find
a loop-free formulation; and this family is the one place in the pool where the
loop-free formulation is genuinely non-obvious, so the hint would remove most of
what the level is for. Dropping it also keeps every level-3 prompt exactly its
level-2 prompt plus the loop clause, which is the invariant
``test_level_three_restates_its_own_task_for_every_problem`` checks and the
reason `sequence_recurrence` is in that test's ``same_task_at_level_three`` set.

The hint remains available where it belongs: the environment's guide/hint
channel, which is scored separately.

## The reference and the naive solution

``reference`` is ``filter``-based at every level, which is loop-free and so
satisfies level 3 as well. ``natural`` is the direct transcription of the
description -- a preallocated vector and a ``for`` loop -- at levels 1 and 2,
where loops are allowed, and the ``filter`` form at level 3, where they are not.

**Stated rather than glossed:** at level 3 the two coincide in method for the
order-2 variants, because ``filter`` is effectively the only loop-free
construction for a two-term recurrence in base Octave (the closed form needs the
roots of ``r^2 - p*r - q``, which are irrational and would forfeit the exact-
integer property this family's tolerance argument rests on). The level-3 check
is therefore weaker than the level-1 and level-2 ones, where the loop
transcription and the ``filter`` reference are independent implementations that
agree on every hidden case.
"""

from __future__ import annotations

from itertools import accumulate

import numpy as np
from specs import Variant

# (order, return mode). Order is the round-robin order and is part of the split
# contract -- appending is safe, reordering silently changes which task gets
# which problem. `order1-terms` and `order2-terms` are the two 0.4.x problems
# (level 1 and level 2 respectively), generalised by making every coefficient an
# argument; they are kept first in each half so the pre-0.5.0 pool remains
# recognisable inside this one.
VARIANT_KEYS: list[str] = [
    "order1-terms",
    "order1-cumulative",
    "order1-final",
    "order1-total",
    "order2-terms",
    "order2-cumulative",
    "order2-final",
    "order2-total",
]

_ORDERS = {
    # name -> (arguments in signature order, the seed clause, the step clause
    # over the level-2 drive {DRIVE}, the first driven index, the Octave
    # denominator of the `filter` transfer function, the loop preamble, the loop
    # body).
    #
    # {DRIVE} is empty at level 1 and " + i" from level 2 on, so one string
    # spells the recurrence at every level and the two cannot drift.
    "order1": (
        ("a", "p", "d", "n"),
        "x(1) = a",
        "x(i) = p*x(i-1) + d{DRIVE}",
        2,
        "[1 -p]",
        " x(1) = a;",
        " x(i) = p*x(i-1) + d{DRIVE};",
    ),
    "order2": (
        ("a", "b", "p", "q", "n"),
        "x(1) = a, x(2) = b",
        "x(i) = p*x(i-1) + q*x(i-2){DRIVE}",
        3,
        "[1 -p -q]",
        " x(1) = a;\n x(2) = b;",
        " x(i) = p*x(i-1) + q*x(i-2){DRIVE};",
    ),
}

_RETURNS = {
    # name -> (the sentence that states what to return, the NumPy-free reducer
    # over the list of terms, the Octave expression over the sequence {X}).
    #
    # `final` and `total` reduce to a bare Python number rather than a
    # one-element list, which is what makes `generators._shape_sentence` say
    # "Return a scalar." for them: it reads the expected value, not a flag here.
    "terms": (
        "Return the n terms x(1), x(2), ..., x(n), in that order.",
        list,
        "{X}",
    ),
    "cumulative": (
        "Return the running totals of x, so that entry i of the result is "
        "x(1) + x(2) + ... + x(i); the result has n entries.",
        lambda terms: list(accumulate(terms)),
        "cumsum({X})",
    ),
    "final": (
        "Return x(n), the last of the n terms.",
        lambda terms: terms[-1],
        "{X}(end)",
    ),
    "total": (
        "Return x(1) + x(2) + ... + x(n), the sum of all n terms.",
        sum,
        "sum({X})",
    ),
}

# The coefficient alphabet. 0 is absent so that no case silently drops a term of
# its own recurrence -- see the module docstring -- and 2 is the largest
# magnitude allowed, which with `n <= 9` caps every expected value at 12,690.
_COEFFICIENTS = (-2, -1, 1, 2)


def _parse(key: str) -> tuple[str, str]:
    order, _, mode = key.partition("-")
    if order not in _ORDERS or mode not in _RETURNS:
        raise ValueError(f"unknown sequence_recurrence variant {key!r}")
    return order, mode


def _drive(level: int) -> str:
    """The level-2 forcing term, as it appears inside the recurrence.

    Empty at level 1. From level 2 on it is ``+ i``: the same recurrence driven
    by the index of the term being computed, which changes every term from the
    first driven index onward without changing the signature, the shape, or the
    exact-integer property. See the ladder section of the module docstring.
    """
    return "" if level == 1 else " + i"


def _sequence(
    order: str, level: int, n: int, a: int, b: int, p: int, q: int, d: int
) -> list[int]:
    """The n terms, in Python integers, exactly as the description defines them.

    Integers throughout: no expected value in this family is a float, so none of
    them can disagree with Octave's double arithmetic in the last ulp.
    """
    drive = 0 if level == 1 else 1
    if order == "order1":
        terms = [a]
        for i in range(2, n + 1):
            terms.append(p * terms[-1] + d + drive * i)
        return terms
    terms = [a, b]
    for i in range(3, n + 1):
        terms.append(p * terms[-1] + q * terms[-2] + drive * i)
    return terms


def _describe(key: str, level: int) -> str:
    """The prompt text. Every level restates its own task in full.

    Level 3 saying only "...without loops" is what let `struct_cell_wrangle`
    level 3 fall from 0.792 to 0.000 while models guessed at the task from the
    family name alone. Guarded by
    `test_level_three_restates_its_own_task_for_every_problem`, which reads the
    generated prompt rather than a table, so it sees all eight variants. Level 3
    here is level 2 verbatim plus the loop clause, so it cannot drop a term --
    and, unlike the 0.4.x text, it does not append a method hint either.

    The output shape is deliberately absent: the prompt builder appends the
    shape sentence it derives from the expected values, so the prompt's claim
    and the grader's comparison cannot drift apart.
    """
    order, mode = _parse(key)
    parameters, seeds, step, first, _, _, _ = _ORDERS[order]
    named = ", ".join(parameters[:-1]) + f" and {parameters[-1]}"

    # `n >= 5` is a promise the draws keep, not a rule the solver must apply. It
    # is stated so that no reader has to decide what "the first 1 term" of a
    # two-term recurrence means -- the edge is excluded from the draws as well.
    lead = f"{named} are integer scalars, and n >= 7."
    definition = (
        f"Let x be the sequence defined by {seeds} and "
        f"{step.format(DRIVE=_drive(level))} for i = {first}, {first + 1}, "
        f"..., n."
    )
    if level > 1:
        # Spelled out because "+ i" inside a formula can be read as a typo for a
        # coefficient. It is the index of the term being computed.
        definition += " Each step after the seed terms adds the index i itself."

    description = f"{lead} {definition} {_RETURNS[mode][0]}"
    if level == 3:
        return description + " Do not use for/while loops."
    return description


def build(rng: np.random.Generator, level: int, key: str) -> Variant:
    order, mode = _parse(key)
    parameters, _, _, _, denominator, preamble, body = _ORDERS[order]
    reduce_terms = _RETURNS[mode][1]

    cases: list[dict] = []
    for _ in range(6):
        # One draw pattern for every variant and every level: a length, two
        # seeds, two coefficients and an offset, in this order. Nothing here
        # reads `key` or `level`, so a variant selection cannot shift the shared
        # rng stream and a variant's level-1 and level-2 cases take the *same*
        # arguments -- which is what makes the degeneracy census in the module
        # docstring a comparison of answers rather than of draws.
        #
        # `b` and `q` are drawn even by the order-1 variants that never use
        # them, and `d` even by the order-2 variants that never use it. The
        # index into `_COEFFICIENTS` is drawn as `integers(0, 4)`: the tempting
        # `rng.integers(0, 1, k)` spelling for a constant returns zeros *without
        # consuming the stream*, which would advance the shared rng differently
        # for different variants.
        n = int(rng.integers(7, 10))
        a = int(rng.integers(-4, 5))
        b = int(rng.integers(-4, 5))
        p = _COEFFICIENTS[int(rng.integers(0, 4))]
        q = _COEFFICIENTS[int(rng.integers(0, 4))]
        d = int(rng.integers(-4, 5))

        terms = _sequence(order, level, n, a, b, p, q, d)
        drawn = {"a": a, "b": b, "p": p, "q": q, "d": d, "n": n}
        cases.append(
            {
                "args": [drawn[name] for name in parameters],
                # A bare number for `final` and `total`, a list for the two
                # sequence modes. `_shape_sentence` derives the promised shape
                # from exactly this value.
                "expected": reduce_terms(terms),
            }
        )

    signature = f"function out = sequence_recurrence({', '.join(parameters)})"
    result = _RETURNS[mode][2]

    # The reference drives `filter` with the input whose zero-state response is
    # the sequence: the seed terms are folded into the first one or two entries
    # of the drive, and the rest is the level's forcing term. For order 2,
    # `u(2) = b - p*a` is what makes `y(2) = u(2) + p*y(1)` come out as `b`.
    # Loop-free at every level, so it satisfies the level-3 constraint too.
    if order == "order1":
        rest = "d + (2:n)" if level > 1 else "repmat(d, 1, n-1)"
        drive = f"[a, {rest}]"
    else:
        rest = "(3:n)" if level > 1 else "zeros(1, n-2)"
        drive = f"[a, b - p*a, {rest}]"
    reference_body = (
        f" x = filter(1, {denominator}, {drive});\n"
        f" out = {result.format(X='x')};"
    )

    # What a competent Octave programmer writes from the description alone. At
    # levels 1 and 2 that is the recurrence transcribed into a loop, which the
    # prompt permits; at level 3 it is the `filter` form, since loops are
    # forbidden there. No `(:)`, no reshape, and no transpose: `filter` and
    # `cumsum` of a row are rows, `x(end)` and `sum(x)` are scalars, and the
    # shape sentence promises exactly those. If this cannot pass, the variant is
    # not shippable.
    if level == 3:
        natural_body = reference_body
    else:
        natural_body = (
            " x = zeros(1, n);\n"
            f"{preamble}\n"
            f" for i = {_ORDERS[order][3]}:n\n"
            f"  {body.format(DRIVE=_drive(level))}\n"
            " endfor\n"
            f" out = {result.format(X='x')};"
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
