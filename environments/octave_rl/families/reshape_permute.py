"""``reshape_permute``: one 3-D dimension reordering, one flattening.

Written in the 0.5.0 variant form described in ``specs.py`` and modelled on
``families/reduce_along_dim.py``, the worked exemplar; the convention prose
follows ``families/sliding_window.py``, which states its windowing rule
precisely enough that a reader cannot land off-by-one. The module exposes
exactly two names: ``VARIANT_KEYS`` and ``build(rng, level, key)``.

**Everything in this family is a convention.** Column-major versus row-major
flattening, whether ``dims`` is the shape before or after permuting, and above
all what "reorder the dimensions to [2 3 1]" means -- does dimension 2 of the
input become the new dimension 1, or does dimension 1 of the input move to
position 2? Those two readings are *inverse operations* and both are defensible
English. This is the family that shipped the undisclosed-convention defect
(PIPELINE_LOG, 2026-08-08 and 2026-08-09): its 0.4.x prompt described a phantom
argument ``A`` against the signature ``reshape_permute(x, dims)``, and its
reference carried an undisclosed ``out(:)'``. ``correct_given_executed`` was
**0.000** across 96 rollouts -- code that runs and cannot be right.

## The spec

Two dimensions, per ``PARAMETERIZATION_DESIGN.md``: which permutation of the
three dimensions, and whether the flattened result is returned as a row or as a
column.

The identity permutation ``[1 2 3]`` is **not shipped**: it returns the input
unchanged, so it is a degenerate problem at every level and would also make
level 1's answer a plain reshape of ``x``. That leaves five permutations and ten
combinations, of which eight ship:

- ``[2 3 1]`` and ``[3 1 2]`` -- the two 3-cycles, and each other's inverse --
  ship with **both** output orientations. They are the only two permutations for
  which the inverse reading of "reorder the dimensions" gives a different
  answer, so they carry the whole weight of the ambiguity this family exists to
  state away. They are also each other's wrong answer: a reader who inverts the
  convention on ``perm231`` computes exactly ``perm312``.
- ``[2 1 3]`` ships with both orientations as the easiest reading of the three
  transpositions (it is a page-wise transpose), so the orientation dimension is
  exercised on a permutation whose values are easy to check by hand.
- ``[1 3 2]`` and ``[3 2 1]`` ship once each, on opposite orientations, so all
  five non-identity permutations appear and neither orientation is a minority
  dialect: the split is four rows and four columns.

**The three transpositions are self-inverse, and that is stated because it
limits a check, not because it is a curiosity.** For ``[2 1 3]``, ``[1 3 2]``
and ``[3 2 1]`` the two readings of the description coincide, so the
"inverse-permutation" wrong solution *passes* those four variants -- measured,
not assumed: it scores **1.000** on ``perm213-row``, ``perm213-column``,
``perm132-row`` and ``perm321-column`` and **0.000** on all four
``perm231``/``perm312`` variants, over three levels and five seeds each. Any
non-vacuity check of the permutation convention has to be read that way round;
the flattening convention is probed separately (see below), and that probe
fails on all eight.

## The four things every description states, in this order

Each is an independent convention, and each has been an off-by-one or an
inversion in some version of this task:

1. **How ``x`` becomes ``A``** -- ``A = reshape(x, dims)``, spelled out
   element-wise (``x(1) = A(1,1,1)``, ``x(2) = A(2,1,1)``,
   ``x(dims(1)+1) = A(1,2,1)``) and then generalised ("the first subscript
   varies fastest along ``x`` and the third slowest"). ``dims`` is stated as the
   size of ``A``, i.e. the size **before** permuting.
2. **What ``B`` is, as an equation** -- ``B(i,j,k) = A(k,i,j)``, generated from
   the same permutation tuple that generates the expected values. An equation
   admits one reading. This is the sentence that makes the inverse reading
   impossible rather than merely discouraged.
3. **The same thing again as a dimension mapping** -- "dimension 1 of B is
   dimension 2 of A, ..." -- and **again as a size** -- "B has size
   ``[dims(2) dims(3) dims(1)]``". Three mutually redundant statements of one
   fact, so a reader who mis-parses one can check it against the other two, and
   so a reader who holds the inverse convention notices the contradiction
   instead of silently computing the inverse.
4. **The flattening order** -- "B's first subscript varying fastest and its
   third slowest, which is the order Octave's ``B(:)`` produces". Naming the
   idiom is deliberate: the flatten order is pure convention with no skill in
   it, whereas translating statement 2 into ``permute`` is the actual task.

What is deliberately **not** printed is the literal call ``permute(A, [2 3 1])``.
The equation already forbids the inverse reading, and printing the call would
reduce every level to transcription -- the model would not have to know which of
``permute`` and ``ipermute`` implements the stated mapping, which is the one
piece of Octave knowledge this family tests.

Output orientation is the one convention this file's prose does **not** state:
the generator appends a shape sentence derived from the expected values
themselves ("Return a row vector (1-by-N)."), so the prompt's claim and the
grader's ``size(actual)`` comparison cannot drift apart. That is
``sliding_window``'s and ``string_parse``'s arrangement, for the same reason.

### About ``(:)`` in the ``natural`` solution

The rule is that ``natural`` carries no defensive coercion and no transpose the
prompt did not ask for. Here ``B(:)`` is not coercion -- it *is* the operation
the description asks for, named in the description as the flattening order, and
it is applied to a value the solution just computed rather than to an argument.
No argument is reshaped or coerced in ``natural``: ``reshape(x, dims)`` uses
both arguments exactly as they arrive. The trailing ``'`` on the row variants is
asked for by the generated shape sentence. The other spelling a competent reader
might write instead -- ``reshape(B, 1, [])`` / ``reshape(B, [], 1)``, which
contains no ``(:)`` and no transpose at all -- was scored against the pinned
interpreter as a second natural reading and passes on all eight variants at all
three levels, 720 hidden cases.

## Hidden case shapes

The three sizes of every case are **distinct** and drawn from ``{2, 3, 4, 5}``,
which matters three times over:

- with two sizes equal, several permutations coincide in the *shape* of ``B``
  though not in its values, and with all three equal the shape carries no signal
  at all -- a solver could then permute wrongly and still produce a
  conformant-looking array;
- a size-1 dimension would make Octave drop a trailing singleton and would make
  some permutations no-ops on the values, an edge no description here states;
- distinct sizes mean a wrong ``dims`` reading raises rather than silently
  returning a plausible answer of the right length.

The flattened answer has the same length ``prod(dims)`` for every permutation,
so the graded shape never leaks which permutation was wanted -- the values do.

## The level ladder

Level 1: reshape, permute, flatten. Level 2: the same, with ``A`` **reversed
along its first dimension** first -- ``R(i,j,k) = A(dims(1)+1-i, j, k)`` -- and
that reversal is stated as an equation for the same reason ``B`` is. Level 3 is
level 2 verbatim plus "Do not use for/while loops".

Level 2 cannot collapse onto level 1: the two answers agree only if
``flip(A, 1) == A``, i.e. only if every fibre along dimension 1 is a palindrome.
With ``dims(1) >= 2`` and entries drawn from 19 integer values that is a
coincidence, and it is measured rather than argued: over **480 seeds x 6 cases
per variant (2,880 cases each, 23,040 in total)** the level-2 answer equals the
level-1 answer of the same case in **0 of 2,880** cases for every one of the
eight variants. The signature is identical at all three levels, so
``test_no_variant_has_a_level_two_its_level_one_solution_already_solves`` can
measure this family for real rather than passing vacuously on an arity error;
the level-1 ``natural`` scores 0.000 on level-2 cases for all eight keys at the
seed that test uses.

**Rejected level-2 steps.**

- *A second permutation applied after the first.* Rejected outright: composing a
  transposition with itself is the identity, so the three self-inverse variants
  would render a level 2 whose answer equals a plain flatten of ``x`` -- the
  ``reduce_along_dim`` symmetric-trim defect (PIPELINE_LOG, 2026-08-10) in a new
  costume, and degenerate for a different subset of the variants than the one an
  author would think to check.
- *Flattening in row-major order at level 2.* Rejected because it changes the
  *convention* between levels rather than the problem, and this is precisely the
  family where a reader must be able to trust that the convention is fixed. It
  would also make the level-3 prompt teach the opposite of the level-1 prompt.
- *A running total along one dimension*, the exemplar's ladder. It works and it
  cannot collapse, but three families in a ten-family pool already train
  ``cumsum``, and this branch exists to measure problem *diversity*. An index
  family's second level should ask for more index work.

## rng discipline

Every key draws the same values in the same order: three distinct sizes, then
``prod(sizes)`` entries. Nothing in the draw depends on ``key`` or on the
orientation, so a variant selection cannot shift the shared stream, which is
what ``test_a_family_generates_the_same_tasks_whichever_variants_are_present``
forbids. Note in passing the trap a sibling family hit today:
``rng.integers(0, 1, n)`` returns zeros **without consuming the stream**, so a
"constant" draw spelled that way is not a draw at all. No such spelling appears
here.

## Tolerance

The default ``1e-9`` is never exercised: every value is an integer in
``[-9, 9]``, and the task only moves values around, so both sides of every
comparison are bit-identical.

## What was measured, on the pinned GNU Octave 10.2.0

8 keys x 3 levels x 5 seeds (0, 1, 7, 41, 2026) = 120 executions per solution,
720 hidden cases per solution, through ``execute_candidate_locally``:

- ``reference`` -- 1.000 on 120 of 120.
- ``natural`` -- 1.000 on 120 of 120.
- second natural reading (``reshape``, no ``(:)``) -- 1.000 on 120 of 120.
- flattened row-major instead of column-major -- **0.000** on 120 of 120.
- right values, other orientation -- **0.000** on 120 of 120.
- inverse permutation -- **0.000** on the four 3-cycle variants; 1.000 on the
  four self-inverse ones, which is the identity noted above rather than a gap in
  the grader.

Level-2 non-collapse: 480 seeds x 6 cases per variant, level-2 expected against
level-1 expected of the same draw -- **0 of 2,880** identical for every key,
23,040 cases in total. The same census also confirms no level equals a plain
flatten of ``x`` (0 of 2,880 at both levels), and that at one seed the eight
keys produce eight distinct expected values. The level-1 ``natural`` run through
Octave against level-2 and level-3 hidden cases at the guard test's own seed
(4242) scores 0.000 with all six cases *executed*, so the zero is a measurement
and not an arity error.

rng discipline: at a fixed seed the eight keys produce identical ``args`` and
leave the generator in an identical state, at every level.

## One note for the change that registers this module

``tests/test_generation.py`` excludes ``reshape_permute`` from its
``same_task_at_level_three`` set, because the 0.4.x level 3 switched to a
*different permutation* than level 2. That is a variant dimension here, not a
level, so ``"reshape_permute"`` should be **added to that set** in the same
change that adds this module to ``generators.VARIANT_MODULES``; level 3 is level
2 verbatim plus the loop clause and passes that test today. Until then the
family is invisible to that guard.
"""

from __future__ import annotations

import numpy as np
from specs import Variant

# (permutation, output orientation). Order is the round-robin order and is part
# of the split contract -- appending is safe, reordering silently changes which
# task gets which problem.
VARIANT_KEYS: list[str] = [
    "perm213-row",
    "perm213-column",
    "perm231-row",
    "perm231-column",
    "perm312-row",
    "perm312-column",
    "perm132-row",
    "perm321-column",
]

# name -> the permutation in Octave's `permute` convention: dimension `k` of the
# result is dimension `perm[k-1]` of the input, so `size(B)(k) = dims(perm(k))`.
# The identity `[1 2 3]` is absent on purpose; see the module docstring.
_PERMUTATIONS: dict[str, tuple[int, int, int]] = {
    "perm213": (2, 1, 3),
    "perm231": (2, 3, 1),
    "perm312": (3, 1, 2),
    "perm132": (1, 3, 2),
    "perm321": (3, 2, 1),
}

# Hidden cases draw three *distinct* sizes from this pool. Distinctness is what
# keeps the shape of `B` a signal and keeps every permutation a real relabelling
# of the data; see the module docstring.
_SIZE_POOL = np.array([2, 3, 4, 5])

# The subscripts the descriptions use for `B`, in order.
_SUBSCRIPTS = ("i", "j", "k")


def _parse(key: str) -> tuple[tuple[int, int, int], str]:
    name, _, orientation = key.partition("-")
    if name not in _PERMUTATIONS or orientation not in ("row", "column"):
        raise ValueError(f"unknown reshape_permute variant {key!r}")
    return _PERMUTATIONS[name], orientation


def _element_identity(perm: tuple[int, int, int], source: str) -> str:
    """``B(i,j,k) = A(k,i,j)`` for this permutation, as the prompt states it.

    Generated from the same tuple that generates the expected values, so the
    equation in the prompt cannot disagree with the arithmetic in the grader --
    which is the entire point of the three-artefacts-from-one-definition rule.

    Subscript `t` of `B` indexes dimension `perm[t]` of the source array, so the
    source's subscript list is the inverse of `perm` applied to (i, j, k).
    """
    arguments = [""] * 3
    for position, dimension in enumerate(perm):
        arguments[dimension - 1] = _SUBSCRIPTS[position]
    return f"B(i,j,k) = {source}({','.join(arguments)})"


def _dimension_mapping(perm: tuple[int, int, int], source: str) -> str:
    """The same fact as a dimension mapping, for the reader to cross-check."""
    clauses = [
        f"dimension {position + 1} of B is dimension {dimension} of {source}"
        for position, dimension in enumerate(perm)
    ]
    return ", ".join(clauses[:-1]) + ", and " + clauses[-1]


def _describe(key: str, level: int) -> str:
    """The prompt text. Every level restates its own task in full.

    Level 3 saying only "...without loops" is what let `struct_cell_wrangle`
    level 3 fall from 0.792 to 0.000 while models guessed at the task from the
    family name alone. Level 3 here is level 2 verbatim plus the loop clause, so
    it cannot drop a term.

    The output orientation is deliberately absent: the generator appends the
    shape sentence it derives from the expected values, so the prompt's claim
    and the grader's comparison cannot drift apart.
    """
    perm, _ = _parse(key)
    size = " ".join(f"dims({dimension})" for dimension in perm)

    # 1. How `x` becomes `A`. `A = reshape(x, dims)` is the whole statement;
    #    "column-major" names the convention it implements. The element-wise
    #    instances that used to follow (x(1)=A(1,1,1), x(2)=A(2,1,1), ...) are
    #    gone -- measured 2026-08-11, they were not what made the prompt
    #    unambiguous, and length was costing real solves: at a 2048-token cap
    #    this family truncated 25% of 4B rollouts against 0-11% elsewhere, and
    #    raising the cap to 4096 moved 4B level 1 from 0.156 to 0.250.
    given = (
        "x is a row vector of dims(1)*dims(2)*dims(3) elements and dims is a "
        "1-by-3 vector of three distinct sizes, each at least 2. Let "
        "A = reshape(x, dims), the 3-D array holding x in column-major order."
    )

    # 2. The reversal, at levels 2 and 3, as an equation over subscripts.
    if level == 1:
        reversal, source = "", "A"
    else:
        reversal = (
            " Let R be A reversed along its first dimension: R has the same "
            "size as A and R(i,j,k) = A(dims(1)+1-i,j,k) for every valid "
            "i, j, k."
        )
        source = "R"

    # 3. What `B` is: the subscript equation and the resulting size. Two
    #    statements of one fact, not three -- the prose dimension mapping was
    #    dropped on 2026-08-11.
    #
    #    The equation is the load-bearing one and stays. "Reorder the dimensions
    #    to [2 1 3]" has two defensible inverse readings and this family shipped
    #    that defect once; a reader holding the wrong one has to contradict
    #    B(i,j,k) = A(...) and the stated size, which no phrasing of the prose
    #    mapping added to. Guarded by the inverse-permutation probe, which must
    #    keep scoring 0.000 on the four 3-cycle variants (the three self-inverse
    #    permutations cannot distinguish the readings and never could).
    build_b = (
        f" Form the 3-D array B of size [{size}] defined by "
        f"{_element_identity(perm, source)} for every valid i, j, k."
    )

    # 4. The flattening order, named as the idiom that produces it. `B(:)` is
    #    the shortest unambiguous statement of it, so the gloss goes.
    flatten = " Return the elements of B in column-major order, as B(:) gives them."

    task = given + reversal + build_b + flatten
    if level == 3:
        return task + " Do not use for/while loops."
    return task


def _oriented(values: list, orientation: str) -> list:
    """The expected value in the orientation the shape sentence will promise.

    A JSON list of scalars is a row (1-by-N); a column (N-by-1) is a list of
    one-element lists. See ``harness.octave_literal``.
    """
    return [[value] for value in values] if orientation == "column" else list(values)


def build(rng: np.random.Generator, level: int, key: str) -> Variant:
    perm, orientation = _parse(key)
    # NumPy's `transpose` axes are Octave's `permute` vector, zero-based:
    # `np.transpose(A, axes)[i,j,k] == A[...]` with `axes[t]` naming the source
    # dimension of result dimension `t`, which is exactly `permute`'s rule.
    axes = [dimension - 1 for dimension in perm]

    cases: list[dict] = []
    for _ in range(6):
        # One draw pattern for every key and level: three distinct sizes, then
        # one entry per element. Nothing here reads `key` or `orientation`, so a
        # variant selection cannot move the shared rng stream.
        dims = [int(value) for value in rng.choice(_SIZE_POOL, 3, replace=False)]
        x = rng.integers(-9, 10, int(np.prod(dims)))

        # `order="F"` is the column-major reading the description states.
        A = np.reshape(x, dims, order="F")
        if level > 1:
            A = A[::-1, :, :]
        B = np.transpose(A, axes)
        out = B.flatten(order="F")

        cases.append({
            "args": [x.tolist(), dims],
            "expected": _oriented(out.tolist(), orientation),
        })

    signature = "function out = reshape_permute(x, dims)"
    octave_perm = "[" + " ".join(str(dimension) for dimension in perm) + "]"
    # The graded orientation, which the appended shape sentence states.
    transpose = "'" if orientation == "row" else ""

    # The reference may coerce, and does: `x(:)` and `dims(:)'` make it correct
    # whatever orientation those arguments arrive in, and `flip` is a different
    # spelling of the reversal than the naive solution's index expression, so
    # the two are not the same code twice.
    reversal = " R = flip(A, 1);\n" if level > 1 else ""
    reference = (
        f"{signature}\n"
        " A = reshape(x(:), dims(:)');\n"
        f"{reversal}"
        f" B = permute({'R' if level > 1 else 'A'}, {octave_perm});\n"
        f" out = B(:){transpose};\n"
        "endfunction"
    )

    # What a competent Octave programmer writes from the description alone: no
    # argument is coerced, the reversal is the index expression the description
    # spells out, and the only transpose is the one the shape sentence asks for.
    # Loop-free at every level, so it also satisfies the level-3 constraint.
    naive_reversal = " R = A(end:-1:1,:,:);\n" if level > 1 else ""
    natural = (
        f"{signature}\n"
        " A = reshape(x, dims);\n"
        f"{naive_reversal}"
        f" B = permute({'R' if level > 1 else 'A'}, {octave_perm});\n"
        f" out = B(:){transpose};\n"
        "endfunction"
    )

    return Variant(
        key=key,
        description=_describe(key, level),
        signature=signature,
        cases=cases,
        reference=reference,
        natural=natural,
        vectorized=level == 3,
    )
