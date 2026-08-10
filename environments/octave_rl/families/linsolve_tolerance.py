"""``linsolve_tolerance``: solve a linear system, and report the solution or its residual.

Written against the worked exemplar in ``reduce_along_dim.py`` -- same two public
names, same level-ladder discipline. Read ``specs.py`` first for why the variant
is an argument rather than an rng draw, and for the rule that
``description``/``reference``/``natural`` are written together so they cannot
drift.

**This is one of the three families that shipped the undisclosed-convention
defect, and the only one that was unsolvable rather than merely hard.** In 0.4.x
``b`` was serialised with ``b.tolist()``, which is a ``1xm`` *row* in Octave, so
``A\\b`` was nonconformant on every hidden case; the reference passed because it
carried an undisclosed ``b=b(:)``. Measured pass rate 0.030/0.000 with
``execution_fraction`` 0.058 -- code that cannot run (PIPELINE_LOG, 2026-08-09).
Every right-hand side here is serialised as a nested list, which is an ``mx1``
**column**, so ``A \\ b`` is conformant exactly as written and no solution below
-- reference or natural -- contains a ``(:)``, a ``reshape`` or a transpose.

## The spec, and the cells of it that cannot ship

``PARAMETERIZATION_DESIGN.md`` proposes ``return in {x, [x; ||Ax-b||], ||Ax-b||
alone, [x; rank]}`` crossed with ``system in {square, over-determined}``. One of
those four returns cannot ship at all, and two more cannot ship in the square
column.

**Every residual cell of the square column is identically zero.** A square
nonsingular system has an exact solution, so ``A*x - b`` is zero to rounding for
*every* draw at *every* level. Shipped, ``residualnorm-square`` would have been
a variant whose answer is the constant 0, gradeable by ``out = 0;`` -- and the
0.4.x level 3 came within one draw of doing exactly that, since it asked for
``[x; norm(A*x-b)]`` over a ``b`` built as ``A @ x0``, in the range of ``A`` by
construction. That final entry was ~1e-15 on every hidden case at every level,
graded against a tolerance of 1e-7. One sixth of the graded output of the
family's hardest level was a constant.

Measured on the shipped draws: the residual norm of a square variant's solution
is at most **1.6e-15** -- zero, as claimed -- while the over-determined variants'
residual norms run **0.037 to 3.66**. So the residual returns ship
over-determined only, and with a right-hand side drawn **independently of**
``A`` rather than as ``A @ x0``, which is what puts ``b`` outside the range of
``A`` and makes the residual a number worth computing.

**``rank`` is dropped, and it is dropped twice over.** ``rank(A)`` is a
tolerance convention rather than a computation -- Octave thresholds the singular
values at ``max(size(A))*eps(max(s))``, NumPy at its own ``rcond`` -- so the two
agree only when the answer is unambiguous. Making it unambiguous means drawing a
full-rank ``A``, and then the rank of every hidden case is ``size(A, 2)``: a
constant, readable off the input's shape without touching a single entry,
exactly the defect ``struct_cell_wrangle`` had to rewrite its ``count`` row to
avoid. Confirmed by execution rather than assumed -- over the 720 matrices five
seeds of the shipped draws produce (5 seeds x 8 keys x 3 levels x 6 cases),
Octave reports:

    rank(A) < columns(A)          0 / 720
    smallest singular value       1.0        (exactly, by construction)
    largest default rank tol      8.0e-15

a margin of fourteen orders of magnitude, and a row of the answer that no
computation can get wrong. The alternative, drawing rank-deficient matrices,
makes ``A\\b`` non-unique: Octave's QR with column
pivoting returns a *basic* solution with at most ``rank(A)`` nonzeros and
``numpy.linalg.lstsq`` returns the *minimum-norm* one, and they are different
vectors. A variant whose answer depends on which algorithm the grader ran is not
a task.

That leaves six returns crossed with two systems, of which the eight below are
the cells that are both well posed and distinct. The square column keeps the two
returns that do not mention a residual.

## The numerics, which is what this family actually rests on

Unlike every other converted family, nothing here is integer-exact. ``A\\b`` in
Octave and ``numpy.linalg.lstsq`` in NumPy are **different algorithms** -- QR
with column pivoting against a divide-and-conquer SVD -- and they agree only to
the conditioning of the system. The tolerance is 1e-7 (inherited from 0.4.x,
which set it for this reason), and the comparison the grader makes is
``abs(actual - expected) <= tol * max(1, abs(expected))``.

So ``A`` is **constructed** with a known condition number rather than drawn and
hoped over. Each case takes two Gaussian blocks, orthonormalises them, and forms
``A = Q * diag(s) * V'`` with ``s`` geometric from 1 to 4: singular values
exactly ``s``, condition number exactly 4, and entries that still look like
ordinary floating-point noise. A bare ``rng.standard_normal((m, n))`` would have
been simpler and has a fat tail -- a few percent of 3x3 Gaussian draws have
condition number above 100 -- which is a task decided by which draw the seed
produced.

Measured by execution on the pinned GNU Octave 10.2.0, comparing what Octave
actually returns for every reference and natural solution against the NumPy
expected values -- 8 keys x 3 levels x 18 seeds x 6 cases, 5,184 graded answers,
25,530 individual numbers:

    worst difference on the grader's own scale, |a-e| / max(1, |e|)   1.3e-15
    worst elementwise true relative difference, |a-e| / |e|           1.4e-12
    tolerance                                                         1e-7

The first number is the one the grader computes, and it clears the tolerance by
eight orders of magnitude. The second is reported beside it because it is the
larger and the more honest: it is a single residual-vector entry of magnitude
~8e-4 -- a near-cancellation, where a ~1e-15 absolute difference is a large
*relative* one. The grader's ``max(1, |e|)`` floor is what makes that
harmless, and every such entry is part of a vector whose norm is order 1.

The headroom that remains is deliberate rather than slack: 1e-7 also forgives a
solver that reaches the same answer by another route -- ``pinv(A)*b``, or the
normal equations ``(A'*A)\\(A'*b)``, which costs ``cond(A)^2 = 16`` times eps and
so also lands near 1e-15. The tolerance is *not* tightened to the 1e-9 default
that the measurement would support, because the margin it buys is over
algorithms, not over noise.

## The level ladder, and the honest word about level 3

Level 2 subtracts the mean of the right-hand side from every one of its entries
and solves for that instead. It is a one-argument step, so this family's level 1
and level 2 have the **same signature** and the repository's guard,
``test_no_variant_has_a_level_two_its_level_one_solution_already_solves``, is a
real measurement here rather than the vacuous arity error it is for
``struct_cell_wrangle`` and ``sliding_window``. Two measurements, because a
guard that can run is still only a guard:

- *The census.* For each variant, the level-2 answer against the level-1 answer
  of the same inputs, on the grader's own comparison, 40 seeds x 6 cases:
  **0 / 240 coincide, for every one of the eight variants**. The smallest gap
  anywhere in the 1,920 cases is **3.9e-4**, which is 3,900 times the 1e-7 that
  would let one through; per variant the smallest gaps are 3.9e-4
  (``residualnorm``), 1.9e-3 (``residualvector``), 3.8e-3 (both
  ``solution``/``solutionresidual`` over-determined), 3.9e-3 (``solution-square``),
  4.0e-3 (``multirhsresidual``), 3.0e-2 and 3.6e-2 (the two ``multirhs``).
- *The guard, actually run.* The level-1 **natural** solution executed on the
  level-2 hidden cases in Octave, 5 seeds x 6 cases: **0/30 for every variant**,
  with ``executed`` 30/30 throughout -- the zeros are wrong answers, not arity
  errors, which is exactly what the four arity-changing families cannot show.

**Three level-2 steps were rejected, each degenerate for a specific variant:**

- *Scale each column of ``A`` to unit norm and solve the scaled system.* The
  prettiest option, standard practice, and fatal: column scaling by an
  invertible ``D`` reparameterises the minimisation without changing its value,
  so ``min ||A D y - b|| = min ||A x - b||`` **exactly**. All three residual
  variants would have had a level 2 whose answer equalled their level 1 by
  construction -- the ``reduce_along_dim`` median defect wearing a different hat.
- *Scale each equation (row) to unit norm.* Kills the other column instead: row
  scaling of a square nonsingular system leaves ``x`` unchanged exactly, so both
  ``square`` variants would have been degenerate.
- *Replace ``b`` by the row sums of ``A``.* Moves every variant, and makes the
  answer the constant vector of ones.

A running total over ``b`` was also available and was left alone: that is
``reduce_along_dim``'s ladder and ``broadcast_arith``'s, and re-using it here
would make a family holdout buy a step the model trained on elsewhere.

**Level 3 is a weak level for this family and this is not papered over.** It is
level 2 plus "Do not use for/while loops", and the natural solution to every
variant here is already a loop-free one-liner or two-liner -- ``A\\b`` *is* the
vectorized form, and there is no loop-shaped way to write it that a competent
reader would reach for first. The one variant where the constraint has any bite
is ``multirhsresidual-overdetermined``, where the obvious answer is a loop over
the columns of ``B`` taking ``norm`` of each; ``vecnorm`` or
``sqrt(sum(R.^2))`` replaces it. For the other seven, level 3 is level 2 with a
sentence that costs nothing, and per-variant pass rates at level 3 should be
read as a restatement of level 2 rather than as a harder rung. Making it a real
level would mean banning the backslash operator, which grades Octave trivia
instead of linear algebra, so the ladder is left short and labelled.

## What was actually run

On the pinned GNU Octave 10.2.0 through ``executors.execute_candidate_locally``,
which is the same path the reward uses:

- ``reference`` and ``natural``, all 8 keys x 3 levels x 18 seeds x 6 hidden
  cases: **5,184 / 5,184 hidden cases passed**, no cell below 1.0. Plus 324/324
  through ``build_tasks`` itself on a nineteenth seed, which is the path that
  assembles the prompt and threads ``natural`` to the validator.
- Non-vacuity, because a grader that accepts everything also accepts a correct
  answer. One deliberately wrong solution per return kind, 3 seeds x 3 levels:
  **0 / 18 for every one of the 24 (key, level) cells**. The wrong answers are
  the ones a competent reader actually produces -- a transposed solution, a
  residual computed before solving (``norm(b)``, i.e. the residual at ``x = 0``),
  the residual with its sign flipped to ``b - A*x``, and a column of per-column
  norms where a row is graded.
- The degeneracy probe and the rank probe recorded above.
"""

from __future__ import annotations

import numpy as np
from specs import Variant

# (return, system). Order is the round-robin order and is part of the split
# contract -- appending is safe, reordering silently changes which task gets
# which problem.
#
# The order is chosen for what `DEFAULT_HELDOUT_VARIANTS` -- the last two keys of
# each family -- takes away. Both held-out keys are recombinations whose parts
# stay in training: `residualvector-overdetermined` holds out the residual *as a
# vector* while the residual norm is still trained by three other keys, and
# `multirhs-overdetermined` holds out the least-squares solve with several
# right-hand sides while both the multi-column layout (`multirhs-square`) and
# the over-determined solve (`solution-overdetermined`) are trained. Ordering
# them the obvious way -- grouped by system -- would have deleted every
# over-determined residual from training at once.
VARIANT_KEYS: list[str] = [
    "solution-square",
    "solution-overdetermined",
    "residualnorm-overdetermined",
    "solutionresidual-overdetermined",
    "multirhs-square",
    "multirhsresidual-overdetermined",
    "residualvector-overdetermined",
    "multirhs-overdetermined",
]

# The valid cells of the (return, system) grid. Five of the twelve conceptual
# cells are missing on purpose and the module docstring says why: the four
# residual returns are identically zero for a square nonsingular system, and
# `rank` is dropped outright.
_PROBLEMS: dict[str, tuple[str, str]] = {
    "solution-square": ("solution", "square"),
    "solution-overdetermined": ("solution", "overdetermined"),
    "residualnorm-overdetermined": ("residualnorm", "overdetermined"),
    "solutionresidual-overdetermined": ("solutionresidual", "overdetermined"),
    "multirhs-square": ("multirhs", "square"),
    "multirhsresidual-overdetermined": ("multirhsresidual", "overdetermined"),
    "residualvector-overdetermined": ("residualvector", "overdetermined"),
    "multirhs-overdetermined": ("multirhs", "overdetermined"),
}

# Draw envelope. Every key draws the *same* fixed-size blocks and slices what it
# needs out of them, so the rng advances by an identical number of values --
# indeed by identical values -- whichever variant is being rendered. Sizing the
# blocks per key would be the obvious optimisation and would make the shared
# stream depend on the variant selection.
_N_MAX = 5  # most unknowns
_M_MAX = 9  # most equations: _N_MAX + the largest surplus
_RHS_COLUMNS = 2  # right-hand sides in the multi-column variants
# Singular values run geometrically from 1 to this, so cond(A) is exactly this
# for every case. Small enough that QR-with-pivoting and SVD agree to ~1e-15,
# large enough that the system is not a disguised orthogonal matrix.
_CONDITION = 4.0
_CASES = 6


def _parse(key: str) -> tuple[str, str]:
    if key not in _PROBLEMS:
        raise ValueError(f"unknown linsolve_tolerance variant {key!r}")
    return _PROBLEMS[key]


def _matrix(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    """An ``m``-by-``n`` matrix with prescribed singular values.

    ``Q`` is an orthonormal basis of the drawn block's column space and ``V`` is
    a random orthogonal matrix, so ``Q * diag(s) * V'`` is an ordinary-looking
    dense matrix whose singular values are exactly ``s`` and whose condition
    number is therefore exactly ``_CONDITION``. This is the whole reason the
    family's floating-point agreement is a property of the generator rather than
    of the seed.
    """
    orthonormal, _ = np.linalg.qr(left)  # m-by-n, orthonormal columns
    rotation, _ = np.linalg.qr(right)  # n-by-n, orthogonal
    spectrum = np.geomspace(1.0, _CONDITION, left.shape[1])
    return (orthonormal * spectrum) @ rotation.T


def _solve(A: np.ndarray, rhs: np.ndarray, system: str) -> np.ndarray:
    """The solution Octave's ``\\`` computes, in the same solution concept.

    Square: the exact solution, LU with partial pivoting on both sides.
    Over-determined and full column rank: *the* least-squares minimiser, which
    is unique, so Octave's pivoted QR and NumPy's SVD are two routes to one
    answer rather than two conventions. Rank deficiency would break that and is
    why no draw here is rank deficient -- see the module docstring.
    """
    if system == "square":
        return np.linalg.solve(A, rhs)
    return np.linalg.lstsq(A, rhs, rcond=None)[0]


def _describe(key: str, level: int) -> str:
    """The prompt text. Every level restates its own task in full.

    Level 3 saying only "...without loops" is what let `struct_cell_wrangle`
    level 3 fall from 0.792 to 0.000 while models guessed the task from the
    family name. Note that this family's name is no help either: nothing here
    involves a tolerance, `linsolve` is not the function being written, and no
    description below leans on either word.

    Two conventions that a reader cannot derive are stated outright rather than
    left to be guessed, because an unstated convention is what made this family
    unsolvable once already: the residual is `A*x - b` **in that order** and not
    `b - A*x`, and the least-squares solution is pinned as the unique minimiser
    of `norm(A*x - b)` under full column rank rather than left to whichever
    solution concept the reader's solver happens to pick. The output *shape* is
    appended by the prompt builder from the expected values, so it cannot drift
    from the grader.
    """
    kind, system = _parse(key)
    multi = kind.startswith("multirhs")
    given = "B" if multi else "b"
    # The right-hand side the task is actually about: the given one at level 1,
    # the centred one at levels 2 and 3.
    rhs = given if level == 1 else ("C" if multi else "c")

    # Sentence 1: what the inputs are. Stated at every level, because the family
    # name says nothing true about them.
    if system == "square":
        setup = "A is a square nonsingular matrix of real numbers."
    else:
        setup = (
            "A is a matrix of real numbers with more rows than columns, and it "
            "has full column rank."
        )
    if multi:
        setup += (
            " B is a matrix of real numbers with one row per row of A and two "
            "columns; treat each of its columns as a separate right-hand side."
        )
    else:
        setup += " b is a column vector with one entry per row of A."

    # Sentence 2, levels 2 and 3 only: the step that makes this a new problem.
    if level == 1:
        transform = ""
    elif multi:
        transform = (
            " Let C be B with each column reduced by that column's own mean, so "
            "that column j of C is column j of B with the scalar mean of column "
            "j of B subtracted from every one of its entries."
        )
    else:
        transform = (
            " Let c be b with the mean of its entries subtracted from every one "
            "of its entries, so that c = b - mean(b)."
        )

    # Sentence 3: which vector the answer is built from. The solution concept is
    # pinned here rather than left to whichever one the reader's solver picks.
    if system == "square" and not multi:
        define = f" Let x be the solution of A*x = {rhs}."
    elif system == "square":
        define = (
            f" Let X be the matrix whose column j solves A*X(:,j) = {rhs}(:,j)."
        )
    elif not multi:
        define = (
            f" The system A*x = {rhs} generally has no exact solution; let x be "
            f"its least-squares solution, the vector minimising "
            f"norm(A*x - {rhs}), which is unique because A has full column rank."
        )
    else:
        define = (
            f" In general no column of {rhs} lies in the range of A; let X be "
            f"the matrix whose column j is the least-squares solution for that "
            f"column, the vector minimising norm(A*X(:,j) - {rhs}(:,j)), which "
            "is unique because A has full column rank."
        )

    # Sentence 4: what to return, in terms of x or X. The output *shape* is
    # appended by the prompt builder; the output *layout* is named here.
    if kind == "solution":
        task = "Return x."
    elif kind == "multirhs":
        task = (
            "Return X, which has one row per column of A and one column per "
            f"column of {rhs}."
        )
    elif kind == "residualnorm":
        task = (
            f"Return the Euclidean norm of the residual, norm(A*x - {rhs}), as "
            "a single number."
        )
    elif kind == "solutionresidual":
        task = (
            f"Return the column vector [x; norm(A*x - {rhs})]: the entries of x "
            "followed by one final entry holding the Euclidean norm of the "
            "residual."
        )
    elif kind == "residualvector":
        task = (
            f"Return the residual vector A*x - {rhs} -- in that order, not "
            f"{rhs} - A*x -- which has one entry per row of A."
        )
    else:  # multirhsresidual
        task = (
            f"Return one entry per column of {rhs}: entry j is the Euclidean "
            f"norm of that column's residual, norm(A*X(:,j) - {rhs}(:,j))."
        )

    described = f"{setup}{transform}{define} {task}"
    if level == 3:
        return described + " Do not use for/while loops."
    return described


def build(rng: np.random.Generator, level: int, key: str) -> Variant:
    kind, system = _parse(key)
    multi = kind.startswith("multirhs")

    cases: list[dict] = []
    for _ in range(_CASES):
        # One draw pattern for every variant and every level: two shape integers
        # and three fixed-size Gaussian blocks, sliced afterwards. Nothing here
        # reads `key`, so a variant selection cannot shift the shared rng stream
        # -- `build_tasks` draws every family's every task from one generator, so
        # a key-dependent draw count would silently reshuffle every later task.
        #
        # Note the shape draws are `integers(low, high)` with `high > low`:
        # `rng.integers(0, 1, n)` returns zeros *without consuming the stream*,
        # which is how a sibling family quietly desynchronised its draws today.
        n = int(rng.integers(3, 6))  # 3 to 5 unknowns
        surplus = int(rng.integers(2, 5))  # 2 to 4 extra equations
        left = rng.standard_normal((_M_MAX, _N_MAX))
        right = rng.standard_normal((_N_MAX, _N_MAX))
        given = rng.standard_normal((_M_MAX, _RHS_COLUMNS))

        m = n if system == "square" else n + surplus
        A = _matrix(left[:m, :n], right[:n, :n])
        # Kept two-dimensional even in the single-right-hand-side case, so
        # `.tolist()` nests and the argument reaches Octave as an m-by-1 COLUMN.
        # A flat list is a 1-by-m row, and `A\b` on a row is the nonconformant
        # error that made every hidden case of this family fail in 0.4.x
        # regardless of what the model wrote.
        rhs = given[:m, :] if multi else given[:m, :1]
        # The level-2 step. `b` itself is still what the function is handed --
        # the transformation is part of the task, not of the input.
        target = rhs - rhs.mean(axis=0) if level > 1 else rhs

        solution = _solve(A, target, system)
        residual = A @ solution - target
        norms = np.linalg.norm(residual, axis=0)

        if kind == "solution" or kind == "multirhs":
            expected = solution.tolist()
        elif kind == "residualnorm":
            # A bare Python float, so `_shape_sentence` says "Return a scalar."
            # and the grader compares against Octave's 1-by-1.
            expected = float(norms[0])
        elif kind == "solutionresidual":
            expected = np.vstack([solution, [[norms[0]]]]).tolist()
        elif kind == "residualvector":
            expected = residual.tolist()
        else:  # multirhsresidual -- a flat list is a 1-by-k ROW
            expected = norms.tolist()

        cases.append({"args": [A.tolist(), rhs.tolist()], "expected": expected})

    given_name = "B" if multi else "b"
    signature = f"function out = linsolve_tolerance(A, {given_name})"

    # `reference` may coerce; it does not need to, and deliberately does not.
    # The right-hand side already arrives with the orientation its signature
    # implies, so the two solutions below differ only where the reference names
    # a dimension or a norm order that the natural solution leaves to Octave's
    # default. There is no `(:)` anywhere in this file, which is the property
    # 0.4.x lacked.
    if level == 1:
        prologue_reference = ""
        prologue_natural = ""
        rhs_name = given_name
    else:
        rhs_name = "C" if multi else "c"
        # `mean(b, 1)` and `mean(B, 1)` name the dimension; every case has at
        # least three rows, so the implicit form the natural solution uses means
        # the same thing.
        prologue_reference = f" {rhs_name} = {given_name} - mean({given_name}, 1);\n"
        prologue_natural = f" {rhs_name} = {given_name} - mean({given_name});\n"

    if kind == "solution" or kind == "multirhs":
        reference_body = f" out = A \\ {rhs_name};"
        natural_body = reference_body
    elif kind == "residualnorm":
        reference_body = f" x = A \\ {rhs_name};\n out = norm(A*x - {rhs_name}, 2);"
        natural_body = f" x = A \\ {rhs_name};\n out = norm(A*x - {rhs_name});"
    elif kind == "solutionresidual":
        reference_body = (
            f" x = A \\ {rhs_name};\n out = [x; norm(A*x - {rhs_name}, 2)];"
        )
        natural_body = f" x = A \\ {rhs_name};\n out = [x; norm(A*x - {rhs_name})];"
    elif kind == "residualvector":
        reference_body = f" x = A \\ {rhs_name};\n out = A*x - {rhs_name};"
        natural_body = reference_body
    else:  # multirhsresidual
        # The one place level 3's loop ban has any bite: the obvious answer is a
        # loop over the columns taking `norm` of each. `vecnorm` reduces along
        # the first non-singleton dimension, which is the column direction here,
        # and returns the 1-by-k row the grader compares against.
        reference_body = (
            f" X = A \\ {rhs_name};\n R = A*X - {rhs_name};"
            "\n out = sqrt(sum(R .^ 2, 1));"
        )
        natural_body = (
            f" X = A \\ {rhs_name};\n out = vecnorm(A*X - {rhs_name});"
        )

    reference = f"{signature}\n{prologue_reference}{reference_body}\nendfunction"
    # What a competent Octave programmer writes from the description alone: no
    # `(:)`, no reshape, no transpose the prompt did not ask for. If this cannot
    # pass, the variant is not shippable.
    natural = f"{signature}\n{prologue_natural}{natural_body}\nendfunction"

    return Variant(
        key=key,
        description=_describe(key, level),
        signature=signature,
        cases=cases,
        reference=reference,
        natural=natural,
        # 1e-7 relative, inherited from 0.4.x, which set it because this family
        # is floating point end to end. The measured worst Octave-versus-NumPy
        # difference is 4.0e-15; the remaining headroom is for a solver that
        # reaches the same answer by another route. See the module docstring.
        tolerance=1e-7,
        vectorized=level == 3,
    )
