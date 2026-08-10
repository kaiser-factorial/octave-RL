"""``signal_identity``: shift, correlate, convolve -- with the convention stated.

Written against the worked exemplar in ``reduce_along_dim.py`` -- same two
public names, same level-ladder discipline. Read ``specs.py`` first for why the
variant is an argument rather than an rng draw, and for the rule that
``description``/``reference``/``natural`` are written together so they cannot
drift.

## The spec

Two dimensions, per ``PARAMETERIZATION_DESIGN.md``: which operation, and which
*convention* that operation is ambiguous about. Four operations times two
conventions is exactly eight, and each pair is chosen so that getting the
convention backwards produces a wrong answer rather than the same one:

| pair | the convention at stake | what the wrong choice gives |
|---|---|---|
| ``shift-forward`` / ``shift-backward`` | does positive ``k`` move entries toward higher or lower indices | the shift by ``-k`` |
| ``autocorr-circular`` / ``autocorr-linear`` | does the index wrap | different values, same length |
| ``xcorr-forward`` / ``xcorr-reversed`` | which argument the lag shifts | the lag-reversed vector |
| ``conv-circular`` / ``conv-linear`` | does the index wrap | different values *and* a different length |

**Lag sign was rejected as the second dimension for autocorrelation**, which is
where this family's obvious eight-way grid falls apart. Circular
autocorrelation of a real vector satisfies ``r(-m) = r(m)`` exactly -- summing
``x(i)*x(i-m)`` over a full period is the same sum as ``x(i)*x(i+m)`` after
re-indexing -- so "lag increasing" and "lag decreasing" would have been two
descriptions of one problem, sharing every hidden expected value. That is worse
than the degeneracy in PIPELINE_LOG's 2026-08-10 entry: not a level whose
answer repeats, but a *variant* whose answer repeats. Wrapping is the second
dimension instead, and it moves every value. The lag-sign convention still gets
tested, on cross-correlation, where ``c_ab(m) = c_ba(-m)`` genuinely differ.

## No FFT, on purpose

The 0.4.x generator computed its expected values with ``np.fft`` and had to
``round(..., 12)`` them and grade at ``1e-7``, because FFT kernels differ by a
few ulps across SIMD code paths and a byte-unstable task is a task whose answer
depends on which machine drew it. Every variant here is defined by an explicit
sum over indices instead, evaluated in integer arithmetic on integer inputs, so
the expected values are exact doubles and the graded answer does not depend on
any FFT implementation. The tolerance is therefore the repo default ``1e-9``
rather than ``1e-7``.

That is a strictly weaker constraint on the *solver*: a candidate that reaches
for ``real(ifft(fft(a).*fft(b)))`` still passes, because its round-off is many
orders of magnitude inside the tolerance -- measured at worst ``4.5e-12``
absolute (relative ``3.4e-16``) over all eight variants and three levels on the
pinned Octave 10.2.0, against a ``1e-9`` budget. What changed is that the
*task* no longer depends on that.

## The level ladder

Level 3 is level 2 plus a vectorization constraint, restating its own task in
full (a bare "...without loops" is what dropped ``struct_cell_wrangle`` level 3
from 0.792 to 0.000). Level 2 replaces each input vector by its **running
total** and then applies the same operation -- the exemplar's ladder, chosen
here for the same reason: it changes the values being combined rather than
which of them are combined, so it moves every one of the four operations, and
it stays in exact integer arithmetic.

Two degeneracies are excluded by construction rather than left to luck:

- ``cumsum(x) == x`` holds exactly when ``x`` is zero everywhere but its last
  entry, which would make level 2 identical to level 1 for the shift variants.
  Every hidden case therefore forces ``x(1) ~= 0`` (and ``b(1) ~= 0``), which
  makes ``cumsum`` move entry 2 of every input.
- ``mod(k, n) == 0`` makes a circular shift the identity, and ``mod(k, n) ==
  n/2`` makes the forward and backward shifts agree -- the second would let a
  solution with the direction backwards score full marks. The drawn ``k`` is
  walked to the next integer avoiding both residues, without consuming rng.

Level 1 and level 2 take the **same arguments** for every variant, so
``test_no_variant_has_a_level_two_its_level_one_solution_already_solves`` runs a
real probe here rather than dying on an arity error. The family's own census
agrees: over 480 cases per variant (80 seeds x 6 cases), the level-2 answer
differs from the level-1 answer on 480/480 for all eight variants.
"""

from __future__ import annotations

import numpy as np
from specs import Variant

# (operation, convention). Order is the round-robin order and is part of the
# split contract -- appending is safe, reordering silently changes which task
# gets which problem.
VARIANT_KEYS: list[str] = [
    "shift-forward",
    "shift-backward",
    "autocorr-circular",
    "autocorr-linear",
    "xcorr-forward",
    "xcorr-reversed",
    "conv-circular",
    "conv-linear",
]

# Hidden inputs are drawn from this range. Small integers keep every product and
# every partial sum exactly representable: at the level-2 extreme a running
# total reaches 9*6 = 54, a product 2916, and a sum of nine of those ~26k --
# integers a double holds without loss, which is what makes the tolerance a
# formality rather than a load-bearing choice.
_VALUE_LOW, _VALUE_HIGH = -6, 7


def _circular_correlation(first: np.ndarray, second: np.ndarray) -> np.ndarray:
    """``c(m) = sum_i first(i) * second(mod(i-1+m, n)+1)``, lags ``m = 0..n-1``.

    The lag shifts ``second`` and leaves ``first`` in place, which is the whole
    content of the ``xcorr-forward`` / ``xcorr-reversed`` distinction:
    ``c_ab(m) = c_ba(mod(-m, n))``, so swapping the arguments reverses the lag
    axis. Written as one function so the two variants cannot disagree about
    what "the lag shifts b" means.
    """
    n = len(first)
    # Rows are lags, columns are i-1: entry (m, i-1) is the index into `second`.
    index = (np.arange(n)[:, None] + np.arange(n)) % n
    return (second[index] * first).sum(axis=1)


def _linear_correlation(x: np.ndarray) -> np.ndarray:
    """``r(m) = sum_{i=1}^{n-m} x(i) * x(i+m)``, lags ``m = 0..n-1``, no wrap.

    Same length as the circular version and different values: the circular
    ``r(m)`` is this one's ``r(m) + r(n-m)``. That is what makes the two
    autocorrelation variants distinct problems rather than distinct sentences.
    """
    n = len(x)
    return np.array([int(np.dot(x[: n - m], x[m:])) for m in range(n)])


def _circular_convolution(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """``y(j) = sum_i a(i) * b(mod(j-i, n)+1)`` for ``j = 1..n``."""
    n = len(a)
    # Rows are j-1, columns are i-1: entry (j-1, i-1) is the index into `b`.
    index = (np.arange(1, n + 1)[:, None] - np.arange(1, n + 1)) % n
    return (b[index] * a).sum(axis=1)


# key -> everything that distinguishes one variant from another.
#
# `params`   the signature's arguments, identical at every level.
# `inputs`   the sentence that introduces them.
# `running`  the sentence that introduces the level-2 running totals.
# `task`     the task itself, over symbols {p} (and {q}), so level 2 can restate
#            it verbatim over the running totals instead of the inputs.
# `symbols`  ((level-1 symbols), (level-2 symbols)) substituted into `task`.
# `compute`  NumPy expected value, over the already-transformed inputs.
# `loop`     Octave body a reader writes while loops are allowed.
# `vector`   Octave body a reader writes once they are not.
_ONE_INPUT = "x is a row vector with n = numel(x) entries."
_TWO_INPUTS = "a and b are row vectors with the same number of entries, n = numel(a)."
_ONE_RUNNING = (
    "First form v, the running total of x from left to right, so that "
    "v(i) = x(1) + x(2) + ... + x(i) for i = 1..n; v also has n entries."
)
_TWO_RUNNING = (
    "First form u and v, the running totals of a and b from left to right, so "
    "that u(i) = a(1) + ... + a(i) and v(i) = b(1) + ... + b(i); u and v also "
    "have n entries each."
)

_LAG_ORDER = (
    "for lags m = 0, 1, ..., n-1 in that order, so lag zero comes first"
)

_VARIANTS: dict[str, dict] = {
    # -- circular shift: the direction convention ------------------------------
    #
    # Octave's `circshift(x, k)` and NumPy's `np.roll(x, k)` both move entries
    # toward *higher* indices for positive k, and a reader may reasonably assume
    # the opposite, so both descriptions give the index map in both directions
    # -- where entry i goes, and where entry j came from -- and neither leans on
    # the word "right".
    "shift-forward": {
        "params": ("x", "k"),
        "inputs": (
            "x is a row vector with n = numel(x) entries and k is an integer, "
            "which may be negative and may exceed n."
        ),
        "running": _ONE_RUNNING,
        "task": (
            "Return the circular shift of {p} by k positions toward higher "
            "indices: the result has n entries, entry i of {p} becomes entry "
            "mod(i-1+k, n)+1 of the result, and equivalently entry j of the "
            "result is {p}(mod(j-1-k, n)+1). The shift wraps around, so every "
            "entry of {p} appears exactly once in the result."
        ),
        "symbols": (("x",), ("v",)),
        "compute": lambda d: np.roll(d["x"], d["k"]),
        "loop": " out = circshift(x, k);",
        "vector": " out = circshift(x, k);",
    },
    "shift-backward": {
        "params": ("x", "k"),
        "inputs": (
            "x is a row vector with n = numel(x) entries and k is an integer, "
            "which may be negative and may exceed n."
        ),
        "running": _ONE_RUNNING,
        "task": (
            "Return the circular shift of {p} by k positions toward lower "
            "indices: the result has n entries, entry i of {p} becomes entry "
            "mod(i-1-k, n)+1 of the result, and equivalently entry j of the "
            "result is {p}(mod(j-1+k, n)+1). The shift wraps around, so every "
            "entry of {p} appears exactly once in the result."
        ),
        "symbols": (("x",), ("v",)),
        "compute": lambda d: np.roll(d["x"], -d["k"]),
        "loop": " out = circshift(x, -k);",
        "vector": " out = circshift(x, -k);",
    },
    # -- autocorrelation: circular against linear -------------------------------
    "autocorr-circular": {
        "params": ("x",),
        "inputs": _ONE_INPUT,
        "running": _ONE_RUNNING,
        "task": (
            "Return the circular autocorrelation of {p}: a vector r with n "
            "entries whose entry r(m+1) = sum over i = 1..n of "
            "{p}(i) * {p}(mod(i-1+m, n)+1), " + _LAG_ORDER + ". The second "
            "index wraps around modulo n, so every lag sums exactly n products."
        ),
        "symbols": (("x",), ("v",)),
        "compute": lambda d: _circular_correlation(d["x"], d["x"]),
        "loop": (
            " n = numel(x);\n"
            " out = zeros(1, n);\n"
            " for m = 0:n-1\n"
            "   s = 0;\n"
            "   for i = 1:n\n"
            "     s = s + x(i) * x(mod(i-1+m, n) + 1);\n"
            "   endfor\n"
            "   out(m+1) = s;\n"
            " endfor"
        ),
        # Rows are lags and columns are i, so summing along dim 2 sums each lag.
        # `x(idx)` takes the shape of `idx`, and the trailing transpose is the
        # row the generated shape sentence asks for.
        "vector": (
            " n = numel(x);\n"
            " idx = mod((0:n-1)' + (0:n-1), n) + 1;\n"
            " out = sum(x(idx) .* x, 2)';"
        ),
    },
    "autocorr-linear": {
        "params": ("x",),
        "inputs": _ONE_INPUT,
        "running": _ONE_RUNNING,
        "task": (
            "Return the linear autocorrelation of {p}: a vector r with n "
            "entries whose entry r(m+1) = sum over i = 1..n-m of "
            "{p}(i) * {p}(i+m), " + _LAG_ORDER + ". Nothing wraps around: lag "
            "m sums exactly n-m products, and the last entry r(n) is the single "
            "product {p}(1)*{p}(n)."
        ),
        "symbols": (("x",), ("v",)),
        "compute": lambda d: _linear_correlation(d["x"]),
        "loop": (
            " n = numel(x);\n"
            " out = zeros(1, n);\n"
            " for m = 0:n-1\n"
            "   s = 0;\n"
            "   for i = 1:n-m\n"
            "     s = s + x(i) * x(i+m);\n"
            "   endfor\n"
            "   out(m+1) = s;\n"
            " endfor"
        ),
        # Zero-padding is how "nothing wraps around" becomes a single indexing
        # expression: y(i+m) is x(i+m) while i+m <= n and 0 past the end, so the
        # products the description does not form contribute nothing.
        "vector": (
            " n = numel(x);\n"
            " y = [x, zeros(1, n)];\n"
            " out = sum(y((0:n-1)' + (1:n)) .* x, 2)';"
        ),
    },
    # -- cross-correlation: which argument the lag shifts -----------------------
    "xcorr-forward": {
        "params": ("a", "b"),
        "inputs": _TWO_INPUTS,
        "running": _TWO_RUNNING,
        "task": (
            "Return the circular cross-correlation of {p} with {q}: a vector c "
            "with n entries whose entry c(m+1) = sum over i = 1..n of "
            "{p}(i) * {q}(mod(i-1+m, n)+1), " + _LAG_ORDER + ". The lag shifts "
            "{q} and leaves {p} in place, and the shifted index wraps around "
            "modulo n."
        ),
        "symbols": (("a", "b"), ("u", "v")),
        "compute": lambda d: _circular_correlation(d["a"], d["b"]),
        "loop": (
            " n = numel(a);\n"
            " out = zeros(1, n);\n"
            " for m = 0:n-1\n"
            "   s = 0;\n"
            "   for i = 1:n\n"
            "     s = s + a(i) * b(mod(i-1+m, n) + 1);\n"
            "   endfor\n"
            "   out(m+1) = s;\n"
            " endfor"
        ),
        "vector": (
            " n = numel(a);\n"
            " idx = mod((0:n-1)' + (0:n-1), n) + 1;\n"
            " out = sum(b(idx) .* a, 2)';"
        ),
    },
    "xcorr-reversed": {
        "params": ("a", "b"),
        "inputs": _TWO_INPUTS,
        "running": _TWO_RUNNING,
        "task": (
            "Return the circular cross-correlation of {q} with {p}: a vector c "
            "with n entries whose entry c(m+1) = sum over i = 1..n of "
            "{q}(i) * {p}(mod(i-1+m, n)+1), " + _LAG_ORDER + ". The lag shifts "
            "{p} and leaves {q} in place, and the shifted index wraps around "
            "modulo n."
        ),
        "symbols": (("a", "b"), ("u", "v")),
        "compute": lambda d: _circular_correlation(d["b"], d["a"]),
        "loop": (
            " n = numel(a);\n"
            " out = zeros(1, n);\n"
            " for m = 0:n-1\n"
            "   s = 0;\n"
            "   for i = 1:n\n"
            "     s = s + b(i) * a(mod(i-1+m, n) + 1);\n"
            "   endfor\n"
            "   out(m+1) = s;\n"
            " endfor"
        ),
        "vector": (
            " n = numel(a);\n"
            " idx = mod((0:n-1)' + (0:n-1), n) + 1;\n"
            " out = sum(a(idx) .* b, 2)';"
        ),
    },
    # -- convolution: circular against linear -----------------------------------
    #
    # These two differ in output *length* as well as in values, which is why the
    # linear one states `na+nb-1` explicitly and draws unequal input lengths:
    # a solution that wrapped would return the wrong number of entries and fail
    # on shape before values are even compared.
    "conv-circular": {
        "params": ("a", "b"),
        "inputs": _TWO_INPUTS,
        "running": _TWO_RUNNING,
        "task": (
            "Return the circular convolution of {p} and {q}: a vector y with n "
            "entries whose entry y(j) = sum over i = 1..n of "
            "{p}(i) * {q}(mod(j-i, n)+1), for j = 1, 2, ..., n in that order. "
            "The second index wraps around modulo n."
        ),
        "symbols": (("a", "b"), ("u", "v")),
        "compute": lambda d: _circular_convolution(d["a"], d["b"]),
        "loop": (
            " n = numel(a);\n"
            " out = zeros(1, n);\n"
            " for j = 1:n\n"
            "   s = 0;\n"
            "   for i = 1:n\n"
            "     s = s + a(i) * b(mod(j-i, n) + 1);\n"
            "   endfor\n"
            "   out(j) = s;\n"
            " endfor"
        ),
        "vector": (
            " n = numel(a);\n"
            " idx = mod((1:n)' - (1:n), n) + 1;\n"
            " out = sum(b(idx) .* a, 2)';"
        ),
    },
    "conv-linear": {
        "params": ("a", "b"),
        "inputs": (
            "a and b are row vectors whose lengths na = numel(a) and "
            "nb = numel(b) need not be equal."
        ),
        "running": (
            "First form u and v, the running totals of a and b from left to "
            "right, so that u(i) = a(1) + ... + a(i) and v(i) = b(1) + ... + "
            "b(i); u has na entries and v has nb entries."
        ),
        "task": (
            "Return the linear convolution of {p} and {q}: a vector y with "
            "na+nb-1 entries whose entry y(j) = sum of {p}(i) * {q}(j-i+1) "
            "over every i satisfying 1 <= i <= na and 1 <= j-i+1 <= nb, for "
            "j = 1, 2, ..., na+nb-1 in that order. Nothing wraps around: an "
            "index outside those ranges contributes no term."
        ),
        "symbols": (("a", "b"), ("u", "v")),
        "compute": lambda d: np.convolve(d["a"], d["b"]),
        "loop": (
            " na = numel(a);\n"
            " nb = numel(b);\n"
            " out = zeros(1, na+nb-1);\n"
            " for j = 1:na+nb-1\n"
            "   s = 0;\n"
            "   for i = max(1, j-nb+1):min(na, j)\n"
            "     s = s + a(i) * b(j-i+1);\n"
            "   endfor\n"
            "   out(j) = s;\n"
            " endfor"
        ),
        # `conv` of two rows is a row of length na+nb-1, which is exactly what
        # the description asks for, so the loop-free reading needs no reshaping.
        "vector": " out = conv(a, b);",
    },
}


def _parse(key: str) -> dict:
    if key not in _VARIANTS:
        raise ValueError(f"unknown signal_identity variant {key!r}")
    return _VARIANTS[key]


def _describe(key: str, level: int) -> str:
    """The prompt text. Every level restates its own task in full.

    Level 3 saying only "...without loops" is what let `struct_cell_wrangle`
    level 3 fall from 0.792 to 0.000 while models guessed at the task from the
    family name alone. Guarded by
    `test_level_three_restates_its_own_task_for_every_problem`, which reads the
    generated prompt rather than a table, so it sees all eight variants. Level 3
    here is level 2 verbatim plus the loop clause, so it cannot drop a term.

    Level 2 restates the *same* index formula over the running totals rather
    than paraphrasing it, which is why `task` is a template over symbols: the
    convention a reader has to get right is stated once, and both levels state
    it identically.

    The output shape is deliberately absent: the prompt builder appends the
    shape sentence it derives from the expected values, so the prompt's claim
    and the grader's comparison cannot drift apart. The output *length* is
    stated here, because the shape sentence says only "row vector" and the
    circular/linear pairs differ in exactly that.
    """
    spec = _parse(key)
    level_one_symbols, running_symbols = spec["symbols"]
    symbols = level_one_symbols if level == 1 else running_symbols
    named = dict(zip("pq", symbols, strict=False))
    task = spec["task"].format(**named)
    if level == 1:
        return f"{spec['inputs']} {task}"
    body = f"{spec['inputs']} {spec['running']} Then {task[0].lower() + task[1:]}"
    if level == 2:
        return body
    return body + " Do not use for/while loops."


def _draw_case(rng: np.random.Generator) -> dict:
    """Draw one hidden case, identically for every variant and every level.

    Every value is drawn unconditionally and with key-independent bounds, so a
    variant selection cannot shift the shared rng stream --
    `test_a_family_generates_the_same_tasks_whichever_variants_are_present`
    forbids that, and it would break silently, since the tasks would still be
    individually valid. The variants that ignore `partner`, `shift` or `trim`
    still pay for them.

    Note what is *not* used to make a draw conditional: `rng.integers(0, 1, m)`
    returns zeros without consuming the stream at all, so "draw m values and
    discard them" written that way is not a draw. Every call below has a
    non-empty range.
    """
    n = int(rng.integers(5, 10))
    x = rng.integers(_VALUE_LOW, _VALUE_HIGH, n)
    partner = rng.integers(_VALUE_LOW, _VALUE_HIGH, n)
    shift = int(rng.integers(-5, 6))
    trim = int(rng.integers(1, 3))

    # Deterministic repairs, after the last draw, so they cost no rng.
    #
    # A leading zero is the one input for which `cumsum(x) == x` is possible
    # (it needs every entry but the last to be zero), and that would make level
    # 2 the same problem as level 1 for the shift variants. Forcing x(1) ~= 0
    # makes the running total differ from the input at entry 2 of every case.
    if x[0] == 0:
        x[0] = 1
    if partner[0] == 0:
        partner[0] = 1
    # `mod(k, n) == 0` is the identity shift; `mod(k, n) == n/2` makes the
    # forward and backward shifts agree, which would let a solution with the
    # direction reversed score full marks. Walk to the next integer avoiding
    # both -- n >= 5 leaves at least three admissible residues, so this
    # terminates in at most two steps.
    k = shift
    while k % n == 0 or (n % 2 == 0 and k % n == n // 2):
        k += 1
    return {"n": n, "x": x, "partner": partner, "k": k, "trim": trim}


def build(rng: np.random.Generator, level: int, key: str) -> Variant:
    spec = _parse(key)
    params = spec["params"]

    cases: list[dict] = []
    for _ in range(6):
        drawn = _draw_case(rng)
        x = drawn["x"]
        # The linear convolution is the one variant whose two inputs have
        # different lengths -- that is what makes "the result has na+nb-1
        # entries" a claim a reader can get wrong, rather than a restatement of
        # n. The trim is drawn for every variant and used only here.
        partner = (
            drawn["partner"][: drawn["n"] - drawn["trim"]]
            if key == "conv-linear"
            else drawn["partner"]
        )
        if level > 1:
            # The level-2 step, in exact integer arithmetic: same operation,
            # applied to the running totals rather than to the inputs.
            values = {"x": np.cumsum(x), "a": np.cumsum(x), "b": np.cumsum(partner)}
        else:
            values = {"x": x, "a": x, "b": partner}
        values["k"] = drawn["k"]
        out = spec["compute"](values)

        args: list = [x.tolist()]
        if "b" in params:
            args.append(partner.tolist())
        if "k" in params:
            args.append(drawn["k"])
        # A flat JSON list, which the harness renders as the 1-by-N row the
        # generated shape sentence promises. Integer-valued throughout, so the
        # expected values are exact and byte-stable across machines.
        cases.append({"args": args, "expected": [int(v) for v in out]})

    signature = f"function out = signal_identity({', '.join(params)})"

    # Level 2 and 3 apply the operation to the running totals. In Octave that is
    # one `cumsum` per input, reusing the parameter name, so the body below it
    # is the level-1 body unchanged -- exactly as the description restates the
    # level-1 formula unchanged over v (and u).
    prologue = (
        "".join(f" {name} = cumsum({name});\n" for name in params if name != "k")
        if level > 1
        else ""
    )
    # The reference may coerce; `x(:)'` is defensive only, since every body
    # below already assumes nothing about orientation beyond what the prompt
    # states.
    coercion = "".join(f" {name} = {name}(:)';\n" for name in params if name != "k")
    reference = f"{signature}\n{coercion}{prologue}{spec['vector']}\nendfunction"

    # The naive solution is the direct transcription of the description, with no
    # coercion and no transpose the prompt did not ask for: the loops that
    # mirror the stated sums while loops are allowed, and the indexing form once
    # they are not. `circshift` and `conv` are the same at both, being already
    # loop-free.
    natural_body = spec["vector"] if level == 3 else spec["loop"]
    natural = f"{signature}\n{prologue}{natural_body}\nendfunction"

    return Variant(
        key=key,
        description=_describe(key, level),
        signature=signature,
        cases=cases,
        reference=reference,
        natural=natural,
        vectorized=level == 3,
    )
