"""Seeded NumPy task generation for ten Octave problem families."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np

Task = dict[str, Any]


def _shape_sentence(cases: list[dict[str, Any]]) -> str:
    """State the graded output shape, derived from the values the grader uses.

    Scoring compares ``size(actual)`` against the expected value's Octave shape
    exactly, so output orientation is part of the task. Deriving the sentence
    from ``expected`` rather than writing it by hand keeps the prompt's claim
    and the grader's comparison from drifting apart -- the failure mode that
    made three families near-unsolvable before 2026-08-09.

    A row count is stated only when every hidden case agrees on it; families
    whose output width follows the inputs get the unquantified form.
    """
    expected = cases[0]["expected"]
    if not isinstance(expected, list):
        return "Return a scalar."
    if expected and all(isinstance(row, list) for row in expected):
        if len(expected[0]) == 1:
            return "Return a column vector (N-by-1)."
        heights = {len(case["expected"]) for case in cases}
        if len(heights) == 1:
            return f"Return a matrix with {heights.pop()} rows."
        return "Return a 2-D matrix."
    return "Return a row vector (1-by-N)."


def _row(family, level, signature, cases, reference, *, tolerance=1e-9, vectorized=False):
    fn = signature.split("=")[1].split("(")[0].strip()
    prompt = (
        f"Write this GNU Octave function:\n\n    {signature}\n\n"
        f"{DESCRIPTIONS[family][level - 1]}\n"
        f"{_shape_sentence(cases)}\n"
        f"Return exactly one fenced `octave` code block. "
        f"Hidden tests include edge cases."
    )
    return {
        "prompt": [{"role": "user", "content": prompt}],
        "info": {
            "family": family, "level": level, "fn_name": fn, "signature": signature,
            "cases": cases, "tolerance": tolerance, "require_vectorized": vectorized,
        },
        "task": f"octave-l{level}-{family}",
        "_reference": reference,
    }


# Every level-3 entry restates its own task in full. Level 3 differs from
# level 2 only by a vectorization constraint, so an entry that says merely
# "...without for/while loops" leaves the model nothing to work from but the
# family name -- which is how `struct_cell_wrangle` level 3 fell from 0.792 to
# 0.000 while models guessed at cell arrays. Do not compress these again.
DESCRIPTIONS = {
    "reduce_along_dim": [
        "Return the arithmetic mean of each column.",
        "Return the k-th largest value in each column; ties count separately.",
        "Return the k-th largest value in each column; ties count separately; no for/while loops.",
    ],
    "logical_index": [
        "Return the positive elements of x, in original order.",
        "Replace values outside inclusive [lo, hi] with NaN.",
        "Replace values outside inclusive [lo, hi] with NaN without for/while loops.",
    ],
    "reshape_permute": [
        "Return the elements of x as a single column.",
        "x holds the elements of a 3-D array of size dims in column-major order. Reshape x to dims, reorder its dimensions to [2 1 3], then flatten the result back to a vector in column-major order.",
        "x holds the elements of a 3-D array of size dims in column-major order. Reshape x to dims, reorder its dimensions to [3 1 2], then flatten the result back to a vector in column-major order, without for/while loops.",
    ],
    "broadcast_arith": [
        "Return the outer sum of column vector a and row vector b.",
        "Return the matrix of squared pairwise differences between column vector a and row vector b.",
        "Return the matrix of squared pairwise differences between column vector a and row vector b, without for/while loops.",
    ],
    "sliding_window": [
        "Return sums of every consecutive window of width w (valid windows only).",
        "Return means of valid windows of width w and stride s.",
        "Return medians of valid windows of width w and stride s, without for/while loops.",
    ],
    "linsolve_tolerance": [
        "Solve the square linear system A*x=b.",
        "Return the least-squares solution of A*x approximately b.",
        "Return [x; norm(A*x-b)] for a least-squares solution.",
    ],
    "sequence_recurrence": [
        "Return the first n terms where x(1)=a and x(i)=x(i-1)+d.",
        "Return n terms where x(1)=a, x(2)=b, x(i)=p*x(i-1)+q*x(i-2).",
        "Return n terms where x(1)=a, x(2)=b, x(i)=p*x(i-1)+q*x(i-2), without for/while loops; use filter or equivalent.",
    ],
    "struct_cell_wrangle": [
        "Given numeric row vectors a and b, return their elementwise sums.",
        "Given a numeric matrix A, return [column minima; column maxima].",
        "Given a numeric matrix A, return [column minima; column maxima], without for/while loops.",
    ],
    "string_parse": [
        "Parse a comma-separated char row such as '1,2,-3' into numbers.",
        "Parse numbers separated by commas with optional surrounding spaces.",
        "Parse comma-separated finite decimal numbers, without for/while loops.",
    ],
    "signal_identity": [
        "Return the circular shift of x by integer k.",
        "Return the real circular autocorrelation using FFT, with lag zero first.",
        "Return the real circular autocorrelation using FFT, with lag zero first, without for/while loops.",
    ],
}


def reduce_along_dim(rng, level):
    cases = []
    for _ in range(6):
        A = rng.integers(-9, 15, (rng.integers(3, 7), rng.integers(2, 5)))
        if level == 1:
            args, out = [A.tolist()], A.mean(axis=0)
        else:
            k = int(rng.integers(1, A.shape[0] + 1))
            args, out = [A.tolist(), k], np.sort(A, axis=0)[-k]
        cases.append({"args": args, "expected": out.tolist()})
    sig = "function out = reduce_along_dim(A)" if level == 1 else "function out = reduce_along_dim(A, k)"
    ref = ("function out = reduce_along_dim(A)\n out = mean(A, 1);\nendfunction" if level == 1 else
           "function out = reduce_along_dim(A,k)\n s=sort(A,1,'descend'); out=s(k,:);\nendfunction")
    return _row("reduce_along_dim", level, sig, cases, ref, vectorized=level == 3)


def logical_index(rng, level):
    cases = []
    for _ in range(6):
        x = rng.integers(-10, 11, rng.integers(5, 12))
        if level == 1:
            args, out = [x.tolist()], x[x > 0]
        else:
            lo, hi = sorted(rng.integers(-6, 7, 2).tolist())
            out = x.astype(float); out[(out < lo) | (out > hi)] = np.nan
            args = [x.tolist(), lo, hi]
        cases.append({"args": args, "expected": out.tolist()})
    sig = "function out = logical_index(x)" if level == 1 else "function out = logical_index(x, lo, hi)"
    ref = ("function out=logical_index(x)\n out=x(x>0);\nendfunction" if level == 1 else
           "function out=logical_index(x,lo,hi)\n out=double(x); out(out<lo | out>hi)=NaN;\nendfunction")
    return _row("logical_index", level, sig, cases, ref, vectorized=level == 3)


def reshape_permute(rng, level):
    cases = []
    for _ in range(6):
        if level == 1:
            x = rng.integers(-9, 10, rng.integers(3, 10))
            args, out = [x.tolist()], x.reshape(-1, 1)
        else:
            # JSON/harness support is intentionally 2-D; represent flattened data + dimensions.
            dims = [int(v) for v in rng.integers(2, 4, 3)]
            x = rng.integers(-9, 10, np.prod(dims))
            order = [1, 0, 2] if level == 2 else [2, 0, 1]
            out = np.transpose(np.reshape(x, dims, order="F"), order).flatten(order="F")
            args = [x.tolist(), dims]
        cases.append({"args": args, "expected": out.tolist()})
    if level == 1:
        sig, ref = "function out = reshape_permute(x)", "function out=reshape_permute(x)\n out=x(:);\nendfunction"
    else:
        sig = "function out = reshape_permute(x, dims)"
        perm = "[2 1 3]" if level == 2 else "[3 1 2]"
        ref = f"function out=reshape_permute(x,dims)\n out=permute(reshape(x,dims),{perm}); out=out(:)';\nendfunction"
    return _row("reshape_permute", level, sig, cases, ref, vectorized=level == 3)


def broadcast_arith(rng, level):
    cases = []
    for _ in range(6):
        a, b = rng.integers(-8, 9, rng.integers(2, 6)), rng.integers(-8, 9, rng.integers(2, 6))
        out = a[:, None] + b[None, :] if level == 1 else (a[:, None] - b[None, :]) ** 2
        # `a` is serialised as a column so the prompt's "column vector a and row
        # vector b" is literally true and bare `a + b` broadcasts correctly.
        # Sending both as rows made the natural answer nonconformant.
        cases.append({
            "args": [a.reshape(-1, 1).tolist(), b.tolist()],
            "expected": out.tolist(),
        })
    sig = "function out = broadcast_arith(a, b)"
    expr = "a(:) + b(:)'" if level == 1 else "(a(:) - b(:)').^2"
    ref = f"function out=broadcast_arith(a,b)\n out={expr};\nendfunction"
    return _row("broadcast_arith", level, sig, cases, ref, vectorized=level == 3)


def sliding_window(rng, level):
    cases = []
    for _ in range(6):
        x = rng.integers(-9, 10, rng.integers(7, 14)); w = int(rng.integers(2, 5))
        s = 1 if level == 1 else int(rng.integers(1, 3))
        windows = np.lib.stride_tricks.sliding_window_view(x, w)[::s]
        out = windows.sum(1) if level == 1 else (windows.mean(1) if level == 2 else np.median(windows, axis=1))
        args = [x.tolist(), w] if level == 1 else [x.tolist(), w, s]
        cases.append({"args": args, "expected": out.tolist()})
    sig = "function out = sliding_window(x, w)" if level == 1 else "function out = sliding_window(x, w, s)"
    if level == 1:
        ref = "function out=sliding_window(x,w)\n out=conv(x,ones(1,w),'valid');\nendfunction"
    else:
        op = "mean" if level == 2 else "median"
        ref = f"function out=sliding_window(x,w,s)\n idx=(1:s:(numel(x)-w+1))'+(0:w-1); out={op}(x(idx),2)';\nendfunction"
    return _row("sliding_window", level, sig, cases, ref, vectorized=level == 3)


def linsolve_tolerance(rng, level):
    cases = []
    for _ in range(6):
        m = int(rng.integers(3, 7)); n = m if level == 1 else int(rng.integers(2, m))
        A = rng.normal(size=(m, n)); x0 = rng.normal(size=n); b = A @ x0
        x = np.linalg.lstsq(A, b, rcond=None)[0]
        out = x if level < 3 else np.r_[x, np.linalg.norm(A @ x - b)]
        # `b` is serialised as a column so `A\b` is conformant as written. As a
        # row it made every hidden case fail with "nonconformant arguments"
        # regardless of what the model wrote, while the reference hid the
        # problem behind an undisclosed `b=b(:)`.
        cases.append({
            "args": [A.tolist(), b.reshape(-1, 1).tolist()],
            "expected": out.reshape(-1, 1).tolist(),
        })
    sig = "function out = linsolve_tolerance(A, b)"
    expr = "A\\b" if level < 3 else "[A\\b; norm(A*(A\\b)-b)]"
    ref = f"function out=linsolve_tolerance(A,b)\n b=b(:); out={expr};\nendfunction"
    return _row("linsolve_tolerance", level, sig, cases, ref, tolerance=1e-7)


def sequence_recurrence(rng, level):
    cases = []
    for _ in range(6):
        n = int(rng.integers(4, 10)); a = int(rng.integers(-4, 5))
        if level == 1:
            d = int(rng.integers(-3, 4)); out = a + d * np.arange(n); args = [a, d, n]
        else:
            b = int(rng.integers(-4, 5)); p, q = int(rng.integers(-2, 3)), int(rng.integers(-2, 3))
            out = [a, b]
            for _i in range(2, n): out.append(p * out[-1] + q * out[-2])
            out = np.array(out); args = [a, b, p, q, n]
        cases.append({"args": args, "expected": out.tolist()})
    if level == 1:
        sig, ref = "function out = sequence_recurrence(a, d, n)", "function out=sequence_recurrence(a,d,n)\n out=a+d*(0:n-1);\nendfunction"
    else:
        sig = "function out = sequence_recurrence(a, b, p, q, n)"
        ref = "function out=sequence_recurrence(a,b,p,q,n)\n out=zeros(1,n); out(1:2)=[a b]; for i=3:n; out(i)=p*out(i-1)+q*out(i-2); endfor\nendfunction"
    return _row("sequence_recurrence", level, sig, cases, ref, vectorized=False)


def struct_cell_wrangle(rng, level):
    cases = []
    for _ in range(6):
        if level == 1:
            a, b = rng.integers(-9, 10, 6), rng.integers(-9, 10, 6); args, out = [a.tolist(), b.tolist()], a + b
        else:
            A = rng.integers(-9, 10, (rng.integers(3, 6), rng.integers(2, 5))); args, out = [A.tolist()], np.vstack([A.min(0), A.max(0)])
        cases.append({"args": args, "expected": out.tolist()})
    if level == 1:
        sig, ref = "function out = struct_cell_wrangle(a, b)", "function out=struct_cell_wrangle(a,b)\n out=a+b;\nendfunction"
    else:
        sig, ref = "function out = struct_cell_wrangle(A)", "function out=struct_cell_wrangle(A)\n out=[min(A,[],1); max(A,[],1)];\nendfunction"
    return _row("struct_cell_wrangle", level, sig, cases, ref, vectorized=level == 3)


def string_parse(rng, level):
    cases = []
    for _ in range(6):
        vals = rng.integers(-99, 100, rng.integers(3, 8))
        sep = "," if level == 1 else ", "
        text = sep.join(map(str, vals))
        cases.append({"args": [text], "expected": vals.tolist()})
    sig = "function out = string_parse(s)"
    ref = "function out=string_parse(s)\n out=sscanf(strrep(s,',',' '),'%f')';\nendfunction"
    return _row("string_parse", level, sig, cases, ref, vectorized=level == 3)


def signal_identity(rng, level):
    cases = []
    for _ in range(6):
        x = rng.integers(-6, 7, rng.integers(4, 10))
        if level == 1:
            k = int(rng.integers(-5, 6)); args, out = [x.tolist(), k], np.roll(x, k)
        else:
            args, out = [x.tolist()], np.fft.ifft(np.abs(np.fft.fft(x)) ** 2).real
            # FFT kernels may differ by a few ulps across SIMD code paths.
            # Quantize well inside the task's 1e-7 tolerance for byte-stable tasks.
            out = np.round(out, 12)
        cases.append({"args": args, "expected": out.tolist()})
    if level == 1:
        sig, ref = "function out = signal_identity(x, k)", "function out=signal_identity(x,k)\n out=circshift(x(:)',[0 k]);\nendfunction"
    else:
        sig, ref = "function out = signal_identity(x)", "function out=signal_identity(x)\n out=real(ifft(abs(fft(x)).^2));\nendfunction"
    return _row("signal_identity", level, sig, cases, ref, tolerance=1e-7, vectorized=level == 3)


FAMILIES: list[Callable] = [
    reduce_along_dim, logical_index, reshape_permute, broadcast_arith, sliding_window,
    linsolve_tolerance, sequence_recurrence, struct_cell_wrangle, string_parse, signal_identity,
]

FAMILY_NAMES: list[str] = [family.__name__ for family in FAMILIES]

# The default generalization split. Holding out a *family* is the only way this
# taskset can support a held-out-problem claim: a pool's prompts are determined
# by (family, level), so two pools drawn with different seeds share every
# prompt and differ only in hidden test inputs.
#
# These two were chosen for dynamic range and for what they test. Both sit
# mid-difficulty for both measured models (2026-08-09: reduce_along_dim
# 0.389/0.326, reshape_permute 0.569/0.292), so neither is floored nor
# ceilinged. `reduce_along_dim` has a near neighbour that stays in training --
# `struct_cell_wrangle` level 2+ is also a column-wise reduction -- so it tests
# transfer of a practiced idiom. `reshape_permute` has none, so it tests
# whether general Octave fluency reaches an unpracticed one.
#
# This is a default, not a recommendation for every experiment. Hold out the
# two hardest families and a real improvement will be invisible against the
# floor; hold out the two easiest and it vanishes into the ceiling.
DEFAULT_HELDOUT_FAMILIES: list[str] = ["reduce_along_dim", "reshape_permute"]


def resolve_families(families: list[str] | None) -> list[str]:
    """Validate a family selection, or return all ten."""
    if families is None:
        return list(FAMILY_NAMES)
    unknown = [name for name in families if name not in FAMILY_NAMES]
    if unknown:
        raise ValueError(
            f"unknown task families {unknown}; valid names are {FAMILY_NAMES}"
        )
    if not families:
        raise ValueError("families must select at least one family")
    return list(dict.fromkeys(families))


def training_families(heldout: list[str] | None = None) -> list[str]:
    """The complement of a holdout, in canonical order."""
    excluded = set(resolve_families(heldout if heldout is not None else DEFAULT_HELDOUT_FAMILIES))
    kept = [name for name in FAMILY_NAMES if name not in excluded]
    if not kept:
        raise ValueError("holding out every family leaves nothing to train on")
    return kept


def build_tasks(
    level=1,
    num_tasks=500,
    seed=0,
    require_vectorized=False,
    include_reference=False,
    families=None,
):
    """Generate ``num_tasks`` tasks, optionally restricted to some families.

    Restricting families *filters* the full ten-family stream rather than
    cycling over the selection, so a given family's k-th task is byte-identical
    whichever other families are present. That is what makes a train split and
    a holdout split drawn from one seed genuinely disjoint *and* individually
    comparable to a full-pool measurement. Task indices come from the full
    stream, so ids stay stable and are not contiguous within a filtered pool.
    """
    selected = set(resolve_families(families))
    rng = np.random.default_rng(seed)
    rows = []
    index = 0
    while len(rows) < num_tasks:
        family = FAMILIES[index % len(FAMILIES)]
        task = family(rng, level)
        task["task"] += f"-{index:05d}"
        index += 1
        if family.__name__ not in selected:
            continue
        if require_vectorized:
            task["info"]["require_vectorized"] = True
        if not include_reference:
            task.pop("_reference")
        rows.append(task)
    return rows
