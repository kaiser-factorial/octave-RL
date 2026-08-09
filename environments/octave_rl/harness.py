"""Parse model output, generate Octave harnesses, and score candidate reports.

Everything in this module is runtime-agnostic and free of Prime dependencies,
so the Prime Sandbox backend and the local subprocess backend in
``executors.py`` share one scoring path. Keeping the comparison here is what
lets the runtime be swapped without moving the trust boundary: no backend ever
receives an expected value or a pass counter.
"""

from __future__ import annotations

import json
import math
import re
import secrets
from typing import Any

# Prime's registry preflight currently rejects the fully qualified GHCR mirror
# even though it serves the same linux/amd64 manifest as Docker Hub.  The
# unqualified reference selects Prime's working Docker Hub resolver path.
OCTAVE_IMAGE = "gnuoctave/octave:10.2.0"
# The Sandbox SDK's default of 60 polls is only about 115 seconds. The pinned
# Octave image has taken roughly five minutes to provision on a cold/queued
# service, so keep an explicit, still-bounded window below the 15-minute
# sandbox lifetime.
SANDBOX_CREATION_MAX_ATTEMPTS = 180
SANDBOX_FINALIZE_TIMEOUT_SECONDS = 420
# The marker includes a fresh per-execution token.  It is distinct from
# ordinary program output and unavailable while the model generates its answer.
RESULT_MARKER_PREFIX = "__OCTAVE_HARNESS_RESULT__"
CANDIDATE_RESULT_MARKER_PREFIX = "__OCTAVE_CANDIDATE_RESULT__"
CODE_RE = re.compile(r"```(?:octave|matlab)?\s*\n(.*?)```", re.IGNORECASE | re.DOTALL)


OPEN_FENCE_RE = re.compile(r"(?s)^.*?```(?:octave|matlab)?[ \t]*\n", re.IGNORECASE)


def extract_code(text: str) -> str:
    matches = CODE_RE.findall(text)
    if len(matches) == 1:
        return matches[0].strip()
    if not matches and re.search(r"(?m)^\s*function\b", text):
        # A generation truncated inside its code block has an opening fence and
        # no closing one, so CODE_RE misses it and this bare-function fallback
        # fires. Without dropping the opening fence the candidate .m file starts
        # with "```octave" and every case dies on "syntax error near line 1",
        # which tells a repair turn nothing about the actual mistake.
        return OPEN_FENCE_RE.sub("", text, count=1).strip()
    return ""


def format_ok(text: str) -> bool:
    return len(CODE_RE.findall(text)) == 1


def new_result_token() -> str:
    return secrets.token_hex(16)


def result_marker(result_token: str) -> str:
    return f"{RESULT_MARKER_PREFIX}{result_token}"


def candidate_result_marker(result_token: str) -> str:
    return f"{CANDIDATE_RESULT_MARKER_PREFIX}{result_token}"


def parse_harness_result(
    output: str,
    *,
    expected_total: int,
    result_token: str,
) -> tuple[int, int] | None:
    """Accept only the final, self-consistent result emitted by the harness.

    Candidate functions can print arbitrary text, including the legacy
    ``RESULT passed=...`` form. The generated harness receives a fresh token
    after model generation and prints the corresponding namespaced protocol
    record after candidate execution. Requiring that record to be terminal and
    validating the expected number of cases prevents ordinary candidate stdout
    from becoming a reward source.
    """
    lines = [line.strip() for line in output.splitlines() if line.strip()]
    if not lines:
        return None
    marker = re.escape(result_marker(result_token))
    match = re.fullmatch(rf"{marker} passed=(\d+) total=(\d+)", lines[-1])
    if match is None:
        return None
    passed, total = map(int, match.groups())
    if total != expected_total or not 0 <= passed <= total:
        return None
    return passed, total


def parse_candidate_records(
    output: str,
    *,
    expected_total: int,
    result_token: str,
) -> list[dict[str, Any]] | None:
    """Decode the terminal candidate report produced without hidden outputs.

    The report is a transport from the isolated candidate process, not a score:
    the trusted Python task process compares its reported actual values with
    hidden expected values after parsing it.
    """
    lines = [line.strip() for line in output.splitlines() if line.strip()]
    if not lines:
        return None
    marker = candidate_result_marker(result_token) + " "
    final = lines[-1]
    if not final.startswith(marker):
        return None
    try:
        records = json.loads(final.removeprefix(marker))
    except json.JSONDecodeError:
        return None
    if (
        not isinstance(records, list)
        or len(records) != expected_total
        or not all(isinstance(record, dict) for record in records)
    ):
        return None
    return records


def octave_literal(value: Any) -> str:
    if value is None:
        return "NaN"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, str):
        return "'" + value.replace("'", "''") + "'"
    if isinstance(value, (int, float)):
        if isinstance(value, float):
            if math.isnan(value):
                return "NaN"
            if value == float("inf"):
                return "Inf"
            if value == float("-inf"):
                return "-Inf"
        return repr(value)
    if isinstance(value, list):
        if not value:
            # A JSON list represents a row vector throughout the task schema.
            # Octave's bare [] has size 0x0, whereas logical row indexing returns
            # 1x0 when no elements match.
            return "zeros(1, 0)"
        if all(not isinstance(x, list) for x in value):
            if all(isinstance(x, str) for x in value):
                return "{" + ", ".join(octave_literal(x) for x in value) + "}"
            return "[" + ", ".join(octave_literal(x) for x in value) + "]"
        return "[" + "; ".join(
            ", ".join(octave_literal(x) for x in row) for row in value
        ) + "]"
    raise TypeError(f"unsupported Octave literal: {type(value)!r}")


def build_harness(info: dict[str, Any], *, result_token: str) -> str:
    lines = [
        "more off; warning('off', 'all');",
        f"passed = 0; total = {len(info['cases'])};",
    ]
    tol = info.get("tolerance", 1e-9)
    for index, case in enumerate(info["cases"], 1):
        args = ", ".join(octave_literal(v) for v in case["args"])
        expected = octave_literal(case["expected"])
        lines += [
            "try",
            f"  actual = {info['fn_name']}({args});",
            f"  expected = {expected};",
            f"  ok = isequal(size(actual), size(expected)) && all((isnan(actual(:)) & isnan(expected(:))) | abs(actual(:) - expected(:)) <= {tol} .* max(1, abs(expected(:))));",
            f"  if ok; passed += 1; else; printf('CASE {index} FAIL value\\n'); endif",
            "catch err",
            f"  printf('CASE {index} ERROR %s\\n', err.message);",
            "end_try_catch",
        ]
    lines += [
        f"printf('{result_marker(result_token)} passed=%d total=%d\\n', passed, total);",
        "fflush(stdout);",
        "exit(passed < total);",
    ]
    return "\n".join(lines) + "\n"


def build_candidate_runner(info: dict[str, Any], *, result_token: str) -> str:
    """Generate an input-only runner for untrusted candidate code.

    Expected values and pass counters intentionally never enter this process.
    Each record preserves the result shape and flattened numeric values so the
    trusted task process can apply the same shape/tolerance comparison outside
    the candidate sandbox.
    """
    lines = [
        "more off; warning('off', 'all');",
        f"records = cell(1, {len(info['cases'])});",
    ]
    for index, case in enumerate(info["cases"], 1):
        args = ", ".join(octave_literal(value) for value in case["args"])
        lines += [
            "try",
            f"  actual = {info['fn_name']}({args});",
            (
                f"  records{{{index}}} = struct('ok', true, "
                "'shape', size(actual), 'values', actual(:)');"
            ),
            "catch err",
            (
                f"  records{{{index}}} = struct('ok', false, "
                "'shape', [], 'values', []);"
            ),
            f"  printf('CASE {index} ERROR %s\\n', err.message);",
            "end_try_catch",
        ]
    lines += [
        f"printf('{candidate_result_marker(result_token)} %s\\n', jsonencode(records));",
        "fflush(stdout);",
    ]
    return "\n".join(lines) + "\n"


def _octave_shape(value: Any) -> list[int]:
    if not isinstance(value, list):
        return [1, 1]
    if not value:
        return [1, 0]
    if all(not isinstance(item, list) for item in value):
        return [1, len(value)]
    if not all(isinstance(item, list) for item in value):
        return []
    widths = {len(item) for item in value}
    return [len(value), widths.pop()] if len(widths) == 1 else []


def _flatten(value: Any) -> list[Any]:
    if isinstance(value, list):
        return [item for child in value for item in _flatten(child)]
    return [value]


def _octave_flatten(value: Any) -> list[Any]:
    """Flatten an expected value in Octave's column-major order.

    The candidate transport reports ``actual(:)'``, which Octave emits column by
    column, while a JSON expected value nests row by row. The two orders
    coincide for scalars, row vectors, and column vectors, so the mismatch is
    invisible until a task returns a genuinely two-dimensional result -- at
    which point a correct answer scores zero on every case.
    """
    if (
        isinstance(value, list)
        and value
        and all(isinstance(row, list) for row in value)
        and len({len(row) for row in value}) == 1
    ):
        return [item for column in zip(*value, strict=True) for item in column]
    return _flatten(value)


def _as_float(value: Any) -> float:
    if value is None:
        return math.nan
    if isinstance(value, bool):
        raise TypeError("boolean outputs are not supported")
    return float(value)


def candidate_record_matches(
    record: dict[str, Any],
    *,
    expected: Any,
    tolerance: float,
) -> bool:
    """Compare one isolated candidate result with the hidden expected value."""
    if record.get("ok") is not True:
        return False
    try:
        actual_shape = [int(item) for item in record["shape"]]
        actual_values = [_as_float(item) for item in _flatten(record["values"])]
        expected_values = [_as_float(item) for item in _octave_flatten(expected)]
    except (KeyError, TypeError, ValueError):
        return False
    if actual_shape != _octave_shape(expected) or len(actual_values) != len(expected_values):
        return False
    for actual, target in zip(actual_values, expected_values, strict=True):
        if math.isnan(target):
            if not math.isnan(actual):
                return False
        elif not math.isfinite(actual) or abs(actual - target) > tolerance * max(1.0, abs(target)):
            return False
    return True


def _octave_rowmajor(value: Any) -> list[Any]:
    """Flatten row by row -- what ``A'(:)`` reports for a transposed answer."""
    if (
        isinstance(value, list)
        and value
        and all(isinstance(row, list) for row in value)
        and len({len(row) for row in value}) == 1
    ):
        return [item for row in value for item in row]
    return _flatten(value)


def candidate_record_is_transpose(
    record: dict[str, Any],
    *,
    expected: Any,
    tolerance: float,
) -> bool:
    """True when the answer is exactly the transpose of the expected value.

    Worth detecting separately because it is the one failure where the model got
    the whole computation right and only the orientation convention wrong. These
    currently score identically to code that does not parse, which throws away
    the most informative near-miss the environment can produce.

    A transposed result reports ``shape`` reversed, and because Octave flattens
    ``actual(:)`` column by column, its reported values equal the *row-major*
    flatten of the expected value.
    """
    if record.get("ok") is not True:
        return False
    expected_shape = _octave_shape(expected)
    if not expected_shape:
        return False
    try:
        actual_shape = [int(item) for item in record["shape"]]
        actual_values = [_as_float(item) for item in _flatten(record["values"])]
        expected_values = [_as_float(item) for item in _octave_rowmajor(expected)]
    except (KeyError, TypeError, ValueError):
        return False
    if actual_shape != [expected_shape[1], expected_shape[0]]:
        return False
    if len(actual_values) != len(expected_values):
        return False
    for actual, target in zip(actual_values, expected_values, strict=True):
        if math.isnan(target):
            if not math.isnan(actual):
                return False
        elif not math.isfinite(actual) or abs(actual - target) > tolerance * max(1.0, abs(target)):
            return False
    return True


def score_candidate_output(
    output: str,
    *,
    cases: list[dict[str, Any]],
    tolerance: float,
    result_token: str,
    exit_code: int,
) -> dict[str, Any]:
    """Turn one candidate transport into the shared execution record.

    This is the only place a score is computed, for every runtime. It runs in
    the trusted task process, which is the sole holder of the expected values,
    so an untrusted interpreter cannot influence the comparison beyond the
    shapes and numbers it reports.
    """
    records = parse_candidate_records(
        output,
        expected_total=len(cases),
        result_token=result_token,
    )
    passed = (
        sum(
            candidate_record_matches(
                record,
                expected=case["expected"],
                tolerance=tolerance,
            )
            for record, case in zip(records, cases, strict=True)
        )
        if records is not None
        else 0
    )
    total = len(cases)
    # Execution and correctness are separate competencies and the transport
    # already distinguishes them: a case with ok=false threw inside Octave,
    # while ok=true with wrong numbers means the code ran and the algorithm was
    # wrong. Collapsing both into `fraction` hides which one failed, and in
    # practice most zeros are the former.
    executed = (
        sum(1 for record in records if record.get("ok") is True)
        if records is not None
        else 0
    )
    transposed = (
        sum(
            not candidate_record_matches(
                record, expected=case["expected"], tolerance=tolerance
            )
            and candidate_record_is_transpose(
                record, expected=case["expected"], tolerance=tolerance
            )
            for record, case in zip(records, cases, strict=True)
        )
        if records is not None
        else 0
    )
    return {
        "passed": passed,
        "total": total,
        "fraction": passed / total if total else 0.0,
        "executed": executed,
        "execution_fraction": executed / total if total else 0.0,
        "transposed": transposed,
        "transposed_fraction": transposed / total if total else 0.0,
        "structured_result": float(records is not None),
        "exit_code": exit_code,
        "feedback": output[-2000:],
    }
