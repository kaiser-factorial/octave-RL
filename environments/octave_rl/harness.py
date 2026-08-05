"""Parse model output and generate deterministic Octave test harnesses."""

from __future__ import annotations

import json
import math
import re
import secrets
from typing import Any

OCTAVE_IMAGE = "ghcr.io/gnu-octave/octave:10.2.0"
# The marker includes a fresh per-execution token.  It is distinct from
# ordinary program output and unavailable while the model generates its answer.
RESULT_MARKER_PREFIX = "__OCTAVE_HARNESS_RESULT__"
CANDIDATE_RESULT_MARKER_PREFIX = "__OCTAVE_CANDIDATE_RESULT__"
CODE_RE = re.compile(r"```(?:octave|matlab)?\s*\n(.*?)```", re.IGNORECASE | re.DOTALL)


def extract_code(text: str) -> str:
    matches = CODE_RE.findall(text)
    if len(matches) == 1:
        return matches[0].strip()
    if not matches and re.search(r"(?m)^\s*function\b", text):
        return text.strip()
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
