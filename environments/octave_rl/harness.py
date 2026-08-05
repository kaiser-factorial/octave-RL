"""Parse model output and generate deterministic Octave test harnesses."""

from __future__ import annotations

import math
import re
from typing import Any

OCTAVE_IMAGE = "ghcr.io/gnu-octave/octave:10.2.0"
RESULT_RE = re.compile(r"RESULT passed=(\d+) total=(\d+)")
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


def build_harness(info: dict[str, Any]) -> str:
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
        "printf('RESULT passed=%d total=%d\\n', passed, total);",
        "fflush(stdout);",
        "exit(passed < total);",
    ]
    return "\n".join(lines) + "\n"
