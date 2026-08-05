import json
import sys
from pathlib import Path

sys.path.insert(
    0, str(Path(__file__).parents[1] / "environments" / "octave_rl")
)

from generators import build_tasks
from harness import extract_code, format_ok, octave_literal


def test_seeded_generation_is_reproducible() -> None:
    left = build_tasks(2, 30, 123, False, True)
    right = build_tasks(2, 30, 123, False, True)
    # JSON text gives stable equality for the intentional NaN literals.
    assert json.dumps(left, sort_keys=True) == json.dumps(right, sort_keys=True)


def test_all_families_and_levels_are_present() -> None:
    expected = {
        "reduce_along_dim",
        "logical_index",
        "reshape_permute",
        "broadcast_arith",
        "sliding_window",
        "linsolve_tolerance",
        "sequence_recurrence",
        "struct_cell_wrangle",
        "string_parse",
        "signal_identity",
    }
    for level in (1, 2, 3):
        tasks = build_tasks(level, 20, 42, False, True)
        assert {task["info"]["family"] for task in tasks} == expected


def test_format_is_observed_but_bare_code_remains_executable() -> None:
    fenced = "```octave\nfunction y=f(x)\ny=x;\nend\n```"
    bare = "function y=f(x)\ny=x;\nend"
    assert format_ok(fenced)
    assert extract_code(fenced).startswith("function")
    assert not format_ok(bare)
    assert extract_code(bare) == bare


def test_empty_json_vector_preserves_row_orientation() -> None:
    assert octave_literal([]) == "zeros(1, 0)"
