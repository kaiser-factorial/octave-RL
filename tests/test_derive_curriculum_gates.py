import importlib.util
import json
import sys
from pathlib import Path

MODULE_PATH = Path(__file__).parents[1] / "scripts" / "derive_curriculum_gates.py"
SPEC = importlib.util.spec_from_file_location("derive_curriculum_gates", MODULE_PATH)
assert SPEC and SPEC.loader
gates = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = gates
SPEC.loader.exec_module(gates)


def _traces(path: Path, per_task: list[list[float]]) -> Path:
    path.write_text(
        "\n".join(
            json.dumps(
                {
                    "task": {"data": {"name": f"task-{index}"}},
                    "metrics": {"raw_case_fraction": score},
                }
            )
            for index, scores in enumerate(per_task)
            for score in scores
        )
        + "\n"
    )
    return path


def test_the_observed_rate_a_gate_demands_exceeds_its_nominal_threshold(tmp_path):
    # The whole point of the script: a nominal threshold is not the bar. At the
    # controller's default evaluation size the gap is large enough to matter.
    assert gates.observed_rate_needed(0.55, 20) == 0.75
    assert gates.observed_rate_needed(0.20, 20) == 0.35

    # And it shrinks with sample size rather than staying fixed.
    for threshold in (0.20, 0.55):
        small = gates.observed_rate_needed(threshold, 20) - threshold
        large = gates.observed_rate_needed(threshold, 200) - threshold
        assert large < small


def test_an_unreachable_gate_reports_none_rather_than_a_rate() -> None:
    # wilson_lower never reaches 1.0, so a threshold at the ceiling cannot be
    # cleared at any sample size. That must not read as "0.0 is enough".
    assert gates.observed_rate_needed(1.0, 20) is None


def test_signal_counts_groups_that_are_not_unanimous(tmp_path):
    # Two tasks: one split, one unanimous. At group size 2 exactly half the
    # groups carry advantage, and the split task contributes all of it.
    path = _traces(tmp_path / "t.jsonl", [[1.0, 0.0], [1.0, 1.0]])
    reading = gates.read_level(path, level=1, group_size=2)
    assert reading.tasks == 2
    assert reading.rollouts == 4
    assert reading.rate == 0.75
    assert reading.signal == 0.5


def test_measured_signal_can_fall_below_the_independence_estimate(tmp_path):
    # Per-task clustering is the reason the script measures instead of assuming.
    # Two tasks at 0.5 overall, but each internally unanimous: independence
    # predicts every group carries advantage, and none of them does.
    path = _traces(tmp_path / "t.jsonl", [[1.0, 1.0], [0.0, 0.0]])
    reading = gates.read_level(path, level=1, group_size=2)
    assert reading.rate == 0.5
    assert reading.signal == 0.0
    assert reading.independent_signal == 0.5


def test_a_threshold_round_trips_through_the_observed_rate_it_demands() -> None:
    # threshold_for_observed is the inverse used to propose new gates: the
    # nominal value it returns must actually be cleared by that observed rate.
    for observed in (0.25, 0.50, 0.75, 0.90):
        nominal = gates.threshold_for_observed(observed, 100)
        assert gates.observed_rate_needed(nominal, 100) <= observed + 1e-9


def test_the_sample_size_search_finds_a_size_that_beats_the_gap() -> None:
    enough = gates.smallest_sample_for_gap(0.05)
    assert enough is not None
    assert gates.observed_rate_needed(0.50, enough) - 0.50 < 0.05
    # And it is the smallest such size on the search's own grid.
    assert gates.observed_rate_needed(0.50, enough - 10) - 0.50 >= 0.05
