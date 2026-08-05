import importlib.util
import json
import sys
from pathlib import Path

MODULE_PATH = Path(__file__).parents[1] / "scripts" / "plot_rollout_dynamics.py"
SPEC = importlib.util.spec_from_file_location("plot_rollout_dynamics", MODULE_PATH)
assert SPEC and SPEC.loader
plot_rollouts = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = plot_rollouts
SPEC.loader.exec_module(plot_rollouts)


def test_standalone_spec_overrides_step_level_and_split(tmp_path):
    path = tmp_path / "traces.jsonl"
    path.write_text(
        json.dumps(
            {
                "run": {"type": "eval"},
                "task": {"data": {"level": 3}},
                "metrics": {"attempts_used": 2},
                "rewards": {"case_fraction": 0.5},
                "calls": [
                    {
                        "finish_reason": "stop",
                        "usage": {"completion_tokens": 12},
                    }
                ],
                "timing": {
                    "start": 10,
                    "generation": {
                        "start": 12,
                        "end": 18,
                        "model": {"duration": 5},
                    },
                    "scoring": {"end": 20},
                },
            }
        )
        + "\n"
    )
    rollouts = plot_rollouts.read_standalone(f"15:1:{path}")
    assert rollouts == [
        plot_rollouts.Rollout(
            step=15,
            split="eval",
            level=1,
            reward=0.5,
            attempts=2,
            generation_seconds=5.0,
            total_seconds=10.0,
            completion_tokens=12,
            truncated=False,
        )
    ]


def test_native_reader_prefers_effective_rollouts(tmp_path):
    split_dir = tmp_path / "run_default" / "rollouts" / "step_2" / "train"
    effective = split_dir / "effective" / "traces.jsonl"
    all_traces = split_dir / "all" / "traces.jsonl"
    effective.parent.mkdir(parents=True)
    all_traces.parent.mkdir(parents=True)
    base = {
        "run": {"step": 2, "type": "train"},
        "task": {"data": {"level": 1}},
        "metrics": {"attempts_used": 1, "raw_case_fraction": 1.0},
        "calls": [],
    }
    effective.write_text(json.dumps(base) + "\n")
    all_traces.write_text(json.dumps(base) + "\n" + json.dumps(base) + "\n")
    rollouts = plot_rollouts.read_rollouts(tmp_path)
    assert len(rollouts) == 1
    assert rollouts[0].step == 2


def test_native_reader_ignores_incomplete_train_all_file(tmp_path):
    path = (
        tmp_path
        / "run_default"
        / "rollouts"
        / "step_3"
        / "train"
        / "all"
        / "traces.jsonl"
    )
    path.parent.mkdir(parents=True)
    path.write_text(
        json.dumps(
            {
                "run": {"step": 3, "type": "train"},
                "task": {"data": {"level": 1}},
                "errors": ["Payment required"],
            }
        )
        + "\n"
    )
    assert plot_rollouts.read_rollouts(tmp_path) == []
