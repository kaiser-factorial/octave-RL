from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(
    0, str(Path(__file__).parents[1] / "environments" / "octave_rl")
)

from octave_rl import load_environment


def test_load_environment_maps_requested_signature() -> None:
    taskset = load_environment(
        level=2,
        num_tasks=7,
        max_turns=3,
        require_vectorized=True,
        seed=42,
        guide_enabled=True,
    )
    assert taskset.config.level == 2
    assert taskset.config.num_tasks == 7
    assert taskset.config.seed == 42
    assert taskset.config.require_vectorized is True
    assert taskset.config.task.user.max_attempts == 3
    assert taskset.config.task.user.guide_enabled is True


def test_load_environment_rejects_unknown_options() -> None:
    with pytest.raises(TypeError, match="Unexpected environment option"):
        load_environment(unknown=True)
