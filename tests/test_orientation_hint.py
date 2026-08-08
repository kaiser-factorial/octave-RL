import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1] / "environments" / "octave_rl"))

import octave_rl as octave_environment


def _user(**kwargs):
    config = octave_environment.OctaveUserConfig(max_attempts=3, **kwargs)
    user = octave_environment.OctaveUser(config)
    return user


def test_orientation_hint_is_off_by_default() -> None:
    # WS3's arms must see exactly what they saw before this feature existed.
    user = _user()
    assert user.config.orientation_hint_enabled is False
    result = {"passed": 0, "total": 6, "transposed": 6}
    assert user._orientation_hint(result) == ""


def test_orientation_hint_fires_when_transposition_explains_every_failure() -> None:
    user = _user(orientation_hint_enabled=True)
    hint = user._orientation_hint({"passed": 0, "total": 6, "transposed": 6})
    assert "transposed" in hint
    assert "rows and columns" in hint


def test_orientation_hint_stays_quiet_when_something_else_is_also_wrong() -> None:
    # Two of six failures are transposes and two are genuinely wrong. Saying
    # "you are transposed" would send the model after the wrong bug.
    user = _user(orientation_hint_enabled=True)
    assert user._orientation_hint({"passed": 2, "total": 6, "transposed": 2}) == ""


def test_orientation_hint_handles_a_result_without_the_field() -> None:
    # The no-fenced-code branch builds a result dict with no transport at all.
    user = _user(orientation_hint_enabled=True)
    assert user._orientation_hint({"passed": 0, "total": 6}) == ""


def test_orientation_hint_is_selectable_through_the_factory() -> None:
    off = octave_environment.load_environment(num_tasks=1, max_turns=3)
    assert off.config.task.user.orientation_hint_enabled is False
    on = octave_environment.load_environment(
        num_tasks=1, max_turns=3, orientation_hint_enabled=True
    )
    assert on.config.task.user.orientation_hint_enabled is True
