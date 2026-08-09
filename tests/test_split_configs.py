"""The shipped configs must agree with the split defined in code.

A holdout is only held out if *every* training config excludes it. That
agreement lives in two places -- `generators.DEFAULT_HELDOUT_FAMILIES` and a
`families` list repeated in each TOML -- and nothing but this test stops them
drifting apart. Drift would not raise: training would quietly include a
held-out family and the generalization number would still look like a
generalization number.
"""

from __future__ import annotations

import sys
import tomllib
from pathlib import Path

ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(ROOT / "environments" / "octave_rl"))

from generators import DEFAULT_HELDOUT_FAMILIES, FAMILY_NAMES, training_families


def _train_envs(config: dict) -> list[dict]:
    return config.get("orchestrator", {}).get("train", {}).get("env", [])


def test_every_training_config_excludes_the_holdout() -> None:
    configs = sorted((ROOT / "configs" / "prime-rl").glob("*.toml"))
    assert configs, "no prime-rl configs found"
    checked = 0
    for path in configs:
        for env in _train_envs(tomllib.loads(path.read_text())):
            families = env["taskset"].get("families")
            assert families is not None, (
                f"{path.name} trains on all ten families; it must set `families` "
                f"to the train split or the holdout is not held out"
            )
            overlap = set(families) & set(DEFAULT_HELDOUT_FAMILIES)
            assert not overlap, f"{path.name} trains on held-out families {sorted(overlap)}"
            assert set(families) == set(training_families()), (
                f"{path.name} families disagree with training_families()"
            )
            checked += 1
    assert checked, "no train env blocks were checked"


def test_split_eval_configs_match_the_code() -> None:
    directory = ROOT / "configs" / "eval"
    validation = tomllib.loads((directory / "octave-split-validation.toml").read_text())
    generalization = tomllib.loads(
        (directory / "octave-split-generalization.toml").read_text()
    )
    assert set(validation["taskset"]["families"]) == set(training_families())
    assert set(generalization["taskset"]["families"]) == set(DEFAULT_HELDOUT_FAMILIES)

    # Disjoint, and together they cover the pool -- otherwise a family is
    # silently measured by neither split.
    both = set(validation["taskset"]["families"]) | set(
        generalization["taskset"]["families"]
    )
    assert both == set(FAMILY_NAMES)
    assert not (
        set(validation["taskset"]["families"])
        & set(generalization["taskset"]["families"])
    )

    # Both read the same seed: the splits must differ by family, not by draw.
    assert validation["taskset"]["seed"] == generalization["taskset"]["seed"]
