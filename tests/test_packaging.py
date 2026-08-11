"""The wheel is what the Hub actually runs, and its file list is hand-written.

`[tool.hatch.build] include` in the environment's pyproject enumerates every
file that reaches the published wheel. Nothing regenerates it, so a module added
to the package is absent from the wheel until someone remembers to list it —
and the local test suite cannot notice, because it imports from the source tree
where the file is present either way.

That is not hypothetical. 0.5.0 moved the whole task pool into `specs.py` and
`families/`, neither of which was listed, so the wheel built from it imported
`generators` and died on `ModuleNotFoundError: No module named 'families'`. The
environment worked perfectly in every local run right up to publication.

These tests read the include list and compare it against the package on disk.
"""

import tomllib
from pathlib import Path

ENVIRONMENT = Path(__file__).parents[1] / "environments" / "octave_rl"


def _include_patterns() -> list[str]:
    manifest = tomllib.loads((ENVIRONMENT / "pyproject.toml").read_text())
    return manifest["tool"]["hatch"]["build"]["include"]


def _shipped(path: Path, patterns: list[str]) -> bool:
    relative = path.relative_to(ENVIRONMENT).as_posix()
    return any(
        relative == pattern or relative.startswith(pattern.rstrip("/") + "/")
        for pattern in patterns
    )


def test_every_module_in_the_package_reaches_the_wheel() -> None:
    patterns = _include_patterns()
    missing = sorted(
        path.relative_to(ENVIRONMENT).as_posix()
        for path in ENVIRONMENT.rglob("*.py")
        if "__pycache__" not in path.parts and not _shipped(path, patterns)
    )
    assert not missing, (
        "these modules exist in the environment package but are not in the "
        f"hatch include list, so `prime env push` would ship a broken wheel: {missing}. "
        "Add them to [tool.hatch.build] include in environments/octave_rl/pyproject.toml."
    )


def test_the_include_list_does_not_name_anything_that_is_gone() -> None:
    # The other direction: a pattern left behind after a rename is a silent
    # no-op in hatch, so the list drifts out of sync without anything failing.
    stale = sorted(
        pattern
        for pattern in _include_patterns()
        if not (ENVIRONMENT / pattern.rstrip("/")).exists()
    )
    assert not stale, f"include list names paths that no longer exist: {stale}"


def test_the_families_package_is_importable_on_its_own() -> None:
    # `families/` only ships as a package if it carries an __init__.py; hatch
    # copies the directory either way, but the import in generators.py needs it.
    assert (ENVIRONMENT / "families" / "__init__.py").exists()
