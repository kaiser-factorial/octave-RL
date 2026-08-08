"""Regressions for the Prime-free local candidate runtime.

The tests that need a real interpreter skip when one is absent, so the suite
stays runnable on a workstation without GNU Octave installed. The tests that
guard the containment contract do not need an interpreter and always run.
"""

import asyncio
import shutil
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parents[1] / "environments" / "octave_rl"))

import executors
import octave_rl as octave_environment
from executors import (
    ALLOW_UNISOLATED_ENV,
    LocalExecutionUnavailable,
    execute_candidate_locally,
)

REFERENCE_SOURCE = "function out=doubler(x)\n  out = 2 .* x;\nendfunction\n"


class _Task:
    """The surface the executor reads from a task, without verifiers types."""

    def __init__(self, cases, fn_name="doubler", tolerance=1e-9):
        self.fn_name = fn_name
        self.cases = cases
        self.tolerance = tolerance

    def model_dump(self):
        return {
            "fn_name": self.fn_name,
            "cases": self.cases,
            "tolerance": self.tolerance,
        }


needs_octave = pytest.mark.skipif(
    shutil.which("octave") is None,
    reason="local runtime tests require a GNU Octave interpreter",
)


@pytest.fixture
def scoring_host(monkeypatch):
    """Let scoring tests run where no network namespace is available.

    ``unshare`` is Linux-only, so on macOS the executor refuses to run rather
    than silently granting candidates the host network. That refusal is correct
    and has its own test below; these tests are about *scoring*, so they opt out
    explicitly rather than being unable to run on a workstation at all.
    """
    monkeypatch.setenv(ALLOW_UNISOLATED_ENV, "1")


@needs_octave
def test_local_runtime_scores_a_correct_candidate(scoring_host) -> None:
    task = _Task([{"args": [[1, 2, 3]], "expected": [2, 4, 6]}])
    record = asyncio.run(execute_candidate_locally(task, REFERENCE_SOURCE))
    assert record["fraction"] == 1.0
    assert record["structured_result"] == 1.0
    assert record["runtime"] == "local"


@needs_octave
def test_local_runtime_scores_a_matrix_result(scoring_host) -> None:
    # The column-major transport must survive a real interpreter, not just the
    # hand-written record in test_generation.
    source = "function out=outer_sum(a,b)\n  out = a(:) + b(:)';\nendfunction\n"
    task = _Task(
        [{"args": [[1, 2], [10, 20, 30]], "expected": [[11, 21, 31], [12, 22, 32]]}],
        fn_name="outer_sum",
    )
    record = asyncio.run(execute_candidate_locally(task, source))
    assert record["fraction"] == 1.0


@needs_octave
def test_local_runtime_reports_partial_credit_without_crashing(scoring_host) -> None:
    task = _Task(
        [
            {"args": [[1, 2, 3]], "expected": [2, 4, 6]},
            {"args": [[1, 2, 3]], "expected": [9, 9, 9]},
        ]
    )
    record = asyncio.run(execute_candidate_locally(task, REFERENCE_SOURCE))
    assert record["passed"] == 1
    assert record["total"] == 2
    assert record["fraction"] == 0.5


@needs_octave
def test_local_runtime_survives_candidate_code_that_raises(scoring_host) -> None:
    broken = "function out=doubler(x)\n  error('boom');\nendfunction\n"
    task = _Task([{"args": [[1]], "expected": [2]}])
    record = asyncio.run(execute_candidate_locally(task, broken))
    # A raising candidate still produces a well-formed transport of failures.
    assert record["fraction"] == 0.0
    assert record["structured_result"] == 1.0


@needs_octave
def test_local_runtime_does_not_leak_the_parent_environment(monkeypatch, scoring_host) -> None:
    monkeypatch.setenv("PRIME_API_KEY", "sentinel-must-not-reach-candidate")
    source = (
        "function out=doubler(x)\n"
        "  printf('LEAK[%s]\\n', getenv('PRIME_API_KEY'));\n"
        "  out = 2 .* x;\n"
        "endfunction\n"
    )
    task = _Task([{"args": [[1]], "expected": [2]}])
    record = asyncio.run(execute_candidate_locally(task, source))
    assert "sentinel-must-not-reach-candidate" not in record["feedback"]
    assert "LEAK[]" in record["feedback"]


def test_local_runtime_refuses_to_run_unisolated_without_explicit_opt_in(monkeypatch) -> None:
    # Silent degradation is the failure mode this guards: a host without user
    # namespaces must raise rather than quietly grant candidates the network.
    monkeypatch.setattr(executors, "network_isolation_prefix", lambda: ())
    monkeypatch.delenv(ALLOW_UNISOLATED_ENV, raising=False)
    task = _Task([{"args": [[1]], "expected": [2]}])
    with pytest.raises(LocalExecutionUnavailable, match="network namespace"):
        asyncio.run(execute_candidate_locally(task, REFERENCE_SOURCE))


@needs_octave
def test_explicit_opt_in_runs_but_records_the_weaker_isolation(monkeypatch) -> None:
    monkeypatch.setattr(executors, "network_isolation_prefix", lambda: ())
    monkeypatch.setenv(ALLOW_UNISOLATED_ENV, "1")
    task = _Task([{"args": [[1, 2, 3]], "expected": [2, 4, 6]}])
    record = asyncio.run(execute_candidate_locally(task, REFERENCE_SOURCE))
    assert record["fraction"] == 1.0
    assert record["network_isolated"] is False


def test_runtime_defaults_to_prime_and_is_selectable() -> None:
    default = octave_environment.load_environment(num_tasks=1)
    assert default.config.task.octave_runtime == "prime"
    assert default.config.task.user.octave_runtime == "prime"

    local = octave_environment.load_environment(num_tasks=1, octave_runtime="local")
    assert local.config.task.octave_runtime == "local"
    assert local.config.task.user.octave_runtime == "local"

    with pytest.raises(ValueError, match="octave_runtime must be"):
        octave_environment.load_environment(num_tasks=1, octave_runtime="somewhere-else")


def test_local_runtime_never_provisions_a_sandbox(monkeypatch) -> None:
    def explode():
        raise AssertionError("the local runtime must not touch Prime")

    monkeypatch.setattr(octave_environment, "AsyncSandboxClient", explode)

    async def fake_local(task, source):
        return {"fraction": 1.0, "runtime": "local"}

    monkeypatch.setattr(octave_environment, "execute_candidate_locally", fake_local)
    result = asyncio.run(
        octave_environment.execute_candidate(
            _Task([{"args": [[1]], "expected": [2]}]),
            REFERENCE_SOURCE,
            runtime="local",
            purpose="candidate",
        )
    )
    assert result == {"fraction": 1.0, "runtime": "local"}


def test_octave_runtime_does_not_shadow_the_verifiers_user_runtime() -> None:
    # vf.UserConfig.runtime holds a RuntimeConfig describing where the simulator
    # process runs. An earlier version of this module named its executor switch
    # `runtime`, which replaced that field with a plain string and made
    # serve_user call .model_dump() on it -- every rollout died with an
    # AttributeError before reaching the model.
    import verifiers.v1 as vf

    config = octave_environment.OctaveUserConfig()
    assert hasattr(config, "octave_runtime")
    assert config.runtime == vf.UserConfig().runtime
    assert hasattr(config.runtime, "model_dump")
