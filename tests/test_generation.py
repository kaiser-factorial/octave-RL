import asyncio
import json
import sys
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(
    0, str(Path(__file__).parents[1] / "environments" / "octave_rl")
)

import octave_rl as octave_environment
from generators import build_tasks
from harness import (
    CANDIDATE_RESULT_MARKER_PREFIX,
    RESULT_MARKER_PREFIX,
    SANDBOX_CREATION_MAX_ATTEMPTS,
    SANDBOX_FINALIZE_TIMEOUT_SECONDS,
    build_candidate_runner,
    build_harness,
    candidate_result_marker,
    extract_code,
    format_ok,
    new_result_token,
    octave_literal,
    parse_candidate_records,
    parse_harness_result,
    result_marker,
)
from octave_rl import (
    attempt_multiplier,
    candidate_record_matches,
    execute_candidate_in_new_sandbox,
    execute_candidate_in_sandbox,
    execute_feedback_in_new_sandbox,
)


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


def test_harness_uses_a_namespaced_terminal_result_protocol() -> None:
    token = "test-token"
    harness = build_harness({"cases": []}, result_token=token)
    assert f"printf('{result_marker(token)} passed=%d total=%d\\n', passed, total);" in harness


def test_candidate_runner_has_inputs_but_not_hidden_expected_values() -> None:
    token = "test-token"
    runner = build_candidate_runner(
        {
            "fn_name": "f",
            "cases": [{"args": [7], "expected": 999}],
        },
        result_token=token,
    )
    assert "999" not in runner
    assert "passed =" not in runner
    assert "total =" not in runner
    assert "expected =" not in runner
    assert candidate_result_marker(token) in runner


def test_result_parser_ignores_candidate_output_before_terminal_harness_record() -> None:
    token = "trusted-token"
    output = "\n".join(
        [
            "RESULT passed=1 total=1",
            f"{RESULT_MARKER_PREFIX}wrong-token passed=1 total=1",
            f"{result_marker(token)} passed=4 total=6",
        ]
    )
    assert parse_harness_result(output, expected_total=6, result_token=token) == (4, 6)


def test_result_parser_rejects_nonterminal_or_invalid_harness_records() -> None:
    token = "trusted-token"
    assert parse_harness_result(
        f"{result_marker(token)} passed=6 total=6\nextra candidate output",
        expected_total=6,
        result_token=token,
    ) is None
    assert parse_harness_result(
        f"{result_marker(token)} passed=6 total=5",
        expected_total=6,
        result_token=token,
    ) is None
    assert parse_harness_result(
        f"{result_marker(token)} passed=7 total=6",
        expected_total=6,
        result_token=token,
    ) is None


def test_candidate_record_parser_requires_a_fresh_terminal_transport() -> None:
    token = "trusted-token"
    output = "\n".join(
        [
            f"{CANDIDATE_RESULT_MARKER_PREFIX}replayed-token []",
            f"{candidate_result_marker(token)} [{{\"ok\": true, \"shape\": [1, 1], \"values\": [4]}}]",
        ]
    )
    assert parse_candidate_records(
        output,
        expected_total=1,
        result_token=token,
    ) == [{"ok": True, "shape": [1, 1], "values": [4]}]
    assert parse_candidate_records(
        f"{candidate_result_marker(token)} []\nextra output",
        expected_total=1,
        result_token=token,
    ) is None


def test_candidate_record_comparison_preserves_shape_and_nan() -> None:
    assert candidate_record_matches(
        {"ok": True, "shape": [1, 2], "values": [1.0, None]},
        expected=[1.0, float("nan")],
        tolerance=1e-9,
    )
    assert not candidate_record_matches(
        {"ok": True, "shape": [2, 1], "values": [1.0, 2.0]},
        expected=[1.0, 2.0],
        tolerance=1e-9,
    )


def test_final_scoring_compares_candidate_values_outside_the_sandbox(monkeypatch) -> None:
    token = "trusted-token"
    monkeypatch.setattr(octave_environment, "new_result_token", lambda: token)

    class Client:
        async def upload_bytes(self, *_args) -> None:
            return None

        async def execute_command(self, *_args, **_kwargs):
            return SimpleNamespace(
                stdout=(
                    f"{candidate_result_marker(token)} "
                    '[{"ok": true, "shape": [1, 1], "values": [999]}]\n'
                ),
                stderr="",
                exit_code=0,
            )

    data = SimpleNamespace(
        fn_name="f",
        cases=[{"args": [], "expected": 0}],
        tolerance=1e-9,
        model_dump=lambda: {
            "fn_name": "f",
            "cases": [{"args": [], "expected": 0}],
        },
    )
    result = asyncio.run(
        execute_candidate_in_sandbox(
            Client(),
            "sandbox-id",
            data,
            "function out=f(); out=0; end",
        )
    )
    assert result["fraction"] == 0.0
    assert result["structured_result"] == 1.0


def test_candidate_provisioning_uses_the_bounded_long_wait(monkeypatch) -> None:
    class Client:
        def __init__(self) -> None:
            self.request = None
            self.wait_args = None
            self.deleted = []
            self.closed = False

        async def create(self, request):
            self.request = request
            return SimpleNamespace(id="sandbox-id")

        async def wait_for_creation(self, *args, **kwargs) -> None:
            self.wait_args = (args, kwargs)

        async def execute_command(self, *_args, **_kwargs) -> None:
            return None

        async def delete(self, sandbox_id) -> None:
            self.deleted.append(sandbox_id)

        async def aclose(self) -> None:
            self.closed = True

    client = Client()

    async def fake_execute(*_args, **_kwargs):
        return {"fraction": 1.0}

    monkeypatch.setattr(octave_environment, "AsyncSandboxClient", lambda: client)
    monkeypatch.setattr(octave_environment, "execute_candidate_in_sandbox", fake_execute)
    result = asyncio.run(
        execute_candidate_in_new_sandbox(
            SimpleNamespace(idx=7),
            "function out=f(); out=0; end",
        )
    )

    assert result == {"fraction": 1.0}
    assert client.request.docker_image == "gnuoctave/octave:10.2.0"
    assert client.request.memory_gb == 2
    assert client.wait_args == (("sandbox-id",), {"max_attempts": SANDBOX_CREATION_MAX_ATTEMPTS})
    assert client.deleted == ["sandbox-id"]
    assert client.closed is True


def test_feedback_provisioning_uses_the_bounded_long_wait_and_deletes(monkeypatch) -> None:
    class Client:
        def __init__(self) -> None:
            self.request = None
            self.wait_args = None
            self.deleted = []
            self.closed = False

        async def create(self, request):
            self.request = request
            return SimpleNamespace(id="feedback-sandbox-id")

        async def wait_for_creation(self, *args, **kwargs) -> None:
            self.wait_args = (args, kwargs)

        async def execute_command(self, *_args, **_kwargs) -> None:
            return None

        async def delete(self, sandbox_id) -> None:
            self.deleted.append(sandbox_id)

        async def aclose(self) -> None:
            self.closed = True

    client = Client()

    async def fake_execute(*_args, **_kwargs):
        return {"fraction": 0.5}

    monkeypatch.setattr(octave_environment, "AsyncSandboxClient", lambda: client)
    monkeypatch.setattr(octave_environment, "execute_candidate_in_sandbox", fake_execute)
    result = asyncio.run(
        execute_feedback_in_new_sandbox(
            SimpleNamespace(idx=8),
            "function out=f(); out=0; end",
        )
    )

    assert result == {"fraction": 0.5}
    assert client.request.docker_image == "gnuoctave/octave:10.2.0"
    assert client.request.labels == ["octave-rl-feedback"]
    assert client.request.memory_gb == 2
    assert client.wait_args == (
        ("feedback-sandbox-id",),
        {"max_attempts": SANDBOX_CREATION_MAX_ATTEMPTS},
    )
    assert client.deleted == ["feedback-sandbox-id"]
    assert client.closed is True


def test_task_default_finalize_timeout_covers_sandbox_provisioning() -> None:
    task = octave_environment.load_environment(num_tasks=1).load()[0]
    assert task.data.timeout.finalize == SANDBOX_FINALIZE_TIMEOUT_SECONDS


def test_task_does_not_request_a_verifiers_container() -> None:
    # Candidate execution provisions the narrowly scoped Octave sandbox itself.
    # A second runtime container would add cost without adding a trust boundary.
    task = octave_environment.load_environment(num_tasks=1).load()[0]
    assert octave_environment.OctaveTask.NEEDS_CONTAINER is False
    assert task.data.image is None
    assert task.data.workdir is None
    assert task.data.resources.cpu is None
    assert task.data.resources.memory is None
    assert task.data.resources.disk is None


def test_result_token_is_fresh() -> None:
    assert new_result_token() != new_result_token()


def test_only_correctness_is_rewarded_and_retry_aware() -> None:
    def reward(raw: float, attempts: int) -> float:
        return raw * attempt_multiplier(
            attempts=attempts,
            second_attempt_multiplier=0.85,
            guided_attempt_multiplier=0.60,
        )

    assert reward(1.0, 1) == 1.0
    assert reward(0.5, 1) == 0.5
    assert reward(0.0, 1) == 0.0
    assert reward(1.0, 2) == 0.85
    assert reward(1.0, 3) == 0.60

    # A transport can be syntactically complete even when every host-side
    # comparison fails. It must still have no standalone reward value.
    task = SimpleNamespace(
        config=SimpleNamespace(
            second_attempt_multiplier=0.85,
            guided_attempt_multiplier=0.60,
        )
    )
    trace = SimpleNamespace(
        info={"octave": {"fraction": 0.0, "structured_result": 1.0}},
        state=SimpleNamespace(attempts=1),
    )
    assert asyncio.run(octave_environment.OctaveTask.case_fraction(task, trace)) == 0.0
