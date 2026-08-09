import asyncio
import json
import sys
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(
    0, str(Path(__file__).parents[1] / "environments" / "octave_rl")
)

import harness as harness_module
import octave_rl as octave_environment
import pytest
from generators import (
    DEFAULT_HELDOUT_FAMILIES,
    DESCRIPTIONS,
    FAMILY_NAMES,
    build_tasks,
    training_families,
)
from harness import (
    CANDIDATE_RESULT_MARKER_PREFIX,
    RESULT_MARKER_PREFIX,
    SANDBOX_CREATION_MAX_ATTEMPTS,
    SANDBOX_FINALIZE_TIMEOUT_SECONDS,
    _octave_shape,
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


def test_family_holdout_splits_are_disjoint_and_cover_the_pool() -> None:
    heldout = DEFAULT_HELDOUT_FAMILIES
    trained = training_families()
    assert set(trained) & set(heldout) == set()
    assert set(trained) | set(heldout) == set(FAMILY_NAMES)

    for level in (1, 2, 3):
        train_rows = build_tasks(level, 40, 5, families=trained)
        test_rows = build_tasks(level, 20, 5, families=heldout)
        assert {row["info"]["family"] for row in train_rows} == set(trained)
        assert {row["info"]["family"] for row in test_rows} == set(heldout)
        assert len(train_rows) == 40 and len(test_rows) == 20
        # The point of the split: no task, and therefore no prompt, is shared.
        train_prompts = {row["prompt"][0]["content"] for row in train_rows}
        test_prompts = {row["prompt"][0]["content"] for row in test_rows}
        assert train_prompts & test_prompts == set()


def test_a_family_generates_the_same_tasks_whichever_others_are_present() -> None:
    """Filtering must not perturb a family's draw.

    A split is only usable as a holdout if the held-out family's tasks are the
    same objects a full-pool measurement would have produced. Cycling over the
    selection instead of filtering the full stream would silently break this,
    and the resulting numbers would look fine.
    """
    for level in (1, 2, 3):
        full = build_tasks(level, 200, 99, include_reference=True)
        for family in ("reduce_along_dim", "string_parse"):
            from_full = [row for row in full if row["info"]["family"] == family]
            filtered = build_tasks(
                level, len(from_full), 99, include_reference=True, families=[family]
            )
            assert json.dumps(filtered, sort_keys=True) == json.dumps(
                from_full, sort_keys=True
            ), f"{family} L{level} differs when generated in isolation"


def test_unknown_family_names_are_rejected() -> None:
    with pytest.raises(ValueError, match="unknown task families"):
        build_tasks(1, 5, 0, families=["reduce_along_dim", "nonexistent_family"])
    with pytest.raises(ValueError, match="at least one family"):
        build_tasks(1, 5, 0, families=[])
    with pytest.raises(ValueError, match="nothing to train on"):
        training_families(FAMILY_NAMES)


def test_every_prompt_states_the_shape_the_grader_compares_against() -> None:
    """The prompt's declared output shape must match the graded expected value.

    Before 2026-08-09 the prompts carried a blanket "Preserve input
    orientation" line while the grader compared ``size(actual)`` exactly
    against a reference that silently coerced orientation. Three families were
    near-unsolvable as a result and nothing failed, because no test related the
    prompt text to the expected values.
    """
    for level in (1, 2, 3):
        for task in build_tasks(level, 40, 7, False, True):
            info = task["info"]
            text = task["prompt"][0]["content"]
            assert "Preserve input orientation" not in text
            claims = [line for line in text.splitlines() if line.startswith("Return a ")]
            assert len(claims) == 1, f"{info['family']} L{level}: {claims}"
            claim = claims[0]
            for case in info["cases"]:
                rows, cols = _octave_shape(case["expected"])
                where = f"{info['family']} L{level} claims {claim!r} but a case is {rows}x{cols}"
                if claim == "Return a scalar.":
                    assert (rows, cols) == (1, 1), where
                elif claim == "Return a row vector (1-by-N).":
                    assert rows == 1, where
                elif claim == "Return a column vector (N-by-1).":
                    assert cols == 1, where
                elif claim == "Return a 2-D matrix.":
                    assert rows > 1 and cols > 1, where
                else:
                    assert claim == f"Return a matrix with {rows} rows.", where


def test_level_three_descriptions_restate_their_own_task() -> None:
    """Level 3 adds a constraint; it must not drop the task definition.

    `struct_cell_wrangle` level 3 once read only "Return [column minima;
    column maxima] without for/while loops", leaving the misleading family name
    as the model's only clue about the input type. It scored 0.000 against
    level 2's 0.792 on the same underlying task.
    """
    # Families whose level 3 is level 2 plus a vectorization constraint. The
    # rest change the task itself between those levels -- linsolve_tolerance
    # switches to [x; norm(A*x-b)], sliding_window from mean to median,
    # reshape_permute to a different permutation, string_parse to decimals --
    # so their wording legitimately differs.
    same_task_at_level_three = {
        "reduce_along_dim",
        "logical_index",
        "broadcast_arith",
        "sequence_recurrence",
        "struct_cell_wrangle",
        "signal_identity",
    }
    stopwords = {"without", "loops", "for/while", "no", "return", "the", "a", "of", "and"}

    def content_words(text: str) -> set[str]:
        cleaned = text.lower().replace(",", " ").replace(";", " ").replace(".", " ")
        return {word for word in cleaned.split() if word not in stopwords}

    for family in sorted(same_task_at_level_three):
        descriptions = DESCRIPTIONS[family]
        missing = content_words(descriptions[1]) - content_words(descriptions[2])
        assert not missing, (
            f"{family} level 3 drops terms its level 2 states: {sorted(missing)}"
        )


def test_orientation_sensitive_arguments_arrive_as_the_prompt_describes() -> None:
    """`A\\b` and `a + b` must be conformant as written.

    Both families used to serialise a vector argument as a row, so the natural
    solution raised "nonconformant arguments" on every hidden case no matter
    what the model wrote.
    """
    for level in (1, 2, 3):
        for task in build_tasks(level, 40, 11, False, True):
            info = task["info"]
            for case in info["cases"]:
                if info["family"] == "linsolve_tolerance":
                    matrix, vector = case["args"]
                    assert _octave_shape(vector) == [len(matrix), 1]
                if info["family"] == "broadcast_arith":
                    left, right = case["args"]
                    assert _octave_shape(left)[1] == 1
                    assert _octave_shape(right)[0] == 1


def test_truncated_code_block_does_not_keep_its_opening_fence() -> None:
    """A generation cut off inside its fence must still yield runnable source.

    Without this the candidate .m file starts with "```octave" and every case
    reports "syntax error near line 1", which tells a repair turn nothing.
    """
    truncated = "Here you go:\n```octave\nfunction y=f(x)\n  y = x + 1;\n"
    extracted = extract_code(truncated)
    assert extracted.startswith("function")
    assert "```" not in extracted


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


def test_matrix_results_are_compared_in_octave_column_major_order() -> None:
    # The runner reports actual(:)', which Octave emits column by column, while
    # the expected value nests row by row. Comparing the two orders directly
    # scored every correct two-dimensional answer as zero on every case.
    expected = [[-4, -3, -4, -8], [3, 4, 3, -1]]
    correct = {
        "ok": True,
        "shape": [2, 4],
        "values": [-4, 3, -3, 4, -4, 3, -8, -1],
    }
    assert candidate_record_matches(correct, expected=expected, tolerance=1e-9)

    # A transposed answer has the same multiset of values and must still fail.
    transposed = {
        "ok": True,
        "shape": [4, 2],
        "values": [-4, -3, -4, -8, 3, 4, 3, -1],
    }
    assert not candidate_record_matches(transposed, expected=expected, tolerance=1e-9)


def test_column_major_reordering_leaves_vectors_and_scalars_alone() -> None:
    # These orders coincide, which is exactly why the matrix bug stayed hidden.
    assert candidate_record_matches(
        {"ok": True, "shape": [1, 3], "values": [1, 2, 3]},
        expected=[1, 2, 3],
        tolerance=1e-9,
    )
    assert candidate_record_matches(
        {"ok": True, "shape": [3, 1], "values": [1, 2, 3]},
        expected=[[1], [2], [3]],
        tolerance=1e-9,
    )
    assert candidate_record_matches(
        {"ok": True, "shape": [1, 1], "values": [7]},
        expected=7,
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


def test_scorer_separates_execution_from_correctness() -> None:
    # A candidate that throws on every case and one that runs but computes the
    # wrong numbers both score fraction 0.0. They are different failures, and
    # in this environment the first is roughly three times more common, so the
    # record has to tell them apart.
    token = "trusted-token"
    cases = [{"args": [], "expected": 1}, {"args": [], "expected": 2}]

    threw = (
        f'{candidate_result_marker(token)} '
        '[{"ok": false, "shape": [], "values": []}, '
        '{"ok": false, "shape": [], "values": []}]'
    )
    record = harness_module.score_candidate_output(
        threw, cases=cases, tolerance=1e-9, result_token=token, exit_code=0
    )
    assert record["fraction"] == 0.0
    assert record["executed"] == 0
    assert record["execution_fraction"] == 0.0

    ran_wrong = (
        f'{candidate_result_marker(token)} '
        '[{"ok": true, "shape": [1, 1], "values": [99]}, '
        '{"ok": true, "shape": [1, 1], "values": [99]}]'
    )
    record = harness_module.score_candidate_output(
        ran_wrong, cases=cases, tolerance=1e-9, result_token=token, exit_code=0
    )
    assert record["fraction"] == 0.0
    assert record["executed"] == 2
    assert record["execution_fraction"] == 1.0


def test_transpose_is_detected_but_not_rewarded() -> None:
    # The prompts require preserving input orientation, so a transposed answer
    # is a real failure and must still score zero. It is worth *naming*, though:
    # the model got the whole computation right and only the convention wrong,
    # which is a different thing from a wrong algorithm.
    token = "trusted-token"
    expected = [[1, 2, 3], [4, 5, 6]]
    transposed = (
        f'{candidate_result_marker(token)} '
        '[{"ok": true, "shape": [3, 2], "values": [1, 2, 3, 4, 5, 6]}]'
    )
    record = harness_module.score_candidate_output(
        transposed,
        cases=[{"expected": expected}],
        tolerance=1e-9,
        result_token=token,
        exit_code=0,
    )
    assert record["fraction"] == 0.0, "orientation is part of the spec"
    assert record["transposed_fraction"] == 1.0
    assert record["execution_fraction"] == 1.0

    correct = (
        f'{candidate_result_marker(token)} '
        '[{"ok": true, "shape": [2, 3], "values": [1, 4, 2, 5, 3, 6]}]'
    )
    record = harness_module.score_candidate_output(
        correct,
        cases=[{"expected": expected}],
        tolerance=1e-9,
        result_token=token,
        exit_code=0,
    )
    assert record["fraction"] == 1.0
    assert record["transposed_fraction"] == 0.0, "a correct answer is not a transpose"


def test_unit_multipliers_make_reward_equal_raw_correctness() -> None:
    # The length-penalty variant moves efficiency pressure out of the reward and
    # into the advantage. With both multipliers at 1.0, case_fraction and
    # raw_case_fraction coincide at every attempt count, so the reward stops
    # double-counting "needed retries" as reduced capability.
    for attempts in (1, 2, 3, 4):
        assert attempt_multiplier(
            attempts=attempts,
            second_attempt_multiplier=1.0,
            guided_attempt_multiplier=1.0,
        ) == 1.0

    # The default configuration still discounts, so existing runs are unchanged.
    assert attempt_multiplier(
        attempts=3, second_attempt_multiplier=0.85, guided_attempt_multiplier=0.60
    ) == 0.60
