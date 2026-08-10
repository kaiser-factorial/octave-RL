import asyncio
import json
import sys
from collections import Counter
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(
    0, str(Path(__file__).parents[1] / "environments" / "octave_rl")
)

import harness as harness_module
import numpy as np
import octave_rl as octave_environment
import pytest
from executors import execute_candidate_locally, runtime_description
from generators import (
    DEFAULT_HELDOUT_FAMILIES,
    DEFAULT_HELDOUT_VARIANTS,
    FAMILY_NAMES,
    VARIANT_MODULES,
    build_tasks,
    declared_variants,
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
from specs import complement


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


_STOPWORDS = {
    "without", "loops", "for/while", "no", "not", "use", "do",
    "return", "the", "a", "an", "of", "and", "that", "in", "to", "its",
}


def _content_words(text: str) -> set[str]:
    cleaned = text.lower().replace(",", " ").replace(";", " ").replace(".", " ")
    return {word for word in cleaned.split() if word not in _STOPWORDS}


def _descriptions_by_problem(level: int) -> dict[tuple[str, str], str]:
    """`(family, variant)` -> the description line of its generated prompt.

    Read off the *generated prompt*, not off a table beside the generator. The
    earlier version of this test read `DESCRIPTIONS` directly, which was
    adequate while a family had one prompt per level and became a blind spot the
    moment it had eight: the dict a converted family no longer consults would
    have gone on passing forever.
    """
    found: dict[tuple[str, str], str] = {}
    for task in build_tasks(level, 400, 5, include_reference=True):
        lines = task["prompt"][0]["content"].splitlines()
        # Layout is fixed by `_row`/`_row_from_variant`: heading, blank,
        # signature, blank, description..., shape sentence, closing line.
        description = "\n".join(lines[4:-2])
        found[(task["info"]["family"], task["info"].get("variant", ""))] = description
    return found


def test_level_three_restates_its_own_task_for_every_problem() -> None:
    """Level 3 adds a constraint; it must not drop the task definition.

    `struct_cell_wrangle` level 3 once read only "Return [column minima; column
    maxima] without for/while loops", leaving the misleading family name as the
    model's only clue about the input type. It scored 0.000 against level 2's
    0.792 on the same underlying task.

    With eight variants per family this has eight times the surface, and a
    variant that drops its task at level 3 would be invisible in an aggregate
    per-family score.
    """
    # Families whose level 3 is level 2 plus a vectorization constraint. The
    # rest change the task itself between those levels -- linsolve_tolerance
    # switches to [x; norm(A*x-b)], reshape_permute to a different permutation,
    # string_parse to decimals -- so their wording legitimately differs.
    same_task_at_level_three = {
        "reduce_along_dim",
        "logical_index",
        "broadcast_arith",
        "sequence_recurrence",
        "struct_cell_wrangle",
        "signal_identity",
        "sliding_window",
    }
    level_two = _descriptions_by_problem(2)
    level_three = _descriptions_by_problem(3)
    checked = 0
    for problem, description in sorted(level_two.items()):
        family, _ = problem
        if family not in same_task_at_level_three:
            continue
        assert problem in level_three, f"{problem} exists at L2 but not L3"
        missing = _content_words(description) - _content_words(level_three[problem])
        assert not missing, (
            f"{family} variant {problem[1]!r} level 3 drops terms its level 2 "
            f"states: {sorted(missing)}"
        )
        checked += 1
    # Guard the guard: a change that stopped generating variants would make the
    # loop body run zero times and the test would still pass.
    assert checked >= len(same_task_at_level_three), checked


def test_a_family_generates_the_same_tasks_whichever_variants_are_present() -> None:
    """The family invariant, extended to the new filter.

    A variant holdout is only a holdout if the kept tasks are the same objects a
    full-pool measurement would have produced. Selecting variants *before*
    generation -- the obvious optimisation -- would consume the rng differently
    and silently make every split incomparable with every other.
    """
    kept = complement(declared_variants(), DEFAULT_HELDOUT_VARIANTS)
    for level in (1, 2, 3):
        held_keys = {name.split(":", 1)[1] for name in DEFAULT_HELDOUT_VARIANTS}
        for selection, wanted in ((kept, False), (DEFAULT_HELDOUT_VARIANTS, True)):
            filtered = build_tasks(
                level, 150, 21, include_reference=True, variants=selection
            )
            # How far the *unfiltered* stream has to run to contain every kept
            # task. Derived from the ids rather than guessed at: a filter that
            # drops more of the stream makes it run further, so any fixed
            # multiple of the row count goes stale as more families convert.
            # Row k of an unfiltered pool carries index k, so this many rows
            # covers every index the filtered pool used.
            needed = max(
                int(row["task"].rsplit("-", 1)[1]) for row in filtered
            ) + 1
            full = {
                row["task"]: row
                for row in build_tasks(level, needed, 21, include_reference=True)
            }
            converted = [
                row for row in filtered
                if row["info"]["family"] in declared_variants()
            ]
            assert converted, "no converted family survived the filter"
            for row in converted:
                assert (row["info"]["variant"] in held_keys) is wanted
                twin = full[row["task"]]
                assert json.dumps(row, sort_keys=True) == json.dumps(
                    twin, sort_keys=True
                ), f"{row['task']} differs when generated under a variant filter"


def test_every_variant_is_drawn_and_counts_are_near_exact() -> None:
    """Round-robin, not sampled: per-variant counts are exact, not multinomial.

    At eight variants and fifty tasks per family, an rng-drawn variant would
    give Binomial(50, 1/8) counts -- a spread of roughly 6 +/- 2.4, and the
    occasional variant missing entirely from a pool. Per-variant pass rates are
    the measurement this change exists to enable, so the counts are made exact
    instead.
    """
    for family, keys in declared_variants().items():
        counts = Counter(
            task["info"]["variant"]
            for task in build_tasks(1, 500, 0)
            if task["info"]["family"] == family
        )
        assert set(counts) == set(keys), f"{family}: {sorted(counts)}"
        assert max(counts.values()) - min(counts.values()) <= 1, dict(counts)


def test_a_converted_family_ships_a_naive_solution_with_every_task() -> None:
    """`natural` travels with the task, because a table beside it did not.

    `validate_natural_solutions.py` is the only check that has ever caught a
    task solvable solely through an undisclosed convention. It used to hold one
    naive solution per (family, level), which would cover one variant of eight
    and report PASS for the other seven.
    """
    for level in (1, 2, 3):
        for task in build_tasks(level, 200, 3, include_reference=True):
            info = task["info"]
            if info["family"] not in declared_variants():
                continue
            natural = info.get("natural", "")
            assert natural, f"{task['task']} carries no naive solution"
            assert info["fn_name"] in natural
            # A naive solution that coerces its ARGUMENTS is not naive: `b(:)`
            # on an input is the defensive reshape that hid the linsolve defect
            # for weeks. `(:)` on a value the function computed is different --
            # it is the flatten `reshape_permute` prompts literally ask for --
            # so the check names the arguments rather than banning the idiom.
            arguments = info["signature"].split("(", 1)[1].rstrip(")").split(",")
            for argument in (name.strip() for name in arguments):
                assert f"{argument}(:)" not in natural, (
                    f"{task['task']} naive solution coerces its argument "
                    f"{argument!r}"
                )


class _VariantTask:
    """The minimal surface ``execute_candidate_locally`` reads from a task."""

    def __init__(self, variant) -> None:
        self.fn_name = variant.signature.split("=")[1].split("(")[0].strip()
        self.cases = variant.cases
        self.tolerance = variant.tolerance

    def model_dump(self):
        return {
            "fn_name": self.fn_name,
            "cases": self.cases,
            "tolerance": self.tolerance,
        }


def _octave_available() -> bool:
    try:
        runtime_description()
    except Exception:  # noqa: BLE001 - any unavailability means skip
        return False
    return True


@pytest.mark.skipif(
    not _octave_available(),
    reason="needs an Octave interpreter; run scripts/fetch_pinned_octave.py",
)
def test_no_variant_has_a_level_two_its_level_one_solution_already_solves() -> None:
    """Level 2 must be a different *problem*, not merely a different sentence.

    `reduce_along_dim` shipped a level-2 step -- trim the k largest and k
    smallest -- that preserves the median exactly, so both median variants had a
    level-2 answer identical to their level 1 on 240 of 240 cases. Neither
    reference-based validator can catch that, and neither can the naive-solution
    validator: all three pass a degenerate level 2, because all three compute
    what the description asks for. It is the description that fails to ask for
    something new. See PIPELINE_LOG, 2026-08-10.

    So the check has to be this one: run each variant's *level-1* naive solution
    against its *level-2* hidden cases. Full marks there means level 2 asked for
    nothing new.

    **What this test cannot measure, stated because an earlier version hid it.**
    When level 2 takes different arguments from level 1 -- a new parameter, a
    second matrix, a different input type -- the level-1 solution dies on an
    arity error before computing anything, and `fraction` is 0.0 for a reason
    that has nothing to do with the mathematics. The assertion passes, and it
    passes vacuously. That was true of four of the six converted families and
    was caught by two family authors independently, not by this test.

    So the arity case is asserted rather than assumed: signatures that differ
    are a structural difference between the two problems, and the test confirms
    the level-1 solution genuinely could not run, rather than accepting a zero
    of unknown provenance. It does NOT confirm those families' level 2 is a new
    problem -- only their own per-variant census can, and each family's module
    docstring records one. Do not read a green run here as covering them.
    """
    measured = 0
    for family, module in VARIANT_MODULES.items():
        for key in module.VARIANT_KEYS:
            level_one = module.build(np.random.default_rng(4242), 1, key)
            level_two = module.build(np.random.default_rng(4242), 2, key)
            record = asyncio.run(
                execute_candidate_locally(_VariantTask(level_two), level_one.natural)
            )
            if level_one.signature == level_two.signature:
                # The probe ran on level 2's inputs, so this is a measurement.
                assert record["fraction"] < 1.0, (
                    f"{family}:{key} level 2 is fully solved by its own level-1 "
                    f"solution -- a distinct prompt that is not a distinct problem"
                )
                measured += 1
                continue
            # Different signature: level 2 asks for different arguments, which
            # is itself a difference between the problems. Confirm the zero came
            # from that and not from something this test would otherwise miss.
            assert record["executed"] == 0, (
                f"{family}:{key} has different level-1 and level-2 signatures, "
                f"yet its level-1 solution ran on level-2 inputs -- this test's "
                f"vacuity assumption is wrong for it, so it needs a real probe"
            )
    # Guard the guard: if every family moved to a differing signature this test
    # would assert nothing about degeneracy at all, and would still be green.
    assert measured >= 8, f"only {measured} variants actually measured"


def test_unknown_variant_names_are_rejected() -> None:
    with pytest.raises(ValueError, match="unknown task variants"):
        build_tasks(1, 5, 0, variants=["reduce_along_dim:no-such-variant"])
    with pytest.raises(ValueError, match="family:key"):
        build_tasks(1, 5, 0, variants=["mean-columns"])
    with pytest.raises(ValueError, match="nothing to train on"):
        complement(
            declared_variants(),
            [f"{fam}:{key}" for fam, keys in declared_variants().items() for key in keys],
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


def test_the_discount_follows_the_hint_not_the_attempt_number() -> None:
    """A hinted solve is priced as guided whenever the hint arrived.

    Since the guide fires on need rather than on attempt 3, discounting purely
    by attempt number would let the same assistance earn 0.85 on attempt 2 and
    0.60 on attempt 3. The discount exists to price assistance, so it tracks
    assistance.
    """
    kw = {"second_attempt_multiplier": 0.85, "guided_attempt_multiplier": 0.60}
    assert attempt_multiplier(attempts=1, **kw) == 1.0
    assert attempt_multiplier(attempts=2, **kw) == 0.85
    assert attempt_multiplier(attempts=2, guided=True, **kw) == 0.60
    assert attempt_multiplier(attempts=3, **kw) == 0.60
    assert attempt_multiplier(attempts=3, guided=True, **kw) == 0.60
    # An unguided first attempt is never discounted, hint flag or not.
    assert attempt_multiplier(attempts=1, guided=False, **kw) == 1.0


def test_retry_feedback_is_a_diagnostic_not_a_transport_dump() -> None:
    """What the model is told between attempts must be information, not protocol.

    Measured over 396 retries on 2026-08-09, the old feedback was the raw Octave
    stdout: 46% of its length was the `__OCTAVE_CANDIDATE_RESULT__` blob, each
    identical error was repeated once per case with a random temp path, and 33%
    of retries carried no diagnostic at all.
    """
    cases = [{"expected": [[1], [2], [3]]} for _ in range(6)]
    raw = "\n".join(
        f"CASE {i} ERROR syntax error near line 3, column 57 "
        f"in file /tmp/vf-a6d/octave-rl-local-qwwf/f.m"
        for i in range(1, 7)
    ) + f"\n{CANDIDATE_RESULT_MARKER_PREFIX}abc [...]"
    text = harness_module.build_retry_feedback(
        raw, records=[{"ok": False}] * 6, cases=cases,
        tolerance=1e-9, passed=0, total=6,
    )
    assert CANDIDATE_RESULT_MARKER_PREFIX not in text, "transport blob leaked into feedback"
    assert "/tmp/" not in text, "random temp path leaked into feedback"
    assert text.count("syntax error") == 1, "identical errors must be deduplicated"
    assert "6 cases did not run" in text
    assert len(text) < 200, f"feedback bloated back up: {len(text)} chars"


def test_running_but_wrong_still_gets_a_diagnostic() -> None:
    """The 33% blind spot: code that runs and is merely wrong.

    It used to produce a message with no diagnostic content whatsoever -- the
    pass count, then the model's own output values echoed back. The three
    failure modes need different fixes and must be distinguishable.
    """
    cases = [{"expected": [[1], [2], [3]]} for _ in range(6)]

    wrong_shape = harness_module.build_retry_feedback(
        f"{CANDIDATE_RESULT_MARKER_PREFIX}abc []",
        records=[{"ok": True, "shape": [1, 3], "values": [1, 2, 3]}] * 6,
        cases=cases, tolerance=1e-9, passed=0, total=6,
    )
    assert "1x3" in wrong_shape and "3x1" in wrong_shape, wrong_shape

    wrong_values = harness_module.build_retry_feedback(
        f"{CANDIDATE_RESULT_MARKER_PREFIX}abc []",
        records=[{"ok": True, "shape": [3, 1], "values": [9, 9, 9]}] * 6,
        cases=cases, tolerance=1e-9, passed=0, total=6,
    )
    assert "wrong values" in wrong_values, wrong_values
    # The two cases must not read alike -- that was the whole defect.
    assert wrong_shape != wrong_values


def test_the_guide_fires_when_the_diagnostic_cannot_help() -> None:
    """Hint timing follows need, not attempt number.

    31% of rollouts reach their first retry with no execution error to report,
    so the ordinary diagnostic can only say "wrong answer". The guide used to
    arrive on the attempt *after* that.
    """
    config = octave_environment.OctaveUserConfig(
        max_attempts=3, guide_enabled=True, octave_runtime="local"
    )
    calls: list[str] = []

    async def fake_guide(_code, diagnostic):
        calls.append(diagnostic)
        return "try dimension 1"

    async def ran_but_wrong(*_a, **_k):
        return {
            "passed": 0, "total": 6, "feedback": "",
            "retry_feedback": "Hidden tests passed 0/6 cases.\n6 cases ran with "
                              "the correct shape but the wrong values.",
            "has_execution_error": False,
        }

    user = octave_environment.OctaveUser(config)
    user._inert_state = octave_environment.OctaveState(attempts=0)
    user.task = SimpleNamespace(prompt="p", cases=[{}] * 6)
    user._guide_hint = fake_guide
    original = octave_environment.execute_candidate
    octave_environment.execute_candidate = ran_but_wrong
    try:
        reply = asyncio.run(user.respond("```octave\nfunction out=f(x)\nout=x;\nend\n```"))
    finally:
        octave_environment.execute_candidate = original

    assert "Guide hint:" in reply[0]["content"], "guide should fire on the first uninformative retry"
    assert user.state.guide_used
    assert CANDIDATE_RESULT_MARKER_PREFIX not in calls[0], "guide must not see the transport blob"


def test_a_failing_guide_degrades_the_retry_instead_of_killing_the_rollout() -> None:
    """A guide failure must not propagate out of `respond`.

    It used to. The exception escaped, the MCP layer returned a contentless
    tool result, and the host reported
    `JSONDecodeError('Expecting value: line 1 column 1 (char 0)')` -- naming
    neither the cause nor this file. Measured on 2026-08-09: 23 of 16 rollouts'
    worth of retry turns lost, and zero once the credential was reachable.
    """
    config = octave_environment.OctaveUserConfig(
        max_attempts=3, guide_enabled=True, octave_runtime="local"
    )
    user = octave_environment.OctaveUser(config)
    # `state` is a read-only property backed by a ContextVar; outside a served
    # call it falls through to `_inert_state`.
    user._inert_state = octave_environment.OctaveState(attempts=1)
    user.task = SimpleNamespace(
        prompt="Write this GNU Octave function:\n\n    function out = f(x)",
        cases=[{"args": [1], "expected": 1}],
    )

    async def boom(*_args, **_kwargs):
        raise RuntimeError("Guide enabled, but no Prime credential was found")

    async def scored(*_args, **_kwargs):
        return {"passed": 0, "total": 1, "feedback": "CASE 1 FAIL value"}

    user._guide_hint = boom
    original = octave_environment.execute_candidate
    octave_environment.execute_candidate = scored
    try:
        reply = asyncio.run(user.respond("```octave\nfunction out=f(x)\nout=x;\nend\n```"))
    finally:
        octave_environment.execute_candidate = original

    assert reply, "respond must still return a usable user turn"
    assert "Guide hint:" not in reply[0]["content"]
    assert "0/1" in reply[0]["content"], "the ordinary diagnostic must survive"
    assert "no Prime credential" in user.state.guide_unavailable, (
        "the reason must be recorded, or a misconfigured run looks healthy"
    )


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
            reward_mode="case_fraction",
        )
    )
    trace = SimpleNamespace(
        info={"octave": {"fraction": 0.0, "structured_result": 1.0}},
        state=SimpleNamespace(attempts=1),
    )
    assert asyncio.run(octave_environment.OctaveTask.case_fraction(task, trace)) == 0.0


def _reward_of(fraction: float, *, reward_mode: str, attempts: int = 1) -> float:
    task = SimpleNamespace(
        config=SimpleNamespace(
            second_attempt_multiplier=0.85,
            guided_attempt_multiplier=0.60,
            reward_mode=reward_mode,
        )
    )
    trace = SimpleNamespace(
        info={"octave": {"fraction": fraction, "structured_result": 1.0}},
        state=SimpleNamespace(attempts=attempts, guide_used=False),
    )
    return asyncio.run(octave_environment.OctaveTask.case_fraction(task, trace))


def test_solved_only_pays_nothing_for_code_that_merely_runs() -> None:
    # The 2026-08-10 ablation. Partial case credit is the only channel through
    # which an answer that is not fully correct can be worth anything -- there
    # has been no execution or structured-output bonus since the 2026-08-05
    # hardening -- so closing it makes the reward strictly "right answer or
    # nothing". Measured on the 2026-07-29 distribution, that channel is 5 of
    # 200 rollouts and about 6% of total reward mass.
    for fraction in (0.0, 1 / 6, 0.5, 5 / 6):
        assert _reward_of(fraction, reward_mode="solved_only") == 0.0
    assert _reward_of(1.0, reward_mode="solved_only") == 1.0

    # The default is unchanged, so every historical config reproduces itself.
    assert _reward_of(0.5, reward_mode="case_fraction") == 0.5


def test_solved_only_still_discounts_by_attempt() -> None:
    # The ablation removes partial credit, not the retry discount. Leaving the
    # discount in place is what keeps this a one-variable change against the
    # paired control run.
    assert _reward_of(1.0, reward_mode="solved_only", attempts=2) == 0.85
    assert _reward_of(1.0, reward_mode="solved_only", attempts=3) == 0.60
    assert _reward_of(5 / 6, reward_mode="solved_only", attempts=2) == 0.0


def test_solved_is_reported_undiscounted_in_both_reward_modes() -> None:
    # Solve rate must never again be recovered by thresholding a discounted
    # reward (see PIPELINE_LOG, 2026-08-09). `solved` is the field to read, and
    # it has to mean the same thing in the ablation arm as in the control.
    def solved(fraction: float, attempts: int) -> float:
        trace = SimpleNamespace(
            info={"octave": {"fraction": fraction}},
            state=SimpleNamespace(attempts=attempts),
        )
        return asyncio.run(octave_environment.OctaveTask.solved(None, trace))

    assert solved(1.0, 3) == 1.0
    assert solved(5 / 6, 1) == 0.0


def test_reward_mode_is_validated_at_load() -> None:
    taskset = octave_environment.load_environment(reward_mode="solved_only")
    assert taskset.config.task.reward_mode == "solved_only"
    assert octave_environment.load_environment().config.task.reward_mode == "case_fraction"
    with pytest.raises(ValueError, match="reward_mode"):
        octave_environment.load_environment(reward_mode="execution_bonus")


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
