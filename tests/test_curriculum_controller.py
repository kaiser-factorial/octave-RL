import importlib.util
import json
import sys
import tomllib
from pathlib import Path

MODULE_PATH = Path(__file__).parents[1] / "scripts" / "curriculum_controller.py"
SPEC = importlib.util.spec_from_file_location("curriculum_controller", MODULE_PATH)
assert SPEC and SPEC.loader
curriculum = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = curriculum
SPEC.loader.exec_module(curriculum)


def metrics(level: int, raw: float, examples: int = 100):
    return curriculum.LevelMetrics(
        level=level,
        examples=examples,
        raw_case_fraction=raw,
        first_attempt_success=raw,
        eventual_success=raw,
        average_attempts=1.5,
        truncation_rate=0.1,
        error_rate=0.0,
    )


def add_eval(state, step: int, level_metrics):
    state.current_step = step
    state.evaluations.append(
        {
            "step": step,
            "levels": {
                str(item.level): curriculum.asdict(item) for item in level_metrics
            },
        }
    )


def test_level1_requires_consecutive_heldout_passes():
    state = curriculum.CurriculumState()
    add_eval(state, 5, [metrics(1, 0.80)])
    assert curriculum.maybe_transition(state, consecutive=2, min_examples=50) is None
    add_eval(state, 10, [metrics(1, 0.82)])
    assert (
        curriculum.maybe_transition(state, consecutive=2, min_examples=50)
        == "level1_mastery"
    )
    assert state.stage_name == "introduce_level2"
    assert state.ratios == (0.8, 0.2, 0.0)


def test_small_eval_cannot_trigger_promotion():
    state = curriculum.CurriculumState()
    add_eval(state, 5, [metrics(1, 1.0, examples=10)])
    add_eval(state, 10, [metrics(1, 1.0, examples=10)])
    assert curriculum.maybe_transition(state, consecutive=2, min_examples=50) is None


def test_regression_precedes_promotion():
    state = curriculum.CurriculumState(stage_index=2)
    add_eval(state, 20, [metrics(1, 0.40), metrics(2, 0.90)])
    assert (
        curriculum.maybe_transition(state, consecutive=1, min_examples=50)
        == "level1_regression"
    )
    assert state.stage_name == "introduce_level2"


def test_level2_regression_demotes_before_harder_promotion():
    state = curriculum.CurriculumState(stage_index=3)
    add_eval(state, 25, [metrics(1, 0.90), metrics(2, 0.05), metrics(3, 0.90)])
    assert (
        curriculum.maybe_transition(state, consecutive=1, min_examples=50)
        == "level2_regression"
    )
    assert state.stage_name == "level2_working_set"


def test_advanced_stage_requires_level3_signal():
    state = curriculum.CurriculumState(stage_index=3)
    add_eval(state, 25, [metrics(1, 0.90), metrics(2, 0.90), metrics(3, 0.0)])
    add_eval(state, 30, [metrics(1, 0.90), metrics(2, 0.90), metrics(3, 0.0)])
    assert curriculum.maybe_transition(state, consecutive=2, min_examples=50) is None
    add_eval(state, 35, [metrics(1, 0.90), metrics(2, 0.90), metrics(3, 0.40)])
    add_eval(state, 40, [metrics(1, 0.90), metrics(2, 0.90), metrics(3, 0.40)])
    assert (
        curriculum.maybe_transition(state, consecutive=2, min_examples=50)
        == "level3_signal"
    )
    assert state.stage_name == "advanced"


def test_next_stage_does_not_reuse_pretransition_evaluations():
    state = curriculum.CurriculumState()
    add_eval(state, 5, [metrics(1, 0.90), metrics(2, 0.90)])
    add_eval(state, 10, [metrics(1, 0.90), metrics(2, 0.90)])
    assert (
        curriculum.maybe_transition(state, consecutive=2, min_examples=50)
        == "level1_mastery"
    )
    add_eval(state, 15, [metrics(1, 0.90), metrics(2, 0.90)])
    assert curriculum.maybe_transition(state, consecutive=2, min_examples=50) is None
    add_eval(state, 20, [metrics(1, 0.90), metrics(2, 0.90)])
    assert (
        curriculum.maybe_transition(state, consecutive=2, min_examples=50)
        == "level2_signal"
    )


def test_transition_records_evaluation_step_not_chunk_tip():
    state = curriculum.CurriculumState(current_step=20)
    state.evaluations = [
        {"step": 10, "levels": {"1": curriculum.asdict(metrics(1, 0.9))}},
        {"step": 15, "levels": {"1": curriculum.asdict(metrics(1, 0.9))}},
    ]
    assert (
        curriculum.maybe_transition(
            state,
            consecutive=2,
            min_examples=50,
            at_step=15,
        )
        == "level1_mastery"
    )
    assert state.current_step == 20
    assert state.transitions[-1]["step"] == 15


def test_render_uses_stage_ratios_and_stage_relevant_evals(tmp_path):
    state = curriculum.CurriculumState(
        stage_index=1,
        current_step=5,
        checkpoint_step=5,
    )
    text = curriculum.render_config(
        state,
        model_path=tmp_path / "model",
        output_dir=tmp_path / "run",
        target_step=10,
        eval_interval=5,
        eval_examples=50,
        learning_rate=5e-6,
        integrated_eval=True,
    )
    assert 'name = "octave-level1-train"' in text
    assert 'name = "octave-level2-train"' in text
    assert 'name = "octave-level3-train"' not in text
    assert "ratio = 0.8" in text
    assert "ratio = 0.2" in text
    assert "max_completion_tokens = 1536" in text
    assert "seq_len = 4096" in text
    assert "max_model_len = 4096" in text
    assert "enable_thinking = false" in text
    assert "clean_output_dir = false" in text
    parsed = tomllib.loads(text)
    assert parsed["trainer"]["optim"]["lr"] == 5e-6
    assert len(parsed["orchestrator"]["train"]["env"]) == 2
    assert len(parsed["orchestrator"]["eval"]["env"]) == 2


def test_eval_scope_expands_only_when_a_gate_needs_it(tmp_path):
    expected = {
        0: ["octave-level1-eval"],
        1: ["octave-level1-eval", "octave-level2-eval"],
        2: ["octave-level1-eval", "octave-level2-eval"],
        3: ["octave-level1-eval", "octave-level2-eval", "octave-level3-eval"],
        4: ["octave-level1-eval", "octave-level2-eval", "octave-level3-eval"],
    }
    for stage_index, names in expected.items():
        state = curriculum.CurriculumState(stage_index=stage_index)
        text = curriculum.render_config(
            state,
            model_path=tmp_path / "model",
            output_dir=tmp_path / "run",
            target_step=5,
            eval_interval=5,
            eval_examples=20,
            integrated_eval=True,
        )
        parsed = tomllib.loads(text)
        assert [item["name"] for item in parsed["orchestrator"]["eval"]["env"]] == names


def test_manual_start_stage_is_recorded_without_a_promotion():
    state = curriculum.initial_state("introduce_level2")
    assert state.stage_name == "introduce_level2"
    assert state.evaluations == []
    assert state.transitions == []
    assert state.initialization == {
        "mode": "manual",
        "selected_stage": "introduce_level2",
    }


def test_init_cli_accepts_a_manual_start_stage(tmp_path, monkeypatch):
    state_path = tmp_path / "manual-state.json"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            str(MODULE_PATH),
            "init",
            "--state",
            str(state_path),
            "--start-stage",
            "introduce_level3",
        ],
    )
    assert curriculum.main() == 0
    state = curriculum.load_state(state_path)
    assert state.stage_name == "introduce_level3"
    assert state.initialization["mode"] == "manual"


def test_assessment_recommends_a_stage_without_promotion_history(tmp_path):
    paths = {}
    for level, raw in ((1, 0.9), (2, 0.5), (3, 0.0)):
        path = tmp_path / f"level-{level}.jsonl"
        path.write_text(
            "\n".join(
                json.dumps(
                    {
                        "metrics": {"raw_case_fraction": raw, "attempts_used": 1},
                        "calls": [],
                        "errors": [],
                    }
                )
                for _ in range(24)
            )
            + "\n"
        )
        paths[level] = [path]

    levels, stage_name, reason, gates = curriculum.assess_trace_levels(
        paths,
        min_examples=24,
    )
    assert set(levels) == {"1", "2", "3"}
    assert stage_name == "level2_working_set"
    assert reason == "level2_signal"
    assert gates["level2_signal"]["passes"] is True
    assert gates["level2_mastery"]["passes"] is False

    state = curriculum.initial_state(
        stage_name,
        mode="assessment",
        assessment={"levels": {key: curriculum.asdict(value) for key, value in levels.items()}},
    )
    assert state.evaluations == []
    assert state.transitions == []
    assert state.initialization["mode"] == "assessment"


def test_assessment_requires_all_three_levels(tmp_path):
    path = tmp_path / "level-1.jsonl"
    path.write_text(
        json.dumps(
            {
                "metrics": {"raw_case_fraction": 1.0, "attempts_used": 1},
                "calls": [],
                "errors": [],
            }
        )
        + "\n"
    )
    try:
        curriculum.assess_trace_levels({1: [path]}, min_examples=1)
    except ValueError as error:
        assert "Levels 1, 2, and 3" in str(error)
    else:
        raise AssertionError("partial assessment was accepted")


def test_train_only_config_omits_integrated_eval(tmp_path):
    text = curriculum.render_config(
        curriculum.CurriculumState(stage_index=1),
        model_path=tmp_path / "model",
        output_dir=tmp_path / "run",
        target_step=10,
        eval_interval=10,
        eval_examples=20,
        integrated_eval=False,
    )
    parsed = tomllib.loads(text)
    assert "eval" not in parsed["orchestrator"]
    assert len(parsed["orchestrator"]["train"]["env"]) == 2
    assert parsed["ckpt"]["interval"] == 10


def test_render_defaults_to_bounded_train_only_configuration(tmp_path):
    text = curriculum.render_config(
        curriculum.CurriculumState(stage_index=1),
        model_path=tmp_path / "model",
        output_dir=tmp_path / "run",
        target_step=10,
        eval_interval=10,
        eval_examples=20,
    )
    parsed = tomllib.loads(text)
    assert "eval" not in parsed["orchestrator"]
    assert parsed["orchestrator"]["batch_size"] == 8
    assert parsed["orchestrator"]["group_size"] == 2
    assert parsed["orchestrator"]["max_inflight_rollouts"] == 2
    assert parsed["orchestrator"]["renderer"]["enable_thinking"] is False
    assert parsed["orchestrator"]["train"]["env"][0]["timeout"]["finalize"] == 420
    assert (
        parsed["orchestrator"]["train"]["env"][0]["harness"]["runtime"]["type"]
        == "subprocess"
    )
    assert {
        item["pool"]["multiplex"] for item in parsed["orchestrator"]["train"]["env"]
    } == {2}


def test_cli_defaults_to_train_only_but_accepts_legacy_disable_flag():
    parser = curriculum.build_parser()
    args = parser.parse_args(
        [
            "render",
            "--state",
            "state.json",
            "--model-path",
            "model",
            "--output-dir",
            "output",
            "--target-step",
            "5",
            "--config",
            "config.toml",
        ]
    )
    assert args.integrated_eval is False
    assert args.group_size == 2
    assert args.max_inflight_rollouts == 2
    legacy = parser.parse_args(
        [
            "render",
            "--state",
            "state.json",
            "--model-path",
            "model",
            "--output-dir",
            "output",
            "--target-step",
            "5",
            "--config",
            "config.toml",
            "--disable-integrated-eval",
        ]
    )
    assert legacy.integrated_eval is False


def test_run_cli_defaults_to_one_train_only_chunk():
    parser = curriculum.build_parser()
    args = parser.parse_args(
        [
            "run",
            "--state",
            "state.json",
            "--prime-rl-dir",
            "prime-rl",
            "--model-path",
            "model",
            "--output-dir",
            "output",
            "--config",
            "config.toml",
            "--price-per-hour",
            "1",
        ]
    )
    assert args.integrated_eval is False
    assert args.continue_train_only is False
    assert args.group_size == 2
    assert args.max_inflight_rollouts == 2


def test_train_only_run_returns_after_one_durable_chunk(tmp_path, monkeypatch):
    state_path = tmp_path / "state.json"
    curriculum.save_state(state_path, curriculum.CurriculumState())
    args = curriculum.build_parser().parse_args(
        [
            "run",
            "--state",
            str(state_path),
            "--prime-rl-dir",
            str(tmp_path),
            "--model-path",
            str(tmp_path / "model"),
            "--output-dir",
            str(tmp_path / "output"),
            "--config",
            str(tmp_path / "generated.toml"),
            "--max-steps",
            "10",
            "--chunk-steps",
            "2",
            "--price-per-hour",
            "1",
        ]
    )
    calls = []

    def fake_run_chunk(**kwargs):
        calls.append(kwargs)
        return 0, False

    monkeypatch.setattr(curriculum, "run_chunk", fake_run_chunk)
    assert curriculum.orchestrate(args) == 0
    assert len(calls) == 1
    assert curriculum.load_state(state_path).current_step == 2


def test_continue_train_only_requires_explicit_opt_in(tmp_path, monkeypatch):
    state_path = tmp_path / "state.json"
    curriculum.save_state(state_path, curriculum.CurriculumState())
    args = curriculum.build_parser().parse_args(
        [
            "run",
            "--state",
            str(state_path),
            "--prime-rl-dir",
            str(tmp_path),
            "--model-path",
            str(tmp_path / "model"),
            "--output-dir",
            str(tmp_path / "output"),
            "--config",
            str(tmp_path / "generated.toml"),
            "--max-steps",
            "4",
            "--chunk-steps",
            "2",
            "--price-per-hour",
            "1",
            "--continue-train-only",
        ]
    )
    calls = []

    def fake_run_chunk(**kwargs):
        calls.append(kwargs)
        return 0, False

    monkeypatch.setattr(curriculum, "run_chunk", fake_run_chunk)
    assert curriculum.orchestrate(args) == 0
    assert len(calls) == 2
    assert curriculum.load_state(state_path).current_step == 4


def test_concurrency_limited_config_preserves_batch_size(tmp_path):
    text = curriculum.render_config(
        curriculum.CurriculumState(stage_index=1),
        model_path=tmp_path / "model",
        output_dir=tmp_path / "run",
        target_step=10,
        eval_interval=10,
        eval_examples=20,
        integrated_eval=False,
        batch_size=8,
        group_size=2,
        max_inflight_rollouts=2,
    )
    parsed = tomllib.loads(text)
    assert parsed["orchestrator"]["batch_size"] == 8
    assert parsed["orchestrator"]["group_size"] == 2
    assert parsed["orchestrator"]["max_inflight_rollouts"] == 2
    assert {
        item["group_size"] for item in parsed["orchestrator"]["train"]["env"]
    } == {2}


def test_full_finetune_uses_nccl_without_lora(tmp_path):
    text = curriculum.render_config(
        curriculum.CurriculumState(),
        model_path=tmp_path / "model",
        output_dir=tmp_path / "run",
        target_step=5,
        eval_interval=5,
        eval_examples=24,
        full_finetune=True,
    )
    parsed = tomllib.loads(text)
    assert "lora" not in parsed["trainer"]["model"]
    assert "ckpt" not in parsed["trainer"]
    assert parsed["weight_broadcast"]["type"] == "nccl"


def test_trace_summary_counts_token_truncation(tmp_path):
    rows = [
        {
            "metrics": {"raw_case_fraction": 1.0, "attempts_used": 1.0},
            "calls": [{"finish_reason": "stop"}],
            "errors": [],
        },
        {
            "metrics": {"raw_case_fraction": 0.5, "attempts_used": 3.0},
            "calls": [{"finish_reason": "length"}],
            "errors": [],
        },
    ]
    path = tmp_path / "traces.jsonl"
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n")
    result = curriculum.summarize_trace_file(path, level=2)
    assert result.raw_case_fraction == 0.75
    assert result.first_attempt_success == 0.5
    assert result.eventual_success == 0.5
    assert result.average_attempts == 2.0
    assert result.truncation_rate == 0.5


def test_discovery_uses_largest_single_policy_cohort(tmp_path):
    trace_path = (
        tmp_path
        / "run_default"
        / "rollouts"
        / "step_5"
        / "eval"
        / "all"
        / "traces.jsonl"
    )
    trace_path.parent.mkdir(parents=True)
    rows = [
        {
            "task": {"data": {"level": 1}},
            "info": {"policy_version": 4},
            "metrics": {"raw_case_fraction": 1.0, "attempts_used": 1},
            "calls": [],
            "errors": [],
        },
        {
            "task": {"data": {"level": 1}},
            "info": {"policy_version": 5},
            "metrics": {"raw_case_fraction": 0.0, "attempts_used": 3},
            "calls": [{"finish_reason": "length"}],
            "errors": [],
        },
        {
            "task": {"data": {"level": 1}},
            "info": {"policy_version": 5},
            "metrics": {"raw_case_fraction": 0.5, "attempts_used": 2},
            "calls": [],
            "errors": [],
        },
    ]
    trace_path.write_text("\n".join(json.dumps(row) for row in rows) + "\n")
    evaluations = curriculum.discover_evaluations(
        tmp_path, curriculum.CurriculumState()
    )
    assert len(evaluations) == 1
    assert evaluations[0].levels["1"].examples == 2
    assert evaluations[0].levels["1"].raw_case_fraction == 0.25


def test_invalid_persisted_stage_is_rejected(tmp_path):
    path = tmp_path / "state.json"
    path.write_text(
        json.dumps(
            {
                "version": 1,
                "stage_index": 99,
                "current_step": 0,
                "observed_steps": [],
                "evaluations": [],
                "transitions": [],
            }
        )
    )
    try:
        curriculum.load_state(path)
    except ValueError as error:
        assert "stage index" in str(error)
    else:
        raise AssertionError("invalid stage was accepted")


def test_version1_state_migrates_checkpoint_progress(tmp_path):
    path = tmp_path / "state.json"
    path.write_text(
        json.dumps(
            {
                "version": 1,
                "stage_index": 0,
                "current_step": 15,
                "observed_steps": [],
                "evaluations": [],
                "transitions": [],
            }
        )
    )
    state = curriculum.load_state(path)
    assert state.version == 3
    assert state.current_step == 15
    assert state.checkpoint_step == 15
    assert state.step_offset == 0


def test_version2_state_migrates_assessment_metadata(tmp_path):
    path = tmp_path / "state.json"
    path.write_text(
        json.dumps(
            {
                "version": 2,
                "stage_index": 0,
                "current_step": 0,
                "checkpoint_step": 0,
                "step_offset": 0,
                "observed_steps": [],
                "evaluations": [],
                "transitions": [],
            }
        )
    )
    state = curriculum.load_state(path)
    assert state.version == 3
    assert state.initialization == {}


def test_standalone_traces_can_trigger_a_live_transition(tmp_path):
    state_path = tmp_path / "state.json"
    curriculum.save_state(state_path, curriculum.CurriculumState())
    rows = [
        {
            "metrics": {"raw_case_fraction": 1.0, "attempts_used": 1},
            "calls": [{"finish_reason": "stop"}],
            "errors": [],
        }
        for _ in range(24)
    ]
    for step in (10, 15):
        trace_path = tmp_path / f"step-{step}.jsonl"
        trace_path.write_text("\n".join(json.dumps(row) for row in rows) + "\n")
        metrics_result, reason = curriculum.ingest_trace_evaluation(
            state_path,
            trace_path,
            step=step,
            level=1,
            consecutive=2,
            min_examples=20,
        )
        assert metrics_result.examples == 24
        assert reason == ("level1_mastery" if step == 15 else None)

    state = curriculum.load_state(state_path)
    assert state.stage_name == "introduce_level2"
    assert state.observed_steps == [10, 15]
    assert state.transitions == [
        {
            "step": 15,
            "from": "level1_only",
            "to": "introduce_level2",
            "reason": "level1_mastery",
        }
    ]


def test_standalone_trace_rejects_duplicate_step(tmp_path):
    state_path = tmp_path / "state.json"
    trace_path = tmp_path / "traces.jsonl"
    trace_path.write_text(
        json.dumps(
            {
                "metrics": {"raw_case_fraction": 1.0, "attempts_used": 1},
                "calls": [],
                "errors": [],
            }
        )
        + "\n"
    )
    curriculum.save_state(
        state_path,
        curriculum.CurriculumState(observed_steps=[5]),
    )
    try:
        curriculum.ingest_trace_evaluation(
            state_path,
            trace_path,
            step=5,
            level=1,
            consecutive=2,
            min_examples=1,
        )
    except ValueError as error:
        assert "already been observed" in str(error)
    else:
        raise AssertionError("duplicate standalone evaluation was accepted")


def test_standalone_evaluation_can_combine_disjoint_trace_files(tmp_path):
    state_path = tmp_path / "state.json"
    curriculum.save_state(state_path, curriculum.CurriculumState())
    paths = []
    for index, raw in enumerate((1.0, 0.0)):
        path = tmp_path / f"seed-{index}.jsonl"
        path.write_text(
            json.dumps(
                {
                    "metrics": {
                        "raw_case_fraction": raw,
                        "attempts_used": index + 1,
                    },
                    "calls": [],
                    "errors": [],
                }
            )
            + "\n"
        )
        paths.append(path)
    metrics_result, reason = curriculum.ingest_trace_evaluation(
        state_path,
        paths,
        step=5,
        level=1,
        consecutive=2,
        min_examples=2,
    )
    assert reason is None
    assert metrics_result.examples == 2
    assert metrics_result.raw_case_fraction == 0.5
    state = curriculum.load_state(state_path)
    assert state.evaluations[0]["sources"] == {
        "1": [str(path.resolve()) for path in paths]
    }


def test_checkpoint_static_evaluation_combines_levels_at_one_step(tmp_path):
    state_path = tmp_path / "state.json"
    curriculum.save_state(
        state_path,
        curriculum.CurriculumState(stage_index=1, current_step=15),
    )
    trace_files_by_level = {}
    for level, raw in ((1, 1.0), (2, 0.5)):
        path = tmp_path / f"level-{level}.jsonl"
        rows = [
            {
                "metrics": {"raw_case_fraction": raw, "attempts_used": 1},
                "calls": [],
                "errors": [],
            }
            for _ in range(20)
        ]
        path.write_text("\n".join(json.dumps(row) for row in rows) + "\n")
        trace_files_by_level[level] = [path]

    levels, reason = curriculum.ingest_trace_levels(
        state_path,
        trace_files_by_level,
        step=25,
        consecutive=2,
        min_examples=20,
    )
    assert reason is None
    assert set(levels) == {"1", "2"}
    state = curriculum.load_state(state_path)
    assert state.observed_steps == [25]
    assert set(state.evaluations[0]["levels"]) == {"1", "2"}


def test_budget_deadline_does_not_report_chunk_complete(tmp_path, monkeypatch):
    class FakeProcess:
        returncode = None

        def poll(self):
            return None

        def send_signal(self, _signal):
            return None

        def wait(self, timeout):
            assert timeout == 60
            return 0

    monkeypatch.setattr(
        curriculum.subprocess, "Popen", lambda *_args, **_kwargs: FakeProcess()
    )
    monkeypatch.setattr(curriculum.time, "monotonic", lambda: 2.0)
    assert curriculum.run_chunk(
        prime_rl_dir=tmp_path,
        config_path=tmp_path / "config.toml",
        resume_step=0,
        deadline=1.0,
    ) == (0, True)


def test_fatal_runtime_log_aborts_chunk(tmp_path, monkeypatch):
    class FakeProcess:
        returncode = None

        def poll(self):
            return None

        def send_signal(self, _signal):
            return None

        def wait(self, timeout):
            assert timeout == 60
            return 0

    orchestrator_log = tmp_path / "orchestrator.log"
    orchestrator_log.write_text("old run completed\n")

    def launch(*_args, **_kwargs):
        with orchestrator_log.open("a") as handle:
            handle.write("Payment required. Check billing status.\n")
        return FakeProcess()

    monkeypatch.setattr(
        curriculum.subprocess,
        "Popen",
        launch,
    )
    assert curriculum.run_chunk(
        prime_rl_dir=tmp_path,
        config_path=tmp_path / "config.toml",
        resume_step=0,
        deadline=100.0,
        fatal_logs=[orchestrator_log],
    ) == (70, False)


def test_rebased_state_maps_local_eval_step_to_global_step(tmp_path):
    state = curriculum.CurriculumState(
        stage_index=1,
        current_step=15,
        checkpoint_step=0,
        step_offset=15,
    )
    trace_path = (
        tmp_path
        / "run_default"
        / "rollouts"
        / "step_5"
        / "eval"
        / "all"
        / "traces.jsonl"
    )
    trace_path.parent.mkdir(parents=True)
    trace_path.write_text(
        json.dumps(
            {
                "task": {"data": {"level": 1}},
                "info": {"policy_version": 5},
                "metrics": {"raw_case_fraction": 1.0, "attempts_used": 1},
                "calls": [],
                "errors": [],
            }
        )
        + "\n"
    )
    evaluations = curriculum.discover_evaluations(tmp_path, state)
    assert [evaluation.step for evaluation in evaluations] == [20]


def test_rebase_can_adopt_stable_broadcast_steps():
    source = curriculum.CurriculumState(
        stage_index=1,
        current_step=15,
        checkpoint_step=0,
        step_offset=15,
        observed_steps=[10, 15],
        transitions=[{"step": 15, "reason": "level1_mastery"}],
    )
    rebased = curriculum.rebase_state(source, advance_steps=2)
    assert rebased.current_step == 17
    assert rebased.step_offset == 17
    assert rebased.checkpoint_step == 0
    assert rebased.observed_steps == [10, 15]
    assert rebased.transitions == source.transitions
