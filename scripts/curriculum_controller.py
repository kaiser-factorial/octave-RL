"""Checkpointed staged curriculum controller for the Octave prime-rl run.

The controller deliberately changes curriculum only between prime-rl chunks.
That keeps each optimizer segment reproducible while allowing the next segment
to resume its trainer checkpoint with new environment sampling ratios.
"""

from __future__ import annotations

import argparse
import json
import math
import signal
import subprocess
import time
from collections.abc import Iterable
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

STATE_VERSION = 3
DEFAULT_STAGES = (
    ("level1_only", (1.0, 0.0, 0.0)),
    ("introduce_level2", (0.8, 0.2, 0.0)),
    ("level2_working_set", (0.3, 0.7, 0.0)),
    ("introduce_level3", (0.2, 0.6, 0.2)),
    ("advanced", (0.1, 0.4, 0.5)),
)
STAGE_INDEX = {name: index for index, (name, _ratios) in enumerate(DEFAULT_STAGES)}
MAX_COMPLETION_TOKENS = 1536


@dataclass(frozen=True)
class LevelMetrics:
    level: int
    examples: int
    raw_case_fraction: float
    first_attempt_success: float
    eventual_success: float
    average_attempts: float
    truncation_rate: float
    error_rate: float


@dataclass(frozen=True)
class Evaluation:
    step: int
    levels: dict[str, LevelMetrics]


@dataclass
class CurriculumState:
    version: int = STATE_VERSION
    stage_index: int = 0
    current_step: int = 0
    checkpoint_step: int = 0
    step_offset: int = 0
    observed_steps: list[int] = field(default_factory=list)
    evaluations: list[dict[str, Any]] = field(default_factory=list)
    transitions: list[dict[str, Any]] = field(default_factory=list)
    initialization: dict[str, Any] = field(default_factory=dict)

    @property
    def stage_name(self) -> str:
        return DEFAULT_STAGES[self.stage_index][0]

    @property
    def ratios(self) -> tuple[float, float, float]:
        return DEFAULT_STAGES[self.stage_index][1]


def load_state(path: Path) -> CurriculumState:
    if not path.exists():
        return CurriculumState()
    data = json.loads(path.read_text())
    if data.get("version") == 1:
        data["version"] = STATE_VERSION
        data["checkpoint_step"] = data["current_step"]
        data["step_offset"] = 0
    elif data.get("version") == 2:
        data["version"] = STATE_VERSION
    elif data.get("version") != STATE_VERSION:
        raise ValueError(f"Unsupported curriculum state version: {data.get('version')}")
    data.setdefault("initialization", {})
    state = CurriculumState(**data)
    if not 0 <= state.stage_index < len(DEFAULT_STAGES):
        raise ValueError(f"Invalid curriculum stage index: {state.stage_index}")
    if state.current_step < 0:
        raise ValueError(f"Invalid current step: {state.current_step}")
    if state.checkpoint_step < 0 or state.step_offset < 0:
        raise ValueError("Checkpoint step and step offset must be non-negative")
    return state


def initial_state(
    stage_name: str = "level1_only",
    *,
    mode: str = "manual",
    assessment: dict[str, Any] | None = None,
) -> CurriculumState:
    """Create a fresh state without manufacturing a promotion observation."""
    try:
        stage_index = STAGE_INDEX[stage_name]
    except KeyError as error:
        valid = ", ".join(STAGE_INDEX)
        raise ValueError(f"Unknown stage {stage_name!r}; choose one of: {valid}") from error
    initialization: dict[str, Any] = {
        "mode": mode,
        "selected_stage": stage_name,
    }
    if assessment is not None:
        initialization["assessment"] = assessment
    return CurriculumState(stage_index=stage_index, initialization=initialization)


def save_state(path: Path, state: CurriculumState) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(asdict(state), indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def rebase_state(
    source: CurriculumState,
    *,
    advance_steps: int = 0,
) -> CurriculumState:
    """Start a new merged-model segment while preserving global history."""
    if advance_steps < 0:
        raise ValueError("--advance-steps must be non-negative")
    current_step = source.current_step + advance_steps
    return CurriculumState(
        stage_index=source.stage_index,
        current_step=current_step,
        checkpoint_step=0,
        step_offset=current_step,
        observed_steps=list(source.observed_steps),
        evaluations=list(source.evaluations),
        transitions=list(source.transitions),
        initialization=dict(source.initialization),
    )


def wilson_lower(successes: float, total: int, z: float = 1.644854) -> float:
    """One-sided 95% Wilson lower bound, accepting fractional case successes."""
    if total <= 0:
        return 0.0
    proportion = min(1.0, max(0.0, successes / total))
    denominator = 1.0 + z * z / total
    center = proportion + z * z / (2.0 * total)
    spread = z * math.sqrt(
        proportion * (1.0 - proportion) / total + z * z / (4.0 * total * total)
    )
    return max(0.0, (center - spread) / denominator)


def _trace_rows(path: Path) -> Iterable[dict[str, Any]]:
    for line in path.read_text().splitlines():
        if line.strip():
            yield json.loads(line)


def summarize_rows(rows: list[dict[str, Any]], level: int) -> LevelMetrics:
    if not rows:
        raise ValueError(f"No Level {level} traces")
    raw = [float(row.get("metrics", {}).get("raw_case_fraction", 0.0)) for row in rows]
    attempts = [float(row.get("metrics", {}).get("attempts_used", 1.0)) for row in rows]
    errors = [1.0 if row.get("errors") else 0.0 for row in rows]
    truncated = [
        1.0
        if any(call.get("finish_reason") == "length" for call in row.get("calls", []))
        else 0.0
        for row in rows
    ]
    return LevelMetrics(
        level=level,
        examples=len(rows),
        raw_case_fraction=sum(raw) / len(raw),
        first_attempt_success=sum(
            fraction >= 1.0 and attempt <= 1.0
            for fraction, attempt in zip(raw, attempts, strict=True)
        )
        / len(rows),
        eventual_success=sum(fraction >= 1.0 for fraction in raw) / len(rows),
        average_attempts=sum(attempts) / len(attempts),
        truncation_rate=sum(truncated) / len(truncated),
        error_rate=sum(errors) / len(errors),
    )


def summarize_trace_file(path: Path, level: int) -> LevelMetrics:
    return summarize_rows(list(_trace_rows(path)), level)


def discover_evaluations(output_dir: Path, state: CurriculumState) -> list[Evaluation]:
    evaluations: list[Evaluation] = []
    rollout_root = output_dir / "run_default" / "rollouts"
    if not rollout_root.exists():
        return evaluations
    for step_dir in sorted(
        rollout_root.glob("step_*"),
        key=lambda path: int(path.name.removeprefix("step_")),
    ):
        checkpoint_step = int(step_dir.name.removeprefix("step_"))
        step = state.step_offset + checkpoint_step
        if step in state.observed_steps:
            continue
        combined = step_dir / "eval" / "all" / "traces.jsonl"
        if not combined.exists():
            continue
        all_rows = list(_trace_rows(combined))
        if not all_rows:
            continue
        policy_counts: dict[int, int] = {}
        for row in all_rows:
            version = int(row.get("info", {}).get("policy_version", -1))
            policy_counts[version] = policy_counts.get(version, 0) + 1
        # Long evals can straddle a weight broadcast. Select the largest
        # internally consistent policy cohort and let min_examples reject it
        # when too few comparable traces remain.
        selected_policy = max(policy_counts, key=policy_counts.get)
        rows_by_level: dict[int, list[dict[str, Any]]] = {}
        for row in all_rows:
            if int(row.get("info", {}).get("policy_version", -1)) != selected_policy:
                continue
            level = int(row["task"]["data"]["level"])
            rows_by_level.setdefault(level, []).append(row)
        levels = {
            str(level): summarize_rows(rows, level)
            for level, rows in rows_by_level.items()
        }
        evaluations.append(Evaluation(step=step, levels=levels))
    return evaluations


def _level_history(state: CurriculumState, level: int) -> list[dict[str, Any]]:
    return [
        evaluation["levels"][str(level)]
        for evaluation in state.evaluations
        if str(level) in evaluation["levels"]
    ]


def _current_stage_history(state: CurriculumState, level: int) -> list[dict[str, Any]]:
    started_after = state.transitions[-1]["step"] if state.transitions else -1
    return [
        evaluation["levels"][str(level)]
        for evaluation in state.evaluations
        if evaluation["step"] > started_after and str(level) in evaluation["levels"]
    ]


def _steady_pass(
    history: list[dict[str, Any]],
    *,
    threshold: float,
    consecutive: int,
    min_examples: int,
    max_error: float = 0.02,
) -> bool:
    if len(history) < consecutive:
        return False
    recent = history[-consecutive:]
    for metrics in recent:
        if metrics["examples"] < min_examples:
            return False
        if metrics["error_rate"] > max_error:
            return False
        successes = metrics["raw_case_fraction"] * metrics["examples"]
        if wilson_lower(successes, metrics["examples"]) < threshold:
            return False
    return True


def _assessment_gate(
    metrics: LevelMetrics,
    *,
    threshold: float,
    min_examples: int,
) -> dict[str, Any]:
    """Describe one conservative placement gate without mutating state."""
    lower_bound = wilson_lower(
        metrics.raw_case_fraction * metrics.examples,
        metrics.examples,
    )
    passes = (
        metrics.examples >= min_examples
        and metrics.error_rate <= 0.02
        and lower_bound >= threshold
    )
    return {
        "threshold": threshold,
        "examples": metrics.examples,
        "error_rate": metrics.error_rate,
        "wilson_lower_one_sided_95": lower_bound,
        "passes": passes,
    }


def recommend_start_stage(
    levels: dict[str, LevelMetrics],
    *,
    min_examples: int,
) -> tuple[str, str, dict[str, dict[str, Any]]]:
    """Recommend a mix from one all-level baseline without claiming promotion.

    A bootstrap assessment is a placement aid, not evidence for the normal
    two-policy promotion gate.  It therefore remains outside ``evaluations``.
    """
    required = {"1", "2", "3"}
    if set(levels) != required:
        missing = sorted(required - set(levels))
        extra = sorted(set(levels) - required)
        details = []
        if missing:
            details.append(f"missing Level(s) {', '.join(missing)}")
        if extra:
            details.append(f"unexpected Level(s) {', '.join(extra)}")
        raise ValueError("Assessment requires exactly Levels 1, 2, and 3: " + "; ".join(details))

    gates = {
        "level1_mastery": _assessment_gate(
            levels["1"], threshold=0.55, min_examples=min_examples
        ),
        "level2_signal": _assessment_gate(
            levels["2"], threshold=0.20, min_examples=min_examples
        ),
        "level2_mastery": _assessment_gate(
            levels["2"], threshold=0.45, min_examples=min_examples
        ),
        "level2_advanced": _assessment_gate(
            levels["2"], threshold=0.60, min_examples=min_examples
        ),
        "level3_signal": _assessment_gate(
            levels["3"], threshold=0.10, min_examples=min_examples
        ),
    }
    if not gates["level1_mastery"]["passes"]:
        return "level1_only", "level1_baseline", gates
    if not gates["level2_signal"]["passes"]:
        return "introduce_level2", "level2_baseline", gates
    if not gates["level2_mastery"]["passes"]:
        return "level2_working_set", "level2_signal", gates
    if not (
        gates["level2_advanced"]["passes"]
        and gates["level3_signal"]["passes"]
    ):
        return "introduce_level3", "level3_baseline", gates
    return "advanced", "all_level_baseline", gates


def maybe_transition(
    state: CurriculumState,
    *,
    consecutive: int,
    min_examples: int,
    at_step: int | None = None,
) -> str | None:
    """Promote or regress with hysteresis; held-out data is the only gate."""
    old_stage = state.stage_index
    level1 = _level_history(state, 1)
    level2 = _level_history(state, 2)
    level1_in_stage = _current_stage_history(state, 1)
    level2_in_stage = _current_stage_history(state, 2)
    level3_in_stage = _current_stage_history(state, 3)

    # Regression protection is deliberately checked before promotion.
    if old_stage > 0 and level1:
        latest = level1[-1]
        if latest["raw_case_fraction"] < 0.55 or latest["error_rate"] > 0.02:
            state.stage_index = max(0, old_stage - 1)
            reason = "level1_regression"
        else:
            reason = ""
    else:
        reason = ""

    if state.stage_index == old_stage and old_stage >= 2 and level2:
        latest = level2[-1]
        if latest["raw_case_fraction"] < 0.10 or latest["error_rate"] > 0.02:
            state.stage_index = old_stage - 1
            reason = "level2_regression"

    if state.stage_index == old_stage:
        if old_stage == 0 and _steady_pass(
            level1_in_stage,
            threshold=0.55,
            consecutive=consecutive,
            min_examples=min_examples,
        ):
            state.stage_index = 1
            reason = "level1_mastery"
        elif old_stage == 1 and _steady_pass(
            level2_in_stage,
            threshold=0.20,
            consecutive=consecutive,
            min_examples=min_examples,
        ):
            state.stage_index = 2
            reason = "level2_signal"
        elif old_stage == 2 and _steady_pass(
            level2_in_stage,
            threshold=0.45,
            consecutive=consecutive,
            min_examples=min_examples,
        ):
            state.stage_index = 3
            reason = "level2_mastery"
        elif (
            old_stage == 3
            and _steady_pass(
                level2_in_stage,
                threshold=0.60,
                consecutive=consecutive,
                min_examples=min_examples,
            )
            and _steady_pass(
                level3_in_stage,
                threshold=0.10,
                consecutive=consecutive,
                min_examples=min_examples,
            )
        ):
            state.stage_index = 4
            reason = "level3_signal"

    if state.stage_index == old_stage:
        return None
    transition = {
        "step": state.current_step if at_step is None else at_step,
        "from": DEFAULT_STAGES[old_stage][0],
        "to": state.stage_name,
        "reason": reason,
    }
    state.transitions.append(transition)
    return reason


def ingest_evaluations(
    state_path: Path,
    output_dir: Path,
    *,
    consecutive: int,
    min_examples: int,
) -> list[Evaluation]:
    state = load_state(state_path)
    found = discover_evaluations(output_dir, state)
    for evaluation in found:
        state.current_step = max(state.current_step, evaluation.step)
        state.observed_steps.append(evaluation.step)
        state.evaluations.append(
            {
                "step": evaluation.step,
                "levels": {
                    key: asdict(metrics) for key, metrics in evaluation.levels.items()
                },
            }
        )
        maybe_transition(
            state,
            consecutive=consecutive,
            min_examples=min_examples,
            at_step=evaluation.step,
        )
    save_state(state_path, state)
    return found


def ingest_trace_evaluation(
    state_path: Path,
    trace_file: Path | Iterable[Path],
    *,
    step: int,
    level: int,
    consecutive: int,
    min_examples: int,
) -> tuple[LevelMetrics, str | None]:
    """Ingest one externally served held-out trace set as a gated evaluation."""
    trace_files = [trace_file] if isinstance(trace_file, Path) else list(trace_file)
    metrics, reason = ingest_trace_levels(
        state_path,
        {level: trace_files},
        step=step,
        consecutive=consecutive,
        min_examples=min_examples,
    )
    return metrics[str(level)], reason


def ingest_trace_levels(
    state_path: Path,
    trace_files_by_level: dict[int, list[Path]],
    *,
    step: int,
    consecutive: int,
    min_examples: int,
) -> tuple[dict[str, LevelMetrics], str | None]:
    """Ingest one checkpoint-static evaluation containing one or more levels."""
    if step < 0:
        raise ValueError("--step must be non-negative")
    if not trace_files_by_level:
        raise ValueError("At least one level trace is required")
    invalid_levels = set(trace_files_by_level) - {1, 2, 3}
    if invalid_levels:
        raise ValueError("--level must be one of 1, 2, or 3")
    state = load_state(state_path)
    if step in state.observed_steps:
        raise ValueError(f"Step {step} has already been observed")
    empty_levels = [
        level for level, trace_files in trace_files_by_level.items() if not trace_files
    ]
    if empty_levels:
        raise ValueError(f"No trace files supplied for level(s): {empty_levels}")
    levels = {
        str(level): summarize_rows(
            [row for path in trace_files for row in _trace_rows(path)],
            level,
        )
        for level, trace_files in sorted(trace_files_by_level.items())
    }
    sources = {
        str(level): [str(path.resolve()) for path in trace_files]
        for level, trace_files in sorted(trace_files_by_level.items())
    }
    state.current_step = max(state.current_step, step)
    state.observed_steps.append(step)
    state.evaluations.append(
        {
            "step": step,
            "sources": sources,
            "levels": {key: asdict(metrics) for key, metrics in levels.items()},
        }
    )
    reason = maybe_transition(
        state,
        consecutive=consecutive,
        min_examples=min_examples,
        at_step=step,
    )
    save_state(state_path, state)
    return levels, reason


def parse_level_trace_specs(specifications: Iterable[str]) -> dict[int, list[Path]]:
    trace_files_by_level: dict[int, list[Path]] = {}
    for specification in specifications:
        try:
            level_literal, path_literal = specification.split(":", 1)
            level = int(level_literal)
        except ValueError as error:
            raise ValueError(
                f"Invalid --trace value {specification!r}; expected LEVEL:PATH"
            ) from error
        trace_files_by_level.setdefault(level, []).append(Path(path_literal))
    return trace_files_by_level


def assess_trace_levels(
    trace_files_by_level: dict[int, list[Path]],
    *,
    min_examples: int,
) -> tuple[dict[str, LevelMetrics], str, str, dict[str, dict[str, Any]]]:
    """Summarize a static Level 1/2/3 baseline and recommend a start stage."""
    levels = {
        str(level): summarize_rows(
            [row for path in trace_files for row in _trace_rows(path)],
            level,
        )
        for level, trace_files in sorted(trace_files_by_level.items())
    }
    stage_name, reason, gates = recommend_start_stage(
        levels,
        min_examples=min_examples,
    )
    return levels, stage_name, reason, gates


def _environment_block(
    level: int,
    ratio: float,
    *,
    train: bool,
    train_group_size: int = 8,
) -> str:
    split = "train" if train else "eval"
    seed = 314159 + level * 1000 if train else 271828 + level * 1000
    name = f"octave-level{level}-{split}"
    ratio_line = f"ratio = {ratio:g}\n" if train else ""
    group_line = (
        f"group_size = {train_group_size}\n" if train else "group_size = 1\n"
    )
    return f"""
[[orchestrator.{"train" if train else "eval"}.env]]
name = "{name}"
{ratio_line}{group_line}taskset = {{ id = "octave-rl", level = {level}, num_tasks = {500 if train else 100}, seed = {seed}, task = {{ second_attempt_multiplier = 0.85, guided_attempt_multiplier = 0.60, user = {{ colocated = false, max_attempts = 3, guide_enabled = true, guide_model = "Qwen/Qwen3.5-35B-A3B" }} }} }}
harness = {{ id = "null", runtime = {{ type = "subprocess" }} }}
max_turns = 3
max_total_tokens = 6144
timeout = {{ rollout = 700, finalize = 420, scoring = 60 }}
pool = {{ type = "elastic", max_workers = 1, multiplex = {train_group_size if train else 1} }}
""".strip()


def evaluation_levels(stage_index: int) -> tuple[int, ...]:
    """Return only the levels required by the current stage's gates."""
    if stage_index == 0:
        return (1,)
    if stage_index < 3:
        return (1, 2)
    return (1, 2, 3)


def render_config(
    state: CurriculumState,
    *,
    model_path: Path,
    output_dir: Path,
    target_step: int,
    eval_interval: int,
    eval_examples: int,
    full_finetune: bool = False,
    learning_rate: float = 1e-5,
    integrated_eval: bool = False,
    batch_size: int = 8,
    group_size: int = 2,
    max_inflight_rollouts: int = 2,
) -> str:
    if group_size < 2:
        raise ValueError("GRPO --group-size must be at least 2")
    if batch_size < group_size or batch_size % group_size:
        raise ValueError("--batch-size must be a multiple of --group-size")
    if max_inflight_rollouts < group_size:
        raise ValueError("--max-inflight-rollouts must be at least --group-size")
    train_blocks = [
        _environment_block(
            level,
            ratio,
            train=True,
            train_group_size=group_size,
        )
        for level, ratio in enumerate(state.ratios, start=1)
        if ratio > 0
    ]
    eval_blocks = []
    if integrated_eval:
        eval_blocks = [
            _environment_block(level, 0.0, train=False).replace(
                "num_tasks = 100", f"num_tasks = {eval_examples}"
            )
            for level in evaluation_levels(state.stage_index)
        ]
    clean = "true" if state.checkpoint_step == 0 else "false"
    model_literal = json.dumps(str(model_path))
    output_literal = json.dumps(str(output_dir))
    lora_block = (
        ""
        if full_finetune
        else """
[trainer.model.lora]
rank = 16
alpha = 32
dropout = 0.0

[trainer.ckpt.weights]
save_adapter_separately = true
"""
    )
    broadcast_block = (
        """
[weight_broadcast]
type = "nccl"
"""
        if full_finetune
        else ""
    )
    eval_section = ""
    if integrated_eval:
        eval_section = f"""
[orchestrator.eval]
interval = {eval_interval}
num_examples = {eval_examples}
group_size = 1

[orchestrator.eval.sampling]
temperature = 0.0
max_completion_tokens = {MAX_COMPLETION_TOKENS}

{chr(10).join(eval_blocks)}
"""
    return f"""# Generated by scripts/curriculum_controller.py
max_steps = {target_step}
seq_len = 4096
output_dir = {output_literal}
clean_output_dir = {clean}

[deployment]
type = "single_node"
num_train_gpus = 1
num_infer_gpus = 1

[model]
name = {model_literal}

[model.vlm]
vision_encoder_attr = "model.visual"
language_model_attr = "model.language_model"

[orchestrator]
batch_size = {batch_size}
group_size = {group_size}
max_inflight_rollouts = {max_inflight_rollouts}
tasks_per_minute = {max_inflight_rollouts}
collect_inference_metrics = false

[orchestrator.renderer]
name = "qwen3.5"
# Qwen3.5-4B defaults to an open thinking block. Octave tasks require a
# compact fenced function, so avoid spending the completion budget on prose.
enable_thinking = false

[orchestrator.train.sampling]
temperature = 1.0
max_completion_tokens = {MAX_COMPLETION_TOKENS}

{chr(10).join(train_blocks)}

{eval_section}

[trainer]

[trainer.model]
impl = "custom"
optimization_dtype = "bfloat16"
reduce_dtype = "bfloat16"
optim_cpu_offload = true

[trainer.optim]
lr = {learning_rate:g}

{lora_block}

[ckpt]
interval = {eval_interval}
keep_last = 2

{broadcast_block}

[inference]
gpu_memory_utilization = 0.85

[inference.model]
enforce_eager = true
max_model_len = 4096

[inference.vllm_extra]
attention_backend = "TRITON_ATTN"
"""


def run_chunk(
    *,
    prime_rl_dir: Path,
    config_path: Path,
    resume_step: int,
    deadline: float,
    fatal_logs: Iterable[Path] = (),
) -> tuple[int, bool]:
    command = ["uv", "run", "rl", "@", str(config_path)]
    if resume_step > 0:
        command += ["--ckpt.resume-step", str(resume_step)]
    watched_logs = tuple(fatal_logs)
    initial_log_sizes = {
        path: path.stat().st_size if path.exists() else 0 for path in watched_logs
    }
    process = subprocess.Popen(command, cwd=prime_rl_dir)
    while process.poll() is None:
        fatal_markers = (
            "EngineCore encountered a fatal error",
            "EngineDeadError",
            "CUDA error: an illegal memory access",
            "Payment required. Check billing status.",
        )
        for log_path in watched_logs:
            if not log_path.exists():
                continue
            log_bytes = log_path.read_bytes()
            initial_size = initial_log_sizes[log_path]
            if len(log_bytes) < initial_size:
                initial_size = 0
            tail_start = max(initial_size, len(log_bytes) - 262_144)
            tail = log_bytes[tail_start:].decode(errors="replace")
            if any(marker in tail for marker in fatal_markers):
                process.send_signal(signal.SIGINT)
                try:
                    process.wait(timeout=60)
                except subprocess.TimeoutExpired:
                    process.terminate()
                    process.wait(timeout=30)
                return 70, False
        if time.monotonic() >= deadline:
            process.send_signal(signal.SIGINT)
            try:
                return process.wait(timeout=60), True
            except subprocess.TimeoutExpired:
                process.terminate()
                return process.wait(timeout=30), True
        time.sleep(5)
    return int(process.returncode), False


def orchestrate(args: argparse.Namespace) -> int:
    if args.price_per_hour <= 0:
        raise ValueError("--price-per-hour must be positive")
    if args.budget_usd <= 0:
        raise ValueError("--budget-usd must be positive")
    if args.chunk_steps <= 0 or args.max_steps <= 0:
        raise ValueError("--chunk-steps and --max-steps must be positive")
    if args.learning_rate <= 0:
        raise ValueError("--learning-rate must be positive")
    if args.group_size < 2:
        raise ValueError("--group-size must be at least 2")
    if args.batch_size < args.group_size or args.batch_size % args.group_size:
        raise ValueError("--batch-size must be a multiple of --group-size")
    if args.max_inflight_rollouts < args.group_size:
        raise ValueError("--max-inflight-rollouts must be at least --group-size")
    integrated_eval = args.integrated_eval
    if integrated_eval and args.eval_examples < args.min_examples:
        raise ValueError("--eval-examples must be at least --min-examples")
    state_path = args.state.resolve()
    state = load_state(state_path)
    output_dir = args.output_dir.resolve()
    started = time.monotonic()
    deadline = started + args.budget_usd / args.price_per_hour * 3600.0
    # Recover completed evals if a previous invocation stopped after saving a
    # trainer checkpoint but before finishing gate ingestion.
    if integrated_eval:
        ingest_evaluations(
            state_path,
            output_dir,
            consecutive=args.consecutive,
            min_examples=args.min_examples,
        )
    state = load_state(state_path)

    while state.current_step < args.max_steps and time.monotonic() < deadline:
        target = min(args.max_steps, state.current_step + args.chunk_steps)
        target_checkpoint = state.checkpoint_step + (target - state.current_step)
        config_text = render_config(
            state,
            model_path=args.model_path.resolve(),
            output_dir=output_dir,
            target_step=target_checkpoint,
            eval_interval=args.chunk_steps,
            eval_examples=args.eval_examples,
            full_finetune=args.full_finetune,
            learning_rate=args.learning_rate,
            integrated_eval=integrated_eval,
            batch_size=args.batch_size,
            group_size=args.group_size,
            max_inflight_rollouts=args.max_inflight_rollouts,
        )
        args.config.parent.mkdir(parents=True, exist_ok=True)
        args.config.write_text(config_text)
        if args.dry_run:
            print(args.config)
            return 0
        code, deadline_hit = run_chunk(
            prime_rl_dir=args.prime_rl_dir.resolve(),
            config_path=args.config.resolve(),
            resume_step=state.checkpoint_step,
            deadline=deadline,
            fatal_logs=(
                output_dir / "logs" / "inference.log",
                output_dir / "logs" / "orchestrator.log",
            ),
        )
        if code != 0:
            return code
        if deadline_hit:
            return 0
        state.current_step = target
        state.checkpoint_step = target_checkpoint
        save_state(state_path, state)
        if not integrated_eval and not args.continue_train_only:
            # A static merged checkpoint must be evaluated and ingested before
            # a later chunk can legitimately transition the curriculum. Return
            # after one durable train-only chunk so that boundary is explicit.
            return 0
        if integrated_eval:
            ingest_evaluations(
                state_path,
                output_dir,
                consecutive=args.consecutive,
                min_examples=args.min_examples,
            )
        state = load_state(state_path)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)

    initialize = subparsers.add_parser("init")
    initialize.add_argument("--state", type=Path, required=True)
    initialize.add_argument(
        "--start-stage",
        choices=tuple(STAGE_INDEX),
        default="level1_only",
        help="start directly with this training mix; no baseline evaluation is run",
    )

    assess = subparsers.add_parser("assess")
    assess.add_argument(
        "--trace",
        action="append",
        required=True,
        metavar="LEVEL:PATH",
        help="static held-out trace; provide Levels 1, 2, and 3",
    )
    assess.add_argument("--min-examples", type=int, default=24)
    assess.add_argument(
        "--state",
        type=Path,
        help="write a fresh state at the recommended stage",
    )

    ingest = subparsers.add_parser("ingest")
    ingest.add_argument("--state", type=Path, required=True)
    ingest.add_argument("--output-dir", type=Path, required=True)
    ingest.add_argument("--consecutive", type=int, default=2)
    ingest.add_argument("--min-examples", type=int, default=20)

    ingest_trace = subparsers.add_parser("ingest-trace")
    ingest_trace.add_argument("--state", type=Path, required=True)
    ingest_trace.add_argument(
        "--trace-file",
        type=Path,
        action="append",
        required=True,
    )
    ingest_trace.add_argument("--step", type=int, required=True)
    ingest_trace.add_argument("--level", type=int, required=True)
    ingest_trace.add_argument("--consecutive", type=int, default=2)
    ingest_trace.add_argument("--min-examples", type=int, default=20)

    ingest_traces = subparsers.add_parser("ingest-traces")
    ingest_traces.add_argument("--state", type=Path, required=True)
    ingest_traces.add_argument(
        "--trace",
        action="append",
        required=True,
        metavar="LEVEL:PATH",
    )
    ingest_traces.add_argument("--step", type=int, required=True)
    ingest_traces.add_argument("--consecutive", type=int, default=2)
    ingest_traces.add_argument("--min-examples", type=int, default=20)

    rebase = subparsers.add_parser("rebase")
    rebase.add_argument("--source-state", type=Path, required=True)
    rebase.add_argument("--state", type=Path, required=True)
    rebase.add_argument("--advance-steps", type=int, default=0)

    render = subparsers.add_parser("render")
    render.add_argument("--state", type=Path, required=True)
    render.add_argument("--model-path", type=Path, required=True)
    render.add_argument("--output-dir", type=Path, required=True)
    render.add_argument("--target-step", type=int, required=True)
    render.add_argument("--eval-interval", type=int, default=5)
    render.add_argument("--eval-examples", type=int, default=20)
    render.add_argument("--full-finetune", action="store_true")
    render.add_argument("--learning-rate", type=float, default=1e-5)
    render.add_argument(
        "--integrated-eval",
        action="store_true",
        help="enable in-process evaluation; unsafe on the tested Qwen/vLLM stack",
    )
    render.add_argument(
        "--disable-integrated-eval",
        dest="integrated_eval",
        action="store_false",
        help="deprecated compatibility flag; train-only is already the default",
    )
    render.set_defaults(integrated_eval=False)
    render.add_argument("--batch-size", type=int, default=8)
    render.add_argument("--group-size", type=int, default=2)
    render.add_argument("--max-inflight-rollouts", type=int, default=2)
    render.add_argument("--config", type=Path, required=True)

    run = subparsers.add_parser("run")
    run.add_argument("--state", type=Path, required=True)
    run.add_argument("--prime-rl-dir", type=Path, required=True)
    run.add_argument("--model-path", type=Path, required=True)
    run.add_argument("--output-dir", type=Path, required=True)
    run.add_argument("--config", type=Path, required=True)
    run.add_argument("--max-steps", type=int, default=50)
    run.add_argument("--chunk-steps", type=int, default=5)
    run.add_argument("--eval-examples", type=int, default=20)
    run.add_argument("--consecutive", type=int, default=2)
    run.add_argument("--min-examples", type=int, default=20)
    run.add_argument("--budget-usd", type=float, default=20.0)
    run.add_argument("--price-per-hour", type=float, required=True)
    run.add_argument("--full-finetune", action="store_true")
    run.add_argument("--learning-rate", type=float, default=1e-5)
    run.add_argument(
        "--integrated-eval",
        action="store_true",
        help="enable in-process evaluation; unsafe on the tested Qwen/vLLM stack",
    )
    run.add_argument(
        "--disable-integrated-eval",
        dest="integrated_eval",
        action="store_false",
        help="deprecated compatibility flag; train-only is already the default",
    )
    run.set_defaults(integrated_eval=False)
    run.add_argument(
        "--continue-train-only",
        action="store_true",
        help=(
            "run further train-only chunks without static evaluation; "
            "the curriculum cannot transition until traces are later ingested"
        ),
    )
    run.add_argument("--batch-size", type=int, default=8)
    run.add_argument("--group-size", type=int, default=2)
    run.add_argument("--max-inflight-rollouts", type=int, default=2)
    run.add_argument("--dry-run", action="store_true")
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    if args.command == "init":
        if args.state.exists():
            raise FileExistsError(args.state)
        save_state(args.state, initial_state(args.start_stage))
        return 0
    if args.command == "assess":
        if args.min_examples <= 0:
            raise ValueError("--min-examples must be positive")
        trace_files_by_level = parse_level_trace_specs(args.trace)
        levels, stage_name, reason, gates = assess_trace_levels(
            trace_files_by_level,
            min_examples=args.min_examples,
        )
        assessment = {
            "levels": {key: asdict(metrics) for key, metrics in levels.items()},
            "gates": gates,
            "reason": reason,
        }
        if args.state is not None:
            if args.state.exists():
                raise FileExistsError(args.state)
            save_state(
                args.state,
                initial_state(
                    stage_name,
                    mode="assessment",
                    assessment=assessment,
                ),
            )
        print(
            json.dumps(
                {
                    "recommended_stage": stage_name,
                    "reason": reason,
                    **assessment,
                    "state_written": str(args.state.resolve()) if args.state else None,
                },
                indent=2,
            )
        )
        return 0
    if args.command == "ingest":
        found = ingest_evaluations(
            args.state,
            args.output_dir,
            consecutive=args.consecutive,
            min_examples=args.min_examples,
        )
        print(json.dumps([asdict(evaluation) for evaluation in found], indent=2))
        return 0
    if args.command == "ingest-trace":
        metrics, reason = ingest_trace_evaluation(
            args.state,
            args.trace_file,
            step=args.step,
            level=args.level,
            consecutive=args.consecutive,
            min_examples=args.min_examples,
        )
        print(
            json.dumps(
                {"metrics": asdict(metrics), "transition_reason": reason},
                indent=2,
            )
        )
        return 0
    if args.command == "ingest-traces":
        trace_files_by_level = parse_level_trace_specs(args.trace)
        metrics_by_level, reason = ingest_trace_levels(
            args.state,
            trace_files_by_level,
            step=args.step,
            consecutive=args.consecutive,
            min_examples=args.min_examples,
        )
        print(
            json.dumps(
                {
                    "levels": {
                        key: asdict(metrics)
                        for key, metrics in metrics_by_level.items()
                    },
                    "transition_reason": reason,
                },
                indent=2,
            )
        )
        return 0
    if args.command == "rebase":
        if args.state.exists():
            raise FileExistsError(args.state)
        source = load_state(args.source_state)
        rebased = rebase_state(
            source,
            advance_steps=args.advance_steps,
        )
        save_state(args.state, rebased)
        return 0
    if args.command == "render":
        state = load_state(args.state)
        args.config.parent.mkdir(parents=True, exist_ok=True)
        args.config.write_text(
            render_config(
                state,
                model_path=args.model_path.resolve(),
                output_dir=args.output_dir.resolve(),
                target_step=args.target_step,
                eval_interval=args.eval_interval,
                eval_examples=args.eval_examples,
                full_finetune=args.full_finetune,
                learning_rate=args.learning_rate,
                integrated_eval=args.integrated_eval,
                batch_size=args.batch_size,
                group_size=args.group_size,
                max_inflight_rollouts=args.max_inflight_rollouts,
            )
        )
        return 0
    if args.command == "run":
        return orchestrate(args)
    parser.error(f"Unknown command: {args.command}")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
