"""Native verifiers.v1 taskset for deterministic GNU Octave RL."""

from __future__ import annotations

import json
import math
import os
import re
from pathlib import Path
from typing import Any

import verifiers.v1 as vf
from generators import build_tasks
from harness import (
    CANDIDATE_RESULT_MARKER_PREFIX,
    OCTAVE_IMAGE,
    SANDBOX_CREATION_MAX_ATTEMPTS,
    SANDBOX_FINALIZE_TIMEOUT_SECONDS,
    build_candidate_runner,
    extract_code,
    format_ok,
    new_result_token,
    parse_candidate_records,
)
from openai import AsyncOpenAI
from prime_sandboxes import AsyncSandboxClient, CreateSandboxRequest

SYSTEM_PROMPT = """Write GNU Octave functions. Return exactly one fenced `octave`
code block containing the requested function. Do not return tests, files, prose, or
shell commands. The function name and signature must exactly match the prompt."""

def attempt_multiplier(
    *,
    attempts: int,
    second_attempt_multiplier: float,
    guided_attempt_multiplier: float,
) -> float:
    """Return the configured discount for the attempt that earned correctness."""
    if attempts >= 3:
        return guided_attempt_multiplier
    if attempts == 2:
        return second_attempt_multiplier
    return 1.0


class OctaveData(vf.TaskData):
    family: str
    level: int
    fn_name: str
    signature: str
    cases: list[dict[str, Any]]
    tolerance: float
    require_vectorized: bool
    reference: str


def _octave_shape(value: Any) -> list[int]:
    if not isinstance(value, list):
        return [1, 1]
    if not value:
        return [1, 0]
    if all(not isinstance(item, list) for item in value):
        return [1, len(value)]
    if not all(isinstance(item, list) for item in value):
        return []
    widths = {len(item) for item in value}
    return [len(value), widths.pop()] if len(widths) == 1 else []


def _flatten(value: Any) -> list[Any]:
    if isinstance(value, list):
        return [item for child in value for item in _flatten(child)]
    return [value]


def _as_float(value: Any) -> float:
    if value is None:
        return math.nan
    if isinstance(value, bool):
        raise TypeError("boolean outputs are not supported")
    return float(value)


def candidate_record_matches(
    record: dict[str, Any],
    *,
    expected: Any,
    tolerance: float,
) -> bool:
    """Compare one isolated candidate result with the hidden expected value."""
    if record.get("ok") is not True:
        return False
    try:
        actual_shape = [int(item) for item in record["shape"]]
        actual_values = [_as_float(item) for item in _flatten(record["values"])]
        expected_values = [_as_float(item) for item in _flatten(expected)]
    except (KeyError, TypeError, ValueError):
        return False
    if actual_shape != _octave_shape(expected) or len(actual_values) != len(expected_values):
        return False
    for actual, target in zip(actual_values, expected_values, strict=True):
        if math.isnan(target):
            if not math.isnan(actual):
                return False
        elif not math.isfinite(actual) or abs(actual - target) > tolerance * max(1.0, abs(target)):
            return False
    return True


async def execute_candidate_in_sandbox(
    client: AsyncSandboxClient,
    sandbox_id: str,
    task: OctaveData,
    source: str,
) -> dict[str, Any]:
    """Run candidate code where only public code and hidden inputs are present.

    The trusted task process retains expected outputs and computes pass/fail
    after decoding the candidate's shape/value report. This keeps both hidden
    values and score state outside the interpreter executing model code.
    """
    result_token = new_result_token()
    await client.upload_bytes(
        sandbox_id,
        f"/sandbox-workspace/task/{task.fn_name}.m",
        source.encode(),
        f"{task.fn_name}.m",
    )
    await client.upload_bytes(
        sandbox_id,
        "/sandbox-workspace/task/run_candidate.m",
        build_candidate_runner(
            task.model_dump(),
            result_token=result_token,
        ).encode(),
        "run_candidate.m",
    )
    proc = await client.execute_command(
        sandbox_id,
        "octave --no-gui --quiet run_candidate.m 2>&1",
        working_dir="/sandbox-workspace/task",
        timeout=60,
    )
    output = (proc.stdout or "") + (proc.stderr or "")
    records = parse_candidate_records(
        output,
        expected_total=len(task.cases),
        result_token=result_token,
    )
    passed = (
        sum(
            candidate_record_matches(
                record,
                expected=case["expected"],
                tolerance=task.tolerance,
            )
            for record, case in zip(records, task.cases, strict=True)
        )
        if records is not None
        else 0
    )
    total = len(task.cases)
    return {
        "passed": passed,
        "total": total,
        "fraction": passed / total if total else 0.0,
        "structured_result": float(records is not None),
        "exit_code": proc.exit_code,
        "feedback": output[-2000:],
    }


async def _execute_in_new_sandbox(
    task: OctaveData,
    source: str,
    *,
    purpose: str,
) -> dict[str, Any]:
    """Provision and always tear down one isolated candidate sandbox."""
    client = AsyncSandboxClient()
    sandbox_id: str | None = None
    try:
        sandbox = await client.create(CreateSandboxRequest(
            name=f"octave-{purpose}-{task.idx}",
            docker_image=OCTAVE_IMAGE,
            start_command="tail -f /dev/null",
            cpu_cores=1,
            memory_gb=2,
            disk_size_gb=5,
            timeout_minutes=15,
            labels=[f"octave-rl-{purpose}"],
        ))
        sandbox_id = sandbox.id
        await client.wait_for_creation(
            sandbox_id,
            max_attempts=SANDBOX_CREATION_MAX_ATTEMPTS,
        )
        await client.execute_command(sandbox_id, "mkdir -p /sandbox-workspace/task")
        return await execute_candidate_in_sandbox(client, sandbox_id, task, source)
    finally:
        try:
            if sandbox_id is not None:
                await client.delete(sandbox_id)
        finally:
            await client.aclose()


async def execute_candidate_in_new_sandbox(
    task: OctaveData,
    source: str,
) -> dict[str, Any]:
    """Execute final scoring in one short-lived candidate sandbox."""
    return await _execute_in_new_sandbox(task, source, purpose="candidate")


async def execute_feedback_in_new_sandbox(
    task: OctaveData,
    source: str,
) -> dict[str, Any]:
    """Execute retry feedback in one short-lived sandbox.

    User MCP subprocesses are terminated as process groups after the final model
    turn, so their async exit callbacks are not a reliable remote-resource
    teardown boundary.  Deleting before returning the feedback prevents the
    last reusable Sandbox from being orphaned when the framework reaches its
    turn cap.
    """
    return await _execute_in_new_sandbox(task, source, purpose="feedback")


class OctaveState(vf.State):
    attempts: int = 0
    done: bool = False


class OctaveUserConfig(vf.UserConfig):
    colocated: bool = False
    max_attempts: int = 2
    guide_enabled: bool = False
    guide_model: str = "Qwen/Qwen3.5-35B-A3B"


class OctaveUser(vf.User[OctaveUserConfig, OctaveState]):
    def __init__(self, config: OctaveUserConfig):
        super().__init__(config)
        self.task: OctaveData | None = None

    def _prime_api_key(self) -> str:
        if key := os.getenv("PRIME_API_KEY"):
            return key
        config_path = Path.home() / ".prime" / "config.json"
        if config_path.exists():
            config = json.loads(config_path.read_text())
            if key := config.get("api_key") or config.get("token"):
                return key
        raise RuntimeError(
            "Guide enabled, but no Prime credential was found in PRIME_API_KEY "
            "or ~/.prime/config.json"
        )

    async def _guide_hint(self, code: str, feedback: str) -> str:
        diagnostic = next(
            (
                line.strip()
                for line in feedback.splitlines()
                if line.strip()
                and not line.startswith("CASE ")
                and not line.startswith(CANDIDATE_RESULT_MARKER_PREFIX)
            ),
            "Some hidden cases still fail.",
        )
        client = AsyncOpenAI(
            api_key=self._prime_api_key(),
            base_url="https://api.pinference.ai/api/v1",
        )
        response = await client.chat.completions.create(
            model=self.config.guide_model,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a GNU Octave debugging guide. Give exactly one "
                        "concise hint identifying the first likely issue. Do not "
                        "write replacement code or reveal a complete solution."
                    ),
                },
                {
                    "role": "user",
                    "content": (
                        f"Task:\n{self.task.prompt}\n\n"
                        f"Candidate:\n{code}\n\n"
                        f"First diagnostic:\n{diagnostic}"
                    ),
                },
            ],
            max_tokens=96,
            reasoning_effort="none",
        )
        return (response.choices[0].message.content or "").strip()

    async def setup_task(self, task: OctaveData) -> None:
        self.task = task

    async def respond(self, message: str) -> vf.Messages:
        if self.task is None:
            raise RuntimeError("Octave task data was not provided")
        self.state.attempts += 1
        if self.config.max_attempts == 1:
            # Final scoring runs the candidate in the task runtime. A one-turn
            # evaluation has no consumer for feedback, so avoid provisioning a
            # second Octave Sandbox solely to construct a reply that cannot be
            # acted on.
            self.state.done = True
            return [{"role": "user", "content": "No retry is available."}]
        code = extract_code(message)
        if not code:
            result = {
                "passed": 0,
                "total": len(self.task.cases),
                "feedback": "Expected one fenced or bare GNU Octave function.",
            }
        else:
            result = await execute_feedback_in_new_sandbox(
                self.task,
                code,
            )
        solved = result["passed"] == result["total"]
        if solved or self.state.attempts >= self.config.max_attempts:
            self.state.done = True
        if solved:
            content = f"All {result['total']} hidden cases passed."
        else:
            content = (
                f"Hidden tests passed {result['passed']}/{result['total']} cases. "
                "Return one corrected replacement function.\n"
                + result["feedback"][-1400:]
            )
            if (
                self.config.guide_enabled
                and self.config.max_attempts >= 3
                and self.state.attempts == 2
            ):
                content += "\nGuide hint: " + await self._guide_hint(
                    code or "(no valid function submitted)", result["feedback"]
                )
        return [{"role": "user", "content": content}]


class OctaveTaskConfig(vf.TaskConfig):
    user: OctaveUserConfig = OctaveUserConfig()
    second_attempt_multiplier: float = 0.85
    guided_attempt_multiplier: float = 0.60


class OctaveTask(vf.Task[OctaveData, OctaveState, OctaveTaskConfig]):
    NEEDS_CONTAINER = False
    user = OctaveUser

    @vf.stop
    async def attempts_complete(self, trace: vf.Trace) -> bool:
        return trace.state.done

    async def _execute(self, source: str) -> dict[str, Any]:
        return await execute_candidate_in_new_sandbox(self.data, source)

    async def validate(self, runtime: vf.Runtime) -> bool:
        result = await self._execute(self.data.reference)
        return result["fraction"] == 1.0 and result["structured_result"] == 1.0

    async def finalize(self, trace: vf.Trace, runtime: vf.Runtime) -> None:
        code = extract_code(trace.last_reply or "")
        if not code:
            trace.info["octave"] = {
                "passed": 0,
                "total": len(self.data.cases),
                "fraction": 0.0,
                "structured_result": 0.0,
                "exit_code": -1,
                "feedback": "Expected exactly one fenced octave code block.",
            }
            return
        trace.info["octave"] = await self._execute(code)
        trace.info["submitted_source"] = code

    @vf.reward(weight=1.0)
    async def case_fraction(self, trace: vf.Trace) -> float:
        raw = float(trace.info["octave"]["fraction"])
        return raw * attempt_multiplier(
            attempts=max(1, trace.state.attempts),
            second_attempt_multiplier=self.config.second_attempt_multiplier,
            guided_attempt_multiplier=self.config.guided_attempt_multiplier,
        )

    @vf.metric
    async def vectorized(self, trace: vf.Trace) -> float:
        if not self.data.require_vectorized:
            return 1.0
        source = trace.info.get("submitted_source", "")
        return float(not re.search(r"(?m)^\s*(for|while)\b", source))

    @vf.metric
    async def format_ok(self, trace: vf.Trace) -> float:
        return float(format_ok(trace.last_reply or ""))

    @vf.metric
    async def raw_case_fraction(self, trace: vf.Trace) -> float:
        return float(trace.info["octave"]["fraction"])

    @vf.metric
    async def attempts_used(self, trace: vf.Trace) -> float:
        return float(max(1, trace.state.attempts))


class OctaveConfig(vf.TasksetConfig):
    task: OctaveTaskConfig = OctaveTaskConfig()
    level: int = 1
    num_tasks: int = 500
    seed: int = 0
    require_vectorized: bool = False


class OctaveTaskset(vf.Taskset[OctaveTask, OctaveConfig]):
    def load(self) -> list[OctaveTask]:
        if self.config.level not in (1, 2, 3):
            raise ValueError("level must be 1, 2, or 3")
        rows = build_tasks(
            level=self.config.level,
            num_tasks=self.config.num_tasks,
            seed=self.config.seed,
            require_vectorized=self.config.require_vectorized,
            include_reference=True,
        )
        tasks = []
        for idx, row in enumerate(rows):
            info = row["info"]
            tasks.append(OctaveTask(OctaveData(
                idx=idx,
                name=row["task"],
                prompt=row["prompt"][0]["content"],
                system_prompt=SYSTEM_PROMPT,
                timeout=vf.TaskTimeout(
                    harness=180,
                    finalize=SANDBOX_FINALIZE_TIMEOUT_SECONDS,
                    scoring=30,
                ),
                reference=row["_reference"],
                **info,
            ), self.config.task))
        return tasks


def load_environment(
    level: int = 1,
    num_tasks: int = 500,
    max_turns: int = 2,
    require_vectorized: bool = False,
    seed: int = 0,
    **kwargs: Any,
) -> OctaveTaskset:
    """Construct the native-v1 equivalent of the brief's environment factory.

    ``max_turns`` controls the persistent user's attempt budget. The rollout
    harness should use the same or a larger turn cap so the taskset, rather
    than the orchestrator, decides when a solved interaction stops.
    """
    if max_turns < 1:
        raise ValueError("max_turns must be at least 1")
    user_config = OctaveUserConfig(
        max_attempts=max_turns,
        guide_enabled=bool(kwargs.pop("guide_enabled", False)),
        guide_model=str(kwargs.pop("guide_model", "Qwen/Qwen3.5-35B-A3B")),
    )
    task_config = OctaveTaskConfig(
        user=user_config,
        second_attempt_multiplier=float(
            kwargs.pop("second_attempt_multiplier", 0.85)
        ),
        guided_attempt_multiplier=float(
            kwargs.pop("guided_attempt_multiplier", 0.60)
        ),
    )
    if kwargs:
        unknown = ", ".join(sorted(kwargs))
        raise TypeError(f"Unexpected environment option(s): {unknown}")
    return OctaveTaskset(OctaveConfig(
        task=task_config,
        level=level,
        num_tasks=num_tasks,
        seed=seed,
        require_vectorized=require_vectorized,
    ))


__all__ = ["OctaveTaskset", "load_environment"]


if __name__ == "__main__":
    OctaveUser.run()
