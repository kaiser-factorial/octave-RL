"""Native verifiers.v1 taskset for deterministic GNU Octave RL."""

from __future__ import annotations

import json
import logging
import os
import re
from pathlib import Path
from typing import Any, Literal

import verifiers.v1 as vf
from executors import execute_candidate_locally
from generators import build_tasks
from harness import (
    CANDIDATE_RESULT_MARKER_PREFIX,
    OCTAVE_IMAGE,
    SANDBOX_CREATION_MAX_ATTEMPTS,
    SANDBOX_FINALIZE_TIMEOUT_SECONDS,
    build_candidate_runner,
    candidate_record_matches,
    extract_code,
    format_ok,
    new_result_token,
    score_candidate_output,
)
from openai import AsyncOpenAI
from prime_sandboxes import AsyncSandboxClient, CreateSandboxRequest

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """Write GNU Octave functions. Return exactly one fenced `octave`
code block containing the requested function. Do not return tests, files, prose, or
shell commands. The function name and signature must exactly match the prompt."""

# Where candidate code runs. "prime" provisions one short-lived Sandbox per
# execution; "local" runs it on the calling host under the bounds in
# ``executors``. Scoring, rewards, and the hidden-value boundary are identical.
OctaveRuntime = Literal["prime", "local"]


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
    record = score_candidate_output(
        output,
        cases=task.cases,
        tolerance=task.tolerance,
        result_token=result_token,
        exit_code=proc.exit_code,
    )
    record["runtime"] = "prime"
    # The installed CPU Sandbox request model does not serialize a network
    # policy, so egress denial cannot be claimed here either. Sandbox
    # containment comes from separate hardware, not from a network namespace.
    record["network_isolated"] = False
    return record


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


async def execute_candidate(
    task: OctaveData,
    source: str,
    *,
    runtime: OctaveRuntime,
    purpose: str,
) -> dict[str, Any]:
    """Route one candidate execution to the configured runtime.

    Both branches produce the same record and enforce the same reward-relevant
    boundary; they differ in where the interpreter runs. See ``executors`` for
    what the local branch does and does not contain.
    """
    if runtime == "local":
        return await execute_candidate_locally(task, source)
    return await _execute_in_new_sandbox(task, source, purpose=purpose)


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
    # Set when the guide was asked for a hint and could not deliver one. The
    # rollout continues without the hint rather than dying: a missing hint is a
    # degraded retry, not an invalid one.
    guide_unavailable: str = ""


class OctaveUserConfig(vf.UserConfig):
    colocated: bool = False
    max_attempts: int = 2
    guide_enabled: bool = False
    guide_model: str = "Qwen/Qwen3.5-35B-A3B"
    # Name a transposed result explicitly in the retry feedback. Off by default:
    # it changes what the model is shown, so a run that enables it is not
    # comparable with one that does not, and WS3's arms are single-turn anyway.
    orientation_hint_enabled: bool = False
    # NOT `runtime`: vf.UserConfig already defines `runtime: RuntimeConfig` for
    # where the simulator process runs. Shadowing it with a string makes
    # serve_user call .model_dump() on "local" and every rollout dies.
    octave_runtime: OctaveRuntime = "prime"


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

    def _orientation_hint(self, result: dict[str, Any]) -> str:
        """Name a transposed result, which the raw diagnostic cannot express.

        A transposed answer is the one failure where the model computed
        everything correctly and only mismatched Octave's orientation
        convention. Nothing it is currently shown reveals that: the transport it
        sees carries its *own* shapes, never the expected ones, so the same
        orientation is typically resubmitted on every remaining attempt and
        three attempts are spent for no signal.

        This does disclose the expected *shape* -- more than the existing
        pass/fail count, though still not any expected value. That is why it is
        opt-in: it is a deliberate trade of a little information for a usable
        retry, and it must be off for any run being used as a benchmark.
        """
        if not self.config.orientation_hint_enabled:
            return ""
        if not result.get("transposed"):
            return ""
        # Only speak up when transposition explains every failing case;
        # otherwise something else is wrong too and this would misdirect.
        if result["transposed"] != result["total"] - result["passed"]:
            return ""
        return (
            "Orientation: your values are correct but transposed -- the "
            "expected result has the rows and columns the other way round. "
            "Check the orientation the prompt asks you to preserve.\n"
        )

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
            result = await execute_candidate(
                self.task,
                code,
                runtime=self.config.octave_runtime,
                purpose="feedback",
            )
        solved = result["passed"] == result["total"]
        orientation = self._orientation_hint(result)
        if solved or self.state.attempts >= self.config.max_attempts:
            self.state.done = True
        if solved:
            content = f"All {result['total']} hidden cases passed."
        else:
            content = (
                f"Hidden tests passed {result['passed']}/{result['total']} cases. "
                "Return one corrected replacement function.\n"
                + orientation
                + result["feedback"][-1400:]
            )
            if (
                self.config.guide_enabled
                and self.config.max_attempts >= 3
                and self.state.attempts == 2
            ):
                # A guide failure must not kill the rollout. It used to: the
                # exception escaped `respond`, and the MCP layer turned it into
                # a contentless tool result that the host reported as
                # `JSONDecodeError('Expecting value: line 1 column 1')` --
                # naming neither the cause nor this file. The common trigger is
                # a missing credential, because `PRIME_API_KEY` is *not*
                # inherited by the user-server subprocess; see the 2026-08-09
                # PIPELINE_LOG entry. Degrade to an unguided retry and record
                # why, so a misconfigured run is visible instead of silently
                # losing every third turn.
                try:
                    hint = await self._guide_hint(
                        code or "(no valid function submitted)", result["feedback"]
                    )
                except Exception as exc:  # noqa: BLE001 - any guide failure degrades
                    self.state.guide_unavailable = f"{type(exc).__name__}: {exc}"
                    logger.warning(
                        "Guide hint unavailable (%s); continuing without it. "
                        "If this is a credential error, note that PRIME_API_KEY "
                        "is not inherited by the user-server subprocess -- write "
                        "it to ~/.prime/config.json instead.",
                        self.state.guide_unavailable,
                    )
                else:
                    if hint:
                        content += "\nGuide hint: " + hint
        return [{"role": "user", "content": content}]


class OctaveTaskConfig(vf.TaskConfig):
    user: OctaveUserConfig = OctaveUserConfig()
    second_attempt_multiplier: float = 0.85
    guided_attempt_multiplier: float = 0.60
    octave_runtime: OctaveRuntime = "prime"


class OctaveTask(vf.Task[OctaveData, OctaveState, OctaveTaskConfig]):
    NEEDS_CONTAINER = False
    user = OctaveUser

    @vf.stop
    async def attempts_complete(self, trace: vf.Trace) -> bool:
        return trace.state.done

    async def _execute(self, source: str) -> dict[str, Any]:
        return await execute_candidate(
            self.data,
            source,
            runtime=self.config.octave_runtime,
            purpose="candidate",
        )

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
    async def execution_fraction(self, trace: vf.Trace) -> float:
        """Share of hidden cases whose candidate call ran without throwing.

        Read together with ``correct_given_executed``: this one says whether the
        model wrote runnable Octave at all, which is a different competency from
        writing *correct* Octave and is the one that fails far more often.
        """
        return float(trace.info["octave"].get("execution_fraction", 0.0))

    @vf.metric
    async def correct_given_executed(self, trace: vf.Trace) -> float:
        """Correct fraction among the cases that actually ran.

        This is the closest thing the environment has to an algorithmic-accuracy
        signal, because it stops charging the model for code that never
        executed. It is 0.0 when nothing ran, so it is only interpretable
        alongside ``execution_fraction`` -- a rollout at (0.0, 0.0) failed to
        run, while (1.0, 0.0) ran cleanly and got every answer wrong.
        """
        record = trace.info["octave"]
        executed = record.get("executed", 0)
        return float(record["passed"] / executed) if executed else 0.0

    @vf.metric
    async def transposed_fraction(self, trace: vf.Trace) -> float:
        """Share of cases answered with exactly the transpose of the expected value.

        This is the model getting the computation right and the orientation
        convention wrong. Reported separately because the prompts explicitly
        require preserving input orientation, so it is a real failure -- but it
        is a categorically different failure from a wrong algorithm, and
        collapsing the two hides which competency an arm actually improved.
        """
        return float(trace.info["octave"].get("transposed_fraction", 0.0))

    @vf.metric
    async def attempts_used(self, trace: vf.Trace) -> float:
        return float(max(1, trace.state.attempts))


class OctaveConfig(vf.TasksetConfig):
    task: OctaveTaskConfig = OctaveTaskConfig()
    level: int = 1
    num_tasks: int = 500
    seed: int = 0
    require_vectorized: bool = False
    # Restricts the pool to these families. `None` means all ten. A pool's
    # prompts are determined by (family, level), so two pools drawn with
    # different seeds share every prompt; holding out a family is the only way
    # to obtain a genuinely held-out *problem*. See `DEFAULT_HELDOUT_FAMILIES`.
    families: list[str] | None = None


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
            families=self.config.families,
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
    families: list[str] | None = None,
    **kwargs: Any,
) -> OctaveTaskset:
    """Construct the native-v1 equivalent of the brief's environment factory.

    ``max_turns`` controls the persistent user's attempt budget. The rollout
    harness should use the same or a larger turn cap so the taskset, rather
    than the orchestrator, decides when a solved interaction stops.

    ``families`` restricts the pool. Because a pool's prompts are determined by
    (family, level), a different ``seed`` holds out hidden inputs but not
    questions; excluding families is what produces a held-out *problem*.
    """
    if max_turns < 1:
        raise ValueError("max_turns must be at least 1")
    runtime: OctaveRuntime = str(kwargs.pop("octave_runtime", "prime"))  # type: ignore[assignment]
    if runtime not in ("prime", "local"):
        raise ValueError("octave_runtime must be 'prime' or 'local'")
    user_config = OctaveUserConfig(
        max_attempts=max_turns,
        guide_enabled=bool(kwargs.pop("guide_enabled", False)),
        guide_model=str(kwargs.pop("guide_model", "Qwen/Qwen3.5-35B-A3B")),
        orientation_hint_enabled=bool(kwargs.pop("orientation_hint_enabled", False)),
        octave_runtime=runtime,
    )
    task_config = OctaveTaskConfig(
        user=user_config,
        second_attempt_multiplier=float(
            kwargs.pop("second_attempt_multiplier", 0.85)
        ),
        guided_attempt_multiplier=float(
            kwargs.pop("guided_attempt_multiplier", 0.60)
        ),
        octave_runtime=runtime,
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
        families=families,
    ))


__all__ = [
    "OctaveTaskset",
    # Re-exported: the comparison itself moved to ``harness`` so both runtimes
    # share one scorer, but it stays reachable here for callers and tests.
    "candidate_record_matches",
    "execute_candidate",
    "load_environment",
]


if __name__ == "__main__":
    OctaveUser.run()
