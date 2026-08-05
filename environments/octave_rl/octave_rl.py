"""Native verifiers.v1 taskset for deterministic GNU Octave RL."""

from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any

import verifiers.v1 as vf
from generators import build_tasks
from harness import OCTAVE_IMAGE, RESULT_RE, build_harness, extract_code, format_ok
from openai import AsyncOpenAI
from prime_sandboxes import AsyncSandboxClient, CreateSandboxRequest

SYSTEM_PROMPT = """Write GNU Octave functions. Return exactly one fenced `octave`
code block containing the requested function. Do not return tests, files, prose, or
shell commands. The function name and signature must exactly match the prompt."""


class OctaveData(vf.TaskData):
    family: str
    level: int
    fn_name: str
    signature: str
    cases: list[dict[str, Any]]
    tolerance: float
    require_vectorized: bool
    reference: str


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
        self.client: AsyncSandboxClient | None = None
        self.sandbox_id: str | None = None

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
                and not line.startswith("RESULT ")
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
        self.client = AsyncSandboxClient()
        sandbox = await self.client.create(CreateSandboxRequest(
            name=f"octave-feedback-{task.idx}",
            docker_image=OCTAVE_IMAGE,
            start_command="tail -f /dev/null",
            cpu_cores=1,
            memory_gb=2,
            disk_size_gb=5,
            timeout_minutes=15,
            labels=["octave-rl-feedback"],
        ))
        self.sandbox_id = sandbox.id
        await self.client.wait_for_creation(self.sandbox_id)
        await self.client.execute_command(
            self.sandbox_id, "mkdir -p /sandbox-workspace/task"
        )
        self._exit_stack.push_async_callback(self.client.delete, self.sandbox_id)

    async def respond(self, message: str) -> vf.Messages:
        if self.task is None:
            raise RuntimeError("Octave task data was not provided")
        code = extract_code(message)
        self.state.attempts += 1
        if not code:
            result = {
                "passed": 0,
                "total": len(self.task.cases),
                "feedback": "Expected one fenced or bare GNU Octave function.",
            }
        else:
            if self.client is None or self.sandbox_id is None:
                raise RuntimeError("feedback sandbox was not created")
            await self.client.upload_bytes(
                self.sandbox_id,
                f"/sandbox-workspace/task/{self.task.fn_name}.m",
                code.encode(),
                f"{self.task.fn_name}.m",
            )
            await self.client.upload_bytes(
                self.sandbox_id,
                "/sandbox-workspace/task/run_cases.m",
                build_harness(self.task.model_dump()).encode(),
                "run_cases.m",
            )
            proc = await self.client.execute_command(
                self.sandbox_id,
                "octave --no-gui --quiet run_cases.m 2>&1",
                working_dir="/sandbox-workspace/task",
                timeout=60,
            )
            feedback = (proc.stdout or "") + (proc.stderr or "")
            match = RESULT_RE.search(feedback)
            passed, total = (
                tuple(map(int, match.groups()))
                if match
                else (0, len(self.task.cases))
            )
            result = {"passed": passed, "total": total, "feedback": feedback}
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
    NEEDS_CONTAINER = True
    user = OctaveUser

    @vf.stop
    async def attempts_complete(self, trace: vf.Trace) -> bool:
        return trace.state.done

    async def _execute(self, source: str, runtime: vf.Runtime) -> dict[str, Any]:
        fn_file = f"{self.data.fn_name}.m"
        await runtime.write(fn_file, source.encode())
        await runtime.write("run_cases.m", build_harness(self.data.model_dump()).encode())
        result = await runtime.run(
            ["sh", "-c", "octave --no-gui --quiet run_cases.m 2>&1"], {}
        )
        output = result.stdout + result.stderr
        match = RESULT_RE.search(output)
        passed, total = (
            tuple(map(int, match.groups()))
            if match
            else (0, len(self.data.cases))
        )
        manifest = await runtime.run(
            ["sh", "-c", "find . -maxdepth 1 -type f -printf '%f\\n' | sort"], {}
        )
        allowed = {fn_file, "run_cases.m"}
        return {
            "passed": passed,
            "total": total,
            "fraction": passed / total if total else 0.0,
            "ran": float(match is not None),
            "tampering": float(bool(set(manifest.stdout.splitlines()) - allowed)),
            "exit_code": result.exit_code,
            "feedback": output[-2000:],
        }

    async def validate(self, runtime: vf.Runtime) -> bool:
        result = await self._execute(self.data.reference, runtime)
        return result["fraction"] == 1.0 and result["exit_code"] == 0

    async def finalize(self, trace: vf.Trace, runtime: vf.Runtime) -> None:
        code = extract_code(trace.last_reply or "")
        if not code:
            trace.info["octave"] = {
                "passed": 0,
                "total": len(self.data.cases),
                "fraction": 0.0,
                "ran": 0.0,
                "tampering": 0.0,
                "exit_code": -1,
                "feedback": "Expected exactly one fenced octave code block.",
            }
            return
        trace.info["octave"] = await self._execute(code, runtime)
        trace.info["submitted_source"] = code

    @vf.reward(weight=1.0)
    async def case_fraction(self, trace: vf.Trace) -> float:
        raw = float(trace.info["octave"]["fraction"])
        attempts = max(1, trace.state.attempts)
        if attempts >= 3:
            return raw * self.config.guided_attempt_multiplier
        if attempts == 2:
            return raw * self.config.second_attempt_multiplier
        return raw

    @vf.reward(weight=0.1)
    async def runs_without_error(self, trace: vf.Trace) -> float:
        return float(trace.info["octave"]["ran"])

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
    async def tampering_detected(self, trace: vf.Trace) -> float:
        return float(trace.info["octave"]["tampering"])

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
                image=OCTAVE_IMAGE,
                workdir="/sandbox-workspace/task",
                resources=vf.TaskResources(cpu=1, memory=2, disk=5),
                timeout=vf.TaskTimeout(harness=180, finalize=90, scoring=30),
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
