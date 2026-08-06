#!/usr/bin/env python3
"""Validate every generated reference solution on the pinned Prime runtime."""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "environments" / "octave_rl"))

from generators import build_tasks
from harness import (
    OCTAVE_IMAGE,
    SANDBOX_CREATION_MAX_ATTEMPTS,
    build_harness,
    new_result_token,
    parse_harness_result,
)
from prime_sandboxes import AsyncSandboxClient, CreateSandboxRequest


async def validate_level(
    client: AsyncSandboxClient, level: int, tasks_per_level: int, seed: int
) -> dict[str, object]:
    sandbox = await client.create(
        CreateSandboxRequest(
            name=f"octave-reference-l{level}-{int(time.time())}",
            docker_image=OCTAVE_IMAGE,
            start_command="tail -f /dev/null",
            cpu_cores=1,
            memory_gb=2,
            disk_size_gb=5,
            timeout_minutes=60,
            labels=["octave-rl-reference-validation"],
        )
    )
    failures: list[dict[str, object]] = []
    family_counts: Counter[str] = Counter()
    started = time.monotonic()
    try:
        await client.wait_for_creation(
            sandbox.id,
            max_attempts=SANDBOX_CREATION_MAX_ATTEMPTS,
        )
        await client.execute_command(sandbox.id, "mkdir -p /sandbox-workspace/task")
        tasks = build_tasks(level, tasks_per_level, seed, False, True)
        for index, task in enumerate(tasks, 1):
            info = task["info"]
            family_counts[info["family"]] += 1
            await client.upload_bytes(
                sandbox.id,
                f"/sandbox-workspace/task/{info['fn_name']}.m",
                task["_reference"].encode(),
                f"{info['fn_name']}.m",
            )
            await client.upload_bytes(
                sandbox.id,
                "/sandbox-workspace/task/run_cases.m",
                build_harness(info, result_token := new_result_token()).encode(),
                "run_cases.m",
            )
            proc = await client.execute_command(
                sandbox.id,
                "octave --no-gui --quiet run_cases.m 2>&1",
                working_dir="/sandbox-workspace/task",
                timeout=60,
            )
            output = (proc.stdout or "") + (proc.stderr or "")
            parsed = parse_harness_result(
                output,
                expected_total=len(info["cases"]),
                result_token=result_token,
            )
            passed, total = parsed or (0, len(info["cases"]))
            if passed != total:
                failures.append(
                    {
                        "task": task["task"],
                        "family": info["family"],
                        "passed": passed,
                        "total": total,
                        "output": output[-2000:],
                    }
                )
            if index % 50 == 0:
                print(
                    f"level {level}: {index}/{tasks_per_level}, "
                    f"failures={len(failures)}",
                    flush=True,
                )
    finally:
        await client.delete(sandbox.id)
    return {
        "level": level,
        "tasks": tasks_per_level,
        "cases": tasks_per_level * 6,
        "seed": seed,
        "family_counts": dict(sorted(family_counts.items())),
        "failures": failures,
        "elapsed_seconds": round(time.monotonic() - started, 3),
    }


async def run(args: argparse.Namespace) -> dict[str, object]:
    client = AsyncSandboxClient()
    levels = await asyncio.gather(
        *(
            validate_level(client, level, args.tasks_per_level, args.seed)
            for level in args.levels
        )
    )
    return {
        "image": OCTAVE_IMAGE,
        "levels": levels,
        "total_tasks": sum(int(level["tasks"]) for level in levels),
        "total_cases": sum(int(level["cases"]) for level in levels),
        "total_failures": sum(len(level["failures"]) for level in levels),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--levels", nargs="+", type=int, default=[1, 2, 3])
    parser.add_argument("--tasks-per-level", type=int, default=500)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--output", type=Path, default=Path("artifacts/reference_validation.json")
    )
    args = parser.parse_args()
    payload = asyncio.run(run(args))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload, indent=2))
    raise SystemExit(1 if payload["total_failures"] else 0)


if __name__ == "__main__":
    main()
