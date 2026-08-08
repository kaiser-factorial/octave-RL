"""Categorise algorithmic failures with a judge model, offline.

This is deliberately **not** a reward. It runs after the fact over retained
traces so its noise and cost cannot reach training, and there is nothing for a
policy to hack. Putting a judge in the reward would make the signal
non-stationary and gameable, which matters especially for WS3, where the
measurand is how routing shifts under an objective.

It looks at one population: rollouts whose candidate **executed cleanly on every
hidden case and still got every answer wrong**. Those are the only rollouts
where the model demonstrably wrote runnable Octave and the algorithm was the
thing that failed. Roughly 12% of baseline rollouts land here; the ~59% that
throw on every case are excluded, because asking a judge whether a program that
never ran has the right algorithm measures the judge's charity, not the model.

Usage:
    uv run python scripts/classify_failures.py --root artifacts/baseline-eval-20260808
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
from collections import Counter
from pathlib import Path
from typing import Any

from openai import AsyncOpenAI

JUDGE_MODEL = "Qwen/Qwen3.5-35B-A3B"
BASE_URL = "https://api.pinference.ai/api/v1"
TRANSPORT_RE = re.compile(r"__OCTAVE_CANDIDATE_RESULT__\w+ (\[.*\])")

# Pre-registered so the judge picks from a fixed set rather than inventing
# categories that cannot be compared across runs.
CATEGORIES = {
    "misread_spec": "solves a different problem than the prompt asked for",
    "wrong_algorithm": "right problem, fundamentally wrong approach",
    "indexing_or_off_by_one": "right approach, indexing or boundary error",
    "orientation_or_shape": "right values, wrong orientation or shape convention",
    "numerical_or_tolerance": "right approach, precision or tolerance problem",
    "other": "none of the above",
}

SYSTEM = (
    "You classify why a GNU Octave function produced wrong answers. The code "
    "ran without errors on every test case, so this is not a syntax or runtime "
    "problem -- the logic is wrong. Reply with strict JSON only: "
    '{"category": "<one of the allowed categories>", "reason": "<one sentence>"}. '
    "Allowed categories: " + ", ".join(f"{k} ({v})" for k, v in CATEGORIES.items())
)


def prime_api_key() -> str:
    if key := os.getenv("PRIME_API_KEY"):
        return key
    config = Path.home() / ".prime" / "config.json"
    if config.exists():
        data = json.loads(config.read_text())
        if key := data.get("api_key") or data.get("token"):
            return key
    raise RuntimeError("no Prime credential in PRIME_API_KEY or ~/.prime/config.json")


def ran_cleanly_but_wrong(root: Path) -> list[dict[str, Any]]:
    """Select rollouts that executed every case and passed none."""
    selected = []
    for path in sorted(root.glob("*/traces.jsonl")):
        if "ctlA" in path.parent.name:
            continue  # thinking-on cell is degenerate: it rarely emits a function
        for line in path.open():
            row = json.loads(line)
            metrics = row.get("metrics") or {}
            if metrics.get("raw_case_fraction") != 0.0:
                continue
            octave = (row.get("info") or {}).get("octave") or {}
            match = TRANSPORT_RE.search(octave.get("feedback", ""))
            if not match:
                continue
            try:
                records = json.loads(match.group(1))
            except json.JSONDecodeError:
                continue
            if not records or not all(r.get("ok") is True for r in records):
                continue
            data = (row.get("task") or {}).get("data") or {}
            source = (row.get("info") or {}).get("submitted_source")
            if not source or not data.get("prompt"):
                continue
            selected.append(
                {
                    "cell": path.parent.name,
                    "task": data.get("name"),
                    "family": data.get("family"),
                    "level": data.get("level"),
                    "prompt": data["prompt"],
                    "source": source,
                }
            )
    return selected


async def classify(client: AsyncOpenAI, item: dict[str, Any]) -> dict[str, Any]:
    response = await client.chat.completions.create(
        model=JUDGE_MODEL,
        messages=[
            {"role": "system", "content": SYSTEM},
            {
                "role": "user",
                "content": (
                    f"Task:\n{item['prompt']}\n\n"
                    f"Submitted function (ran without error, every answer wrong):\n"
                    f"{item['source']}"
                ),
            },
        ],
        max_tokens=200,
        reasoning_effort="none",
    )
    text = (response.choices[0].message.content or "").strip()
    match = re.search(r"\{.*\}", text, re.DOTALL)
    verdict = {"category": "unparsed", "reason": text[:160]}
    if match:
        try:
            parsed = json.loads(match.group(0))
            category = parsed.get("category", "other")
            verdict = {
                "category": category if category in CATEGORIES else "other",
                "reason": str(parsed.get("reason", ""))[:200],
            }
        except json.JSONDecodeError:
            pass
    return {**{k: v for k, v in item.items() if k != "prompt"}, **verdict}


async def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--concurrency", type=int, default=4)
    args = parser.parse_args()

    items = ran_cleanly_but_wrong(args.root)
    print(f"rollouts that ran cleanly and got everything wrong: {len(items)}")
    if not items:
        return 0

    client = AsyncOpenAI(api_key=prime_api_key(), base_url=BASE_URL)
    semaphore = asyncio.Semaphore(args.concurrency)

    async def guarded(item):
        async with semaphore:
            try:
                return await classify(client, item)
            except Exception as error:  # noqa: BLE001 - one bad call must not sink the pass
                return {**{k: v for k, v in item.items() if k != "prompt"},
                        "category": "judge_error", "reason": str(error)[:160]}

    verdicts = await asyncio.gather(*(guarded(i) for i in items))

    counts = Counter(v["category"] for v in verdicts)
    print("\ncategory                    n   share")
    for category, n in counts.most_common():
        print(f"  {category:<24}{n:>3}{n / len(verdicts):>8.1%}")
    print("\nby task family:")
    for family, n in Counter(v["family"] for v in verdicts).most_common():
        print(f"  {family:<24}{n:>3}")

    print("\nexamples:")
    seen = set()
    for v in verdicts:
        if v["category"] in seen:
            continue
        seen.add(v["category"])
        print(f"  [{v['category']}] {v['task']}: {v['reason']}")

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(
            {"judge_model": JUDGE_MODEL, "n": len(verdicts),
             "counts": dict(counts), "verdicts": verdicts}, indent=2) + "\n")
        print(f"\nreport: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
