"""Merge a PEFT LoRA adapter into a standalone Qwen vision-language model."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file


def _adapter_weights(adapter: Path) -> tuple[float, dict[str, torch.Tensor]]:
    config = json.loads((adapter / "adapter_config.json").read_text())
    unsupported = [
        key for key in ("use_rslora", "use_dora", "fan_in_fan_out") if config.get(key)
    ]
    if config.get("rank_pattern") or config.get("alpha_pattern"):
        unsupported.append("rank_pattern/alpha_pattern")
    if unsupported:
        raise ValueError(f"Unsupported LoRA scaling/layout: {', '.join(unsupported)}")
    scale = float(config["lora_alpha"]) / int(config["r"])
    path = adapter / "adapter_model.safetensors"
    with safe_open(path, framework="pt", device="cpu") as handle:
        tensors = {
            key: handle.get_tensor(key)
            for key in handle.keys()  # noqa: SIM118 - safe_open is not iterable
        }
    return scale, tensors


def _base_key(adapter_key: str) -> str:
    prefix = "base_model.model."
    if adapter_key.startswith(prefix):
        adapter_key = adapter_key.removeprefix(prefix)
    elif not adapter_key.startswith("model."):
        raise ValueError(f"Unexpected adapter key: {adapter_key}")
    return adapter_key.replace(".lora_A.weight", ".weight")


def merge_checkpoint(base_model: Path, adapter: Path, output: Path) -> None:
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite existing output: {output}")
    temporary = output.with_name(output.name + ".tmp")
    if temporary.exists():
        raise FileExistsError(f"Refusing to overwrite temporary output: {temporary}")
    temporary.mkdir(parents=True)

    scale, lora = _adapter_weights(adapter)
    a_keys = {key for key in lora if key.endswith(".lora_A.weight")}
    b_keys = {key for key in lora if key.endswith(".lora_B.weight")}
    expected_b = {key.replace(".lora_A.weight", ".lora_B.weight") for key in a_keys}
    if b_keys != expected_b:
        raise ValueError("Adapter does not contain exactly paired LoRA A/B tensors")
    adapters_by_base = {
        _base_key(key): (
            value,
            lora[key.replace(".lora_A.weight", ".lora_B.weight")],
        )
        for key, value in lora.items()
        if key in a_keys
    }
    if not adapters_by_base:
        raise ValueError("Adapter contains no LoRA A/B pairs")

    merged_keys: set[str] = set()
    shards = sorted(base_model.glob("*.safetensors"))
    if not shards:
        raise FileNotFoundError(f"No safetensors shards found in {base_model}")
    for shard in shards:
        with safe_open(shard, framework="pt", device="cpu") as handle:
            metadata = handle.metadata()
            tensors = {
                key: handle.get_tensor(key)
                for key in handle.keys()  # noqa: SIM118 - safe_open is not iterable
            }
        for key, (a_weight, b_weight) in adapters_by_base.items():
            if key not in tensors:
                continue
            weight = tensors[key]
            delta = torch.matmul(b_weight.float(), a_weight.float()).mul_(scale)
            if not torch.isfinite(delta).all():
                raise ValueError(f"Non-finite LoRA delta for {key}")
            tensors[key] = weight + delta.to(dtype=weight.dtype)
            merged_keys.add(key)
        save_file(tensors, temporary / shard.name, metadata=metadata)

    missing = set(adapters_by_base) - merged_keys
    if missing:
        raise KeyError(f"Adapters did not match base tensors: {sorted(missing)[:5]}")

    for source in base_model.iterdir():
        if source.is_file() and source.suffix != ".safetensors":
            shutil.copy2(source, temporary / source.name)
    temporary.replace(output)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-model", type=Path, required=True)
    parser.add_argument("--adapter", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    merge_checkpoint(
        args.base_model.resolve(),
        args.adapter.resolve(),
        args.output.resolve(),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
