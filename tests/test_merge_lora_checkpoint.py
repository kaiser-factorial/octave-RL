import importlib.util
import json
import sys
from pathlib import Path

import pytest
import torch
from safetensors.torch import load_file, save_file

MODULE_PATH = Path(__file__).parents[1] / "scripts" / "merge_lora_checkpoint.py"
SPEC = importlib.util.spec_from_file_location("merge_lora_checkpoint", MODULE_PATH)
assert SPEC and SPEC.loader
merge_lora = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = merge_lora
SPEC.loader.exec_module(merge_lora)


@pytest.mark.parametrize(
    "adapter_prefix",
    ["model.", "base_model.model.model."],
)
def test_merge_lora_checkpoint_applies_scaled_delta(tmp_path, adapter_prefix):
    base = tmp_path / "base"
    adapter = tmp_path / "adapter"
    output = tmp_path / "merged"
    base.mkdir()
    adapter.mkdir()

    base_key = "model.layers.0.self_attn.q_proj.weight"
    original = torch.arange(12, dtype=torch.float32).reshape(3, 4)
    save_file({base_key: original}, base / "model.safetensors")
    (base / "config.json").write_text('{"model_type": "qwen3_5"}\n')

    a_weight = torch.tensor([[1.0, 0.0, -1.0, 2.0], [0.5, 1.0, 0.0, -0.5]])
    b_weight = torch.tensor([[1.0, 0.0], [0.0, 2.0], [-1.0, 1.0]])
    stem = f"{adapter_prefix}layers.0.self_attn.q_proj"
    save_file(
        {
            f"{stem}.lora_A.weight": a_weight,
            f"{stem}.lora_B.weight": b_weight,
        },
        adapter / "adapter_model.safetensors",
    )
    (adapter / "adapter_config.json").write_text(json.dumps({"lora_alpha": 4, "r": 2}))

    merge_lora.merge_checkpoint(base, adapter, output)

    expected = original + 2.0 * (b_weight @ a_weight)
    assert torch.equal(load_file(output / "model.safetensors")[base_key], expected)
    assert (output / "config.json").read_text() == '{"model_type": "qwen3_5"}\n'


def test_merge_refuses_to_overwrite_nonempty_output(tmp_path):
    output = tmp_path / "merged"
    output.mkdir()
    (output / "keep.txt").write_text("user data")
    with pytest.raises(FileExistsError):
        merge_lora.merge_checkpoint(tmp_path / "base", tmp_path / "adapter", output)


def test_merge_rejects_unsupported_lora_scaling(tmp_path):
    adapter = tmp_path / "adapter"
    adapter.mkdir()
    (adapter / "adapter_config.json").write_text(
        json.dumps({"lora_alpha": 4, "r": 2, "use_rslora": True})
    )
    with pytest.raises(ValueError, match="use_rslora"):
        merge_lora._adapter_weights(adapter)
