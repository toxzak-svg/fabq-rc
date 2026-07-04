import subprocess
import sys
from pathlib import Path
import os

from benchmark_unified_fabq import (
    estimate_mix_bpw,
    precision_mix_for_target,
)


BENCHMARK_DIR = Path(__file__).resolve().parent


def run_torch_check(source: str) -> None:
    env = os.environ.copy()
    env.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
    completed = subprocess.run(
        [sys.executable, "-c", source],
        cwd=str(BENCHMARK_DIR),
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr


def test_precision_mix_for_target_tracks_requested_budget():
    mix = precision_mix_for_target(3.0)

    assert set(mix) == {"int8", "int4", "int2", "binary"}
    assert abs(sum(mix.values()) - 1.0) < 1e-12
    assert estimate_mix_bpw(mix) <= 3.05
    assert mix["int4"] > mix["binary"]


def test_precision_mix_for_high_quality_floor_avoids_int2_and_binary():
    mix = precision_mix_for_target(4.5)

    assert mix["int2"] == 0.0
    assert mix["binary"] == 0.0
    assert estimate_mix_bpw(mix) == 4.4


def test_global_error_budget_respects_target_precision_floor():
    run_torch_check(
        """
import torch
from benchmark_unified_fabq import build_global_precision_plan

layers = [
    {"name": "model.layers.0.self_attn.v_proj", "weight": torch.ones(4, 128), "importance": torch.ones(128)},
    {"name": "model.layers.0.mlp.down_proj", "weight": torch.ones(4, 128) * 0.1, "importance": torch.ones(128)},
]
plan = build_global_precision_plan(layers, target_bpw=4.5, blocksize=128)
all_precisions = [precision for allocation in plan["allocations"].values() for precision in allocation]
assert "binary" not in all_precisions
assert "int2" not in all_precisions
"""
    )


def test_weighted_mse_uses_imatrix_input_importance():
    run_torch_check(
        """
import torch
from benchmark_unified_fabq import weighted_mse
weight = torch.zeros(2, 3)
recon = torch.tensor([[1.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
importance = torch.tensor([10.0, 1.0, 1.0])
assert weighted_mse(weight, recon, importance) > weighted_mse(weight, recon, torch.ones(3))
"""
    )


def test_allocate_precision_by_damage_promotes_high_damage_rows():
    run_torch_check(
        """
import torch
from benchmark_unified_fabq import allocate_precision_by_damage, precision_mix_for_target
weight = torch.tensor([
    [0.01, -0.01, 0.01, -0.01],
    [4.0, -3.0, 2.0, -1.0],
    [0.02, 0.02, -0.02, -0.02],
    [0.03, -0.03, 0.03, -0.03],
])
allocation = allocate_precision_by_damage(weight, torch.ones(4), precision_mix_for_target(3.0))
assert allocation[1] in {"int8", "int4"}
assert allocation.count("binary") >= 1
"""
    )


def test_quantize_symmetric_two_bit_is_more_accurate_than_binary():
    run_torch_check(
        """
import torch
from benchmark_unified_fabq import quantize_symmetric, weighted_mse
weight = torch.tensor([[0.1, -0.4, 0.9, -1.7]])
importance = torch.ones(4)
binary = quantize_symmetric(weight, bits=1, input_importance=importance, blocksize=4)
int2 = quantize_symmetric(weight, bits=2, input_importance=importance, blocksize=4)
assert weighted_mse(weight, int2, importance) < weighted_mse(weight, binary, importance)
"""
    )


def test_block_residual_correction_reduces_binary_error():
    run_torch_check(
        """
import torch
from benchmark_unified_fabq import apply_block_residual_correction, quantize_symmetric, weighted_mse
weight = torch.tensor([[0.2, 0.4, 1.2, -2.4]])
importance = torch.ones(4)
recon = quantize_symmetric(weight, bits=1, input_importance=importance, blocksize=4)
corrected = apply_block_residual_correction(weight, recon, blocksize=4)
assert weighted_mse(weight, corrected, importance) < weighted_mse(weight, recon, importance)
"""
    )


def test_global_error_budget_keeps_legacy_storage_budget():
    run_torch_check(
        """
import torch
try:
    from benchmark_unified_fabq import build_global_precision_plan
except ImportError as exc:
    raise AssertionError("global precision planner is not implemented") from exc

layers = [
    {"name": "model.layers.0.self_attn.v_proj", "weight": torch.ones(4, 128), "importance": torch.ones(128)},
    {"name": "model.layers.0.mlp.down_proj", "weight": torch.ones(4, 128) * 0.1, "importance": torch.ones(128)},
]
plan = build_global_precision_plan(layers, target_bpw=3.0, blocksize=128)
assert plan["estimated_bpw"] <= plan["legacy_estimated_bpw"]
assert plan["budget_bits"] <= plan["legacy_budget_bits"]
"""
    )


def test_global_error_budget_promotes_high_damage_rows_across_layers():
    run_torch_check(
        """
import torch
try:
    from benchmark_unified_fabq import build_global_precision_plan
except ImportError as exc:
    raise AssertionError("global precision planner is not implemented") from exc

layers = [
    {"name": "model.layers.0.mlp.down_proj", "weight": torch.tensor([[0.01] * 128, [0.02] * 128, [0.03] * 128, [0.04] * 128]), "importance": torch.ones(128)},
    {"name": "model.layers.1.mlp.down_proj", "weight": torch.tensor([[9.0] * 128, [0.02] * 128, [0.03] * 128, [0.04] * 128]), "importance": torch.ones(128)},
]
plan = build_global_precision_plan(layers, target_bpw=2.05, blocksize=128)
high_damage_precision = plan["allocations"]["model.layers.1.mlp.down_proj"][0]
low_damage_precisions = plan["allocations"]["model.layers.0.mlp.down_proj"]
assert high_damage_precision in {"int2", "int4", "int8"}
assert low_damage_precisions.count("binary") >= 1
"""
    )


def test_module_precision_floor_prevents_binary_for_fragile_layers():
    run_torch_check(
        """
import torch
try:
    from benchmark_unified_fabq import build_global_precision_plan, precision_floor_for_layer
except ImportError as exc:
    raise AssertionError("global precision planner floors are not implemented") from exc

assert precision_floor_for_layer("model.layers.4.self_attn.v_proj") == "int2"
assert precision_floor_for_layer("model.language_model.layers.22.linear_attn.in_proj_a") == "int8"

layers = [
    {"name": "model.layers.4.self_attn.v_proj", "weight": torch.ones(4, 128) * 0.01, "importance": torch.ones(128)},
    {"name": "model.layers.4.mlp.down_proj", "weight": torch.ones(4, 128) * 0.01, "importance": torch.ones(128)},
]
plan = build_global_precision_plan(layers, target_bpw=2.05, blocksize=128)
assert "binary" not in plan["allocations"]["model.layers.4.self_attn.v_proj"]
"""
    )


def test_global_error_budget_relaxes_floors_to_preserve_budget_when_needed():
    run_torch_check(
        """
import torch
from benchmark_unified_fabq import build_global_precision_plan

layers = [
    {"name": "model.layers.0.self_attn.q_proj", "weight": torch.ones(4, 128), "importance": torch.ones(128)},
    {"name": "model.layers.1.self_attn.k_proj", "weight": torch.ones(4, 128), "importance": torch.ones(128)},
]
plan = build_global_precision_plan(layers, target_bpw=2.05, blocksize=128)
assert plan["estimated_bpw"] <= plan["legacy_estimated_bpw"]
assert plan["floor_relaxed_rows"] > 0
"""
    )


def test_apply_unified_dequantized_uses_global_error_budget_plan():
    run_torch_check(
        """
import torch
from benchmark_unified_fabq import apply_unified_dequantized

class SelfAttn(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.v_proj = torch.nn.Linear(128, 4, bias=False)

class Mlp(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.down_proj = torch.nn.Linear(128, 4, bias=False)

class Block(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.self_attn = SelfAttn()
        self.mlp = Mlp()

class ToyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = torch.nn.ModuleList([Block()])

model = ToyModel()
with torch.no_grad():
    model.layers[0].self_attn.v_proj.weight.fill_(0.01)
    model.layers[0].mlp.down_proj.weight.fill_(0.01)

result = apply_unified_dequantized(
    model,
    {
        "layers.0.self_attn.v_proj": torch.ones(128),
        "layers.0.mlp.down_proj": torch.ones(128),
    },
    target_bpw=2.05,
    blocksize=128,
)

assert result["allocation_strategy"] == "global_error_budget_with_module_floors"
assert result["estimated_bpw"] <= result["legacy_estimated_bpw"]
v_proj = next(layer for layer in result["layers"] if layer["name"] == "layers.0.self_attn.v_proj")
assert v_proj["precision_counts"]["binary"] == 0
"""
    )
