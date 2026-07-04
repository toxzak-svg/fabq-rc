#!/usr/bin/env python3
"""Unified FABQ-VP/EBQ dense-dequantized validation benchmark.

This is a CPU-friendly prototype of the unified spec:
- forward-only imatrix calibration for input-feature importance
- variable precision allocation across int8/int4/int2/binary rows
- residual block correction for int2/binary rows
- dense dequantized weights for perplexity and generation validation

It is not a native compressed-kernel throughput benchmark.
"""

from __future__ import annotations

import argparse
import heapq
import json
import math
import os
import platform
import sys
import time
from pathlib import Path

import psutil

sys.path.insert(0, str(Path(__file__).resolve().parent))
from benchmark_fabq_runtime import ascii_preview, is_target_linear_name  # noqa: E402
from benchmark_quality import (  # noqa: E402
    aggregate_quality,
    format_quality_report,
    load_baseline_summary,
    load_quality_tasks,
    quality_gate,
    score_multiple_choice_task,
)
from benchmark_qwen35_runtime import (  # noqa: E402
    DEFAULT_PROMPT,
    auto_model_kind,
    load_eval_text,
    run_forward_throughput,
    run_generation,
    run_perplexity,
)


BIT_WIDTHS = {"int8": 8, "int4": 4, "int2": 2, "binary": 1}
PRECISION_ORDER = ("binary", "int2", "int4", "int8")
PRECISION_RANK = {name: rank for rank, name in enumerate(PRECISION_ORDER)}
PRECISION_ERROR_WEIGHT = {"binary": 1.0, "int2": 0.45, "int4": 0.12, "int8": 0.02}
INT8_FLOOR_PATTERNS = (
    ".linear_attn.in_proj_a",
    ".linear_attn.in_proj_b",
)
INT2_FLOOR_PATTERNS = (
    ".self_attn.q_proj",
    ".self_attn.k_proj",
    ".self_attn.v_proj",
    ".self_attn.o_proj",
    ".linear_attn.out_proj",
    ".linear_attn.in_proj_qkv",
    ".linear_attn.in_proj_z",
    ".mlp.gate_proj",
    ".mlp.up_proj",
)


def rss_gb() -> float:
    return psutil.Process(os.getpid()).memory_info().rss / 1e9


def count_parameters(model) -> int:
    return sum(p.numel() for p in model.parameters())


def dtype_parameter_counts(model) -> dict[str, int]:
    counts: dict[str, int] = {}
    for p in model.parameters():
        key = str(p.dtype)
        counts[key] = counts.get(key, 0) + p.numel()
    return counts


def load_model_low_memory(model_cls, config, repo_id: str, torch):
    """Load a model, using a manual .bin path when Transformers dispatch exits."""
    repo_path = Path(repo_id)
    bin_path = repo_path / "pytorch_model.bin"
    if repo_path.exists() and bin_path.exists():
        dtype = getattr(config, "torch_dtype", None) or torch.bfloat16
        model = model_cls.from_config(config, trust_remote_code=True, dtype=dtype)
        state = torch.load(bin_path, map_location="cpu", weights_only=True)
        model.load_state_dict(state, strict=True)
        del state
        return model
    return model_cls.from_pretrained(
        repo_id,
        dtype="auto",
        trust_remote_code=True,
        low_cpu_mem_usage=True,
    )


def precision_mix_for_target(target_bpw: float) -> dict[str, float]:
    """Return conservative row fractions for a requested unified-spec budget."""
    if target_bpw <= 2.05:
        return {"int8": 0.02, "int4": 0.18, "int2": 0.30, "binary": 0.50}
    if target_bpw <= 3.05:
        return {"int8": 0.03, "int4": 0.49, "int2": 0.24, "binary": 0.24}
    if target_bpw >= 4.5:
        return {"int8": 0.10, "int4": 0.90, "int2": 0.0, "binary": 0.0}
    return {"int8": 0.05, "int4": 0.85, "int2": 0.10, "binary": 0.0}


def estimate_mix_bpw(mix: dict[str, float]) -> float:
    return sum(BIT_WIDTHS[name] * frac for name, frac in mix.items())


def precision_floor_for_layer(name: str) -> str:
    """Return the minimum precision allowed for fragile module types."""
    if any(pattern in name for pattern in INT8_FLOOR_PATTERNS):
        return "int8"
    if any(pattern in name for pattern in INT2_FLOOR_PATTERNS):
        return "int2"
    return "binary"


def _next_precision(precision: str) -> str | None:
    rank = PRECISION_RANK[precision] + 1
    if rank >= len(PRECISION_ORDER):
        return None
    return PRECISION_ORDER[rank]


def _previous_precision(precision: str) -> str | None:
    rank = PRECISION_RANK[precision] - 1
    if rank < 0:
        return None
    return PRECISION_ORDER[rank]


def _max_precision(left: str, right: str) -> str:
    return left if PRECISION_RANK[left] >= PRECISION_RANK[right] else right


def _precision_storage_bits_for_row(in_features: int, precision: str, blocksize: int) -> int:
    width = BIT_WIDTHS[precision]
    n_blocks = math.ceil(in_features / blocksize)
    bits = in_features * width
    scale_blocks = n_blocks if width <= 4 else 1
    bits += scale_blocks * 16
    if width <= 2:
        bits += n_blocks * 16
    return bits


def _allocation_storage_bits(in_features: int, allocation: list[str], blocksize: int) -> int:
    bits = len(allocation) * 16
    for precision in allocation:
        bits += _precision_storage_bits_for_row(in_features, precision, blocksize)
    return bits


def _precision_counts_from_mix(n_rows: int, mix: dict[str, float]) -> dict[str, int]:
    counts = {name: int(round(mix.get(name, 0.0) * n_rows)) for name in BIT_WIDTHS}
    delta = n_rows - sum(counts.values())
    counts["binary"] = max(0, counts.get("binary", 0) + delta)
    excess = sum(counts.values()) - n_rows
    for name in ("binary", "int2", "int4", "int8"):
        if excess <= 0:
            break
        take = min(counts.get(name, 0), excess)
        counts[name] -= take
        excess -= take
    return counts


def minimum_precision_for_mix(mix: dict[str, float]) -> str:
    for precision in PRECISION_ORDER:
        if mix.get(precision, 0.0) > 0.0:
            return precision
    return "int8"


def _allocate_precision_by_damage_scores(row_damage: list[float], mix: dict[str, float]) -> list[str]:
    n_rows = len(row_damage)
    counts = _precision_counts_from_mix(n_rows, mix)
    order = sorted(range(n_rows), key=lambda idx: row_damage[idx], reverse=True)
    allocation = ["binary"] * n_rows
    cursor = 0
    for name in ("int8", "int4", "int2", "binary"):
        for row in order[cursor : cursor + counts.get(name, 0)]:
            allocation[row] = name
        cursor += counts.get(name, 0)
    return allocation


def _normalized_importance(input_importance):
    import torch

    imp = input_importance.detach().to(torch.float32).cpu().flatten()
    if imp.numel() == 0:
        return torch.ones(1, dtype=torch.float32)
    imp = torch.nan_to_num(imp, nan=0.0, posinf=0.0, neginf=0.0).clamp_min(0.0)
    mean = imp.mean().clamp_min(1e-8)
    return imp / mean


def weighted_mse(weight, recon, input_importance) -> float:
    import torch

    imp = _normalized_importance(input_importance).to(weight.device)
    while imp.numel() < weight.shape[1]:
        imp = torch.cat([imp, imp.new_ones(weight.shape[1] - imp.numel())])
    imp = imp[: weight.shape[1]]
    diff = (weight.to(torch.float32) - recon.to(torch.float32)) ** 2
    return float((diff * imp.view(1, -1)).mean().detach().cpu())


def row_damage_scores(weight, input_importance) -> list[float]:
    import torch

    w = weight.detach().to(torch.float32).cpu()
    imp = _normalized_importance(input_importance)
    if imp.numel() < w.shape[1]:
        imp = torch.cat([imp, imp.new_ones(w.shape[1] - imp.numel())])
    imp = imp[: w.shape[1]]
    return (w * w * imp.view(1, -1)).mean(dim=1).tolist()


def quantize_symmetric(weight, bits: int, input_importance=None, blocksize: int = 128):
    import torch

    w = weight.detach().to(torch.float32).cpu()
    recon = torch.empty_like(w)
    if input_importance is None:
        imp = torch.ones(w.shape[1], dtype=torch.float32)
    else:
        imp = _normalized_importance(input_importance)
        if imp.numel() < w.shape[1]:
            imp = torch.cat([imp, imp.new_ones(w.shape[1] - imp.numel())])
        imp = imp[: w.shape[1]]

    for start in range(0, w.shape[1], blocksize):
        end = min(start + blocksize, w.shape[1])
        block = w[:, start:end]
        block_imp = imp[start:end].view(1, -1)
        if bits == 1:
            denom = block_imp.sum(dim=1, keepdim=True).clamp_min(1e-8)
            scale = (block.abs() * block_imp).sum(dim=1, keepdim=True) / denom
            recon[:, start:end] = torch.where(block >= 0, scale, -scale)
            continue

        qmin = -(2 ** (bits - 1))
        qmax = 2 ** (bits - 1) - 1
        denom = max(abs(qmin), abs(qmax))
        scale = (block.abs().amax(dim=1, keepdim=True) / max(denom, 1)).clamp_min(1e-8)
        q = torch.clamp(torch.round(block / scale), qmin, qmax)
        recon[:, start:end] = q * scale
    return recon


def apply_block_residual_correction(weight, recon, blocksize: int = 128):
    import torch

    w = weight.detach().to(torch.float32).cpu()
    out = recon.detach().to(torch.float32).cpu().clone()
    for start in range(0, w.shape[1], blocksize):
        end = min(start + blocksize, w.shape[1])
        residual = w[:, start:end] - out[:, start:end]
        out[:, start:end] += residual.mean(dim=1, keepdim=True)
    return out


def allocate_precision_by_damage(weight, input_importance, mix: dict[str, float]) -> list[str]:
    return _allocate_precision_by_damage_scores(row_damage_scores(weight, input_importance), mix)


def build_global_precision_plan(layers: list[dict], target_bpw: float, blocksize: int) -> dict:
    """Allocate precision globally while staying within the legacy fixed-mix budget."""
    mix = precision_mix_for_target(target_bpw)
    normalized = []
    legacy_budget_bits = 0
    storage_bits = 0
    total_weights = 0
    precision_hist = {name: 0 for name in BIT_WIDTHS}
    heap = []
    tie_breaker = 0
    floor_relaxed_rows = 0
    target_floor = minimum_precision_for_mix(mix)

    for layer_idx, layer in enumerate(layers):
        name = layer["name"]
        weight = layer.get("weight")
        if weight is not None:
            out_features = int(weight.shape[0])
            in_features = int(weight.shape[1])
            importance = layer.get("importance")
            if importance is None:
                import torch

                importance = torch.ones(in_features, dtype=torch.float32)
            damage = row_damage_scores(weight, importance)
        else:
            out_features = int(layer["out_features"])
            in_features = int(layer["in_features"])
            damage = [float(value) for value in layer["row_damage"]]

        if len(damage) != out_features:
            raise ValueError(f"row_damage length mismatch for {name}")

        legacy_allocation = _allocate_precision_by_damage_scores(damage, mix)
        floor = _max_precision(precision_floor_for_layer(name), target_floor)
        allocation = [floor] * out_features
        layer_bits = _allocation_storage_bits(in_features, allocation, blocksize)
        legacy_bits = _allocation_storage_bits(in_features, legacy_allocation, blocksize)
        storage_bits += layer_bits
        legacy_budget_bits += legacy_bits
        total_weights += out_features * in_features
        normalized.append(
            {
                "name": name,
                "out_features": out_features,
                "in_features": in_features,
                "damage": damage,
                "allocation": allocation,
            }
        )

        for row_idx, precision in enumerate(allocation):
            precision_hist[precision] += 1
    relax_heap = []
    for layer_idx, layer in enumerate(normalized):
        for row_idx, precision in enumerate(layer["allocation"]):
            previous_precision = _previous_precision(precision)
            if previous_precision is None or PRECISION_RANK[previous_precision] < PRECISION_RANK[target_floor]:
                continue
            current_bits = _precision_storage_bits_for_row(layer["in_features"], precision, blocksize)
            previous_bits = _precision_storage_bits_for_row(layer["in_features"], previous_precision, blocksize)
            saved_bits = current_bits - previous_bits
            if saved_bits <= 0:
                continue
            damage_cost = max(0.0, layer["damage"][row_idx]) * (
                PRECISION_ERROR_WEIGHT[previous_precision] - PRECISION_ERROR_WEIGHT[precision]
            )
            efficiency_cost = damage_cost / max(saved_bits, 1)
            heapq.heappush(
                relax_heap,
                (efficiency_cost, damage_cost, tie_breaker, layer_idx, row_idx, previous_precision, saved_bits),
            )
            tie_breaker += 1

    while storage_bits > legacy_budget_bits and relax_heap:
        _, _, _, layer_idx, row_idx, previous_precision, saved_bits = heapq.heappop(relax_heap)
        layer = normalized[layer_idx]
        current_precision = layer["allocation"][row_idx]
        if _previous_precision(current_precision) != previous_precision:
            continue

        layer["allocation"][row_idx] = previous_precision
        precision_hist[current_precision] -= 1
        precision_hist[previous_precision] += 1
        storage_bits -= saved_bits
        floor_relaxed_rows += 1

        next_previous = _previous_precision(previous_precision)
        if next_previous is None or PRECISION_RANK[next_previous] < PRECISION_RANK[target_floor]:
            continue
        current_bits = _precision_storage_bits_for_row(layer["in_features"], previous_precision, blocksize)
        previous_bits = _precision_storage_bits_for_row(layer["in_features"], next_previous, blocksize)
        next_saved_bits = current_bits - previous_bits
        if next_saved_bits <= 0:
            continue
        damage_cost = max(0.0, layer["damage"][row_idx]) * (
            PRECISION_ERROR_WEIGHT[next_previous] - PRECISION_ERROR_WEIGHT[previous_precision]
        )
        efficiency_cost = damage_cost / max(next_saved_bits, 1)
        heapq.heappush(
            relax_heap,
            (efficiency_cost, damage_cost, tie_breaker, layer_idx, row_idx, next_previous, next_saved_bits),
        )
        tie_breaker += 1

    for layer_idx, layer in enumerate(normalized):
        for row_idx, precision in enumerate(layer["allocation"]):
            next_precision = _next_precision(precision)
            if next_precision is None:
                continue
            current_bits = _precision_storage_bits_for_row(layer["in_features"], precision, blocksize)
            next_bits = _precision_storage_bits_for_row(layer["in_features"], next_precision, blocksize)
            delta_bits = next_bits - current_bits
            benefit = max(0.0, layer["damage"][row_idx]) * (
                PRECISION_ERROR_WEIGHT[precision] - PRECISION_ERROR_WEIGHT[next_precision]
            )
            efficiency = benefit / max(delta_bits, 1)
            heapq.heappush(heap, (-efficiency, -benefit, tie_breaker, layer_idx, row_idx, next_precision, delta_bits))
            tie_breaker += 1

    while heap:
        _, _, _, layer_idx, row_idx, next_precision, delta_bits = heapq.heappop(heap)
        layer = normalized[layer_idx]
        current_precision = layer["allocation"][row_idx]
        if _next_precision(current_precision) != next_precision:
            continue
        if delta_bits > 0 and storage_bits + delta_bits > legacy_budget_bits:
            continue

        layer["allocation"][row_idx] = next_precision
        precision_hist[current_precision] -= 1
        precision_hist[next_precision] += 1
        storage_bits += delta_bits

        following = _next_precision(next_precision)
        if following is None:
            continue
        current_bits = _precision_storage_bits_for_row(layer["in_features"], next_precision, blocksize)
        following_bits = _precision_storage_bits_for_row(layer["in_features"], following, blocksize)
        next_delta = following_bits - current_bits
        benefit = max(0.0, layer["damage"][row_idx]) * (
            PRECISION_ERROR_WEIGHT[next_precision] - PRECISION_ERROR_WEIGHT[following]
        )
        efficiency = benefit / max(next_delta, 1)
        heapq.heappush(heap, (-efficiency, -benefit, tie_breaker, layer_idx, row_idx, following, next_delta))
        tie_breaker += 1

    allocations = {layer["name"]: layer["allocation"] for layer in normalized}
    return {
        "strategy": "global_error_budget_with_module_floors",
        "target_bpw": target_bpw,
        "mix": mix,
        "mix_nominal_bpw": estimate_mix_bpw(mix),
        "target_precision_floor": target_floor,
        "allocations": allocations,
        "storage_bits": storage_bits,
        "budget_bits": legacy_budget_bits,
        "legacy_budget_bits": legacy_budget_bits,
        "estimated_bpw": storage_bits / max(total_weights, 1),
        "legacy_estimated_bpw": legacy_budget_bits / max(total_weights, 1),
        "precision_histogram": precision_hist,
        "floor_relaxed_rows": floor_relaxed_rows,
    }


def storage_bits_for_layer(out_features: int, in_features: int, allocation: list[str], blocksize: int) -> int:
    if len(allocation) != out_features:
        raise ValueError("allocation length must match out_features")
    return _allocation_storage_bits(in_features, allocation, blocksize)


def collect_imatrix(model, tokenizer, text: str, max_tokens: int, block_size: int, max_batches: int) -> dict:
    import torch

    sums: dict[str, torch.Tensor] = {}
    counts: dict[str, int] = {}
    hooks = []

    for name, module in model.named_modules():
        if not isinstance(module, torch.nn.Linear):
            continue
        if not is_target_linear_name(name):
            continue

        def _hook(mod, inputs, output, layer_name=name):
            if not inputs:
                return
            x = inputs[0].detach().to(torch.float32).cpu()
            if x.shape[-1] != mod.in_features:
                return
            flat = x.reshape(-1, x.shape[-1])
            val = (flat * flat).sum(dim=0)
            if layer_name not in sums:
                sums[layer_name] = val
                counts[layer_name] = flat.shape[0]
            else:
                sums[layer_name] += val
                counts[layer_name] += flat.shape[0]

        hooks.append(module.register_forward_hook(_hook))

    encoded = tokenizer(text, return_tensors="pt", add_special_tokens=False)
    input_ids = encoded["input_ids"][:, :max_tokens]
    with torch.inference_mode():
        seen = 0
        for start in range(0, input_ids.size(1), block_size):
            if seen >= max_batches:
                break
            chunk = input_ids[:, start : min(start + block_size, input_ids.size(1))]
            if chunk.numel() == 0:
                continue
            _ = model(input_ids=chunk)
            seen += 1

    for h in hooks:
        h.remove()

    return {name: sums[name] / max(counts.get(name, 1), 1) for name in sums}


def _quantize_rows(weight, input_importance, allocation: list[str], blocksize: int):
    import torch

    w = weight.detach().to(torch.float32).cpu()
    recon = torch.empty_like(w)
    for name, bits in BIT_WIDTHS.items():
        rows = torch.tensor([i for i, p in enumerate(allocation) if p == name], dtype=torch.long)
        if rows.numel() == 0:
            continue
        q = quantize_symmetric(w[rows], bits, input_importance, blocksize)
        if bits <= 2:
            q = apply_block_residual_correction(w[rows], q, blocksize)
        recon[rows] = q
    return recon


def apply_unified_dequantized(
    model,
    imatrix: dict,
    target_bpw: float,
    blocksize: int,
    max_layers: int = 0,
) -> dict:
    import torch

    mix = precision_mix_for_target(target_bpw)
    started = time.perf_counter()
    target_layers = []
    layers = []

    with torch.no_grad():
        for name, module in model.named_modules():
            if not isinstance(module, torch.nn.Linear):
                continue
            if not is_target_linear_name(name):
                continue
            if max_layers and len(target_layers) >= max_layers:
                break

            w = module.weight.detach().to(torch.float32).cpu()
            imp = imatrix.get(name)
            if imp is None:
                imp = torch.ones(w.shape[1], dtype=torch.float32)
            target_layers.append(
                {
                    "name": name,
                    "module": module,
                    "out_features": w.shape[0],
                    "in_features": w.shape[1],
                    "row_damage": row_damage_scores(w, imp),
                    "importance": imp,
                }
            )

    plan_inputs = [
        {
            "name": layer["name"],
            "out_features": layer["out_features"],
            "in_features": layer["in_features"],
            "row_damage": layer["row_damage"],
        }
        for layer in target_layers
    ]
    plan = build_global_precision_plan(plan_inputs, target_bpw=target_bpw, blocksize=blocksize)

    precision_hist = {name: 0 for name in BIT_WIDTHS}
    with torch.no_grad():
        for layer in target_layers:
            name = layer["name"]
            module = layer["module"]
            w = module.weight.detach().to(torch.float32).cpu()
            imp = layer["importance"]
            allocation = plan["allocations"][name]
            recon = _quantize_rows(w, imp, allocation, blocksize)
            module.weight.data.copy_(recon.to(device=module.weight.device, dtype=module.weight.dtype))

            diff = w - recon
            sse = float((diff.double() * diff.double()).sum())
            signal = float((w.double() * w.double()).sum())
            wmse = weighted_mse(w, recon, imp)
            layer_bits = storage_bits_for_layer(w.shape[0], w.shape[1], allocation, blocksize)
            for p in allocation:
                precision_hist[p] += 1
            layers.append(
                {
                    "name": name,
                    "out_features": w.shape[0],
                    "in_features": w.shape[1],
                    "weights": w.numel(),
                    "sse": sse,
                    "signal": signal,
                    "weighted_mse": wmse,
                    "storage_bits": layer_bits,
                    "precision_floor": precision_floor_for_layer(name),
                    "precision_counts": {p: allocation.count(p) for p in BIT_WIDTHS},
                }
            )

    total_weights = sum(layer["weights"] for layer in layers)
    total_sse = sum(layer["sse"] for layer in layers)
    total_signal = sum(layer["signal"] for layer in layers)
    total_bits = sum(layer["storage_bits"] for layer in layers)
    mse = total_sse / max(total_weights, 1)
    signal_mse = total_signal / max(total_weights, 1)
    return {
        "method": "unified_fabq_vp_ebq_dequantized",
        "method_note": (
            "Forward-only imatrix calibration scores row damage. A global error-budget "
            "allocator with module precision floors spends the legacy fixed-mix storage "
            "budget across all target layers. int2/binary rows receive block residual "
            "mean correction. Weights are dequantized back to dense tensors for CPU "
            "validation."
        ),
        "target_bpw": target_bpw,
        "mix": mix,
        "mix_nominal_bpw": estimate_mix_bpw(mix),
        "allocation_strategy": plan["strategy"],
        "budget_bits": plan["budget_bits"],
        "legacy_budget_bits": plan["legacy_budget_bits"],
        "legacy_estimated_bpw": plan["legacy_estimated_bpw"],
        "floor_relaxed_rows": plan["floor_relaxed_rows"],
        "blocksize": blocksize,
        "layers_quantized": len(layers),
        "target_weights": total_weights,
        "mse": mse,
        "sqnr_db": 10.0 * math.log10(max(signal_mse, 1e-30) / max(mse, 1e-30)),
        "estimated_bpw": total_bits / max(total_weights, 1),
        "precision_histogram": precision_hist,
        "elapsed_sec": time.perf_counter() - started,
        "layers": layers,
    }


def main() -> int:
    import torch
    from transformers import AutoConfig, AutoModelForCausalLM, AutoModelForImageTextToText, AutoTokenizer

    ap = argparse.ArgumentParser()
    ap.add_argument("--repo-id", default="Qwen/Qwen3-0.6B")
    ap.add_argument("--hf-home", default=str(Path.cwd() / ".hf_cache"))
    ap.add_argument("--out", default="results/qwen3_06b_unified_fabq_benchmark.json")
    ap.add_argument("--max-eval-tokens", type=int, default=256)
    ap.add_argument("--block-size", type=int, default=128)
    ap.add_argument("--max-new-tokens", type=int, default=24)
    ap.add_argument("--forward-repeats", type=int, default=3)
    ap.add_argument("--dataset-name", default="wikitext")
    ap.add_argument("--dataset-config", default="wikitext-2-raw-v1")
    ap.add_argument("--dataset-split", default="test")
    ap.add_argument("--dataset-max-chars", type=int, default=20000)
    ap.add_argument("--prompt", default=DEFAULT_PROMPT)
    ap.add_argument("--target-bpw", type=float, default=3.0)
    ap.add_argument("--imatrix-tokens", type=int, default=256)
    ap.add_argument("--imatrix-batches", type=int, default=2)
    ap.add_argument("--quant-blocksize", type=int, default=128)
    ap.add_argument("--max-layers", type=int, default=0)
    ap.add_argument("--run-quality", action="store_true")
    ap.add_argument("--quality-tasks")
    ap.add_argument("--max-quality-tasks", type=int, default=0)
    ap.add_argument("--baseline-quality-json")
    ap.add_argument("--max-accuracy-drop", type=float, default=0.05)
    ap.add_argument("--min-quality-tasks", type=int, default=3)
    args = ap.parse_args()

    os.environ.setdefault("HF_HOME", args.hf_home)
    torch.set_num_threads(max(1, min(8, os.cpu_count() or 1)))

    started = time.perf_counter()
    rss_start = rss_gb()
    config = AutoConfig.from_pretrained(args.repo_id, trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained(args.repo_id, trust_remote_code=True)
    model_kind = auto_model_kind(getattr(config, "model_type", None), getattr(config, "architectures", None))
    model_cls = AutoModelForImageTextToText if model_kind == "image_text_to_text" else AutoModelForCausalLM

    load_t0 = time.perf_counter()
    model = load_model_low_memory(model_cls, config, args.repo_id, torch)
    model.eval()
    load_sec = time.perf_counter() - load_t0
    rss_loaded = rss_gb()

    eval_text, dataset_id = load_eval_text(
        args.dataset_max_chars,
        args.dataset_name,
        args.dataset_config,
        args.dataset_split,
    )
    imatrix_t0 = time.perf_counter()
    imatrix = collect_imatrix(model, tokenizer, eval_text, args.imatrix_tokens, args.block_size, args.imatrix_batches)
    imatrix_sec = time.perf_counter() - imatrix_t0
    rss_imatrix = rss_gb()

    quantization = apply_unified_dequantized(
        model,
        imatrix,
        target_bpw=args.target_bpw,
        blocksize=args.quant_blocksize,
        max_layers=args.max_layers,
    )
    rss_quantized = rss_gb()

    validation = {"loaded": True, "imatrix_layers": len(imatrix), "quantized": quantization["layers_quantized"] > 0, "can_forward": False, "can_generate": False}
    ppl = run_perplexity(model, tokenizer, eval_text, args.max_eval_tokens, args.block_size)
    validation["can_forward"] = True
    forward = run_forward_throughput(model, tokenizer, args.prompt, args.forward_repeats)
    generation = run_generation(model, tokenizer, args.prompt, args.max_new_tokens)
    validation["can_generate"] = generation["new_tokens"] > 0
    quality = None
    gate = None
    quality_records = None
    if args.run_quality:
        quality_tasks = load_quality_tasks(args.quality_tasks, args.max_quality_tasks)
        quality_records = [score_multiple_choice_task(model, tokenizer, task) for task in quality_tasks]
        quality = aggregate_quality(quality_records)
        baseline_quality = load_baseline_summary(args.baseline_quality_json)
        gate = quality_gate(
            quality,
            baseline_quality,
            max_accuracy_drop=args.max_accuracy_drop,
            min_tasks=args.min_quality_tasks,
        )

    result = {
        "repo_id": args.repo_id,
        "benchmark_kind": "unified_fabq_vp_ebq_dequantized_runtime_validation",
        "architecture": getattr(config, "architectures", None),
        "model_type": getattr(config, "model_type", None),
        "auto_model_kind": model_kind,
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "torch": torch.__version__,
            "cuda_available": torch.cuda.is_available(),
            "torch_threads": torch.get_num_threads(),
        },
        "validation": validation,
        "parameter_count": count_parameters(model),
        "dtype_parameter_counts": dtype_parameter_counts(model),
        "rss_gb": {
            "start": rss_start,
            "after_load": rss_loaded,
            "after_imatrix": rss_imatrix,
            "after_unified_quantize": rss_quantized,
            "after_benchmark": rss_gb(),
        },
        "load_sec": load_sec,
        "imatrix_sec": imatrix_sec,
        "dataset": dataset_id,
        "quantization": quantization,
        "perplexity": ppl,
        "forward_throughput": forward,
        "generation": generation,
        "elapsed_sec": time.perf_counter() - started,
    }
    if args.run_quality:
        result["quality"] = quality
        result["quality_gate"] = gate
        result["quality_tasks"] = quality_records

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    printable = {
        "repo_id": result["repo_id"],
        "validation": validation,
        "quantization": {k: v for k, v in quantization.items() if k != "layers"},
        "perplexity": ppl,
        "forward_throughput": forward,
        "generation": {k: v for k, v in generation.items() if k != "output"},
        "rss_gb": result["rss_gb"],
    }
    if args.run_quality:
        printable["quality"] = quality
        printable["quality_gate"] = gate
    print(json.dumps(printable, indent=2))
    if args.run_quality:
        print(format_quality_report(quality, gate, perplexity=ppl))
    print(f"Output preview: {ascii_preview(generation['output'])}")
    print(f"Wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
