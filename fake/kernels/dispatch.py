from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn

from fake.kernels.modules import (
    Conv1x1AsLinear,
    NVFP4Linear,
    NVFP4SemiSparseLinear,
    SemiSparseLinear,
)
from fake.kernels.pack import k_aligned, required_k_multiple
from fake.kernels.registry import DENSE_FALLBACK_METHODS, get_entry


def materialize(
    model: nn.Module,
    metadata: dict[str, Any],
    masks: dict[str, Any] | None = None,
) -> nn.Module:
    """Replace compressible modules with kernel-backed equivalents.

    For each nn.Linear / pointwise Conv2d in the model:
    - Checks K-dimension alignment for the target kernel.
    - Replaces with the appropriate *Linear module.
    - Falls back to dense (leaves module unchanged) when K is misaligned or
      mask data is missing.

    Writes kernel_path, kernel_name, and kernel_fallback_layers into metadata.
    masks: {module_name: {"mask": bool_tensor}} as produced by compress_model().
    """
    method = metadata.get("method", "")

    if method in DENSE_FALLBACK_METHODS:
        metadata["kernel_path"] = "dense_fallback"
        return model

    entry = get_entry(method)
    if entry is None:
        metadata["kernel_path"] = "dense_fallback"
        return model

    fallback_layers: list[str] = []
    for name, module in list(model.named_modules()):
        cols = _in_cols(module)
        if cols is None:
            continue
        required = required_k_multiple(method)
        if cols % required != 0:
            fallback_layers.append(f"{name}:k_misaligned({cols},need={required})")
            continue
        mask = _get_mask(masks, name)
        replacement = _make_replacement(method, module, mask)
        if replacement is None:
            fallback_layers.append(f"{name}:no_mask")
            continue
        _set_submodule(model, name, replacement)

    metadata["kernel_path"] = entry.operation
    metadata["kernel_name"] = entry.kernels
    metadata["kernel_fallback_layers"] = fallback_layers

    if fallback_layers:
        print(f"[materialize] {len(fallback_layers)} layers fell back to dense (K not aligned or missing mask):")
        for name in fallback_layers[:5]:
            print(f"  - {name}")
        if len(fallback_layers) > 5:
            print(f"  ... and {len(fallback_layers) - 5} more")

    matched = sum(1 for m in model.modules() if type(m).__name__ in {
        "NVFP4Linear", "SemiSparseLinear", "NVFP4SemiSparseLinear", "Conv1x1AsLinear"
    })
    print(f"[materialize] kernel modules: {matched} matched, {len(fallback_layers)} fallback")

    return model


# ── helpers ───────────────────────────────────────────────────────────────────

def _in_cols(module: nn.Module) -> int | None:
    if isinstance(module, nn.Linear):
        return module.in_features
    if (
        isinstance(module, nn.Conv2d)
        and tuple(module.kernel_size) == (1, 1)
        and module.groups == 1
    ):
        return module.in_channels
    return None


def _get_mask(masks: dict | None, name: str) -> torch.Tensor | None:
    if not masks:
        return None
    entry = masks.get(name, {})
    return entry.get("mask") if isinstance(entry, dict) else None


def _make_replacement(method: str, module: nn.Module, mask) -> nn.Module | None:
    is_conv = isinstance(module, nn.Conv2d)

    if method == "nvfp4":
        if is_conv:
            return Conv1x1AsLinear.from_conv(module, NVFP4Linear)
        return NVFP4Linear.from_linear(module)

    if method == "semi_structured_sparse":
        if mask is None:
            return None
        if is_conv:
            return Conv1x1AsLinear.from_conv(module, SemiSparseLinear, mask=mask)
        return SemiSparseLinear.from_linear(module, mask)

    if method == "nvfp4_semi_structured_sparse":
        if mask is None:
            return None
        if is_conv:
            return Conv1x1AsLinear.from_conv(module, NVFP4SemiSparseLinear, mask=mask)
        return NVFP4SemiSparseLinear.from_linear(module, mask)

    return None


def _set_submodule(root: nn.Module, name: str, new_module: nn.Module) -> None:
    parts = name.split(".")
    parent = root
    for part in parts[:-1]:
        parent = getattr(parent, part)
    setattr(parent, parts[-1], new_module)