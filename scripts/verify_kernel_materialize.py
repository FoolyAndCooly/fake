#!/usr/bin/env python
"""Verify that materialize() actually replaces modules with kernel versions."""
from __future__ import annotations

import torch
import torch.nn as nn

from fake.kernels import kernel_info
from fake.kernels.dispatch import materialize


def count_kernel_modules(model: nn.Module) -> dict[str, int]:
    """Count how many kernel-backed modules are in the model."""
    counts = {
        "NVFP4Linear": 0,
        "SemiSparseLinear": 0,
        "NVFP4SemiSparseLinear": 0,
        "Conv1x1AsLinear": 0,
    }
    for module in model.modules():
        name = type(module).__name__
        if name in counts:
            counts[name] += 1
    return counts


if __name__ == "__main__":
    if not torch.cuda.is_available():
        print("❌ CUDA not available")
        exit(1)

    print("=== Kernel Extension Status ===")
    info = kernel_info()
    print(f"CUTLASS_ROOT: {info['cutlass_root'] or '(not set)'}")
    print(f"Extension available: {info['extension_available']}")
    if not info['extension_available']:
        print(f"Load error: {info['load_error']}")
        print("\n⚠️  Extension not available — materialize will skip all replacements")
    print()

    # Create a toy model with some Linear layers
    print("=== Creating toy model ===")
    model = nn.Sequential(
        nn.Linear(64, 128),
        nn.ReLU(),
        nn.Linear(128, 256),
        nn.ReLU(),
        nn.Linear(256, 10),
    ).cuda()

    print(f"Before materialize: {sum(1 for _ in model.modules() if isinstance(_, nn.Linear))} Linear layers")
    before_counts = count_kernel_modules(model)
    print(f"Kernel modules before: {before_counts}")
    print()

    # Materialize with NVFP4 method
    print("=== Running materialize(method='nvfp4') ===")
    metadata = {"method": "nvfp4"}
    model = materialize(model, metadata, masks=None)
    print()

    after_counts = count_kernel_modules(model)
    print(f"After materialize: {sum(1 for _ in model.modules() if isinstance(_, nn.Linear))} Linear layers")
    print(f"Kernel modules after: {after_counts}")
    print(f"Metadata: {metadata}")
    print()

    if after_counts["NVFP4Linear"] > 0:
        print(f"✅ {after_counts['NVFP4Linear']} layers replaced with NVFP4Linear")
    else:
        print("❌ No layers were replaced — check K-dimension alignment or extension availability")
