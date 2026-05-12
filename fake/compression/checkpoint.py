from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
import torch.nn as nn


def load_checkpoint_into_model(
    model: nn.Module,
    checkpoint_path: str | Path | None,
    strict: bool = True,
) -> dict[str, Any]:
    if checkpoint_path is None:
        return {}
    payload = torch.load(Path(checkpoint_path), map_location="cpu")
    if isinstance(payload, dict) and "state_dict" in payload:
        state_dict = payload["state_dict"]
        metadata = payload.get("metadata", {})
    else:
        state_dict = payload
        metadata = {}
    missing, unexpected = model.load_state_dict(state_dict, strict=strict)
    if missing or unexpected:
        raise RuntimeError(f"Failed to load checkpoint cleanly: missing={missing}, unexpected={unexpected}")
    return metadata


def checkpoint_csv_fields(metadata: dict[str, Any], checkpoint_path: str | None, method: str) -> dict[str, object]:
    return {
        "checkpoint_path": checkpoint_path or "",
        "compression_method": metadata.get("method", method),
        "sparsity": metadata.get("sparsity", ""),
        "nvfp4_group_size": metadata.get("nvfp4_group_size", ""),
        "calib_samples": metadata.get("calib_samples", ""),
        "kernel_path": metadata.get("kernel_path", ""),
        "kernel_name": metadata.get("kernel_name", ""),
    }


def materialize_from_checkpoint(
    model: nn.Module,
    metadata: dict[str, Any],
    masks_path: str | Path | None = None,
) -> nn.Module:
    """Replace model modules with kernel-backed equivalents based on checkpoint metadata.

    Loads masks from masks_path when present (required for sparse methods).
    Mutates metadata in-place to record kernel_path / kernel_name.
    """
    from fake.kernels.dispatch import materialize

    masks: dict[str, Any] | None = None
    if masks_path is not None:
        payload = torch.load(Path(masks_path), map_location="cpu")
        masks = payload.get("modules") if isinstance(payload, dict) else None

    return materialize(model, metadata, masks)

