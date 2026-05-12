#!/usr/bin/env python
"""Kernel micro-benchmark: measure single-GEMM latency via cutlass_profiler.

Enumerates real (M, N, K) shapes from MaxViT / DINOv3 compressible layers
and benchmarks each against the three kernel paths (nvfp4, semi_structured_sparse,
nvfp4_semi_structured_sparse), plus dense bf16 as baseline.

Usage (on compute node with CUTLASS_PROFILER set):
    CUTLASS_PROFILER=/path/to/cutlass_profiler \
        PYTHONPATH=. python scripts/bench_kernel.py \
        --model maxvit --variant tiny \
        --output artifacts/results/kernel_bench/shape_latency.csv
"""
from __future__ import annotations

import argparse
import csv
import os
import subprocess
import tempfile
from datetime import datetime
from pathlib import Path

import torch
import torch.nn as nn

from fake.kernels.registry import REGISTRY
from fake.kernels import CUTLASS_PROFILER, is_profiler_available
from fake.utils.csv_io import append_csv_row

# Dense bf16 baseline entry (bench_02 style)
_DENSE_BF16_ENTRY = (
    "gemm",
    "cutlass_tensorop_bf16_s16816gemm_bf16_256x128_32x3_tn_align8",
    "dense_bf16",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="CUTLASS kernel shape micro-benchmark.")
    parser.add_argument("--model", choices=["maxvit", "dinov3_vit7b16"], required=True)
    parser.add_argument("--variant", default="tiny", help="MaxViT variant (ignored for dinov3)")
    parser.add_argument("--model-path", default=None)
    parser.add_argument("--dinov3-backbone-path", default=None)
    parser.add_argument("--batch-sizes", type=int, nargs="+", default=[1, 8, 32],
                        help="M dimension (batch * spatial tokens)")
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iters", type=int, default=50)
    parser.add_argument("--output", default="artifacts/results/kernel_bench/shape_latency.csv")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if not is_profiler_available():
        raise RuntimeError(
            "cutlass_profiler not found. Set CUTLASS_PROFILER env var to its path."
        )

    shapes = _collect_shapes(args)
    print(f"[bench_kernel] {len(shapes)} unique (N, K) shapes from {args.model}")

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)

    kernel_entries = [
        (e.operation, e.kernels, e.method) for e in REGISTRY
    ] + [_DENSE_BF16_ENTRY]

    for m in args.batch_sizes:
        for (n, k) in shapes:
            for (operation, kernels, label) in kernel_entries:
                runtime_ms = _run_profiler(
                    operation=operation,
                    kernels=kernels,
                    m=m, n=n, k=k,
                    warmup=args.warmup,
                    iters=args.iters,
                )
                row = {
                    "timestamp": datetime.now().isoformat(timespec="seconds"),
                    "model": args.model,
                    "variant": args.variant,
                    "kernel": label,
                    "operation": operation,
                    "m": m, "n": n, "k": k,
                    "warmup": args.warmup,
                    "iters": args.iters,
                    "runtime_ms": f"{runtime_ms:.4f}" if runtime_ms is not None else "NA",
                    "gflops": f"{_gflops(m, n, k, runtime_ms):.2f}" if runtime_ms else "NA",
                    "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "",
                }
                append_csv_row(args.output, list(row.keys()), row)
                print(
                    f"  {label:35s} M={m:5d} N={n:5d} K={k:5d} "
                    f"runtime={row['runtime_ms']}ms gflops={row['gflops']}"
                )

    print(f"[bench_kernel] done → {args.output}")


# ── shape collection ─────────────────────────────────────────────────────────

def _collect_shapes(args: argparse.Namespace) -> list[tuple[int, int]]:
    """Return sorted unique (out_features, in_features) from compressible layers."""
    model = _load_model_cpu(args)
    shapes: set[tuple[int, int]] = set()
    for module in model.modules():
        if isinstance(module, nn.Linear):
            shapes.add((module.out_features, module.in_features))
        elif (
            isinstance(module, nn.Conv2d)
            and tuple(module.kernel_size) == (1, 1)
            and module.groups == 1
        ):
            shapes.add((module.out_channels, module.in_channels))
    return sorted(shapes)


def _load_model_cpu(args: argparse.Namespace) -> nn.Module:
    if args.model == "maxvit":
        from fake.models.maxvit import load_maxvit_dense
        model, _ = load_maxvit_dense(args.model_path, dtype="auto", device="cpu", variant=args.variant)
        return model
    from fake.models.dinov3 import (
        DEFAULT_DINOV3_BACKBONE_PATH,
        load_dinov3_vit7b16_dense_classifier,
    )
    from fake.models.dinov3 import DEFAULT_DINOV3_HEAD_PATH
    backbone = args.dinov3_backbone_path or str(DEFAULT_DINOV3_BACKBONE_PATH)
    model, _ = load_dinov3_vit7b16_dense_classifier(backbone, str(DEFAULT_DINOV3_HEAD_PATH), device="cpu")
    return model


# ── profiler runner ───────────────────────────────────────────────────────────

def _run_profiler(
    operation: str,
    kernels: str,
    m: int, n: int, k: int,
    warmup: int,
    iters: int,
) -> float | None:
    with tempfile.TemporaryDirectory() as tmp:
        out_prefix = os.path.join(tmp, "prof")
        cmd = [
            CUTLASS_PROFILER,
            f"--operation={operation}",
            f"--kernels={kernels}",
            f"--m={m}", f"--n={n}", f"--k={k}",
            "--providers=cutlass",
            "--verification-enabled=false",
            f"--warmup-iterations={warmup}",
            f"--profiling-iterations={iters}",
            f"--output={out_prefix}",
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            return None
        return _parse_best_runtime(out_prefix, operation)


def _parse_best_runtime(out_prefix: str, operation: str) -> float | None:
    csv_path = f"{out_prefix}.{operation}.csv"
    if not Path(csv_path).exists():
        return None
    best: float | None = None
    with open(csv_path) as f:
        for row in csv.DictReader(f):
            raw = (row.get("Runtime") or "").strip()
            try:
                val = float(raw)
                if best is None or val < best:
                    best = val
            except ValueError:
                pass
    return best


def _gflops(m: int, n: int, k: int, runtime_ms: float | None) -> float:
    if not runtime_ms or runtime_ms <= 0:
        return 0.0
    return 2 * m * n * k / (runtime_ms * 1e-3) / 1e12


if __name__ == "__main__":
    main()
