#!/usr/bin/env python
from __future__ import annotations

import argparse
from datetime import datetime

import torch

from fake.compression.checkpoint import checkpoint_csv_fields, load_checkpoint_into_model
from fake.evaluation.speed import benchmark_forward
from fake.models.maxvit import (
    MAXVIT_VARIANT_CHOICES,
    get_maxvit_variant,
    load_maxvit_dense,
    maxvit_input_size,
    model_input_dtype,
)
from fake.utils.csv_io import append_csv_row


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark dense MaxViT forward speed.")
    parser.add_argument("--variant", choices=MAXVIT_VARIANT_CHOICES, default="tiny")
    parser.add_argument("--model-path", default=None)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--input-size", type=int, nargs=3, default=None)
    parser.add_argument("--warmup", type=int, default=50)
    parser.add_argument("--iters", type=int, default=200)
    parser.add_argument("--dtype", choices=["auto", "fp32", "bf16", "fp16"], default="auto")
    parser.add_argument("--output", default=None)
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--method", default="dense")
    parser.add_argument("--kernel", action="store_true", help="Materialize kernel modules after loading checkpoint")
    parser.add_argument("--masks", default=None, help="Path to masks.pt (required for sparse kernel methods)")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required. Submit this script to a GPU compute node.")

    device = torch.device("cuda")
    variant_info = get_maxvit_variant(args.variant)
    output = args.output or f"artifacts/results/{variant_info.result_key}_dense/speed.csv"
    model, config = load_maxvit_dense(args.model_path, dtype=args.dtype, device=device, variant=args.variant)
    checkpoint_metadata = load_checkpoint_into_model(model, args.checkpoint)
    if args.kernel:
        from fake.compression.checkpoint import materialize_from_checkpoint
        model = materialize_from_checkpoint(model, checkpoint_metadata, args.masks)
    input_dtype = model_input_dtype(model)
    input_size = tuple(args.input_size) if args.input_size is not None else maxvit_input_size(config)
    result = benchmark_forward(
        model=model,
        batch_size=args.batch_size,
        input_size=input_size,
        input_dtype=input_dtype,
        device=device,
        warmup=args.warmup,
        iters=args.iters,
    )
    gpu_name = torch.cuda.get_device_name(device)
    row = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "model": variant_info.model_id,
        "model_variant": variant_info.variant,
        "method": args.method,
        "task": "forward_speed",
        "speed_scope": "random_input_forward_only",
        "dtype_arg": args.dtype,
        "runtime_dtype": str(input_dtype).replace("torch.", ""),
        "device": gpu_name,
        "batch_size": args.batch_size,
        "input_c": input_size[0],
        "input_h": input_size[1],
        "input_w": input_size[2],
        "warmup": args.warmup,
        "iters": args.iters,
        "latency_mean_ms": f"{result.latency_mean_ms:.6f}",
        "latency_p50_ms": f"{result.latency_p50_ms:.6f}",
        "latency_p90_ms": f"{result.latency_p90_ms:.6f}",
        "latency_min_ms": f"{result.latency_min_ms:.6f}",
        "latency_max_ms": f"{result.latency_max_ms:.6f}",
        "images_per_sec": f"{result.images_per_sec:.3f}",
        "model_path": args.model_path or str(variant_info.model_path),
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        **checkpoint_csv_fields(checkpoint_metadata, args.checkpoint, args.method),
    }
    fieldnames = list(row.keys())
    append_csv_row(output, fieldnames, row)
    print(
        "speed done: "
        f"variant={args.variant} batch_size={args.batch_size} dtype={row['runtime_dtype']} "
        f"mean_ms={row['latency_mean_ms']} images_per_sec={row['images_per_sec']} "
        f"warmup={args.warmup} iters={args.iters} output={output}"
    )


if __name__ == "__main__":
    main()
