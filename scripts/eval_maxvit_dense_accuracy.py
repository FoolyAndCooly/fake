#!/usr/bin/env python
from __future__ import annotations

import argparse
from datetime import datetime

import torch
from torch.utils.data import DataLoader

from fake.compression.checkpoint import checkpoint_csv_fields, load_checkpoint_into_model
from fake.data.imagenet_zip import DEFAULT_IMAGENET_ROOT, ImageNetZipDataset
from fake.evaluation.accuracy import evaluate_topk
from fake.models.maxvit import MAXVIT_VARIANT_CHOICES, get_maxvit_variant, load_maxvit_dense, model_input_dtype
from fake.utils.csv_io import append_csv_row


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate dense MaxViT on ImageNet val zip dataset.")
    parser.add_argument("--variant", choices=MAXVIT_VARIANT_CHOICES, default="tiny")
    parser.add_argument("--model-path", default=None)
    parser.add_argument("--dataset-root", default=str(DEFAULT_IMAGENET_ROOT))
    parser.add_argument("--csv", default="val.csv")
    parser.add_argument("--zip", default="imagenet_val.zip")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--dtype", choices=["auto", "fp32", "bf16", "fp16"], default="auto")
    parser.add_argument("--log-interval", type=int, default=50)
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
    output = args.output or f"artifacts/results/{variant_info.result_key}_dense/accuracy.csv"
    model, config = load_maxvit_dense(args.model_path, dtype=args.dtype, device=device, variant=args.variant)
    checkpoint_metadata = load_checkpoint_into_model(model, args.checkpoint)
    if args.kernel:
        from fake.compression.checkpoint import materialize_from_checkpoint
        model = materialize_from_checkpoint(model, checkpoint_metadata, args.masks)
    input_dtype = model_input_dtype(model)
    dataset = ImageNetZipDataset(args.dataset_root, args.csv, args.zip, config)
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=args.num_workers > 0,
    )

    result = evaluate_topk(model, dataloader, device, input_dtype, args.log_interval)
    gpu_name = torch.cuda.get_device_name(device)
    row = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "model": variant_info.model_id,
        "model_variant": variant_info.variant,
        "method": args.method,
        "task": "imagenet_accuracy",
        "dtype_arg": args.dtype,
        "runtime_dtype": str(input_dtype).replace("torch.", ""),
        "device": gpu_name,
        "num_samples": result.num_samples,
        "batch_size": args.batch_size,
        "num_workers": args.num_workers,
        "top1": f"{result.top1:.6f}",
        "top5": f"{result.top5:.6f}",
        "elapsed_sec": f"{result.elapsed_sec:.3f}",
        "images_per_sec": f"{result.images_per_sec:.3f}",
        "model_path": args.model_path or str(variant_info.model_path),
        "dataset_root": args.dataset_root,
        "csv": args.csv,
        "zip": args.zip,
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        **checkpoint_csv_fields(checkpoint_metadata, args.checkpoint, args.method),
    }
    fieldnames = list(row.keys())
    append_csv_row(output, fieldnames, row)
    print(
        "accuracy done: "
        f"top1={row['top1']} top5={row['top5']} samples={result.num_samples} "
        f"variant={args.variant} batch_size={args.batch_size} dtype={row['runtime_dtype']} output={output}"
    )


if __name__ == "__main__":
    main()
