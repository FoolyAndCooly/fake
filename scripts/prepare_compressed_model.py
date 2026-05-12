#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from fake.compression.pipeline import (
    SUPPORTED_METHODS,
    CompressionConfig,
    compress_model,
    default_calib_batch_size,
    default_calib_samples,
    default_nvfp4_group_size,
)
from fake.data.dinov3_transforms import build_dinov3_lvd1689m_transform
from fake.data.imagenet_zip import DEFAULT_IMAGENET_ROOT, ImageNetZipDataset
from fake.models.dinov3 import (
    DEFAULT_DINOV3_BACKBONE_PATH,
    DEFAULT_DINOV3_HEAD_PATH,
    load_dinov3_vit7b16_dense_classifier,
    model_input_dtype as dinov3_input_dtype,
)
from fake.models.maxvit import (
    MAXVIT_VARIANT_CHOICES,
    get_maxvit_variant,
    load_maxvit_dense,
    model_input_dtype as maxvit_input_dtype,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare fake-quant/pruned compressed checkpoints.")
    parser.add_argument("--model", choices=["maxvit", "dinov3_vit7b16"], required=True)
    parser.add_argument("--method", choices=SUPPORTED_METHODS, required=True)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--dataset-root", default=str(DEFAULT_IMAGENET_ROOT))
    parser.add_argument("--csv", default="val.csv")
    parser.add_argument("--zip", default="imagenet_val.zip")
    parser.add_argument("--calib-samples", type=int, default=None)
    parser.add_argument("--calib-batch-size", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--sparsity", type=float, default=0.5)
    parser.add_argument("--nvfp4-group-size", type=int, default=None)
    parser.add_argument("--nvfp4-scale-precision", default="fp16")
    parser.add_argument("--nvfp4-scale-remap", default="none")
    parser.add_argument("--save-full-masks", action="store_true")
    parser.add_argument("--save-full-scales", action="store_true")
    parser.add_argument("--maxvit-variant", choices=MAXVIT_VARIANT_CHOICES, default="tiny")
    parser.add_argument("--maxvit-model-path", default=None)
    parser.add_argument("--dinov3-backbone-path", default=str(DEFAULT_DINOV3_BACKBONE_PATH))
    parser.add_argument("--dinov3-head-path", default=str(DEFAULT_DINOV3_HEAD_PATH))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required. Submit this script to a GPU compute node.")

    device = torch.device("cuda")
    model, model_config, input_dtype = _load_model(args, device)
    calib_samples = args.calib_samples if args.calib_samples is not None else default_calib_samples(args.model)
    calib_batch_size = args.calib_batch_size if args.calib_batch_size is not None else _default_calib_batch_size(args)
    group_size = args.nvfp4_group_size if args.nvfp4_group_size is not None else default_nvfp4_group_size(args.method)
    output_dir = Path(args.output_dir or _default_output_dir(args))
    output_dir.mkdir(parents=True, exist_ok=True)

    dataset = _build_dataset(args, model_config)
    dataloader = DataLoader(
        dataset,
        batch_size=calib_batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=args.num_workers > 0,
    )
    # Kernel path (SemiSparseLinear / NVFP4SemiSparseLinear) requires full masks
    # for materialize_from_checkpoint to reconstruct the 2:4 sparsity pattern.
    # For non-sparse methods, save_full_masks remains user-controlled.
    save_full_masks = args.save_full_masks or args.method in {"semi_structured_sparse", "nvfp4_semi_structured_sparse"}
    config = CompressionConfig(
        model_name=args.model,
        method=args.method,
        calib_samples=calib_samples,
        sparsity=args.sparsity,
        nvfp4_group_size=group_size,
        nvfp4_scale_precision=args.nvfp4_scale_precision,
        nvfp4_scale_remap=args.nvfp4_scale_remap,
        save_full_masks=save_full_masks,
        save_full_scales=args.save_full_scales,
    )
    metadata, masks, scales = compress_model(
        model=model,
        dataloader=dataloader,
        device=device,
        input_dtype=input_dtype,
        config=config,
    )
    metadata.update(
        {
            "checkpoint_path": str(output_dir / "model.pt"),
            "masks_path": str(output_dir / "masks.pt"),
            "scales_path": str(output_dir / "scales.pt"),
            "dataset_root": args.dataset_root,
            "csv": args.csv,
            "zip": args.zip,
            **_model_metadata(args),
        }
    )

    torch.save({"state_dict": model.state_dict(), "metadata": metadata}, output_dir / "model.pt")
    torch.save(masks, output_dir / "masks.pt")
    torch.save(scales, output_dir / "scales.pt")
    with (output_dir / "metadata.json").open("w") as f:
        json.dump(metadata, f, indent=2)
    print(
        "compression done: "
        f"model={args.model} variant={metadata.get('model_variant', '')} "
        f"method={args.method} modules={metadata['compressed_modules']} "
        f"calib_samples={calib_samples} output={output_dir}"
    )


def _load_model(args: argparse.Namespace, device: torch.device):
    if args.model == "maxvit":
        model, config = load_maxvit_dense(
            args.maxvit_model_path,
            dtype="auto",
            device=device,
            variant=args.maxvit_variant,
        )
        return model, config, maxvit_input_dtype(model)
    if args.model == "dinov3_vit7b16":
        model, config = load_dinov3_vit7b16_dense_classifier(
            args.dinov3_backbone_path,
            args.dinov3_head_path,
            device=device,
        )
        return model, config, dinov3_input_dtype(model)
    raise ValueError(f"Unsupported model: {args.model}")


def _build_dataset(args: argparse.Namespace, model_config: dict):
    if args.model == "dinov3_vit7b16":
        return ImageNetZipDataset(
            args.dataset_root,
            args.csv,
            args.zip,
            transform=build_dinov3_lvd1689m_transform(256),
        )
    return ImageNetZipDataset(args.dataset_root, args.csv, args.zip, model_config)


def _default_calib_batch_size(args: argparse.Namespace) -> int:
    if args.model == "maxvit" and args.maxvit_variant == "large":
        return 4
    return default_calib_batch_size(args.model)


def _default_output_dir(args: argparse.Namespace) -> str:
    if args.model == "maxvit":
        return f"artifacts/checkpoints/maxvit_{args.maxvit_variant}/{args.method}"
    return f"artifacts/checkpoints/{args.model}/{args.method}"


def _model_metadata(args: argparse.Namespace) -> dict[str, str]:
    if args.model == "maxvit":
        variant_info = get_maxvit_variant(args.maxvit_variant)
        return {
            "model_id": variant_info.model_id,
            "model_variant": variant_info.variant,
            "result_key": variant_info.result_key,
            "model_path": str(args.maxvit_model_path or variant_info.model_path),
        }
    return {}


if __name__ == "__main__":
    main()
