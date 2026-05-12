"""JIT-compile the CUTLASS kernel extension.

Uses the same CUTLASS_ROOT convention as cutlass_5090_my/test/common.sh.

The extension bundles three kernels:
- nvfp4_gemm              (sm_120 only)
- sparse24_gemm_bf16      (sm_80+)
- sparse24_nvfp4_gemm     (sm_120 only)

load() picks a gencode based on the current GPU. If compilation fails,
load() returns None and modules.py uses the Python fallback path.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import torch

_ext_cache: Any = None
_load_error: str | None = None


def has_blackwell() -> bool:
    if not torch.cuda.is_available():
        return False
    cap = torch.cuda.get_device_capability(0)
    return cap[0] >= 12


def has_ampere_or_newer() -> bool:
    if not torch.cuda.is_available():
        return False
    cap = torch.cuda.get_device_capability(0)
    return cap[0] >= 8


def _cutlass_root() -> Path | None:
    root = os.environ.get("CUTLASS_ROOT", "")
    if not root:
        return None
    p = Path(root)
    if not (p / "include" / "cutlass" / "cutlass.h").is_file():
        return None
    return p


def load() -> Any | None:
    """JIT-build and return the extension module, or None on any failure."""
    global _ext_cache, _load_error

    if _ext_cache is not None:
        return _ext_cache

    if not has_ampere_or_newer():
        _load_error = "CUDA device capability < 8.0; no supported kernels"
        return None

    root = _cutlass_root()
    if root is None:
        _load_error = "CUTLASS_ROOT not set or does not point to a CUTLASS source tree"
        return None

    here = Path(__file__).resolve().parent
    sources = [
        str(here / "csrc" / "bindings.cpp"),
        str(here / "csrc" / "nvfp4_gemm.cu"),
        str(here / "csrc" / "sparse_2_4_bf16_gemm.cu"),
        str(here / "csrc" / "sparse_2_4_nvfp4_gemm.cu"),
    ]
    include_dirs = [
        str(root / "include"),
        str(root / "tools" / "util" / "include"),
    ]

    gencode_flags = ["-gencode=arch=compute_80,code=sm_80"]
    if has_blackwell():
        gencode_flags = ["-gencode=arch=compute_120,code=sm_120"]

    from torch.utils.cpp_extension import load as torch_load
    try:
        _ext_cache = torch_load(
            name="fake_kernels_cutlass",
            sources=sources,
            extra_include_paths=include_dirs,
            extra_cuda_cflags=[
                "-O3",
                "-std=c++17",
                *gencode_flags,
                "-DCUTLASS_ENABLE_TENSOR_CORE_MMA=1",
                "--expt-relaxed-constexpr",
                "--expt-extended-lambda",
            ],
            extra_cflags=["-O3", "-std=c++17"],
            verbose=False,
        )
        return _ext_cache
    except Exception as e:
        _load_error = f"JIT compile failed: {e}"
        _ext_cache = None
        return None


def load_error() -> str | None:
    return _load_error
