from __future__ import annotations

import os
from pathlib import Path

CUTLASS_ROOT = os.environ.get("CUTLASS_ROOT", "")
CUTLASS_PROFILER = os.environ.get("CUTLASS_PROFILER", "")


def is_available() -> bool:
    """True when the kernel extension can be JIT-compiled in this process."""
    from fake.kernels import build as _build
    return _build.load() is not None


def is_profiler_available() -> bool:
    """True when cutlass_profiler binary is reachable (used by bench_kernel.py)."""
    return bool(CUTLASS_PROFILER) and Path(CUTLASS_PROFILER).is_file()


def kernel_info() -> dict:
    from fake.kernels import build as _build
    return {
        "cutlass_root": CUTLASS_ROOT,
        "cutlass_profiler": CUTLASS_PROFILER,
        "extension_available": is_available(),
        "load_error": _build.load_error(),
    }