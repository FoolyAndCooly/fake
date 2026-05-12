from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class KernelEntry:
    method: str
    operation: str
    kernels: str


REGISTRY: list[KernelEntry] = [
    # NOTE: The actual kernel instantiated in csrc/*.cu is hard-coded via template
    # parameters (tile/cluster/stage). The `kernels` field here is a *label* written
    # to CSV and used by cutlass_profiler for micro-bench; it does NOT control which
    # kernel the extension JIT-compiles. To change the kernel, edit the .cu file.
    KernelEntry(
        method="nvfp4",
        operation="block_scaled_gemm",
        kernels="*sm120*bstensorop*ue4m3xe2m1*ue4m3xe2m1*f32_void_f32*cooperative*",
    ),
    KernelEntry(
        method="semi_structured_sparse",
        operation="spgemm",
        kernels="cutlass_tensorop_s16832spgemm_bf16_64x128_64x6_tn_align8",
    ),
    KernelEntry(
        method="nvfp4_semi_structured_sparse",
        operation="block_scaled_gemm",
        kernels="*sm120*bssptensorop*ue4m3xe2m1*ue4m3xe2m1*f32_void_f32*",
    ),
]

# These methods have no GPU-accelerated kernel; forward stays dense.
DENSE_FALLBACK_METHODS = {"unstructured_sparse", "nvfp4_unstructured_sparse"}


def get_entry(method: str) -> KernelEntry | None:
    for e in REGISTRY:
        if e.method == method:
            return e
    return None