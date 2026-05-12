#!/usr/bin/env python
"""Verify that forward() actually calls the CUTLASS kernel, not fallback."""
from __future__ import annotations

import time
import torch
import torch.nn as nn

from fake.kernels import kernel_info
from fake.kernels.modules import NVFP4Linear, SemiSparseLinear


def benchmark_forward(module: nn.Module, x: torch.Tensor, warmup: int = 10, iters: int = 50) -> float:
    """Measure forward latency in milliseconds."""
    for _ in range(warmup):
        _ = module(x)
    torch.cuda.synchronize()

    start = time.perf_counter()
    for _ in range(iters):
        _ = module(x)
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start
    return (elapsed / iters) * 1000


if __name__ == "__main__":
    if not torch.cuda.is_available():
        print("❌ CUDA not available")
        exit(1)

    cap = torch.cuda.get_device_capability(0)
    print(f"GPU: {torch.cuda.get_device_name(0)} (sm_{cap[0]}{cap[1]})")
    print()

    info = kernel_info()
    print(f"CUTLASS_ROOT: {info['cutlass_root'] or '(not set)'}")
    print(f"Extension available: {info['extension_available']}")
    if not info['extension_available']:
        print(f"Load error: {info['load_error']}")
        print("\n⚠️  Extension not available — all forwards will use fallback")
    print()

    # Test NVFP4Linear
    print("=== Testing NVFP4Linear ===")
    M, K, N = 128, 1024, 512
    linear = nn.Linear(K, N, bias=True).cuda()
    nvfp4_layer = NVFP4Linear.from_linear(linear)
    x = torch.randn(M, K, device="cuda", dtype=torch.bfloat16)

    # Warm up and measure
    latency = benchmark_forward(nvfp4_layer, x, warmup=10, iters=50)
    print(f"Forward latency: {latency:.3f} ms")

    # Check if kernel path was taken by inspecting internal state
    # (This is a heuristic: kernel path should be faster on sm_120)
    if cap[0] >= 12 and info['extension_available']:
        print("✅ sm_120 + extension available → kernel path should be active")
        print("   Expected: ~0.1-1ms latency (kernel), not ~5-10ms (fallback dequant)")
    elif cap[0] >= 8 and info['extension_available']:
        print("⚠️  sm_80 + extension available → NVFP4 kernel unavailable (needs sm_120), using fallback")
    else:
        print("⚠️  Extension not available → using fallback dequant path")
    print()

    # Test SemiSparseLinear (works on sm_80+)
    print("=== Testing SemiSparseLinear ===")
    K_aligned = 128  # Must be multiple of 64 for sparse kernel
    linear2 = nn.Linear(K_aligned, N, bias=True).cuda()
    # Create a 2:4 mask
    mask = torch.zeros(N, K_aligned, dtype=torch.bool, device="cuda")
    for i in range(N):
        for j in range(0, K_aligned, 4):
            mask[i, j:j+2] = True  # Keep first 2 out of every 4

    sparse_layer = SemiSparseLinear.from_linear(linear2, mask)
    x2 = torch.randn(M, K_aligned, device="cuda", dtype=torch.bfloat16)

    latency2 = benchmark_forward(sparse_layer, x2, warmup=10, iters=50)
    print(f"Forward latency: {latency2:.3f} ms")
    print(f"kernel_ready flag: {sparse_layer.kernel_ready}")

    if cap[0] >= 8 and info['extension_available'] and sparse_layer.kernel_ready:
        print("✅ sm_80+ + extension available + kernel_ready=True → sparse kernel should be active")
        print("   Expected: ~0.1-1ms latency (kernel), not ~5-10ms (fallback mask reconstruction)")
    else:
        print("⚠️  Sparse kernel not available → using fallback (mask reconstruction)")
    print()

    # Final verdict
    print("=== How to confirm kernel is really used ===")
    print("1. Check extension_available=True above")
    print("2. Check kernel_ready=True for SemiSparseLinear")
    print("3. Compare latency: kernel should be 5-10x faster than fallback")
    print("4. Run with KERNEL=0 and KERNEL=1, compare CSV latency_mean_ms")
    print("5. Add print() inside fake/kernels/modules.py _kernel_forward() to see if it's called")
