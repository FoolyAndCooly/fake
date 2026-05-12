#!/usr/bin/env python
"""Comprehensive verification that CUTLASS kernels are actually being used.

This script performs multiple checks to confirm kernel usage:
1. Extension compilation and loading
2. Module replacement during materialize
3. Forward path selection (kernel vs fallback)
4. Performance comparison

Usage:
    CUTLASS_ROOT=/path/to/cutlass python scripts/verify_kernel_usage.py
"""
from __future__ import annotations

import time
import torch
import torch.nn as nn

from fake.kernels import kernel_info, is_available
from fake.kernels.modules import NVFP4Linear, SemiSparseLinear, NVFP4SemiSparseLinear
from fake.kernels.dispatch import materialize


def print_section(title: str) -> None:
    print(f"\n{'='*70}")
    print(f"  {title}")
    print('='*70)


def check_extension() -> bool:
    """Check if CUTLASS extension can be loaded."""
    print_section("1. Extension Availability Check")

    if not torch.cuda.is_available():
        print("❌ CUDA not available")
        return False

    cap = torch.cuda.get_device_capability(0)
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Compute capability: sm_{cap[0]}{cap[1]}")

    info = kernel_info()
    print(f"\nCUTLASS_ROOT: {info['cutlass_root'] or '(not set)'}")
    print(f"Extension available: {info['extension_available']}")

    if not info['extension_available']:
        print(f"❌ Load error: {info['load_error']}")
        print("\n⚠️  Kernels will NOT be used — all forwards will use fallback")
        return False

    print("✅ Extension loaded successfully")

    # Check which kernels are available based on compute capability
    if cap[0] >= 12:
        print("✅ sm_120 — All kernels available (NVFP4, Sparse BF16, NVFP4 Sparse)")
    elif cap[0] >= 8:
        print("⚠️  sm_80-89 — Only Sparse BF16 kernel available (NVFP4 needs sm_120)")
    else:
        print("❌ sm < 80 — No kernels available")
        return False

    return True


def check_materialize() -> bool:
    """Verify that materialize() replaces modules correctly."""
    print_section("2. Module Replacement Check")

    # Create a toy model
    model = nn.Sequential(
        nn.Linear(64, 128),
        nn.ReLU(),
        nn.Linear(128, 256),
        nn.ReLU(),
        nn.Linear(256, 64),
    ).cuda()

    linear_count = sum(1 for m in model.modules() if isinstance(m, nn.Linear))
    print(f"Created toy model with {linear_count} Linear layers")

    # Test NVFP4 materialize
    print("\nTesting materialize(method='nvfp4')...")
    metadata = {"method": "nvfp4"}
    model_nvfp4 = materialize(model, metadata, masks=None)

    nvfp4_count = sum(1 for m in model_nvfp4.modules() if type(m).__name__ == "NVFP4Linear")
    print(f"\nResult: {nvfp4_count} layers replaced with NVFP4Linear")
    print(f"Metadata: kernel_path={metadata.get('kernel_path')}, kernel_name={metadata.get('kernel_name')}")

    if nvfp4_count > 0:
        print("✅ Module replacement successful")
        return True
    else:
        print("❌ No modules were replaced — check K-dimension alignment")
        return False


def check_forward_path() -> bool:
    """Verify that forward() calls kernel, not fallback."""
    print_section("3. Forward Path Verification")

    cap = torch.cuda.get_device_capability(0)

    # Test NVFP4Linear
    print("\n--- Testing NVFP4Linear ---")
    M, K, N = 32, 128, 256  # K=128 is aligned for NVFP4 (multiple of 16)
    linear = nn.Linear(K, N, bias=True).cuda()
    nvfp4_layer = NVFP4Linear.from_linear(linear)
    x = torch.randn(M, K, device="cuda", dtype=torch.bfloat16)

    # Monkey-patch to detect which path is taken
    kernel_called = [False]
    fallback_called = [False]

    original_kernel = nvfp4_layer._kernel_forward
    original_fallback = nvfp4_layer._fallback_forward

    def tracked_kernel(x2d):
        kernel_called[0] = True
        return original_kernel(x2d)

    def tracked_fallback(x2d):
        fallback_called[0] = True
        return original_fallback(x2d)

    nvfp4_layer._kernel_forward = tracked_kernel
    nvfp4_layer._fallback_forward = tracked_fallback

    # Run forward
    out = nvfp4_layer(x)

    print(f"Output shape: {out.shape}")
    print(f"_kernel_forward called: {kernel_called[0]}")
    print(f"_fallback_forward called: {fallback_called[0]}")

    if cap[0] >= 12 and is_available():
        if kernel_called[0] and not fallback_called[0]:
            print("✅ NVFP4 kernel path is active (sm_120)")
        else:
            print("❌ Expected kernel path but got fallback")
            return False
    else:
        if fallback_called[0]:
            print("⚠️  Fallback path used (expected on sm < 120)")
        else:
            print("❌ Unexpected path behavior")
            return False

    # Test SemiSparseLinear
    print("\n--- Testing SemiSparseLinear ---")
    K_aligned = 128  # Must be multiple of 64
    linear2 = nn.Linear(K_aligned, N, bias=True).cuda()

    # Create 2:4 mask
    mask = torch.zeros(N, K_aligned, dtype=torch.bool, device="cuda")
    for i in range(N):
        for j in range(0, K_aligned, 4):
            mask[i, j:j+2] = True

    sparse_layer = SemiSparseLinear.from_linear(linear2, mask)
    x2 = torch.randn(M, K_aligned, device="cuda", dtype=torch.bfloat16)

    print(f"kernel_ready flag: {sparse_layer.kernel_ready}")

    # Track sparse forward
    sparse_kernel_called = [False]
    sparse_fallback_called = [False]

    original_sparse_kernel = sparse_layer._kernel_forward
    original_sparse_fallback = sparse_layer._fallback_forward

    def tracked_sparse_kernel(x2d):
        sparse_kernel_called[0] = True
        return original_sparse_kernel(x2d)

    def tracked_sparse_fallback(x2d):
        sparse_fallback_called[0] = True
        return original_sparse_fallback(x2d)

    sparse_layer._kernel_forward = tracked_sparse_kernel
    sparse_layer._fallback_forward = tracked_sparse_fallback

    out2 = sparse_layer(x2)

    print(f"Output shape: {out2.shape}")
    print(f"_kernel_forward called: {sparse_kernel_called[0]}")
    print(f"_fallback_forward called: {sparse_fallback_called[0]}")

    if cap[0] >= 8 and is_available() and sparse_layer.kernel_ready:
        if sparse_kernel_called[0] and not sparse_fallback_called[0]:
            print("✅ Sparse BF16 kernel path is active (sm_80+)")
        else:
            print("❌ Expected kernel path but got fallback")
            return False
    else:
        if sparse_fallback_called[0]:
            print("⚠️  Fallback path used (expected when kernel_ready=False)")
        else:
            print("❌ Unexpected path behavior")
            return False

    return True


def benchmark_comparison() -> None:
    """Compare kernel vs fallback performance."""
    print_section("4. Performance Comparison")

    cap = torch.cuda.get_device_capability(0)
    if cap[0] < 8 or not is_available():
        print("⚠️  Skipping benchmark — kernels not available")
        return

    print("\nBenchmarking SemiSparseLinear (available on sm_80+)...")
    M, K, N = 128, 1024, 512
    warmup, iters = 20, 100

    linear = nn.Linear(K, N, bias=True).cuda()
    mask = torch.zeros(N, K, dtype=torch.bool, device="cuda")
    for i in range(N):
        for j in range(0, K, 4):
            mask[i, j:j+2] = True

    sparse_layer = SemiSparseLinear.from_linear(linear, mask)
    x = torch.randn(M, K, device="cuda", dtype=torch.bfloat16)

    # Benchmark with kernel_ready=True (kernel path)
    if sparse_layer.kernel_ready:
        for _ in range(warmup):
            _ = sparse_layer(x)
        torch.cuda.synchronize()

        start = time.perf_counter()
        for _ in range(iters):
            _ = sparse_layer(x)
        torch.cuda.synchronize()
        kernel_time = (time.perf_counter() - start) / iters * 1000
        print(f"Kernel path latency: {kernel_time:.3f} ms")
    else:
        print("⚠️  kernel_ready=False, cannot benchmark kernel path")
        kernel_time = None

    # Force fallback by setting kernel_ready=False
    sparse_layer.kernel_ready = False
    for _ in range(warmup):
        _ = sparse_layer(x)
    torch.cuda.synchronize()

    start = time.perf_counter()
    for _ in range(iters):
        _ = sparse_layer(x)
    torch.cuda.synchronize()
    fallback_time = (time.perf_counter() - start) / iters * 1000
    print(f"Fallback path latency: {fallback_time:.3f} ms")

    if kernel_time is not None:
        speedup = fallback_time / kernel_time
        print(f"\nSpeedup: {speedup:.2f}x")
        if speedup > 1.5:
            print("✅ Kernel is significantly faster — confirms kernel is being used")
        else:
            print("⚠️  Speedup is small — kernel may not be active or shape is too small")


def main() -> None:
    print("="*70)
    print("  CUTLASS Kernel Usage Verification")
    print("="*70)

    # Run all checks
    ext_ok = check_extension()
    if not ext_ok:
        print("\n" + "="*70)
        print("❌ VERIFICATION FAILED: Extension not available")
        print("="*70)
        print("\nTo fix:")
        print("1. Set CUTLASS_ROOT environment variable")
        print("2. Ensure CUTLASS source tree has include/cutlass/cutlass.h")
        print("3. Check GPU compute capability >= 8.0")
        return

    mat_ok = check_materialize()
    fwd_ok = check_forward_path()

    benchmark_comparison()

    # Final verdict
    print_section("Final Verdict")
    if ext_ok and mat_ok and fwd_ok:
        print("✅ ALL CHECKS PASSED — Kernels are being used correctly")
    else:
        print("❌ SOME CHECKS FAILED — Review output above")

    print("\n" + "="*70)
    print("Additional verification methods:")
    print("1. Compare KERNEL=0 vs KERNEL=1 in CSV output (latency_mean_ms)")
    print("2. Add print() in fake/kernels/modules.py _kernel_forward()")
    print("3. Use nsys profile to see CUTLASS kernel names in timeline")
    print("="*70)


if __name__ == "__main__":
    main()
