#!/usr/bin/env python
"""Quick check: can the CUTLASS kernel extension be loaded?"""
from __future__ import annotations

import torch
from fake.kernels import kernel_info, is_available

if __name__ == "__main__":
    if not torch.cuda.is_available():
        print("❌ CUDA not available")
        exit(1)

    cap = torch.cuda.get_device_capability(0)
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Compute capability: sm_{cap[0]}{cap[1]}")

    info = kernel_info()
    print(f"\nCUTLASS_ROOT: {info['cutlass_root'] or '(not set)'}")
    print(f"Extension available: {info['extension_available']}")

    if not info['extension_available']:
        print(f"Load error: {info['load_error']}")
        exit(1)

    print("\n✅ Kernel extension loaded successfully")

    # Try a minimal forward pass
    from fake.kernels.modules import NVFP4Linear
    import torch.nn as nn

    linear = nn.Linear(64, 128).cuda()
    nvfp4_layer = NVFP4Linear.from_linear(linear)
    x = torch.randn(4, 64, device="cuda")

    # Check if kernel path is taken
    out = nvfp4_layer(x)
    print(f"Forward output shape: {out.shape}")

    if cap[0] >= 12:
        print("✅ sm_120 detected — NVFP4 kernel should be active")
    else:
        print(f"⚠️  sm_{cap[0]}{cap[1]} — NVFP4 kernel will fallback (needs sm_120)")
