#!/bin/bash
#SBATCH --job-name=smoke_cutlass
#SBATCH --partition=gpu_5090
#SBATCH --gpus=1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --output=out/smoke_cutlass_%j.out
#SBATCH --error=err/smoke_cutlass_%j.err

set -euo pipefail

echo "Running on $(hostname)"

module load cuda/12.8
source ~/run/miniconda3/etc/profile.d/conda.sh
conda activate wja-cospaq

# Point to the CUTLASS source tree you built cutlass_profiler from.
# Same convention as cutlass_5090_my/test/common.sh.
export CUTLASS_ROOT="${CUTLASS_ROOT:-/data/home/scxj523/run/wja/cutlass}"

cd /data/home/scxj523/run/wja/project/my/fake/

PYTHONPATH=. python -c "
import torch, torch.nn as nn
from fake.kernels import kernel_info
from fake.kernels.modules import NVFP4Linear, SemiSparseLinear, NVFP4SemiSparseLinear

print('=== kernel_info ===')
for k, v in kernel_info().items():
    print(f'  {k}: {v}')

assert torch.cuda.is_available()
cap = torch.cuda.get_device_capability(0)
print(f'device capability: {cap}')
device = torch.device('cuda')

# ── NVFP4Linear (sm_120) ─────────────────────────────────
if cap[0] >= 12:
    print('--- NVFP4Linear kernel path ---')
    lin = nn.Linear(1024, 2048, bias=True).to(device).to(torch.bfloat16)
    m = NVFP4Linear.from_linear(lin).to(device)
    x = torch.randn(256, 1024, device=device, dtype=torch.bfloat16)
    y_kernel = m(x)
    y_ref = m._fallback_forward(x)
    err = (y_kernel.float() - y_ref.float()).abs()
    max_err = err.max().item()
    mean_err = err.mean().item()
    print(f'  NVFP4Linear: shape={tuple(y_kernel.shape)} dtype={y_kernel.dtype}')
    print(f'    kernel vs fallback: max_err={max_err:.4f} mean_err={mean_err:.4f}')
    # Kernel quantizes activation to NVFP4 at runtime (W4A4), fallback uses bf16 activation.
    # Expect moderate difference. Loose sanity bound: max_err < 5.0.
    assert torch.isfinite(y_kernel).all(), 'kernel output has NaN/Inf'
    assert max_err < 5.0, f'NVFP4Linear max_err {max_err} >= 5.0 — layout mismatch?'
    print('  ✓ NVFP4Linear kernel OK')
else:
    print('!! skipping NVFP4Linear kernel (need sm_120)')

# ── SemiSparseLinear (sm_80+) ────────────────────────────
print('--- SemiSparseLinear kernel path ---')
in_f, out_f = 1024, 512
lin = nn.Linear(in_f, out_f, bias=True).to(device)
mask = torch.zeros(out_f, in_f, dtype=torch.bool, device=device)
mask[:, 0::4] = True
mask[:, 2::4] = True
with torch.no_grad():
    lin.weight.data[:, 1::4] = 0
    lin.weight.data[:, 3::4] = 0
m = SemiSparseLinear.from_linear(lin, mask).to(device)
print(f'  kernel_ready={m.kernel_ready}')
x = torch.randn(256, in_f, device=device, dtype=torch.bfloat16)
y_kernel = m(x)
y_ref = m._fallback_forward(x)
err = (y_kernel.float() - y_ref.float()).abs()
max_err = err.max().item()
mean_err = err.mean().item()
print(f'  SemiSparseLinear: shape={tuple(y_kernel.shape)} dtype={y_kernel.dtype}')
print(f'    kernel vs fallback: max_err={max_err:.4f} mean_err={mean_err:.4f}')
assert torch.isfinite(y_kernel).all()
# Kernel and fallback use same compressed weight + bf16 activation; expect near-exact match.
if m.kernel_ready:
    assert max_err < 1e-2, f'SemiSparseLinear max_err {max_err} >= 1e-2 — metadata layout issue?'
print('  ✓ SemiSparseLinear kernel OK')

# ── NVFP4SemiSparseLinear (sm_120) ───────────────────────
if cap[0] >= 12:
    print('--- NVFP4SemiSparseLinear kernel path ---')
    in_f, out_f = 1024, 512
    lin = nn.Linear(in_f, out_f, bias=False).to(device).to(torch.bfloat16)
    mask = torch.zeros(out_f, in_f, dtype=torch.bool, device=device)
    mask[:, 0::4] = True; mask[:, 2::4] = True
    with torch.no_grad():
        lin.weight.data[:, 1::4] = 0
        lin.weight.data[:, 3::4] = 0
    m = NVFP4SemiSparseLinear.from_linear(lin, mask, group_size=32).to(device)
    x = torch.randn(256, in_f, device=device, dtype=torch.bfloat16)
    y_kernel = m(x)
    y_ref = m._fallback_forward(x)
    err = (y_kernel.float() - y_ref.float()).abs()
    max_err = err.max().item()
    mean_err = err.mean().item()
    print(f'  NVFP4SemiSparseLinear: shape={tuple(y_kernel.shape)} dtype={y_kernel.dtype}')
    print(f'    kernel vs fallback: max_err={max_err:.4f} mean_err={mean_err:.4f}')
    assert torch.isfinite(y_kernel).all()
    assert max_err < 5.0, f'NVFP4SemiSparseLinear max_err {max_err} >= 5.0'
    print('  ✓ NVFP4SemiSparseLinear kernel OK')
else:
    print('!! skipping NVFP4SemiSparseLinear kernel (need sm_120)')

print('=== all kernel smokes PASSED ===')
"
