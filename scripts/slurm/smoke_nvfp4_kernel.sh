#!/bin/bash
#SBATCH --job-name=smoke_nvfp4
#SBATCH --partition=gpu_5090
#SBATCH --gpus=1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --output=out/smoke_nvfp4_%j.out
#SBATCH --error=err/smoke_nvfp4_%j.err

set -euo pipefail

echo "Running on $(hostname)"

module load cuda/12.8
source ~/run/miniconda3/etc/profile.d/conda.sh
conda activate wja-cospaq

# Point to the CUTLASS source tree you built cutlass_profiler from.
# Default matches the layout used by cutlass_5090_my/test/common.sh.
export CUTLASS_ROOT="${CUTLASS_ROOT:-/data/home/scxj523/run/wja/cutlass}"

cd /data/home/scxj523/run/wja/project/my/fake/

PYTHONPATH=. python -c "
import torch
from fake.kernels import kernel_info
from fake.kernels.modules import NVFP4Linear
import torch.nn as nn

print('=== kernel_info ===')
for k, v in kernel_info().items():
    print(f'  {k}: {v}')

assert torch.cuda.is_available()
cap = torch.cuda.get_device_capability(0)
print(f'device capability: {cap}')
if cap[0] < 12:
    print('!! skipping kernel smoke: need sm_120+ for NVFP4')
    raise SystemExit(0)

device = torch.device('cuda')

# Build a layer that meets NVFP4 constraints:
# M, N, K aligned to 128 (CUTLASS block-scaled cooperative tile), K % 16 == 0.
in_features, out_features = 1024, 2048
lin = nn.Linear(in_features, out_features, bias=True).to(device).to(torch.bfloat16)
m = NVFP4Linear.from_linear(lin).to(device)

x = torch.randn(256, in_features, device=device, dtype=torch.bfloat16)

y_kernel = m(x)
y_ref    = m._fallback_forward(x)

err = (y_kernel.float() - y_ref.float()).abs()
print(f'kernel out: shape={tuple(y_kernel.shape)} dtype={y_kernel.dtype}')
print(f'err max: {err.max().item():.4f}  mean: {err.mean().item():.4f}')
print('  (kernel/fallback differ because kernel additionally quantizes activation to NVFP4)')

# Sanity: no NaN / Inf
assert torch.isfinite(y_kernel).all(), 'kernel output has NaN/Inf'
print('=== NVFP4 kernel smoke PASSED ===')
"
