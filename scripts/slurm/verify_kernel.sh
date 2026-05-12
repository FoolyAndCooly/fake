#!/bin/bash
#SBATCH --job-name=verify_kernel
#SBATCH --partition=gpu_5090
#SBATCH --gpus=1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --output=out/verify_kernel_%j.out
#SBATCH --error=err/verify_kernel_%j.err

set -euo pipefail

echo "Running on $(hostname)"

module load cuda/12.8
source ~/run/miniconda3/etc/profile.d/conda.sh
conda activate wja-cospaq

export HF_HOME=/data/home/scxj523/.cache/huggingface/
export HF_DATASETS_OFFLINE="1"
export TRANSFORMERS_OFFLINE="1"

export CUTLASS_ROOT="${CUTLASS_ROOT:-/data/home/scxj523/run/wja/cutlass}"

cd /data/home/scxj523/run/wja/project/my/fake/

echo "============================================================"
echo "  Step 1: Extension availability check"
echo "============================================================"
PYTHONPATH=. python scripts/check_kernel_availability.py

echo ""
echo "============================================================"
echo "  Step 2: Module replacement check (materialize)"
echo "============================================================"
PYTHONPATH=. python scripts/verify_kernel_materialize.py

echo ""
echo "============================================================"
echo "  Step 3: Forward path verification (kernel vs fallback)"
echo "============================================================"
PYTHONPATH=. python scripts/verify_kernel_forward.py

echo ""
echo "============================================================"
echo "  Step 4: Comprehensive verification"
echo "============================================================"
PYTHONPATH=. python scripts/verify_kernel_usage.py

echo ""
echo "============================================================"
echo "  All kernel verification steps completed"
echo "============================================================"
