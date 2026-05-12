#!/bin/bash
#SBATCH --job-name=bench_kernel
#SBATCH --partition=gpu_5090
#SBATCH --gpus=1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --output=out/bench_kernel_%j.out
#SBATCH --error=err/bench_kernel_%j.err

set -euo pipefail

echo "Running on $(hostname)"

module load cuda/12.8
source ~/run/miniconda3/etc/profile.d/conda.sh
conda activate wja-cospaq

export HF_HOME=/data/home/scxj523/.cache/huggingface/
export HF_DATASETS_OFFLINE="1"
export TRANSFORMERS_OFFLINE="1"

# Must point to compiled cutlass_profiler binary
export CUTLASS_PROFILER="${CUTLASS_PROFILER:-/data/home/scxj523/run/wja/cutlass/build/tools/profiler/cutlass_profiler}"

cd /data/home/scxj523/run/wja/project/my/fake/

MODEL="${MODEL:-maxvit}"
MAXVIT_VARIANT="${MAXVIT_VARIANT:-tiny}"

PYTHONPATH=. python scripts/bench_kernel.py \
  --model "${MODEL}" \
  --variant "${MAXVIT_VARIANT}" \
  --batch-sizes 1 8 32 128 \
  --warmup 10 \
  --iters 50 \
  --output "artifacts/results/kernel_bench/${MODEL}_${MAXVIT_VARIANT}_shape_latency.csv"
