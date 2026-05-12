#!/bin/bash
#SBATCH --job-name=bench_compress
#SBATCH --partition=gpu_5090
#SBATCH --gpus=1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --output=out/bench_compress_%j.out
#SBATCH --error=err/bench_compress_%j.err

set -euo pipefail

echo "Running on $(hostname)"

module load cuda/12.8
source ~/run/miniconda3/etc/profile.d/conda.sh
conda activate wja-cospaq

export HF_HOME=/data/home/scxj523/.cache/huggingface/
export HF_DATASETS_OFFLINE="1"
export TRANSFORMERS_OFFLINE="1"

# When KERNEL=1, the kernel modules JIT-compile via CUTLASS_ROOT (same convention
# as cutlass_5090_my/test/common.sh). Set to your CUTLASS source tree.
export CUTLASS_ROOT="${CUTLASS_ROOT:-/data/home/scxj523/run/wja/cutlass}"

cd /data/home/scxj523/run/wja/project/my/fake/

MODEL="${MODEL:-maxvit}"
MAXVIT_VARIANT="${MAXVIT_VARIANT:-tiny}"
METHOD="${METHOD:-nvfp4}"
# KERNEL=1 → load CUTLASS kernels (needs CUTLASS_ROOT). KERNEL=0 → dequant fallback.
KERNEL_FLAG=""
if [[ "${KERNEL:-0}" == "1" ]]; then
  KERNEL_FLAG="--kernel"
fi

if [[ "${MODEL}" == "maxvit" ]]; then
  CHECKPOINT="${CHECKPOINT:-artifacts/checkpoints/maxvit_${MAXVIT_VARIANT}/${METHOD}/model.pt}"
  MASKS="${MASKS:-artifacts/checkpoints/maxvit_${MAXVIT_VARIANT}/${METHOD}/masks.pt}"
  if [[ "${MAXVIT_VARIANT}" == "large" ]]; then
    DEFAULT_BATCH_SIZE=16
  else
    DEFAULT_BATCH_SIZE=128
  fi
  BATCH_SIZE="${BATCH_SIZE:-${DEFAULT_BATCH_SIZE}}"
  PYTHONPATH=. python scripts/bench_maxvit_dense_speed.py \
    --variant "${MAXVIT_VARIANT}" \
    --batch-size "${BATCH_SIZE}" \
    --checkpoint "${CHECKPOINT}" \
    --method "${METHOD}" \
    --masks "${MASKS}" \
    ${KERNEL_FLAG} \
    --output "artifacts/results/maxvit_${MAXVIT_VARIANT}_compressed/speed.csv"
elif [[ "${MODEL}" == "dinov3_vit7b16" ]]; then
  CHECKPOINT="${CHECKPOINT:-artifacts/checkpoints/${MODEL}/${METHOD}/model.pt}"
  MASKS="${MASKS:-artifacts/checkpoints/${MODEL}/${METHOD}/masks.pt}"
  PYTHONPATH=. python scripts/bench_dinov3_vit7b16_dense_speed.py \
    --checkpoint "${CHECKPOINT}" \
    --method "${METHOD}" \
    --masks "${MASKS}" \
    ${KERNEL_FLAG} \
    --output "artifacts/results/dinov3_vit7b16_compressed/speed.csv"
else
  echo "Unsupported MODEL=${MODEL}" >&2
  exit 1
fi
