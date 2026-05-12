#!/bin/bash
#SBATCH --job-name=maxvit_acc
#SBATCH --partition=gpu_5090
#SBATCH --gpus=1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --output=out/maxvit_acc_%j.out
#SBATCH --error=err/maxvit_acc_%j.err

set -euo pipefail

echo "Running on $(hostname)"

module load cuda/12.8
source ~/run/miniconda3/etc/profile.d/conda.sh
conda activate wja-cospaq

export HF_HOME=/data/home/scxj523/.cache/huggingface/
export HF_DATASETS_OFFLINE="1"

# When KERNEL=1, the kernel modules JIT-compile via CUTLASS_ROOT (same convention
# as cutlass_5090_my/test/common.sh). Set to your CUTLASS source tree.
export CUTLASS_ROOT="${CUTLASS_ROOT:-/data/home/scxj523/run/wja/cutlass}"

cd /data/home/scxj523/run/wja/project/my/fake/

MAXVIT_VARIANT="${MAXVIT_VARIANT:-tiny}"
METHOD="${METHOD:-dense}"
CHECKPOINT="${CHECKPOINT:-}"
MASKS="${MASKS:-}"

if [[ "${MAXVIT_VARIANT}" == "large" ]]; then
  DEFAULT_BATCH_SIZE=16
else
  DEFAULT_BATCH_SIZE=128
fi
BATCH_SIZE="${BATCH_SIZE:-${DEFAULT_BATCH_SIZE}}"
NUM_WORKERS="${NUM_WORKERS:-8}"

# KERNEL=1 → materialize CUTLASS kernel modules (needs CUTLASS_ROOT). KERNEL=0 → dequant fallback or pure dense.
KERNEL_FLAG=""
if [[ "${KERNEL:-0}" == "1" ]]; then
  KERNEL_FLAG="--kernel"
fi

# Default checkpoint/masks paths when METHOD != dense and caller did not override.
if [[ "${METHOD}" != "dense" && -z "${CHECKPOINT}" ]]; then
  CHECKPOINT="artifacts/checkpoints/maxvit_${MAXVIT_VARIANT}/${METHOD}/model.pt"
fi
if [[ "${METHOD}" != "dense" && -z "${MASKS}" ]]; then
  MASKS="artifacts/checkpoints/maxvit_${MAXVIT_VARIANT}/${METHOD}/masks.pt"
fi

CHECKPOINT_ARG=()
if [[ -n "${CHECKPOINT}" ]]; then
  CHECKPOINT_ARG=(--checkpoint "${CHECKPOINT}")
fi
MASKS_ARG=()
if [[ -n "${MASKS}" ]]; then
  MASKS_ARG=(--masks "${MASKS}")
fi

if [[ "${METHOD}" == "dense" ]]; then
  OUTPUT="artifacts/results/maxvit_${MAXVIT_VARIANT}_dense/accuracy.csv"
else
  OUTPUT="artifacts/results/maxvit_${MAXVIT_VARIANT}_compressed/accuracy.csv"
fi

PYTHONPATH=. python scripts/eval_maxvit_dense_accuracy.py \
  --variant "${MAXVIT_VARIANT}" \
  --batch-size "${BATCH_SIZE}" \
  --num-workers "${NUM_WORKERS}" \
  --dtype auto \
  --method "${METHOD}" \
  "${CHECKPOINT_ARG[@]}" \
  "${MASKS_ARG[@]}" \
  ${KERNEL_FLAG} \
  --output "${OUTPUT}"
