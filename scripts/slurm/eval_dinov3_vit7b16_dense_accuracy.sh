#!/bin/bash
#SBATCH --job-name=dinov3_acc
#SBATCH --partition=gpu_5090
#SBATCH --gpus=1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --output=out/dinov3_acc_%j.out
#SBATCH --error=err/dinov3_acc_%j.err

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

METHOD="${METHOD:-dense}"
CHECKPOINT="${CHECKPOINT:-}"
MASKS="${MASKS:-}"

# KERNEL=1 → materialize CUTLASS kernel modules (needs CUTLASS_ROOT). KERNEL=0 → dequant fallback or pure dense.
KERNEL_FLAG=""
if [[ "${KERNEL:-0}" == "1" ]]; then
  KERNEL_FLAG="--kernel"
fi

# Default checkpoint/masks paths when METHOD != dense and caller did not override.
if [[ "${METHOD}" != "dense" && -z "${CHECKPOINT}" ]]; then
  CHECKPOINT="artifacts/checkpoints/dinov3_vit7b16/${METHOD}/model.pt"
fi
if [[ "${METHOD}" != "dense" && -z "${MASKS}" ]]; then
  MASKS="artifacts/checkpoints/dinov3_vit7b16/${METHOD}/masks.pt"
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
  OUTPUT="artifacts/results/dinov3_vit7b16_dense/accuracy.csv"
else
  OUTPUT="artifacts/results/dinov3_vit7b16_compressed/accuracy.csv"
fi

PYTHONPATH=. python scripts/eval_dinov3_vit7b16_dense_accuracy.py \
  --batch-size 1 \
  --num-workers 4 \
  --resize-size 256 \
  --method "${METHOD}" \
  "${CHECKPOINT_ARG[@]}" \
  "${MASKS_ARG[@]}" \
  ${KERNEL_FLAG} \
  --output "${OUTPUT}"

