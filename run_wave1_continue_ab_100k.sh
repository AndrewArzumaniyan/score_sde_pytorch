#!/usr/bin/env bash
# Wave 1 / training: resume controlled arms A and B from 50k to 100k,
# sequentially on the single GPU visible inside the container.
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${ROOT_DIR}"
exec > >(tee -a "${ROOT_DIR}/wave1_continue_ab_100k.log") 2>&1

export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-9.0}"
export TFDS_DATA_DIR="${TFDS_DATA_DIR:-/workspace/score_sde_pytorch/.tfds}"
export TORCH_EXTENSIONS_DIR="${TORCH_EXTENSIONS_DIR:-/workspace/score_sde_pytorch/.torch_extensions}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-/workspace/score_sde_pytorch/.cache}"
export MAX_JOBS="${MAX_JOBS:-8}"
mkdir -p "${TFDS_DATA_DIR}" "${TORCH_EXTENSIONS_DIR}" "${XDG_CACHE_HOME}"

TRAIN_ITERS="${WAVE1_TRAIN_ITERS:-100000}"
SNAPSHOT_FREQ="${FACTORIAL_SNAPSHOT_FREQ:-5000}"
SOURCE_CKPT="${WAVE1_SOURCE_CKPT:-10}"
SEED="${FACTORIAL_SEED:-42}"
ALLOW_TF32="${FACTORIAL_ALLOW_TF32:-True}"
CUDNN_BENCHMARK="${FACTORIAL_CUDNN_BENCHMARK:-True}"

ARMS=(A B)
CONFIGS=(
  configs/factorial/celeba_a_vp_pvp.py
  configs/factorial/celeba_b_vp_pfox.py
)
WORKDIRS=(
  workdirs/celeba_factorial_A_vp_pvp_50k
  workdirs/celeba_factorial_B_vp_pfox_50k
)

echo "=== Wave 1 A/B continuation started: $(date --iso-8601=seconds) ==="
echo "Target: ${TRAIN_ITERS} updates; expected source checkpoint: ${SOURCE_CKPT}"

for index in "${!ARMS[@]}"; do
  checkpoint="${WORKDIRS[index]}/checkpoints/checkpoint_${SOURCE_CKPT}.pth"
  manifest="${WORKDIRS[index]}/run_manifest.json"
  if [[ ! -f "${checkpoint}" || ! -f "${manifest}" ]]; then
    echo "Refusing to start ${ARMS[index]} from scratch." >&2
    echo "Missing ${checkpoint} or ${manifest}." >&2
    exit 1
  fi

  echo "=== Continuing arm ${ARMS[index]} -> ${TRAIN_ITERS} updates ==="
  python -u main.py \
    --mode=train \
    --config="${CONFIGS[index]}" \
    --workdir="${WORKDIRS[index]}" \
    --config.training.batch_size=128 \
    --config.training.n_iters="${TRAIN_ITERS}" \
    --config.training.snapshot_freq="${SNAPSHOT_FREQ}" \
    --config.training.snapshot_freq_for_preemption="${SNAPSHOT_FREQ}" \
    --config.training.snapshot_sampling=False \
    --config.training.log_freq=50 \
    --config.training.eval_freq=500 \
    --config.seed="${SEED}" \
    --config.allow_tf32="${ALLOW_TF32}" \
    --config.cudnn_benchmark="${CUDNN_BENCHMARK}"
done

echo "=== Wave 1 A/B continuation finished: $(date --iso-8601=seconds) ==="

