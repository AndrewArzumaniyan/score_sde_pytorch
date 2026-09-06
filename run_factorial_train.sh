#!/usr/bin/env bash
# Sequential 50k-step training of the controlled CelebA arms A-E.
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${ROOT_DIR}"
exec > >(tee -a "${ROOT_DIR}/factorial_train.log") 2>&1

export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-9.0}"
export TFDS_DATA_DIR="${TFDS_DATA_DIR:-/workspace/score_sde_pytorch/.tfds}"
export TORCH_EXTENSIONS_DIR="${TORCH_EXTENSIONS_DIR:-/workspace/score_sde_pytorch/.torch_extensions}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-/workspace/score_sde_pytorch/.cache}"
export MAX_JOBS="${MAX_JOBS:-8}"
mkdir -p "${TFDS_DATA_DIR}" "${TORCH_EXTENSIONS_DIR}" "${XDG_CACHE_HOME}" workdirs

TRAIN_ITERS="${FACTORIAL_TRAIN_ITERS:-50000}"
SNAPSHOT_FREQ="${FACTORIAL_SNAPSHOT_FREQ:-5000}"
SEED="${FACTORIAL_SEED:-42}"
ALLOW_TF32="${FACTORIAL_ALLOW_TF32:-True}"
CUDNN_BENCHMARK="${FACTORIAL_CUDNN_BENCHMARK:-True}"

ARMS=(A B C D E)
CONFIGS=(
  configs/factorial/celeba_a_vp_pvp.py
  configs/factorial/celeba_b_vp_pfox.py
  configs/factorial/celeba_c_normfox_pvp.py
  configs/factorial/celeba_d_normfox_pfox.py
  configs/factorial/celeba_e_realfox_pfox.py
)
WORKDIRS=(
  workdirs/celeba_factorial_A_vp_pvp_50k
  workdirs/celeba_factorial_B_vp_pfox_50k
  workdirs/celeba_factorial_C_normfox_pvp_50k
  workdirs/celeba_factorial_D_normfox_pfox_50k
  workdirs/celeba_factorial_E_realfox_pfox_50k
)

echo "=== Factorial training started: $(date --iso-8601=seconds) ==="
for index in "${!ARMS[@]}"; do
  echo "=== Training arm ${ARMS[index]} -> ${WORKDIRS[index]} ==="
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
echo "=== Factorial training finished: $(date --iso-8601=seconds) ==="
