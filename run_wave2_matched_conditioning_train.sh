#!/usr/bin/env bash
# Wave 2 / training: test matched time/quantile conditioning on the shared
# log-SNR support. F uses VP + P_VP + VP time; G uses normalized-FOX + P_FOX
# + FOX time. Both are trained to 100k so checkpoints 10 and 20 give the
# 50k/100k learning-curve comparison.
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${ROOT_DIR}"
exec > >(tee -a "${ROOT_DIR}/wave2_matched_conditioning_train.log") 2>&1

export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-9.0}"
export TFDS_DATA_DIR="${TFDS_DATA_DIR:-/workspace/score_sde_pytorch/.tfds}"
export TORCH_EXTENSIONS_DIR="${TORCH_EXTENSIONS_DIR:-/workspace/score_sde_pytorch/.torch_extensions}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-/workspace/score_sde_pytorch/.cache}"
export MAX_JOBS="${MAX_JOBS:-8}"
mkdir -p "${TFDS_DATA_DIR}" "${TORCH_EXTENSIONS_DIR}" "${XDG_CACHE_HOME}" workdirs

TRAIN_ITERS="${WAVE2_TRAIN_ITERS:-100000}"
SNAPSHOT_FREQ="${FACTORIAL_SNAPSHOT_FREQ:-5000}"
SEED="${WAVE2_SEED:-42}"
ALLOW_TF32="${FACTORIAL_ALLOW_TF32:-True}"
CUDNN_BENCHMARK="${FACTORIAL_CUDNN_BENCHMARK:-True}"

ARMS=(F G)
CONFIGS=(
  configs/factorial/celeba_a_vp_pvp.py
  configs/factorial/celeba_d_normfox_pfox.py
)
ARM_NAMES=(
  F_vp_pvp_vptime_common
  G_normfox_pfox_foxtime_common
)
WORKDIRS=(
  workdirs/celeba_wave2_F_vp_pvp_vptime_common_100k
  workdirs/celeba_wave2_G_normfox_pfox_foxtime_common_100k
)

echo "=== Wave 2 matched-conditioning training started: $(date --iso-8601=seconds) ==="
for index in "${!ARMS[@]}"; do
  echo "=== Training arm ${ARMS[index]} -> ${WORKDIRS[index]} ==="
  python -u main.py \
    --mode=train \
    --config="${CONFIGS[index]}" \
    --workdir="${WORKDIRS[index]}" \
    --config.training.factorial_arm="${ARM_NAMES[index]}" \
    --config.training.batch_size=128 \
    --config.training.n_iters="${TRAIN_ITERS}" \
    --config.training.snapshot_freq="${SNAPSHOT_FREQ}" \
    --config.training.snapshot_freq_for_preemption="${SNAPSHOT_FREQ}" \
    --config.training.snapshot_sampling=False \
    --config.training.log_freq=50 \
    --config.training.eval_freq=500 \
    --config.model.noise_conditioning=time \
    --config.seed="${SEED}" \
    --config.allow_tf32="${ALLOW_TF32}" \
    --config.cudnn_benchmark="${CUDNN_BENCHMARK}"
done
echo "=== Wave 2 matched-conditioning training finished: $(date --iso-8601=seconds) ==="

