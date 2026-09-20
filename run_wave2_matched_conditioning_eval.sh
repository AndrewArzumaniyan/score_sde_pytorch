#!/usr/bin/env bash
# Wave 2 / evaluation: evaluate F and G at 50k and 100k with the same common
# log-SNR endpoints, uniform-log-SNR grid, NFE=1000, and sampling seed 0.
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${ROOT_DIR}"
exec > >(tee -a "${ROOT_DIR}/wave2_matched_conditioning_eval.log") 2>&1

export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-9.0}"
export TFDS_DATA_DIR="${TFDS_DATA_DIR:-/workspace/score_sde_pytorch/.tfds}"
export TORCH_EXTENSIONS_DIR="${TORCH_EXTENSIONS_DIR:-/workspace/score_sde_pytorch/.torch_extensions}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-/workspace/score_sde_pytorch/.cache}"
export MAX_JOBS="${MAX_JOBS:-8}"
mkdir -p "${TFDS_DATA_DIR}" "${TORCH_EXTENSIONS_DIR}" "${XDG_CACHE_HOME}"

TRAIN_ITERS="${WAVE2_TRAIN_ITERS:-100000}"
SNAPSHOT_FREQ="${FACTORIAL_SNAPSHOT_FREQ:-5000}"
NUM_SAMPLES="${WAVE2_NUM_SAMPLES:-5000}"
NFE="${WAVE2_NFE:-1000}"
ALLOW_TF32="${FACTORIAL_ALLOW_TF32:-True}"
CUDNN_BENCHMARK="${FACTORIAL_CUDNN_BENCHMARK:-True}"
read -r -a CHECKPOINTS <<< "${WAVE2_EVAL_CKPTS:-10 20}"

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

run_eval() {
  local arm="$1"
  local arm_name="$2"
  local config="$3"
  local workdir="$4"
  local checkpoint="$5"
  local folder="$6"

  local checkpoint_path="${workdir}/checkpoints/checkpoint_${checkpoint}.pth"
  if [[ ! -f "${checkpoint_path}" ]]; then
    echo "Missing ${checkpoint_path}; refusing to wait indefinitely." >&2
    exit 1
  fi

  echo "=== Wave 2 eval ${arm}: checkpoint ${checkpoint}, NFE=${NFE} ==="
  python -u main.py \
    --mode=eval \
    --config="${config}" \
    --workdir="${workdir}" \
    --eval_folder="${folder}" \
    --config.training.factorial_arm="${arm_name}" \
    --config.training.n_iters="${TRAIN_ITERS}" \
    --config.training.snapshot_freq="${SNAPSHOT_FREQ}" \
    --config.model.noise_conditioning=time \
    --config.eval.begin_ckpt="${checkpoint}" \
    --config.eval.end_ckpt="${checkpoint}" \
    --config.eval.batch_size=512 \
    --config.eval.num_samples="${NUM_SAMPLES}" \
    --config.eval.enable_loss=False \
    --config.eval.enable_sampling=True \
    --config.eval.enable_bpd=False \
    --config.eval.sampling_num_scales="${NFE}" \
    --config.eval.sampling_seed=0 \
    --config.sampling.method=pc \
    --config.sampling.predictor=euler_maruyama \
    --config.sampling.corrector=none \
    --config.sampling.time_grid=uniform_logsnr \
    --config.allow_tf32="${ALLOW_TF32}" \
    --config.cudnn_benchmark="${CUDNN_BENCHMARK}"
}

echo "=== Wave 2 matched-conditioning evaluation started: $(date --iso-8601=seconds) ==="
for index in "${!ARMS[@]}"; do
  folder="eval_wave2_matched_time_logsnr_nfe${NFE}_${NUM_SAMPLES}samples"
  for checkpoint in "${CHECKPOINTS[@]}"; do
    run_eval \
      "${ARMS[index]}" "${ARM_NAMES[index]}" "${CONFIGS[index]}" \
      "${WORKDIRS[index]}" "${checkpoint}" "${folder}"
  done
done
echo "=== Wave 2 matched-conditioning evaluation finished: $(date --iso-8601=seconds) ==="

