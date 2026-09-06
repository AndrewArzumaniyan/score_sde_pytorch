#!/usr/bin/env bash
# Sequential CelebA evaluation: A-E plus the requested sampler-only control.
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${ROOT_DIR}"
exec > >(tee -a "${ROOT_DIR}/factorial_eval.log") 2>&1

export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-9.0}"
export TFDS_DATA_DIR="${TFDS_DATA_DIR:-/workspace/score_sde_pytorch/.tfds}"
export TORCH_EXTENSIONS_DIR="${TORCH_EXTENSIONS_DIR:-/workspace/score_sde_pytorch/.torch_extensions}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-/workspace/score_sde_pytorch/.cache}"
export MAX_JOBS="${MAX_JOBS:-8}"
mkdir -p "${TFDS_DATA_DIR}" "${TORCH_EXTENSIONS_DIR}" "${XDG_CACHE_HOME}" workdirs

TRAIN_ITERS="${FACTORIAL_TRAIN_ITERS:-50000}"
SNAPSHOT_FREQ="${FACTORIAL_SNAPSHOT_FREQ:-5000}"
CHECKPOINT="${FACTORIAL_CKPT:-$((TRAIN_ITERS / SNAPSHOT_FREQ))}"
NUM_SAMPLES="${FACTORIAL_NUM_SAMPLES:-5000}"
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

run_eval() {
  local config="$1"
  local workdir="$2"
  local checkpoint="$3"
  local folder="$4"
  local nfe="$5"
  local grid="$6"
  python -u main.py \
    --mode=eval \
    --config="${config}" \
    --workdir="${workdir}" \
    --eval_folder="${folder}" \
    --config.eval.begin_ckpt="${checkpoint}" \
    --config.eval.end_ckpt="${checkpoint}" \
    --config.eval.batch_size=512 \
    --config.eval.num_samples="${NUM_SAMPLES}" \
    --config.eval.enable_loss=False \
    --config.eval.enable_sampling=True \
    --config.eval.enable_bpd=False \
    --config.eval.sampling_num_scales="${nfe}" \
    --config.eval.sampling_seed=0 \
    --config.sampling.method=pc \
    --config.sampling.predictor=euler_maruyama \
    --config.sampling.corrector=none \
    --config.sampling.time_grid="${grid}" \
    --config.allow_tf32="${ALLOW_TF32}" \
    --config.cudnn_benchmark="${CUDNN_BENCHMARK}"
}

echo "=== Factorial evaluation started: $(date --iso-8601=seconds) ==="
for index in "${!ARMS[@]}"; do
  echo "=== Evaluating arm ${ARMS[index]}: log-SNR grid, NFE=1000 ==="
  run_eval \
    "${CONFIGS[index]}" "${WORKDIRS[index]}" "${CHECKPOINT}" \
    "eval_factorial_logsnr_nfe1000_${NUM_SAMPLES}samples" 1000 uniform_logsnr
done

# Sampler-only control on the pre-existing time-conditioned checkpoints.
# Set RUN_EXISTING_SAMPLER_SWEEP=False to skip it.
if [[ "${RUN_EXISTING_SAMPLER_SWEEP:-True}" == "True" ]]; then
  SWEEP_CONFIGS=(
    configs/vp/celeba_ncsnpp_continuous.py
    configs/fox/celeba_ncsnpp_continuous_normalized.py
  )
  SWEEP_WORKDIRS=(
    "${SWEEP_VP_WORKDIR:-workdirs/celeba_vp_continuous_50k}"
    "${SWEEP_NORMFOX_WORKDIR:-workdirs/celeba_fox_normalized_kappa1000_100k}"
  )
  SWEEP_CKPTS=("${SWEEP_VP_CKPT:-20}" "${SWEEP_NORMFOX_CKPT:-20}")
  SWEEP_NAMES=(vp normalized_fox)
  NFE_VALUES=(20 50 100 250 1000)
  GRIDS=(uniform_time uniform_logsnr)
  for model_index in "${!SWEEP_NAMES[@]}"; do
    for grid in "${GRIDS[@]}"; do
      for nfe in "${NFE_VALUES[@]}"; do
        echo "=== Sampler control ${SWEEP_NAMES[model_index]}: ${grid}, NFE=${nfe} ==="
        run_eval \
          "${SWEEP_CONFIGS[model_index]}" \
          "${SWEEP_WORKDIRS[model_index]}" \
          "${SWEEP_CKPTS[model_index]}" \
          "eval_sampler_${grid}_nfe${nfe}_${NUM_SAMPLES}samples" \
          "${nfe}" "${grid}"
      done
    done
  done
fi

echo "=== Factorial evaluation finished: $(date --iso-8601=seconds) ==="
