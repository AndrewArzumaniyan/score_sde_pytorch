#!/usr/bin/env bash
# Wave 1 / evaluation:
#   1. evaluate the old time-conditioned VP and normalized-FOX at checkpoint 10;
#   2. evaluate the continued controlled A and B at checkpoint 20.
# Existing old checkpoint-20 and controlled checkpoint-10 results are the
# reference values from the completed factorial/sampler run.
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${ROOT_DIR}"
exec > >(tee -a "${ROOT_DIR}/wave1_eval_old_and_ab_100k.log") 2>&1

export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-9.0}"
export TFDS_DATA_DIR="${TFDS_DATA_DIR:-/workspace/score_sde_pytorch/.tfds}"
export TORCH_EXTENSIONS_DIR="${TORCH_EXTENSIONS_DIR:-/workspace/score_sde_pytorch/.torch_extensions}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-/workspace/score_sde_pytorch/.cache}"
export MAX_JOBS="${MAX_JOBS:-8}"
mkdir -p "${TFDS_DATA_DIR}" "${TORCH_EXTENSIONS_DIR}" "${XDG_CACHE_HOME}"

NUM_SAMPLES="${WAVE1_NUM_SAMPLES:-5000}"
NFE="${WAVE1_NFE:-1000}"
OLD_CKPT="${WAVE1_OLD_CKPT:-10}"
NEW_CKPT="${WAVE1_NEW_CKPT:-20}"
ALLOW_TF32="${FACTORIAL_ALLOW_TF32:-True}"
CUDNN_BENCHMARK="${FACTORIAL_CUDNN_BENCHMARK:-True}"

OLD_VP_CONFIG="configs/vp/celeba_ncsnpp_continuous.py"
OLD_VP_WORKDIR="${SWEEP_VP_WORKDIR:-workdirs/celeba_vp_continuous_50k}"
OLD_FOX_CONFIG="configs/fox/celeba_ncsnpp_continuous_normalized.py"
OLD_FOX_WORKDIR="${SWEEP_NORMFOX_WORKDIR:-workdirs/celeba_fox_normalized_kappa1000_100k}"

run_eval() {
  local name="$1"
  local config="$2"
  local workdir="$3"
  local checkpoint="$4"
  local folder="$5"

  local checkpoint_path="${workdir}/checkpoints/checkpoint_${checkpoint}.pth"
  if [[ ! -f "${checkpoint_path}" ]]; then
    echo "Missing ${checkpoint_path}; refusing to wait indefinitely." >&2
    exit 1
  fi

  echo "=== Wave 1 eval ${name}: checkpoint ${checkpoint}, NFE=${NFE} ==="
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
    --config.eval.sampling_num_scales="${NFE}" \
    --config.eval.sampling_seed=0 \
    --config.sampling.method=pc \
    --config.sampling.predictor=euler_maruyama \
    --config.sampling.corrector=none \
    --config.sampling.time_grid=uniform_logsnr \
    --config.allow_tf32="${ALLOW_TF32}" \
    --config.cudnn_benchmark="${CUDNN_BENCHMARK}"
}

echo "=== Wave 1 evaluation started: $(date --iso-8601=seconds) ==="

# These retain each old model's native log-SNR endpoints. Together with the
# already available checkpoint-20 sweep, they isolate the 50k -> 100k effect.
run_eval \
  old_vp "${OLD_VP_CONFIG}" "${OLD_VP_WORKDIR}" "${OLD_CKPT}" \
  "eval_wave1_old_ckpt${OLD_CKPT}_logsnr_nfe${NFE}_${NUM_SAMPLES}samples"
run_eval \
  old_normalized_fox "${OLD_FOX_CONFIG}" "${OLD_FOX_WORKDIR}" "${OLD_CKPT}" \
  "eval_wave1_old_ckpt${OLD_CKPT}_logsnr_nfe${NFE}_${NUM_SAMPLES}samples"

# The controlled A/B configs retain their common endpoints and log-SNR labels.
run_eval \
  controlled_A_100k configs/factorial/celeba_a_vp_pvp.py \
  workdirs/celeba_factorial_A_vp_pvp_50k "${NEW_CKPT}" \
  "eval_wave1_100k_logsnr_nfe${NFE}_${NUM_SAMPLES}samples"
run_eval \
  controlled_B_100k configs/factorial/celeba_b_vp_pfox.py \
  workdirs/celeba_factorial_B_vp_pfox_50k "${NEW_CKPT}" \
  "eval_wave1_100k_logsnr_nfe${NFE}_${NUM_SAMPLES}samples"

echo "=== Wave 1 evaluation finished: $(date --iso-8601=seconds) ==="

