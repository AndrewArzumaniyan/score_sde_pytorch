#!/usr/bin/env bash
# Evaluation-only endpoint-swap experiment (no training):
#   1. old normalized-FOX -> common factorial endpoints;
#   2. controlled G       -> extended/native FOX clean endpoint;
#   3. matched VP F       -> the same extended clean endpoint (control).
#
# By default, checkpoints 10 (50k) and 20 (100k) are evaluated with 5k
# samples, uniform-log-SNR Euler-Maruyama sampling, NFE=1000, and seed 0.
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${ROOT_DIR}"
exec > >(tee -a "${ROOT_DIR}/endpoint_swap_eval.log") 2>&1

export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-9.0}"
export TFDS_DATA_DIR="${TFDS_DATA_DIR:-/workspace/score_sde_pytorch/.tfds}"
export TORCH_EXTENSIONS_DIR="${TORCH_EXTENSIONS_DIR:-/workspace/score_sde_pytorch/.torch_extensions}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-/workspace/score_sde_pytorch/.cache}"
export MAX_JOBS="${MAX_JOBS:-8}"
mkdir -p "${TFDS_DATA_DIR}" "${TORCH_EXTENSIONS_DIR}" "${XDG_CACHE_HOME}"

NUM_SAMPLES="${ENDPOINT_SWAP_NUM_SAMPLES:-5000}"
NFE="${ENDPOINT_SWAP_NFE:-1000}"
ALLOW_TF32="${FACTORIAL_ALLOW_TF32:-True}"
CUDNN_BENCHMARK="${FACTORIAL_CUDNN_BENCHMARK:-True}"
read -r -a CHECKPOINTS <<< "${ENDPOINT_SWAP_CKPTS:-10 20}"

OLD_FOX_WORKDIR="${SWEEP_NORMFOX_WORKDIR:-workdirs/celeba_fox_normalized_kappa1000_100k}"
F_WORKDIR="${WAVE2_F_WORKDIR:-workdirs/celeba_wave2_F_vp_pvp_vptime_common_100k}"
G_WORKDIR="${WAVE2_G_WORKDIR:-workdirs/celeba_wave2_G_normfox_pfox_foxtime_common_100k}"

# Exact endpoints used by the completed experiments.
COMMON_LOGSNR_MIN="-10.0"
COMMON_LOGSNR_MAX="9.115429865459795"
FOX_NATIVE_LOGSNR_MAX="10.575749165617202"

if [[ "${#CHECKPOINTS[@]}" -eq 0 ]]; then
  echo "ENDPOINT_SWAP_CKPTS must contain at least one checkpoint number." >&2
  exit 2
fi

# Config fields cannot be added through ml_collections CLI overrides because
# main.py locks configs.  Build one ephemeral eval-only config outside the
# repository, so the experiment does not alter the source fingerprint of any
# completed training run.  The resolved config is still stored in each eval
# manifest.
TMP_CONFIG="$(mktemp /tmp/score_sde_endpoint_swap.XXXXXX)"
trap 'rm -f "${TMP_CONFIG}"' EXIT

{
  printf '%s\n' '"""Ephemeral configs for the endpoint-swap evaluation."""'
  printf '%s\n' 'from configs.factorial.celeba_a_vp_pvp import get_config as get_f_base'
  printf '%s\n' 'from configs.factorial.celeba_d_normfox_pfox import get_config as get_g_base'
  printf '%s\n' 'from configs.fox.celeba_ncsnpp_continuous_normalized import get_config as get_old_fox_base'
  printf '\nCOMMON_MIN = %s\n' "${COMMON_LOGSNR_MIN}"
  printf 'COMMON_MAX = %s\n' "${COMMON_LOGSNR_MAX}"
  printf 'FOX_NATIVE_MAX = %s\n\n' "${FOX_NATIVE_LOGSNR_MAX}"
  printf '%s\n' \
    'def _sampling(config, lower, upper):' \
    "  config.sampling.time_grid = 'uniform_logsnr'" \
    '  config.sampling.logsnr_min = lower' \
    '  config.sampling.logsnr_max = upper' \
    '  return config' \
    '' \
    'def _wave2(config, arm):' \
    '  config.training.factorial_arm = arm' \
    '  config.training.n_iters = 100000' \
    '  config.training.snapshot_freq = 5000' \
    "  config.model.noise_conditioning = 'time'" \
    '  return config' \
    '' \
    'def get_config(name):' \
    "  if name == 'old_fox_common':" \
    '    return _sampling(get_old_fox_base(), COMMON_MIN, COMMON_MAX)' \
    "  if name == 'f_vp_extended':" \
    "    config = _wave2(get_f_base(), 'F_vp_pvp_vptime_common')" \
    '    return _sampling(config, COMMON_MIN, FOX_NATIVE_MAX)' \
    "  if name == 'g_fox_extended':" \
    "    config = _wave2(get_g_base(), 'G_normfox_pfox_foxtime_common')" \
    '    return _sampling(config, COMMON_MIN, FOX_NATIVE_MAX)' \
    "  raise ValueError(f'Unknown endpoint-swap arm: {name!r}')"
} > "${TMP_CONFIG}"

ARMS=(old_fox_common g_fox_extended f_vp_extended)
WORKDIRS=("${OLD_FOX_WORKDIR}" "${G_WORKDIR}" "${F_WORKDIR}")
FOLDERS=(
  "eval_endpoint_swap_common_max${COMMON_LOGSNR_MAX}_logsnr_nfe${NFE}_${NUM_SAMPLES}samples"
  "eval_endpoint_swap_foxmax${FOX_NATIVE_LOGSNR_MAX}_logsnr_nfe${NFE}_${NUM_SAMPLES}samples"
  "eval_endpoint_swap_foxmax${FOX_NATIVE_LOGSNR_MAX}_logsnr_nfe${NFE}_${NUM_SAMPLES}samples"
)

# Fail before launching any expensive evaluation if a requested checkpoint is
# absent.  This also catches a wrong bind mount or workdir override early.
for index in "${!ARMS[@]}"; do
  for checkpoint in "${CHECKPOINTS[@]}"; do
    checkpoint_path="${WORKDIRS[index]}/checkpoints/checkpoint_${checkpoint}.pth"
    if [[ ! -f "${checkpoint_path}" ]]; then
      echo "Missing ${checkpoint_path}; refusing to start a partial sweep." >&2
      exit 1
    fi
  done
done

run_eval() {
  local arm="$1"
  local workdir="$2"
  local folder="$3"
  local checkpoint="$4"

  echo "=== Endpoint swap: ${arm}, checkpoint ${checkpoint}, NFE=${NFE}, samples=${NUM_SAMPLES} ==="
  python -u main.py \
    --mode=eval \
    --config="${TMP_CONFIG}:${arm}" \
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
    --config.allow_tf32="${ALLOW_TF32}" \
    --config.cudnn_benchmark="${CUDNN_BENCHMARK}"
}

echo "=== Endpoint-swap evaluation started: $(date --iso-8601=seconds) ==="
echo "Common endpoints: [${COMMON_LOGSNR_MIN}, ${COMMON_LOGSNR_MAX}]"
echo "Extended endpoints: [${COMMON_LOGSNR_MIN}, ${FOX_NATIVE_LOGSNR_MAX}]"
echo "Checkpoints: ${CHECKPOINTS[*]}"

for index in "${!ARMS[@]}"; do
  for checkpoint in "${CHECKPOINTS[@]}"; do
    run_eval \
      "${ARMS[index]}" "${WORKDIRS[index]}" \
      "${FOLDERS[index]}" "${checkpoint}"
  done
done

echo "=== Endpoint-swap evaluation finished: $(date --iso-8601=seconds) ==="
