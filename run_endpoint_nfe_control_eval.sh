#!/usr/bin/env bash
# Evaluation-only endpoint/NFE control at checkpoint 20.
#
# Evaluate old VP, matched VP F, and normalized-FOX G at both common and
# extended endpoints with NFE=1077. Together with their existing NFE=1000
# evaluations, this completes the 2x2 endpoint x NFE table. No training.
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${ROOT_DIR}"
exec > >(tee -a "${ROOT_DIR}/endpoint_nfe_control_eval.log") 2>&1

export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-9.0}"
export TFDS_DATA_DIR="${TFDS_DATA_DIR:-/workspace/score_sde_pytorch/.tfds}"
export TORCH_EXTENSIONS_DIR="${TORCH_EXTENSIONS_DIR:-/workspace/score_sde_pytorch/.torch_extensions}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-/workspace/score_sde_pytorch/.cache}"
export MAX_JOBS="${MAX_JOBS:-8}"
mkdir -p "${TFDS_DATA_DIR}" "${TORCH_EXTENSIONS_DIR}" "${XDG_CACHE_HOME}"

NUM_SAMPLES="${ENDPOINT_NFE_CONTROL_NUM_SAMPLES:-5000}"
NFE="${ENDPOINT_NFE_CONTROL_NFE:-1077}"
CHECKPOINT="${ENDPOINT_NFE_CONTROL_CKPT:-20}"
ALLOW_TF32="${FACTORIAL_ALLOW_TF32:-True}"
CUDNN_BENCHMARK="${FACTORIAL_CUDNN_BENCHMARK:-True}"

OLD_VP_WORKDIR="${SWEEP_VP_WORKDIR:-workdirs/celeba_vp_continuous_50k}"
F_WORKDIR="${WAVE2_F_WORKDIR:-workdirs/celeba_wave2_F_vp_pvp_vptime_common_100k}"
G_WORKDIR="${WAVE2_G_WORKDIR:-workdirs/celeba_wave2_G_normfox_pfox_foxtime_common_100k}"

COMMON_LOGSNR_MIN="-10.0"
COMMON_LOGSNR_MAX="9.115429865459795"
EXTENDED_LOGSNR_MAX="10.575749165617202"

# Use an eval-only config outside the repository. This avoids changing the
# Python source fingerprint embedded in the completed training runs.
TMP_CONFIG="$(mktemp /tmp/score_sde_endpoint_nfe_control.XXXXXX)"
trap 'rm -f "${TMP_CONFIG}"' EXIT

{
  printf '%s\n' '"""Ephemeral endpoint/NFE-control evaluation configs."""'
  printf '%s\n' 'from configs.vp.celeba_ncsnpp_continuous import get_config as get_old_vp_base'
  printf '%s\n' 'from configs.factorial.celeba_a_vp_pvp import get_config as get_f_base'
  printf '%s\n' 'from configs.factorial.celeba_d_normfox_pfox import get_config as get_g_base'
  printf '\nCOMMON_MIN = %s\n' "${COMMON_LOGSNR_MIN}"
  printf 'COMMON_MAX = %s\n' "${COMMON_LOGSNR_MAX}"
  printf 'EXTENDED_MAX = %s\n\n' "${EXTENDED_LOGSNR_MAX}"
  printf '%s\n' \
    'def _wave2(config, arm):' \
    '  config.training.factorial_arm = arm' \
    '  config.training.n_iters = 100000' \
    '  config.training.snapshot_freq = 5000' \
    "  config.model.noise_conditioning = 'time'" \
    '  return config' \
    '' \
    'def get_config(name):' \
    "  model_name, endpoint = name.rsplit('_', 1)" \
    "  if model_name == 'old_vp':" \
    '    config = get_old_vp_base()' \
    "  elif model_name == 'f':" \
    "    config = _wave2(get_f_base(), 'F_vp_pvp_vptime_common')" \
    "  elif model_name == 'g':" \
    "    config = _wave2(get_g_base(), 'G_normfox_pfox_foxtime_common')" \
    '  else:' \
    "    raise ValueError(f'Unknown endpoint/NFE model: {model_name!r}')" \
    "  config.sampling.time_grid = 'uniform_logsnr'" \
    '  config.sampling.logsnr_min = COMMON_MIN' \
    "  if endpoint == 'common':" \
    '    config.sampling.logsnr_max = COMMON_MAX' \
    "  elif endpoint == 'extended':" \
    '    config.sampling.logsnr_max = EXTENDED_MAX' \
    '  else:' \
    "    raise ValueError(f'Unknown endpoint/NFE endpoint: {endpoint!r}')" \
    '  return config'
} > "${TMP_CONFIG}"

MODELS=(old_vp f g)
WORKDIRS=("${OLD_VP_WORKDIR}" "${F_WORKDIR}" "${G_WORKDIR}")
ENDPOINTS=(common extended)

for workdir in "${WORKDIRS[@]}"; do
  checkpoint_path="${workdir}/checkpoints/checkpoint_${CHECKPOINT}.pth"
  if [[ ! -f "${checkpoint_path}" ]]; then
    echo "Missing ${checkpoint_path}; refusing to start a partial NFE control." >&2
    exit 1
  fi
done

run_eval() {
  local model_name="$1"
  local endpoint="$2"
  local workdir="$3"
  local folder="$4"

  echo "=== Endpoint/NFE control: ${model_name}, ${endpoint}, checkpoint ${CHECKPOINT}, NFE=${NFE}, samples=${NUM_SAMPLES} ==="
  python -u main.py \
    --mode=eval \
    --config="${TMP_CONFIG}:${model_name}_${endpoint}" \
    --workdir="${workdir}" \
    --eval_folder="${folder}" \
    --config.eval.begin_ckpt="${CHECKPOINT}" \
    --config.eval.end_ckpt="${CHECKPOINT}" \
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

echo "=== Endpoint/NFE-control evaluation started: $(date --iso-8601=seconds) ==="
echo "Common endpoints: [${COMMON_LOGSNR_MIN}, ${COMMON_LOGSNR_MAX}]"
echo "Extended endpoints: [${COMMON_LOGSNR_MIN}, ${EXTENDED_LOGSNR_MAX}]"
echo "Checkpoint: ${CHECKPOINT}; NFE: ${NFE}"

for index in "${!MODELS[@]}"; do
  for endpoint in "${ENDPOINTS[@]}"; do
    if [[ "${endpoint}" == "common" ]]; then
      endpoint_max="${COMMON_LOGSNR_MAX}"
    else
      endpoint_max="${EXTENDED_LOGSNR_MAX}"
    fi
    folder="eval_endpoint_nfe_control_${endpoint}_max${endpoint_max}_nfe${NFE}_${NUM_SAMPLES}samples"
    run_eval "${MODELS[index]}" "${endpoint}" "${WORKDIRS[index]}" "${folder}"
  done
done

echo "=== Endpoint/NFE-control evaluation finished: $(date --iso-8601=seconds) ==="
