#!/usr/bin/env bash
# Evaluation-only old-VP endpoint pair.
#
# For checkpoints 10 (50k) and 20 (100k), evaluate exactly the same old VP
# model at:
#   common:   lambda in [-10, 9.115429865459795]
#   extended: lambda in [-10, 10.575749165617202]
#
# Only the clean endpoint changes inside each checkpoint pair. No training.
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${ROOT_DIR}"
exec > >(tee -a "${ROOT_DIR}/old_vp_endpoint_pair_eval.log") 2>&1

export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-9.0}"
export TFDS_DATA_DIR="${TFDS_DATA_DIR:-/workspace/score_sde_pytorch/.tfds}"
export TORCH_EXTENSIONS_DIR="${TORCH_EXTENSIONS_DIR:-/workspace/score_sde_pytorch/.torch_extensions}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-/workspace/score_sde_pytorch/.cache}"
export MAX_JOBS="${MAX_JOBS:-8}"
mkdir -p "${TFDS_DATA_DIR}" "${TORCH_EXTENSIONS_DIR}" "${XDG_CACHE_HOME}"

NUM_SAMPLES="${OLD_VP_ENDPOINT_NUM_SAMPLES:-5000}"
NFE="${OLD_VP_ENDPOINT_NFE:-1000}"
ALLOW_TF32="${FACTORIAL_ALLOW_TF32:-True}"
CUDNN_BENCHMARK="${FACTORIAL_CUDNN_BENCHMARK:-True}"
read -r -a CHECKPOINTS <<< "${OLD_VP_ENDPOINT_CKPTS:-10 20}"

OLD_VP_WORKDIR="${SWEEP_VP_WORKDIR:-workdirs/celeba_vp_continuous_50k}"
COMMON_LOGSNR_MIN="-10.0"
COMMON_LOGSNR_MAX="9.115429865459795"
EXTENDED_LOGSNR_MAX="10.575749165617202"

if [[ "${#CHECKPOINTS[@]}" -eq 0 ]]; then
  echo "OLD_VP_ENDPOINT_CKPTS must contain at least one checkpoint number." >&2
  exit 2
fi

# sampling.logsnr_min/max do not exist in the legacy VP config, and main.py
# locks configs against adding fields from the CLI. Create an ephemeral config
# outside the repository so completed training source fingerprints stay intact.
TMP_CONFIG="$(mktemp /tmp/score_sde_old_vp_endpoint_pair.XXXXXX)"
trap 'rm -f "${TMP_CONFIG}"' EXIT

{
  printf '%s\n' '"""Ephemeral old-VP endpoint-pair evaluation configs."""'
  printf '%s\n' 'from configs.vp.celeba_ncsnpp_continuous import get_config as get_base_config'
  printf '\nCOMMON_MIN = %s\n' "${COMMON_LOGSNR_MIN}"
  printf 'COMMON_MAX = %s\n' "${COMMON_LOGSNR_MAX}"
  printf 'EXTENDED_MAX = %s\n\n' "${EXTENDED_LOGSNR_MAX}"
  printf '%s\n' \
    'def get_config(name):' \
    '  config = get_base_config()' \
    "  config.sampling.time_grid = 'uniform_logsnr'" \
    '  config.sampling.logsnr_min = COMMON_MIN' \
    "  if name == 'common':" \
    '    config.sampling.logsnr_max = COMMON_MAX' \
    "  elif name == 'extended':" \
    '    config.sampling.logsnr_max = EXTENDED_MAX' \
    '  else:' \
    "    raise ValueError(f'Unknown old-VP endpoint variant: {name!r}')" \
    '  return config'
} > "${TMP_CONFIG}"

for checkpoint in "${CHECKPOINTS[@]}"; do
  checkpoint_path="${OLD_VP_WORKDIR}/checkpoints/checkpoint_${checkpoint}.pth"
  if [[ ! -f "${checkpoint_path}" ]]; then
    echo "Missing ${checkpoint_path}; refusing to start a partial endpoint pair." >&2
    exit 1
  fi
done

run_eval() {
  local variant="$1"
  local endpoint_max="$2"
  local checkpoint="$3"
  local folder="$4"

  echo "=== Old VP ${variant}: checkpoint ${checkpoint}, lambda_max=${endpoint_max}, NFE=${NFE}, samples=${NUM_SAMPLES} ==="
  python -u main.py \
    --mode=eval \
    --config="${TMP_CONFIG}:${variant}" \
    --workdir="${OLD_VP_WORKDIR}" \
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

COMMON_FOLDER="eval_old_vp_endpoint_common_logsnr_nfe${NFE}_${NUM_SAMPLES}samples"
EXTENDED_FOLDER="eval_old_vp_endpoint_extended_logsnr_nfe${NFE}_${NUM_SAMPLES}samples"

echo "=== Old-VP endpoint-pair evaluation started: $(date --iso-8601=seconds) ==="
echo "Common endpoints: [${COMMON_LOGSNR_MIN}, ${COMMON_LOGSNR_MAX}]"
echo "Extended endpoints: [${COMMON_LOGSNR_MIN}, ${EXTENDED_LOGSNR_MAX}]"
echo "Checkpoints: ${CHECKPOINTS[*]}"

for checkpoint in "${CHECKPOINTS[@]}"; do
  run_eval common "${COMMON_LOGSNR_MAX}" "${checkpoint}" "${COMMON_FOLDER}"
  run_eval extended "${EXTENDED_LOGSNR_MAX}" "${checkpoint}" "${EXTENDED_FOLDER}"
done

echo "=== Old-VP endpoint-pair evaluation finished: $(date --iso-8601=seconds) ==="
