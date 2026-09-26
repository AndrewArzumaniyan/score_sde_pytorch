#!/usr/bin/env bash
# Eval-only controls for the historical VP and real (unnormalized) FOX models.
# core: FOX at both endpoints, NFE 1000 (matched to completed VP evaluations).
# nfe: VP and FOX at both endpoints, NFE 1077 (grid-spacing control).
# all: both groups. No training or checkpoint changes.
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

STAGE="${ENDPOINT_1300K_STAGE:-core}"
case "$STAGE" in core|nfe|all) ;; *) echo "ENDPOINT_1300K_STAGE must be core, nfe, or all" >&2; exit 2 ;; esac
NUM_SAMPLES="${ENDPOINT_1300K_NUM_SAMPLES:-10000}"
CHECKPOINT="${ENDPOINT_1300K_CKPT:-260}"
SEED="${ENDPOINT_1300K_SEED:-0}"
ALLOW_TF32="${FACTORIAL_ALLOW_TF32:-True}"
CUDNN_BENCHMARK="${FACTORIAL_CUDNN_BENCHMARK:-True}"
VP_WORKDIR="${ENDPOINT_1300K_VP_WORKDIR:-workdirs/celeba_vp_continuous_50k}"
FOX_WORKDIR="${ENDPOINT_1300K_FOX_WORKDIR:-workdirs/celeba_fox_matern32_kappa1000_50k}"

export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-9.0}"
export TFDS_DATA_DIR="${TFDS_DATA_DIR:-/workspace/score_sde_pytorch/.tfds}"
export TORCH_EXTENSIONS_DIR="${TORCH_EXTENSIONS_DIR:-/workspace/score_sde_pytorch/.torch_extensions}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-/workspace/score_sde_pytorch/.cache}"
export MAX_JOBS="${MAX_JOBS:-8}"

if [[ "$STAGE" == core || "$STAGE" == all ]]; then
  [[ -f "$FOX_WORKDIR/checkpoints/checkpoint_${CHECKPOINT}.pth" ]] || {
    echo "Missing FOX checkpoint ${CHECKPOINT} in $FOX_WORKDIR" >&2; exit 1;
  }
fi
if [[ "$STAGE" == nfe || "$STAGE" == all ]]; then
  for workdir in "$VP_WORKDIR" "$FOX_WORKDIR"; do
    [[ -f "$workdir/checkpoints/checkpoint_${CHECKPOINT}.pth" ]] || {
      echo "Missing checkpoint ${CHECKPOINT} in $workdir" >&2; exit 1;
    }
  done
fi

# main.py locks config keys, so add eval-only sampling endpoints in a temporary
# config. The resolved config and checkpoint identity are saved in each manifest.
TMP_CONFIG="$(mktemp /tmp/score_sde_1300k_endpoint.XXXXXX)"
trap 'rm -f "$TMP_CONFIG"' EXIT
cat > "$TMP_CONFIG" <<'PY'
from configs.vp.celeba_ncsnpp_continuous import get_config as get_vp
from configs.fox.celeba_ncsnpp_continuous_matern_3_2 import get_config as get_fox

ENDPOINTS = {'common': 9.115429865459795, 'extended': 10.575749165617202}

def get_config(name):
  model, endpoint = name.split('_')
  if model == 'vp':
    config = get_vp()
  elif model == 'fox':
    config = get_fox()
    config.model.fox_u = -5.0
    config.model.fox_matern_length_scale = 346.4101615138
    config.model.fox_target_terminal_variance = 1.0
  else:
    raise ValueError(name)
  config.sampling.time_grid = 'uniform_logsnr'
  config.sampling.logsnr_min = -10.0
  config.sampling.logsnr_max = ENDPOINTS[endpoint]
  return config
PY

mkdir -p "$TFDS_DATA_DIR" "$TORCH_EXTENSIONS_DIR" "$XDG_CACHE_HOME"
run_eval() {
  local model="$1" endpoint="$2" nfe="$3" workdir="$4"
  local folder="eval_1300k_${model}_${endpoint}_logsnr_nfe${nfe}_${NUM_SAMPLES}samples_seed${SEED}"
  echo "Evaluating ${model}/${endpoint}: checkpoint=${CHECKPOINT}, NFE=${nfe}, samples=${NUM_SAMPLES}, seed=${SEED}"
  python -u main.py \
    --mode=eval --config="${TMP_CONFIG}:${model}_${endpoint}" \
    --workdir="$workdir" --eval_folder="$folder" \
    --config.eval.begin_ckpt="$CHECKPOINT" \
    --config.eval.end_ckpt="$CHECKPOINT" \
    --config.eval.batch_size=512 \
    --config.eval.num_samples="$NUM_SAMPLES" \
    --config.eval.enable_loss=False \
    --config.eval.enable_sampling=True \
    --config.eval.enable_bpd=False \
    --config.eval.sampling_num_scales="$nfe" \
    --config.eval.sampling_seed="$SEED" \
    --config.sampling.method=pc \
    --config.sampling.predictor=euler_maruyama \
    --config.sampling.corrector=none \
    --config.allow_tf32="$ALLOW_TF32" \
    --config.cudnn_benchmark="$CUDNN_BENCHMARK"
}

if [[ "$STAGE" == core || "$STAGE" == all ]]; then
  run_eval fox common 1000 "$FOX_WORKDIR"
  run_eval fox extended 1000 "$FOX_WORKDIR"
fi
if [[ "$STAGE" == nfe || "$STAGE" == all ]]; then
  for endpoint in common extended; do
    run_eval vp "$endpoint" 1077 "$VP_WORKDIR"
    run_eval fox "$endpoint" 1077 "$FOX_WORKDIR"
  done
fi
