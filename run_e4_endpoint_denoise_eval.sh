#!/usr/bin/env bash
# E4: paired x_mean, noisy endpoint x_t and explicit Tweedie(x_t) evaluation.
# Run VP and FOX independently in two GPU-pinned containers:
#   E4_ENDPOINT_DENOISE_MODEL=vp  bash run_e4_endpoint_denoise_eval.sh
#   E4_ENDPOINT_DENOISE_MODEL=fox bash run_e4_endpoint_denoise_eval.sh
# Smoke: E4_ENDPOINT_DENOISE_NUM_SAMPLES=16 E4_ENDPOINT_DENOISE_BATCH_SIZE=16
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

MODEL="${E4_ENDPOINT_DENOISE_MODEL:-all}"
ENDPOINT="${E4_ENDPOINT_DENOISE_ENDPOINT:-all}"
CHECKPOINT="${E4_ENDPOINT_DENOISE_CKPT:-260}"
NUM_SAMPLES="${E4_ENDPOINT_DENOISE_NUM_SAMPLES:-10000}"
BATCH_SIZE="${E4_ENDPOINT_DENOISE_BATCH_SIZE:-512}"
SEED="${E4_ENDPOINT_DENOISE_SEED:-0}"
ALLOW_TF32="${E4_ENDPOINT_DENOISE_ALLOW_TF32:-True}"
CUDNN_BENCHMARK="${E4_ENDPOINT_DENOISE_CUDNN_BENCHMARK:-True}"
RUN_TAG="${E4_ENDPOINT_DENOISE_RUN_TAG:-v1}"
VP_WORKDIR="${E4_ENDPOINT_DENOISE_VP_WORKDIR:-workdirs/celeba_vp_continuous_50k}"
FOX_WORKDIR="${E4_ENDPOINT_DENOISE_FOX_WORKDIR:-workdirs/celeba_fox_matern32_kappa1000_50k}"

case "$MODEL" in vp|fox|all) ;; *) echo "Invalid E4 model: $MODEL" >&2; exit 2 ;; esac
case "$ENDPOINT" in common|extended|all) ;; *) echo "Invalid E4 endpoint: $ENDPOINT" >&2; exit 2 ;; esac
for value in "$CHECKPOINT" "$NUM_SAMPLES" "$BATCH_SIZE" "$SEED"; do
  [[ "$value" =~ ^[0-9]+$ ]] || { echo "E4 numeric settings must be nonnegative integers" >&2; exit 2; }
done
(( NUM_SAMPLES > 0 && BATCH_SIZE > 0 )) || { echo "Samples and batch size must be positive" >&2; exit 2; }
[[ "$RUN_TAG" =~ ^[a-zA-Z0-9_-]+$ ]] || { echo "Invalid E4 run tag" >&2; exit 2; }

export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-9.0}"
export TFDS_DATA_DIR="${TFDS_DATA_DIR:-/workspace/score_sde_pytorch/.tfds}"
export TORCH_EXTENSIONS_DIR="${TORCH_EXTENSIONS_DIR:-/workspace/score_sde_pytorch/.torch_extensions}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-/workspace/score_sde_pytorch/.cache}"
export MAX_JOBS="${MAX_JOBS:-8}"
mkdir -p "$TFDS_DATA_DIR" "$TORCH_EXTENSIONS_DIR" "$XDG_CACHE_HOME"

MODELS=(vp fox)
ENDPOINTS=(common extended)
[[ "$MODEL" == all ]] || MODELS=("$MODEL")
[[ "$ENDPOINT" == all ]] || ENDPOINTS=("$ENDPOINT")
for model in "${MODELS[@]}"; do
  if [[ "$model" == vp ]]; then workdir="$VP_WORKDIR"; else workdir="$FOX_WORKDIR"; fi
  [[ -f "$workdir/checkpoints/checkpoint_${CHECKPOINT}.pth" ]] || {
    echo "Missing checkpoint: $workdir/checkpoints/checkpoint_${CHECKPOINT}.pth" >&2
    exit 1
  }
done

# The endpoint fields are absent from the historical training configs.
# A temporary eval-only config leaves their training provenance untouched.
TMP_CONFIG="$(mktemp /tmp/score_sde_e4_endpoint_denoise.XXXXXX)"
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
  config.sampling.noise_removal = True
  config.eval.endpoint_denoise_variants = True
  return config
PY

for model in "${MODELS[@]}"; do
  if [[ "$model" == vp ]]; then workdir="$VP_WORKDIR"; else workdir="$FOX_WORKDIR"; fi
  for endpoint in "${ENDPOINTS[@]}"; do
    folder="eval_e4_endpoint_denoise_${RUN_TAG}_${model}_${endpoint}_ckpt${CHECKPOINT}_nfe1000_${NUM_SAMPLES}samples_batch${BATCH_SIZE}_seed${SEED}"
    echo "=== E4 $model/$endpoint: ckpt=$CHECKPOINT, 1000 EM steps + 1 denoise, samples=$NUM_SAMPLES, seed=$SEED ==="
    python -u main.py \
      --mode=eval --config="${TMP_CONFIG}:${model}_${endpoint}" \
      --workdir="$workdir" --eval_folder="$folder" \
      --config.eval.begin_ckpt="$CHECKPOINT" \
      --config.eval.end_ckpt="$CHECKPOINT" \
      --config.eval.batch_size="$BATCH_SIZE" \
      --config.eval.num_samples="$NUM_SAMPLES" \
      --config.eval.enable_loss=False \
      --config.eval.enable_sampling=True \
      --config.eval.enable_bpd=False \
      --config.eval.sampling_num_scales=1000 \
      --config.eval.sampling_seed="$SEED" \
      --config.sampling.method=pc \
      --config.sampling.predictor=euler_maruyama \
      --config.sampling.corrector=none \
      --config.allow_tf32="$ALLOW_TF32" \
      --config.cudnn_benchmark="$CUDNN_BENCHMARK"
  done
done
