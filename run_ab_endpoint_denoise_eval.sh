#!/usr/bin/env bash
# Controlled A/B (VP process, P_VP vs P_FOX training clock) under the E4 protocol:
# paired x_mean, noisy endpoint x_t and explicit Tweedie(x_t) at common or
# extended clean endpoints.  Evaluation only; no training, no checkpoint writes.
#   AB_ENDPOINT_DENOISE_RUNS="A:extended B:extended A:common B:common"
# Smoke: AB_ENDPOINT_DENOISE_NUM_SAMPLES=16 AB_ENDPOINT_DENOISE_BATCH_SIZE=16
#        AB_ENDPOINT_DENOISE_RUNS="A:extended" AB_ENDPOINT_DENOISE_RUN_TAG=smoke
# The common-endpoint x_mean values are a regression check against the earlier
# Wave 1 A/B numbers (same seed, samples, batch size, NFE).
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

RUNS="${AB_ENDPOINT_DENOISE_RUNS:-A:extended B:extended A:common B:common}"
CHECKPOINT="${AB_ENDPOINT_DENOISE_CKPT:-20}"
NUM_SAMPLES="${AB_ENDPOINT_DENOISE_NUM_SAMPLES:-5000}"
BATCH_SIZE="${AB_ENDPOINT_DENOISE_BATCH_SIZE:-512}"
SEED="${AB_ENDPOINT_DENOISE_SEED:-0}"
ALLOW_TF32="${AB_ENDPOINT_DENOISE_ALLOW_TF32:-True}"
CUDNN_BENCHMARK="${AB_ENDPOINT_DENOISE_CUDNN_BENCHMARK:-True}"
RUN_TAG="${AB_ENDPOINT_DENOISE_RUN_TAG:-v1}"
A_WORKDIR="${AB_ENDPOINT_DENOISE_A_WORKDIR:-workdirs/celeba_factorial_A_vp_pvp_50k}"
B_WORKDIR="${AB_ENDPOINT_DENOISE_B_WORKDIR:-workdirs/celeba_factorial_B_vp_pfox_50k}"

for value in "$CHECKPOINT" "$NUM_SAMPLES" "$BATCH_SIZE" "$SEED"; do
  [[ "$value" =~ ^[0-9]+$ ]] || { echo "A/B numeric settings must be nonnegative integers" >&2; exit 2; }
done
(( NUM_SAMPLES > 0 && BATCH_SIZE > 0 )) || { echo "Samples and batch size must be positive" >&2; exit 2; }
[[ "$RUN_TAG" =~ ^[a-zA-Z0-9_-]+$ ]] || { echo "Invalid A/B run tag" >&2; exit 2; }
for spec in $RUNS; do
  case "$spec" in A:common|A:extended|B:common|B:extended) ;; *) echo "Invalid run: $spec" >&2; exit 2 ;; esac
done
for workdir in "$A_WORKDIR" "$B_WORKDIR"; do
  [[ -f "$workdir/checkpoints/checkpoint_${CHECKPOINT}.pth" ]] || {
    echo "Missing checkpoint: $workdir/checkpoints/checkpoint_${CHECKPOINT}.pth" >&2
    exit 1
  }
done

export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-9.0}"
export TFDS_DATA_DIR="${TFDS_DATA_DIR:-/workspace/score_sde_pytorch/.tfds}"
export TORCH_EXTENSIONS_DIR="${TORCH_EXTENSIONS_DIR:-/workspace/score_sde_pytorch/.torch_extensions}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-/workspace/score_sde_pytorch/.cache}"
export MAX_JOBS="${MAX_JOBS:-8}"
mkdir -p "$TFDS_DATA_DIR" "$TORCH_EXTENSIONS_DIR" "$XDG_CACHE_HOME"

# The endpoint-denoise fields are absent from the training configs and main.py
# locks configs, so use one ephemeral eval-only config outside the repository.
# The training configs themselves are used unchanged, as in the Wave 1 A/B eval.
TMP_CONFIG="$(mktemp /tmp/score_sde_ab_endpoint_denoise.XXXXXX)"
trap 'rm -f "$TMP_CONFIG"' EXIT
cat > "$TMP_CONFIG" <<'PY'
from configs.factorial.celeba_a_vp_pvp import get_config as get_a
from configs.factorial.celeba_b_vp_pfox import get_config as get_b

COMMON_MIN = -10.0
ENDPOINTS = {'common': 9.115429865459795, 'extended': 10.575749165617202}

def get_config(name):
  arm, endpoint = name.split('_')
  if arm == 'a':
    config = get_a()
  elif arm == 'b':
    config = get_b()
  else:
    raise ValueError(name)
  config.sampling.time_grid = 'uniform_logsnr'
  config.sampling.logsnr_min = COMMON_MIN
  config.sampling.logsnr_max = ENDPOINTS[endpoint]
  config.sampling.noise_removal = True
  config.eval.endpoint_denoise_variants = True
  return config
PY

for spec in $RUNS; do
  arm="${spec%%:*}"
  endpoint="${spec#*:}"
  if [[ "$arm" == A ]]; then workdir="$A_WORKDIR"; else workdir="$B_WORKDIR"; fi
  lower_arm="$(echo "$arm" | tr 'A-Z' 'a-z')"
  folder="eval_ab_endpoint_denoise_${RUN_TAG}_${arm}_${endpoint}_ckpt${CHECKPOINT}_nfe1000_${NUM_SAMPLES}samples_batch${BATCH_SIZE}_seed${SEED}"
  echo "=== A/B $arm/$endpoint: ckpt=$CHECKPOINT, 1000 EM steps + 1 denoise, samples=$NUM_SAMPLES, seed=$SEED ==="
  python -u main.py \
    --mode=eval --config="${TMP_CONFIG}:${lower_arm}_${endpoint}" \
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
