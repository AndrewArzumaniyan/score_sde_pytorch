#!/usr/bin/env bash
# CIFAR-10 VP / real FOX: common log-SNR endpoints and paired final denoising.
# Run inside the RTX container: ./docker/run_rtx.sh
# Smoke: CIFAR_DENOISE_NUM_SAMPLES=128 CIFAR_DENOISE_RUN_TAG=smoke bash "$0"
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

MODEL="${CIFAR_DENOISE_MODEL:-all}"
ENDPOINT="${CIFAR_DENOISE_ENDPOINT:-common}"
CHECKPOINT="${CIFAR_DENOISE_CKPT:-4}"
NUM_SAMPLES="${CIFAR_DENOISE_NUM_SAMPLES:-10000}"
BATCH_SIZE="${CIFAR_DENOISE_BATCH_SIZE:-128}"
SEED="${CIFAR_DENOISE_SEED:-0}"
RUN_TAG="${CIFAR_DENOISE_RUN_TAG:-v1}"
VP_WORKDIR="${CIFAR_DENOISE_VP_WORKDIR:-workdirs/cifar10_vp_kappa1_50k}"
FOX_WORKDIR="${CIFAR_DENOISE_FOX_WORKDIR:-workdirs/cifar10_fox_matern32_kappa1000_100k}"

case "$MODEL" in vp|fox|all) ;; *) echo "Invalid CIFAR model: $MODEL" >&2; exit 2 ;; esac
case "$ENDPOINT" in common|extended|all) ;; *) echo "Invalid CIFAR endpoint: $ENDPOINT" >&2; exit 2 ;; esac
for value in "$CHECKPOINT" "$NUM_SAMPLES" "$BATCH_SIZE" "$SEED"; do
  [[ "$value" =~ ^[0-9]+$ ]] || { echo 'Numeric settings must be nonnegative integers.' >&2; exit 2; }
done
(( NUM_SAMPLES > 0 && BATCH_SIZE > 0 )) || { echo 'Samples and batch size must be positive.' >&2; exit 2; }
[[ "$RUN_TAG" =~ ^[a-zA-Z0-9_-]+$ ]] || { echo 'Invalid CIFAR run tag.' >&2; exit 2; }

export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-12.0}"
export TFDS_DATA_DIR="${TFDS_DATA_DIR:-$ROOT_DIR/.tfds}"
export TORCH_EXTENSIONS_DIR="${TORCH_EXTENSIONS_DIR:-$ROOT_DIR/.torch_extensions_rtx}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-$ROOT_DIR/.cache_rtx}"
export MAX_JOBS="${MAX_JOBS:-8}"
export TF_ENABLE_ONEDNN_OPTS=0
export TF_NUM_INTRAOP_THREADS="${TF_NUM_INTRAOP_THREADS:-8}"
export TF_NUM_INTEROP_THREADS="${TF_NUM_INTEROP_THREADS:-2}"
export TFHUB_CACHE_DIR="${TFHUB_CACHE_DIR:-$ROOT_DIR/.cache/tfhub_smoke}"
mkdir -p "$TFDS_DATA_DIR" "$TORCH_EXTENSIONS_DIR" "$XDG_CACHE_HOME"

MODELS=(vp fox)
ENDPOINTS=(common extended)
[[ "$MODEL" == all ]] || MODELS=("$MODEL")
[[ "$ENDPOINT" == all ]] || ENDPOINTS=("$ENDPOINT")
for model in "${MODELS[@]}"; do
  if [[ "$model" == vp ]]; then workdir="$VP_WORKDIR"; else workdir="$FOX_WORKDIR"; fi
  [[ -s "$workdir/checkpoints/checkpoint_${CHECKPOINT}.pth" ]] || {
    echo "Missing or empty checkpoint: $workdir/checkpoints/checkpoint_${CHECKPOINT}.pth" >&2
    exit 1
  }
done

# Temporary eval config restores historical FOX overrides without editing
# the original training configs. It is removed when this script exits.
TMP_CONFIG="$(mktemp /tmp/score_sde_cifar_endpoint_denoise.XXXXXX)"
trap 'rm -f "$TMP_CONFIG"' EXIT
cat > "$TMP_CONFIG" <<'PY_CONFIG'
import torch
from configs.vp.cifar10_ncsnpp_continuous import get_config as get_vp
from configs.fox.cifar10_ncsnpp_continuous_matern_3_2 import get_config as get_fox

def base_config(model):
  if model == 'vp':
    return get_vp()
  if model != 'fox':
    raise ValueError(model)
  config = get_fox()
  config.model.fox_u = -5.0
  config.model.fox_drift_schedule = 'constant'
  config.model.fox_matern_length_scale = 346.4101615138
  config.model.fox_target_terminal_variance = 1.0
  # CIFAR legacy configs omit this field; get_sde defaults it to False.
  if hasattr(config.model, 'fox_normalize_scale'):
    config.model.fox_normalize_scale = False
  return config

def get_config(name):
  from run_lib import get_sde
  model, endpoint = name.split('_')
  if endpoint not in ('common', 'extended'):
    raise ValueError(endpoint)
  config = base_config(model)
  bounds = []
  for arm in ('vp', 'fox'):
    sde, eps = get_sde(base_config(arm))
    native = sde.log_snr(torch.tensor([sde.T, eps], dtype=torch.float64))
    bounds.append((float(native[0]), float(native[1])))
  lambda_min = max(bound[0] for bound in bounds)
  clean_bounds = [bound[1] for bound in bounds]
  lambda_max = min(clean_bounds) if endpoint == 'common' else max(clean_bounds)
  print('CIFAR native (lambda(T), lambda(eps)): VP=%s FOX=%s; selected=%s/%s' %
        (bounds[0], bounds[1], lambda_min, lambda_max), flush=True)
  config.sampling.time_grid = 'uniform_logsnr'
  config.sampling.logsnr_min = lambda_min
  config.sampling.logsnr_max = lambda_max
  config.sampling.noise_removal = True
  config.eval.endpoint_denoise_variants = True
  return config
PY_CONFIG

# Initialize Torch CUDA first, then keep the validated Inception path on CPU.
ENTRYPOINT_CODE="$(cat <<'PY_ENTRY'
import runpy
import sys
import torch
if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
  raise RuntimeError('Run this eval in a container with exactly one CUDA GPU.')
torch.cuda.init()
import tensorflow as tf
tf.config.set_visible_devices([], 'GPU')
import evaluation
evaluation.jax = None
sys.argv[0] = 'main.py'
runpy.run_path('main.py', run_name='__main__')
PY_ENTRY
)"

for model in "${MODELS[@]}"; do
  if [[ "$model" == vp ]]; then workdir="$VP_WORKDIR"; else workdir="$FOX_WORKDIR"; fi
  for endpoint in "${ENDPOINTS[@]}"; do
    folder="eval_cifar_endpoint_denoise_${RUN_TAG}_${model}_${endpoint}_ckpt${CHECKPOINT}_nfe1000_${NUM_SAMPLES}samples_batch${BATCH_SIZE}_seed${SEED}"
    echo "=== CIFAR $model/$endpoint: ckpt=$CHECKPOINT, 1000 EM + 1 denoise, samples=$NUM_SAMPLES, batch=$BATCH_SIZE, seed=$SEED ==="
    python -u -c "$ENTRYPOINT_CODE" \
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
      --config.seed=42 \
      --config.allow_tf32=False \
      --config.cudnn_benchmark=False \
      --config.allow_fp16_reduced_precision_reduction=False
  done
done
