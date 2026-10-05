#!/usr/bin/env bash
# Controlled A/B splice under one VP process and shared log-SNR conditioning.
# Primary is A; alternate is B. head:-3.3 uses B for lambda < -3.3,
# tail:-3.3 uses B for lambda >= -3.3. none/full reproduce pure A/B.
set -euo pipefail
cd "$(dirname "$0")"
RUNS="${AB_SPLICE_RUNS:-none full head:-3.3 tail:-3.3}"
CHECKPOINT="${AB_SPLICE_CKPT:-40}"
ALT_CHECKPOINT="${AB_SPLICE_ALT_CKPT:-40}"
NUM_SAMPLES="${AB_SPLICE_NUM_SAMPLES:-10000}"
NFE="${AB_SPLICE_NFE:-1000}"
BATCH_SIZE="${AB_SPLICE_BATCH_SIZE:-512}"
SEED="${AB_SPLICE_SEED:-1}"
RUN_TAG="${AB_SPLICE_RUN_TAG:-common_seed1}"
A_WORKDIR="${AB_SPLICE_A_WORKDIR:-workdirs/celeba_factorial_A_vp_pvp_50k}"
B_WORKDIR="${AB_SPLICE_B_WORKDIR:-workdirs/celeba_factorial_B_vp_pfox_50k}"
for value in "$CHECKPOINT" "$ALT_CHECKPOINT" "$NUM_SAMPLES" "$NFE" "$BATCH_SIZE" "$SEED"; do
  [[ "$value" =~ ^[0-9]+$ ]] || { echo "Numeric settings must be nonnegative integers" >&2; exit 2; }
done
(( NUM_SAMPLES > 0 && NFE > 0 && BATCH_SIZE > 0 )) || { echo "Samples, NFE, and batch size must be positive" >&2; exit 2; }
[[ "$RUN_TAG" =~ ^[a-zA-Z0-9_-]+$ ]] || { echo "Invalid run tag" >&2; exit 2; }
for spec in $RUNS; do
  case "$spec" in none|full|head:-*|head:[0-9]*|tail:-*|tail:[0-9]*) ;; *) echo "Invalid splice run: $spec" >&2; exit 2 ;; esac
done
for path in "$A_WORKDIR/checkpoints/checkpoint_${CHECKPOINT}.pth" "$B_WORKDIR/checkpoints/checkpoint_${ALT_CHECKPOINT}.pth"; do
  [[ -f "$path" ]] || { echo "Missing checkpoint: $path" >&2; exit 1; }
done
export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-9.0}"
export TFDS_DATA_DIR="${TFDS_DATA_DIR:-/workspace/score_sde_pytorch/.tfds}"
export TORCH_EXTENSIONS_DIR="${TORCH_EXTENSIONS_DIR:-/workspace/score_sde_pytorch/.torch_extensions}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-/workspace/score_sde_pytorch/.cache}"
export MAX_JOBS="${MAX_JOBS:-8}"
mkdir -p "$TFDS_DATA_DIR" "$TORCH_EXTENSIONS_DIR" "$XDG_CACHE_HOME"
TMP_CONFIG="$(mktemp /tmp/score_sde_ab_splice.XXXXXX)"
trap 'rm -f "$TMP_CONFIG"' EXIT
cat > "$TMP_CONFIG" <<'PY'
import os
from configs.factorial.celeba_a_vp_pvp import get_config as get_a
from configs.factorial.celeba_b_vp_pfox import get_config as get_b

def get_config(name):
  parts = name.split('_')
  mode = parts[0]
  if mode not in ('none', 'full', 'head', 'tail') or len(parts) > 2:
    raise ValueError(name)
  a = get_a()
  b = get_b()
  a.sampling.time_grid = 'uniform_logsnr'
  a.sampling.logsnr_min = -10.0
  a.sampling.logsnr_max = 9.115429865459795
  a.sampling.noise_removal = True
  a.eval.endpoint_denoise_variants = True
  a.eval.splice_enabled = True
  a.eval.splice_shared_logsnr_conditioning = True
  a.eval.splice_mode = mode
  a.eval.splice_lambda = float(parts[1]) if len(parts) == 2 else None
  a.eval.splice_alt = b
  a.eval.splice_alt_workdir = os.environ['AB_SPLICE_B_WORKDIR']
  a.eval.splice_alt_ckpt = int(os.environ['AB_SPLICE_ALT_CKPT'])
  return a
PY
export AB_SPLICE_B_WORKDIR="$B_WORKDIR"
export AB_SPLICE_ALT_CKPT="$ALT_CHECKPOINT"
for spec in $RUNS; do
  if [[ "$spec" == none || "$spec" == full ]]; then mode="$spec"; threshold=""; else mode="${spec%%:*}"; threshold="${spec#*:}"; fi
  if [[ -n "$threshold" ]]; then config_name="${mode}_${threshold}"; folder_mode="${mode}${threshold}"; else config_name="$mode"; folder_mode="$mode"; fi
  folder="eval_ab_splice_${RUN_TAG}_${folder_mode}_ckpt${CHECKPOINT}_nfe${NFE}_${NUM_SAMPLES}samples_batch${BATCH_SIZE}_seed${SEED}"
  echo "=== A/B splice $spec: A ckpt=$CHECKPOINT, B ckpt=$ALT_CHECKPOINT, NFE=$NFE, samples=$NUM_SAMPLES, seed=$SEED ==="
  python -u main.py --mode=eval --config="$TMP_CONFIG:$config_name" \
    --workdir="$A_WORKDIR" --eval_folder="$folder" \
    --config.eval.begin_ckpt="$CHECKPOINT" --config.eval.end_ckpt="$CHECKPOINT" \
    --config.eval.batch_size="$BATCH_SIZE" --config.eval.num_samples="$NUM_SAMPLES" \
    --config.eval.enable_loss=False --config.eval.enable_sampling=True \
    --config.eval.enable_bpd=False --config.eval.sampling_num_scales="$NFE" \
    --config.eval.sampling_seed="$SEED" --config.sampling.method=pc \
    --config.sampling.predictor=euler_maruyama --config.sampling.corrector=none \
    --config.allow_tf32="${AB_SPLICE_ALLOW_TF32:-True}" \
    --config.cudnn_benchmark="${AB_SPLICE_CUDNN_BENCHMARK:-True}"
done
