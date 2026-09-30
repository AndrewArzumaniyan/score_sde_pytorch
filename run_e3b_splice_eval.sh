#!/usr/bin/env bash
# E3b VP-process splice. Set E3B_SPLICE_RUNS separately in each GPU container.
set -euo pipefail
cd "$(dirname "$0")"
RUNS="${E3B_SPLICE_RUNS:-full tail:8 tail:4 tail:9.5 head:-3.3 head:8 tail:0 tail:-3.3}"
CHECKPOINT="${E3B_SPLICE_CKPT:-260}"
ALT_CHECKPOINT="${E3B_SPLICE_ALT_CKPT:-260}"
NUM_SAMPLES="${E3B_SPLICE_NUM_SAMPLES:-10000}"
BATCH_SIZE="${E3B_SPLICE_BATCH_SIZE:-512}"
SEED="${E3B_SPLICE_SEED:-0}"
RUN_TAG="${E3B_SPLICE_RUN_TAG:-v1}"
VP_WORKDIR="${E3B_SPLICE_VP_WORKDIR:-workdirs/celeba_vp_continuous_50k}"
FOX_WORKDIR="${E3B_SPLICE_FOX_WORKDIR:-workdirs/celeba_fox_matern32_kappa1000_50k}"
for value in "$CHECKPOINT" "$ALT_CHECKPOINT" "$NUM_SAMPLES" "$BATCH_SIZE" "$SEED"; do
  [[ "$value" =~ ^[0-9]+$ ]] || { echo "Numeric settings must be nonnegative integers" >&2; exit 2; }
done
(( NUM_SAMPLES > 0 && BATCH_SIZE > 0 )) || { echo "Samples and batch size must be positive" >&2; exit 2; }
[[ "$RUN_TAG" =~ ^[a-zA-Z0-9_-]+$ ]] || { echo "Invalid run tag" >&2; exit 2; }
[[ -f "$VP_WORKDIR/checkpoints/checkpoint_$CHECKPOINT.pth" ]] || { echo "Missing VP checkpoint" >&2; exit 1; }
[[ -f "$FOX_WORKDIR/checkpoints/checkpoint_$ALT_CHECKPOINT.pth" ]] || { echo "Missing FOX checkpoint" >&2; exit 1; }
export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-9.0}"
export TFDS_DATA_DIR="${TFDS_DATA_DIR:-/workspace/score_sde_pytorch/.tfds}"
export TORCH_EXTENSIONS_DIR="${TORCH_EXTENSIONS_DIR:-/workspace/score_sde_pytorch/.torch_extensions}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-/workspace/score_sde_pytorch/.cache}"
export MAX_JOBS="${MAX_JOBS:-8}"
mkdir -p "$TFDS_DATA_DIR" "$TORCH_EXTENSIONS_DIR" "$XDG_CACHE_HOME"
TMP_CONFIG="$(mktemp /tmp/score_sde_e3b_splice.XXXXXX)"
trap 'rm -f "$TMP_CONFIG"' EXIT
cat > "$TMP_CONFIG" <<'PY'
import os
from configs.vp.celeba_ncsnpp_continuous import get_config as get_vp
from configs.fox.celeba_ncsnpp_continuous_matern_3_2 import get_config as get_fox

def get_config(name):
  parts = name.split('_')
  mode = parts[0]
  if mode not in ('full', 'none', 'tail', 'head'):
    raise ValueError(name)
  if (mode in ('tail', 'head')) != (len(parts) == 2):
    raise ValueError(name)
  vp = get_vp()
  fox = get_fox()
  fox.model.fox_u = -5.0
  fox.model.fox_matern_length_scale = 346.4101615138
  fox.model.fox_target_terminal_variance = 1.0
  vp.sampling.time_grid = 'uniform_logsnr'
  vp.sampling.logsnr_min = -10.0
  vp.sampling.logsnr_max = 10.575749165617202
  vp.sampling.noise_removal = True
  vp.eval.endpoint_denoise_variants = True
  vp.eval.splice_enabled = True
  vp.eval.splice_mode = mode
  vp.eval.splice_lambda = float(parts[1]) if len(parts) == 2 else None
  vp.eval.splice_alt = fox
  vp.eval.splice_alt_workdir = os.environ['E3B_SPLICE_FOX_WORKDIR']
  vp.eval.splice_alt_ckpt = int(os.environ['E3B_SPLICE_ALT_CKPT'])
  return vp
PY
export E3B_SPLICE_FOX_WORKDIR="$FOX_WORKDIR"
export E3B_SPLICE_ALT_CKPT="$ALT_CHECKPOINT"
for spec in $RUNS; do
  case "$spec" in
    full|none) mode="$spec"; threshold="" ;;
    tail:*|head:*) mode="${spec%%:*}"; threshold="${spec#*:}" ;;
    *) echo "Invalid splice run: $spec" >&2; exit 2 ;;
  esac
  if [[ -n "$threshold" ]]; then
    [[ "$threshold" =~ ^-?[0-9]+([.][0-9]+)?$ ]] || { echo "Invalid threshold: $threshold" >&2; exit 2; }
    config_name="${mode}_${threshold}"
    folder_mode="${mode}${threshold}"
  else
    config_name="$mode"
    folder_mode="$mode"
  fi
  folder="eval_e3b_splice_${RUN_TAG}_${folder_mode}_ckpt${CHECKPOINT}_nfe1000_${NUM_SAMPLES}samples_batch${BATCH_SIZE}_seed${SEED}"
  echo "=== E3b $spec: VP ckpt=$CHECKPOINT, FOX ckpt=$ALT_CHECKPOINT, samples=$NUM_SAMPLES ==="
  python -u main.py \
    --mode=eval --config="$TMP_CONFIG:$config_name" \
    --workdir="$VP_WORKDIR" --eval_folder="$folder" \
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
    --config.allow_tf32="${E3B_SPLICE_ALLOW_TF32:-True}" \
    --config.cudnn_benchmark="${E3B_SPLICE_CUDNN_BENCHMARK:-True}"
done
