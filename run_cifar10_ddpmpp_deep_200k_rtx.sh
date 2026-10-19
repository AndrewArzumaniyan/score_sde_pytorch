#!/usr/bin/env bash
# Train a fresh DDPM++ deep VP / real FOX pair sequentially to 200k.
# Run inside score-sde-rtx with exactly one visible GPU (physical GPU 2).
# Existing workdirs are refused; this script does not resume old experiments.
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"
if (( $# != 0 )); then
  echo "Usage: bash $0" >&2
  exit 2
fi

VP_CONFIG=configs/vp/cifar10_ddpmpp_deep_continuous_matched.py
FOX_CONFIG=configs/fox/cifar10_ddpmpp_deep_continuous_matern_3_2_kappa1000.py
VP_WORKDIR=workdirs/cifar10_vp_ddpmpp_deep_200k_rtx_seed42
FOX_WORKDIR=workdirs/cifar10_fox_ddpmpp_deep_matern32_kappa1000_200k_rtx_seed42

export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-12.0}"
export TFDS_DATA_DIR="${TFDS_DATA_DIR:-$ROOT_DIR/.tfds}"
export TORCH_EXTENSIONS_DIR="${TORCH_EXTENSIONS_DIR:-$ROOT_DIR/.torch_extensions_rtx}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-$ROOT_DIR/.cache_rtx}"
export MAX_JOBS="${MAX_JOBS:-8}"
export TF_ENABLE_ONEDNN_OPTS=0
export TF_NUM_INTRAOP_THREADS="${TF_NUM_INTRAOP_THREADS:-8}"
export TF_NUM_INTEROP_THREADS="${TF_NUM_INTEROP_THREADS:-2}"

for config in "$VP_CONFIG" "$FOX_CONFIG"; do
  [[ -s "$config" ]] || { echo "Missing config: $config" >&2; exit 1; }
done
[[ -d "$TFDS_DATA_DIR/cifar10" ]] || {
  echo "Missing prepared CIFAR-10: $TFDS_DATA_DIR/cifar10" >&2
  exit 1
}
for workdir in "$VP_WORKDIR" "$FOX_WORKDIR"; do
  if [[ -e "$workdir" || -L "$workdir" ]]; then
    echo "Workdir already exists; refusing to reuse it: $workdir" >&2
    exit 1
  fi
done

run_model() {
  local arm="$1" config="$2" workdir="$3" log="$4"
  echo "=== Training CIFAR DDPM++ deep $arm from scratch to 200k; batch=128; seed=42 ==="
  echo "Config: $config; workdir: $workdir; log: $log"
  python -u - \
    --mode=train \
    --config="$config" \
    --workdir="$workdir" \
    --config.training.n_iters=200000 \
    --config.training.batch_size=128 \
    --config.seed=42 \
    <<'PY_TRAIN' 2>&1 | tee -a "$log"
import runpy

import torch
if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
    raise RuntimeError('Run inside the RTX container with exactly one visible CUDA GPU.')
torch.cuda.init()
import tensorflow as tf
tf.config.set_visible_devices([], 'GPU')

# Keep the same PyTorch/TensorFlow startup as previous RTX training scripts.
# main.py receives the config/workdir arguments above and runs normal training.
runpy.run_path('main.py', run_name='__main__')
PY_TRAIN
  echo "=== $arm finished; checkpoint: $workdir/checkpoints/checkpoint_4.pth ==="
}

run_model vp "$VP_CONFIG" "$VP_WORKDIR" \
  run_cifar10_vp_ddpmpp_deep_200k_rtx.log
run_model fox "$FOX_CONFIG" "$FOX_WORKDIR" \
  run_cifar10_fox_ddpmpp_deep_matern32_kappa1000_200k_rtx.log

echo '=== VP and FOX DDPM++ deep training to 200k completed ==='
