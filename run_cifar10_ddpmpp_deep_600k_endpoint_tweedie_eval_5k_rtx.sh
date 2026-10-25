#!/usr/bin/env bash
# DDPM++ deep VP / real FOX at 600k: the same CIFAR endpoint/Tweedie screening.
# Four sequential evals: seed 0 VP/FOX, then seed 1 VP/FOX; 5000 images each.
# Run inside score-sde-rtx with exactly one visible GPU. No training is run.
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
export TFHUB_CACHE_DIR="${TFHUB_CACHE_DIR:-$ROOT_DIR/.cache/tfhub_smoke}"

for config in "$VP_CONFIG" "$FOX_CONFIG"; do
  [[ -s "$config" ]] || { echo "Missing config: $config" >&2; exit 1; }
done
[[ -s assets/stats/cifar10_stats.npz ]] || {
  echo 'Missing CIFAR reference statistics: assets/stats/cifar10_stats.npz' >&2
  exit 1
}
[[ -d "$TFDS_DATA_DIR/cifar10" ]] || {
  echo "Missing prepared CIFAR-10: $TFDS_DATA_DIR/cifar10" >&2
  exit 1
}
for workdir in "$VP_WORKDIR" "$FOX_WORKDIR"; do
  [[ -s "$workdir/checkpoints/checkpoint_12.pth" ]] || {
    echo "Missing or empty 600k checkpoint: $workdir/checkpoints/checkpoint_12.pth" >&2
    exit 1
  }
done

# Author DDPM++ snapshots are every 50k; 600k uses checkpoint 12.
# Check both checkpoint steps before spending GPU time on generation.
python -u - "$VP_WORKDIR" "$FOX_WORKDIR" <<'PY_CHECK'
from pathlib import Path
import sys
import torch

for workdir in sys.argv[1:]:
    path = Path(workdir) / 'checkpoints' / 'checkpoint_12.pth'
    state = torch.load(path, map_location='cpu', weights_only=False)
    step = int(state['step'])
    if step != 600001:
        raise ValueError(f'{path}: expected state.step=600001, got {step}')
    print(f'600k checkpoint checked: {path}; state.step={step}', flush=True)
    del state
PY_CHECK

run_model() {
  local arm="$1" config="$2" workdir="$3" seed="$4" folder log
  folder="eval_cifar_endpoint_denoise_ddpmpp_deep600k_screen_${arm}_common_ckpt12_nfe1000_5000samples_batch512_seed${seed}"
  log="run_cifar10_ddpmpp_deep600k_${arm}_common_tweedie_5k_batch512_seed${seed}.log"
  echo "=== CIFAR DDPM++ deep $arm/common: ckpt=12, 1000 EM + 1 denoise, samples=5000, batch=512, seed=$seed ==="
  echo "log-SNR: [-10, 9.115429865459795]; workdir: $workdir; eval folder: $folder"
  python -u - \
    --mode=eval \
    --config="${config}:eval" \
    --workdir="$workdir" \
    --eval_folder="$folder" \
    --config.eval.begin_ckpt=12 \
    --config.eval.end_ckpt=12 \
    --config.eval.batch_size=512 \
    --config.eval.num_samples=5000 \
    --config.eval.enable_loss=False \
    --config.eval.enable_sampling=True \
    --config.eval.enable_bpd=False \
    --config.eval.endpoint_denoise_variants=True \
    --config.eval.sampling_num_scales=1000 \
    --config.eval.sampling_seed="$seed" \
    --config.sampling.method=pc \
    --config.sampling.predictor=euler_maruyama \
    --config.sampling.corrector=none \
    --config.sampling.time_grid=uniform_logsnr \
    --config.sampling.logsnr_min=-10.0 \
    --config.sampling.logsnr_max=9.115429865459795 \
    --config.sampling.noise_removal=True \
    --config.seed=42 \
    --config.allow_tf32=False \
    --config.cudnn_benchmark=False \
    --config.allow_fp16_reduced_precision_reduction=False \
    <<'PY_EVAL' 2>&1 | tee -a "$log"
import runpy
import sys
import torch
if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
    raise RuntimeError('Run this eval inside the RTX container with exactly one CUDA GPU.')
torch.cuda.init()
import tensorflow as tf
tf.config.set_visible_devices([], 'GPU')
import evaluation
# Match the previous CIFAR eval: run Inception on CPU, generation on GPU.
evaluation.jax = None
print('DDPM++ evaluation arguments:', ' '.join(sys.argv[1:]), flush=True)
sys.argv[0] = 'main.py'
runpy.run_path('main.py', run_name='__main__')
PY_EVAL
  echo "=== $arm seed $seed finished; report: $workdir/$folder/report_12.npz ==="
}

for seed in 0 1; do
  run_model vp "$VP_CONFIG" "$VP_WORKDIR" "$seed"
  run_model fox "$FOX_CONFIG" "$FOX_WORKDIR" "$seed"
done
echo '=== DDPM++ deep 600k eval finished: VP/FOX, seeds 0/1, 5000 samples each ==='
