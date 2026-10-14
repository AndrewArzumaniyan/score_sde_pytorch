#!/usr/bin/env bash
# Same CIFAR protocol as the 100k/200k screening, now on checkpoint 16.
# Run inside score-sde-rtx with one visible GPU. Four evals: VP/FOX x seeds 0/1.
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"
if (( $# != 0 )); then
  echo "Usage: bash $0" >&2
  exit 2
fi

export CIFAR_DENOISE_VP_WORKDIR="${CIFAR_DENOISE_VP_WORKDIR:-workdirs/cifar10_vp_200k_rtx_seed42}"
export CIFAR_DENOISE_FOX_WORKDIR="${CIFAR_DENOISE_FOX_WORKDIR:-workdirs/cifar10_fox_matern32_kappa1000_200k_rtx_seed42}"
export CIFAR_DENOISE_MODEL=all
export CIFAR_DENOISE_ENDPOINT=common
export CIFAR_DENOISE_CKPT=16
export CIFAR_DENOISE_NUM_SAMPLES=5000
export CIFAR_DENOISE_BATCH_SIZE=512
export CIFAR_DENOISE_RUN_TAG=cifar400k_screen

[[ -s assets/stats/cifar10_stats.npz ]] || {
  echo 'Missing CIFAR reference statistics: assets/stats/cifar10_stats.npz' >&2
  exit 1
}
[[ -f run_cifar10_endpoint_denoise_eval.sh ]] || {
  echo 'Missing evaluator runner: run_cifar10_endpoint_denoise_eval.sh' >&2
  exit 1
}
for workdir in "$CIFAR_DENOISE_VP_WORKDIR" "$CIFAR_DENOISE_FOX_WORKDIR"; do
  [[ -s "$workdir/checkpoints/checkpoint_16.pth" ]] || {
    echo "Missing or empty 400k checkpoint: $workdir/checkpoints/checkpoint_16.pth" >&2
    exit 1
  }
done

# Check both actual checkpoint steps before spending time on sampling.
python -u - "$CIFAR_DENOISE_VP_WORKDIR" "$CIFAR_DENOISE_FOX_WORKDIR" <<'PY_CHECK'
from pathlib import Path
import sys
import torch

for workdir in sys.argv[1:]:
    path = Path(workdir) / 'checkpoints' / 'checkpoint_16.pth'
    state = torch.load(path, map_location='cpu', weights_only=False)
    step = int(state['step'])
    if step != 400001:
        raise ValueError(f'{path}: expected state.step=400001, got {step}')
    print(f'400k checkpoint checked: {path}; state.step={step}', flush=True)
    del state
PY_CHECK

for seed in 0 1; do
  export CIFAR_DENOISE_SEED="$seed"
  log="run_cifar400k_endpoint_tweedie_5k_batch512_seed${seed}.log"
  echo "=== CIFAR 400k: VP + FOX, sampling seed $seed; log: $log ==="
  bash run_cifar10_endpoint_denoise_eval.sh 2>&1 | tee -a "$log"
done
echo '=== CIFAR 400k eval finished: VP/FOX, seeds 0/1, 5000 samples each ==='
