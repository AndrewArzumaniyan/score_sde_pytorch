#!/usr/bin/env bash
# DDPM++ deep VP / real FOX at 600k: original EM/Tweedie eval, extended endpoint.
# Only numerical protocol change: lambda_max 9.115429865459795 -> 10.575749165617202.
# Also save unclipped float endpoint states for later diagnostics.
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
export PYTHONDONTWRITEBYTECODE=1
export CIFAR_EM_EXTENDED_DRIVER="$ROOT_DIR/$(basename "${BASH_SOURCE[0]}")"
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


# Add persistence fields in a temporary config without editing trained configs.
TMP_CONFIG="$(mktemp /tmp/cifar_ddpmpp_600k_em_extended.XXXXXX)"
trap 'rm -f "$TMP_CONFIG"' EXIT
cat > "$TMP_CONFIG" <<'PY_CONFIG'
import hashlib
import os
from pathlib import Path
from configs.vp.cifar10_ddpmpp_deep_continuous_matched import get_config as get_vp
from configs.fox.cifar10_ddpmpp_deep_continuous_matern_3_2_kappa1000 import get_config as get_fox


def get_config(name):
    if name not in ('vp', 'fox'):
        raise ValueError('Unknown CIFAR arm: ' + name)
    cfg = (get_vp if name == 'vp' else get_fox)('eval')
    cfg.sampling.probability_flow = False
    cfg.eval.endpoint_denoise_outputs = ('x_mean', 'x_t', 'tweedie_xt')
    cfg.eval.save_endpoint_states = True
    cfg.eval.em_extended_driver_sha256 = hashlib.sha256(
        Path(os.environ['CIFAR_EM_EXTENDED_DRIVER']).read_bytes()).hexdigest()
    return cfg
PY_CONFIG
VP_CONFIG="$TMP_CONFIG"
FOX_CONFIG="$TMP_CONFIG"

run_model() {
  local arm="$1" config="$2" workdir="$3" seed="$4" folder log
  folder="eval_cifar_endpoint_denoise_ddpmpp_deep600k_em_extended_${arm}_extended_ckpt12_nfe1000_5000samples_batch512_seed${seed}"
  log="run_cifar10_ddpmpp_deep600k_${arm}_extended_tweedie_5k_batch512_seed${seed}.log"
  echo "=== CIFAR DDPM++ deep $arm/extended: ckpt=12, 1000 EM + 1 denoise, samples=5000, batch=512, seed=$seed ==="
  echo "log-SNR: [-10, 10.575749165617202]; workdir: $workdir; eval folder: $folder"
  python -u - \
    --mode=eval \
    --config="${config}:${arm}" \
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
    --config.sampling.probability_flow=False \
    --config.sampling.time_grid=uniform_logsnr \
    --config.sampling.logsnr_min=-10.0 \
    --config.sampling.logsnr_max=10.575749165617202 \
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
echo '=== DDPM++ deep 600k extended EM eval finished: VP/FOX, seeds 0/1, 5000 samples each ==='


python -u - "$VP_WORKDIR" "$FOX_WORKDIR" <<'PY_SUMMARY'
import json
from pathlib import Path
import sys
import numpy as np

workdirs = dict(zip(('vp', 'fox'), map(Path, sys.argv[1:])))
rows, identities, references, sources = {}, {}, set(), set()
for seed in (0, 1):
    rows[seed] = {}
    for arm, workdir in workdirs.items():
        folder = workdir / (
            f'eval_cifar_endpoint_denoise_ddpmpp_deep600k_em_extended_{arm}_extended_ckpt12_'
            f'nfe1000_5000samples_batch512_seed{seed}')
        manifest = json.loads((folder/'manifest.json').read_text())
        cfg = manifest['config']
        sampling, ev = cfg['sampling'], cfg['eval']
        expected_sampling = dict(method='pc', predictor='euler_maruyama', corrector='none',
                                 probability_flow=False, time_grid='uniform_logsnr',
                                 noise_removal=True, logsnr_min=-10.,
                                 logsnr_max=10.575749165617202)
        if any(sampling.get(k) != v for k, v in expected_sampling.items()):
            raise ValueError('Unexpected extended EM sampling protocol: ' + str(folder))
        if (ev['sampling_seed'] != seed or ev['batch_size'] != 512
                or ev['num_samples'] != 5000 or ev['sampling_num_scales'] != 1000
                or ev['save_endpoint_states'] is not True
                or tuple(ev['endpoint_denoise_outputs']) != ('x_mean', 'x_t', 'tweedie_xt')):
            raise ValueError('Unexpected extended EM eval settings: ' + str(folder))
        sources.add(manifest['source_sha256'])
        with np.load(folder/'report_12.npz', allow_pickle=False) as z:
            if (int(z['checkpoint_step']) != 600001 or int(z['num_samples']) != 5000
                    or int(z['nfe']) != 1001 or z['protocol_sha256'].item() != manifest['protocol_sha256']):
                raise ValueError('Unexpected report identity: ' + str(folder))
            identity = (z['checkpoint_id'].item(), z['training_protocol_sha256'].item())
            if identities.setdefault(arm, identity) != identity:
                raise ValueError('Checkpoint differs across seeds: ' + arm)
            references.add(z['dataset_stats_id'].item())
            metrics = {m+'_'+v: float(z[m+'_'+v]) for m in ('fid', 'kid', 'IS')
                       for v in ('x_mean', 'x_t', 'tweedie_xt')}
            if not all(np.isfinite(v) for v in metrics.values()):
                raise ValueError('Non-finite metric: ' + str(folder))
        if len(list((folder/'ckpt_12').glob('statistics_*.npz'))) != 10:
            raise ValueError('Expected 10 saved feature batches: ' + str(folder))
        for i in (0, 9):
            with np.load(folder/'ckpt_12'/f'samples_{i}.npz', allow_pickle=False) as z:
                state = z['float_x_t']
                if state.dtype != np.float32 or state.shape != (512, 3, 32, 32) or not np.isfinite(state).all():
                    raise ValueError('Invalid unclipped float endpoint cache: ' + str(folder))
        rows[seed][arm] = metrics
if len(sources) != 1 or len(references) != 1:
    raise ValueError('Mixed runtime sources or reference statistics.')
print('\n=== CIFAR DDPM++ 600k EM: lambda [-10, 10.575749165617202], 1000 + 1 calls ===')
print('seed\tVP x_mean\tFOX x_mean\tVP Tweedie\tFOX Tweedie\tFOX-VP Tweedie')
for seed, pair in rows.items():
    a, b = pair['vp'], pair['fox']
    print('%d\t%.6f\t%.6f\t%.6f\t%.6f\t%+.6f' %
          (seed, a['fid_x_mean'], b['fid_x_mean'], a['fid_tweedie_xt'],
           b['fid_tweedie_xt'], b['fid_tweedie_xt']-a['fid_tweedie_xt']))
for variant in ('x_mean', 'x_t', 'tweedie_xt'):
    a = np.mean([p['vp']['fid_'+variant] for p in rows.values()])
    b = np.mean([p['fox']['fid_'+variant] for p in rows.values()])
    print('%s mean of separate 5k FIDs: VP=%.6f FOX=%.6f FOX-VP=%+.6f' % (variant, a, b, b-a))
print('Finished all 4 extended EM evals. Separate 5k seed estimates, not pooled 10k FID.')
PY_SUMMARY
