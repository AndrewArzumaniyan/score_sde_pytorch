#!/usr/bin/env bash
# 8 sequential evals: DDPM++ deep VP / native FOX, DPM++ 2M / Heun, seeds 0/1.
# Fixed checkpoint 12 (600001), 5000 images, batch 512: 10 rounds, trim to 5000.
# Common lambda [-10, 9.115429865459795], 256 trajectory calls + 1 Tweedie.
# Uses the existing ode_sampling.py; no training, source copies or old-script edits.
set -euo pipefail
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"
if (( $# != 0 )); then
  echo "Usage: bash $0" >&2
  exit 2
fi

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
export PYTHONDONTWRITEBYTECODE=1
export CIFAR_ODE_DRIVER="$ROOT_DIR/$(basename "${BASH_SOURCE[0]}")"
for file in ode_sampling.py configs/vp/cifar10_ddpmpp_deep_continuous_matched.py \
            configs/fox/cifar10_ddpmpp_deep_continuous_matern_3_2_kappa1000.py \
            assets/stats/cifar10_stats.npz; do
  [[ -s "$file" ]] || { echo "Missing or empty: $file" >&2; exit 1; }
done
[[ -d "$TFDS_DATA_DIR/cifar10" ]] || { echo "Missing prepared CIFAR-10: $TFDS_DATA_DIR/cifar10" >&2; exit 1; }
for workdir in "$VP_WORKDIR" "$FOX_WORKDIR"; do
  for file in checkpoints/checkpoint_12.pth run_manifest.json; do
    [[ -s "$workdir/$file" ]] || { echo "Missing or empty: $workdir/$file" >&2; exit 1; }
  done
done

# Temporary eval configuration only; model fields and training configs stay intact.
TMP_CONFIG="$(mktemp /tmp/cifar_ddpmpp_600k_ode.XXXXXX)"
trap 'rm -f "$TMP_CONFIG"' EXIT
cat > "$TMP_CONFIG" <<'PY_CONFIG'
import hashlib
import os
from pathlib import Path
from configs.vp.cifar10_ddpmpp_deep_continuous_matched import get_config as get_vp
from configs.fox.cifar10_ddpmpp_deep_continuous_matern_3_2_kappa1000 import get_config as get_fox


def get_config(name):
    method, arm = name.rsplit('_', 1)
    if method not in ('ode_dpmpp_2m', 'ode_heun') or arm not in ('vp', 'fox'):
        raise ValueError('Unknown CIFAR ODE cell: ' + name)
    cfg = (get_vp if arm == 'vp' else get_fox)('eval')
    cfg.eval.begin_ckpt = cfg.eval.end_ckpt = 12
    cfg.eval.sampling_num_scales = 256  # Network calls, not the number of Heun steps.
    cfg.eval.num_samples = 5000
    cfg.eval.batch_size = 512
    cfg.eval.enable_sampling = True
    cfg.eval.enable_loss = cfg.eval.enable_bpd = False
    cfg.eval.endpoint_denoise_variants = True
    cfg.eval.endpoint_denoise_outputs = ('x_t', 'tweedie_xt')
    cfg.eval.save_endpoint_states = True
    cfg.eval.ode_driver_sha256 = hashlib.sha256(Path(os.environ['CIFAR_ODE_DRIVER']).read_bytes()).hexdigest()
    cfg.sampling.method = method
    cfg.sampling.probability_flow = True
    cfg.sampling.predictor = 'none'
    cfg.sampling.corrector = 'none'
    cfg.sampling.noise_removal = True
    cfg.sampling.time_grid = 'uniform_logsnr'
    cfg.sampling.logsnr_min = -10.0
    cfg.sampling.logsnr_max = 9.115429865459795
    cfg.seed = 42
    cfg.allow_tf32 = cfg.cudnn_benchmark = False
    cfg.allow_fp16_reduced_precision_reduction = False
    return cfg
PY_CONFIG

# Validate both parents and smoke both solvers on EMA weights before full evals.
python -u - "$TMP_CONFIG" "$VP_WORKDIR" "$FOX_WORKDIR" <<'PY_CHECK'
import gc
import hashlib
import json
from pathlib import Path
import runpy
import sys
import torch
if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
    raise RuntimeError('Run in score-sde-rtx with exactly one visible CUDA GPU.')
torch.cuda.init()
import tensorflow as tf
tf.config.set_visible_devices([], 'GPU')
import run_lib
from artifact_utils import build_model_manifest
from models import utils as mutils
from models.ema import ExponentialMovingAverage
from losses import get_optimizer
from ode_sampling import get_vp_ode_sampler
from reproducibility import configure_reproducibility
from utils import restore_checkpoint

def digest(path):
    result = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            result.update(block)
    return result.hexdigest()

reference = 'assets/stats/cifar10_stats.npz'
if digest(reference) != 'c464743bce16cb0f64408eb36bdc7a7ef0e7b4dfd618c38f8119d4e29ef0c9d5':
    raise ValueError('CIFAR reference differs from the reference used in the previous evals.')
get_config = runpy.run_path(sys.argv[1])['get_config']
for arm, workdir in zip(('vp', 'fox'), sys.argv[2:]):
    cfg = get_config('ode_heun_' + arm)
    configure_reproducibility(cfg, tensorflow_module=tf)
    manifest = json.loads((Path(workdir)/'run_manifest.json').read_text())
    payload = {k: v for k, v in manifest.items() if k != 'training_protocol_sha256'}
    expected = hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(',', ':'), ensure_ascii=True).encode()).hexdigest()
    if expected != manifest['training_protocol_sha256']:
        raise ValueError('Corrupt training manifest: ' + workdir)
    model_manifest = build_model_manifest(cfg)
    if model_manifest != manifest['model_manifest']:
        raise ValueError('ODE config changes the trained model: ' + arm)
    model = mutils.create_model(cfg)
    ema = ExponentialMovingAverage(model.parameters(), decay=cfg.model.ema_rate)
    optimizer = get_optimizer(cfg, model.parameters())
    state = dict(model=model, ema=ema, optimizer=optimizer, step=0,
                 model_protocol_sha256=model_manifest['model_protocol_sha256'],
                 training_protocol_sha256=expected)
    path = Path(workdir)/'checkpoints/checkpoint_12.pth'
    state = restore_checkpoint(str(path), state, cfg.device, restore_rng=False)
    if int(state['step']) != 600001 or not state.get('checkpoint_id'):
        raise ValueError('Expected identified 600k checkpoint: ' + str(path))
    ema.copy_to(model.parameters())
    model.eval()
    if not all(bool(torch.isfinite(p).all()) for p in model.parameters()):
        raise ValueError('Non-finite EMA weights: ' + arm)
    sde, eps = run_lib.get_sde(cfg, n_override=4)
    z = torch.zeros((2, 3, 32, 32), device=cfg.device)
    for method in ('ode_dpmpp_2m', 'ode_heun'):
        outputs, nfe = get_vp_ode_sampler(sde, z.shape, lambda x: x, method, eps, cfg.device)(model, z=z)
        if nfe != 5 or set(outputs) != {'x_t', 'tweedie_xt'} or not all(
                x.shape == z.shape and bool(torch.isfinite(x).all()) for x in outputs.values()):
            raise ValueError('Failed real-weights ODE smoke: ' + arm + '/' + method)
        print(f'EMA smoke passed: {arm}/{method}; trajectory NFE=4 + Tweedie=1', flush=True)
    print(f'600k ODE preflight passed: {path}; step={state["step"]}; id={state["checkpoint_id"]}; SHA256={digest(path)}', flush=True)
    del state, optimizer, ema, model, z, outputs
    gc.collect()
    torch.cuda.empty_cache()
PY_CHECK

run_cell() {
  local method="$1" arm="$2" seed="$3" workdir folder log
  if [[ "$arm" == vp ]]; then workdir="$VP_WORKDIR"; else workdir="$FOX_WORKDIR"; fi
  folder="eval_cifar_ddpmpp_deep600k_ode_v1_${method}_${arm}_common_ckpt12_nfe256_5000samples_batch512_seed${seed}"
  log="run_cifar10_ddpmpp_deep600k_${method}_${arm}_5k_seed${seed}.log"
  echo "=== CIFAR DDPM++ ODE: method=$method arm=$arm seed=$seed; ckpt12; samples=5000 batch=512; 256+1 NFE ==="
  echo 'Common lambda [-10, 9.115429865459795]; native conditioning; outputs x_t / Tweedie'
  python -u - --mode=eval --config="$TMP_CONFIG:${method}_${arm}" \
    --workdir="$workdir" --eval_folder="$folder" \
    --config.eval.sampling_seed="$seed" <<'PY_EVAL' 2>&1 | tee -a "$log"
import runpy
import sys
import torch
if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
    raise RuntimeError('Run in score-sde-rtx with exactly one visible CUDA GPU.')
torch.cuda.init()
import tensorflow as tf
tf.config.set_visible_devices([], 'GPU')
import evaluation
evaluation.jax = None
print('CIFAR ODE arguments:', ' '.join(sys.argv[1:]), flush=True)
sys.argv[0] = 'main.py'
runpy.run_path('main.py', run_name='__main__')
PY_EVAL
  echo "=== Finished $method/$arm seed=$seed; report: $workdir/$folder/report_12.npz ==="
}

# Complete matched VP/FOX pairs and both seeds for each solver.
for method in ode_dpmpp_2m ode_heun; do
  for seed in 0 1; do
    run_cell "$method" vp "$seed"
    run_cell "$method" fox "$seed"
  done
done

python -u - "$VP_WORKDIR" "$FOX_WORKDIR" <<'PY_SUMMARY'
import json
from pathlib import Path
import sys
import numpy as np

workdirs = dict(zip(('vp', 'fox'), map(Path, sys.argv[1:])))
cells = []
identities, runtimes, references = {}, set(), set()
for method in ('ode_dpmpp_2m', 'ode_heun'):
    for seed in (0, 1):
        for arm, workdir in workdirs.items():
            folder = workdir / (
                f'eval_cifar_ddpmpp_deep600k_ode_v1_{method}_{arm}_common_ckpt12_'
                f'nfe256_5000samples_batch512_seed{seed}')
            manifest = json.loads((folder/'manifest.json').read_text())
            config = manifest['config']
            if (config['sampling']['method'] != method
                    or config['sampling']['probability_flow'] is not True
                    or config['sampling']['logsnr_min'] != -10.
                    or config['sampling']['logsnr_max'] != 9.115429865459795
                    or config['eval']['sampling_seed'] != seed
                    or config['eval']['batch_size'] != 512
                    or config['eval']['sampling_num_scales'] != 256
                    or config['eval']['save_endpoint_states'] is not True
                    or tuple(config['eval']['endpoint_denoise_outputs']) != ('x_t', 'tweedie_xt')):
                raise ValueError('Unexpected ODE protocol: ' + str(folder))
            runtimes.add(manifest['source_sha256'])
            with np.load(folder/'report_12.npz', allow_pickle=False) as z:
                if (int(z['checkpoint_step']) != 600001 or int(z['num_samples']) != 5000
                        or int(z['nfe']) != 257 or z['protocol_sha256'].item() != manifest['protocol_sha256']):
                    raise ValueError('Unexpected report identity: ' + str(folder))
                identity = (z['checkpoint_id'].item(), z['training_protocol_sha256'].item())
                if identities.setdefault(arm, identity) != identity:
                    raise ValueError('Checkpoint changed between ODE cells: ' + arm)
                references.add(z['dataset_stats_id'].item())
                metrics = {m+'_'+v: float(z[m+'_'+v])
                           for m in ('fid', 'kid', 'IS') for v in ('x_t', 'tweedie_xt')}
                if not all(np.isfinite(x) for x in metrics.values()):
                    raise ValueError('Non-finite ODE metric: ' + str(folder))
            if len(list((folder/'ckpt_12').glob('statistics_*.npz'))) != 10:
                raise ValueError('Expected 10 saved feature batches: ' + str(folder))
            # float_x_t is UNCLIPPED NCHW float32 in inverse-scaled data coordinates.
            for i in (0, 9):
                with np.load(folder/'ckpt_12'/f'samples_{i}.npz', allow_pickle=False) as z:
                    x = z['float_x_t']
                    if x.shape != (512, 3, 32, 32) or x.dtype != np.float32 or not np.isfinite(x).all():
                        raise ValueError('Invalid saved endpoint float state: ' + str(folder))
            cells.append(dict(method=method, arm=arm, sampling_seed=seed, count=5000,
                              nfe=257, folder=str(folder), **metrics))
if len(cells) != 8 or len(runtimes) != 1 or len(references) != 1:
    raise ValueError('Incomplete matrix or mixed runtime/reference.')
print('\n=== CIFAR DDPM++ 600k ODE results: mean of separate 5k FIDs, not pooled FID ===')
print('method\tseed\tVP raw\tFOX raw\tVP Tweedie\tFOX Tweedie\tFOX-VP Tweedie')
for method in ('ode_dpmpp_2m', 'ode_heun'):
    pairs = {seed: {r['arm']: r for r in cells if r['method'] == method and r['sampling_seed'] == seed}
             for seed in (0, 1)}
    for seed, pair in pairs.items():
        a, b = pair['vp'], pair['fox']
        print('%s\t%d\t%.6f\t%.6f\t%.6f\t%.6f\t%+.6f' %
              (method, seed, a['fid_x_t'], b['fid_x_t'], a['fid_tweedie_xt'],
               b['fid_tweedie_xt'], b['fid_tweedie_xt']-a['fid_tweedie_xt']))
    a = np.mean([p['vp']['fid_tweedie_xt'] for p in pairs.values()])
    b = np.mean([p['fox']['fid_tweedie_xt'] for p in pairs.values()])
    print('%s mean: VP=%.6f FOX=%.6f FOX-VP=%+.6f' % (method, a, b, b-a))
print('Finished all 8 ODE evals. No training or EM regeneration was run. Reports/features/float states are in the new eval folders.')
PY_SUMMARY
