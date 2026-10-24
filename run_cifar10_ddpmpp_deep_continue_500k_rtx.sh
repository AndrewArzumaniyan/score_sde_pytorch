#!/usr/bin/env bash
# Continue the fresh DDPM++ deep VP/real FOX pair in its existing workdirs.
# Parent: checkpoint_8.pth at state.step=400001; target: checkpoint_10.pth.
# --check checks both arms without training or workdir writes.
set -euo pipefail
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"
if (( $# > 1 )) || { (( $# == 1 )) && [[ "$1" != --check ]]; }; then
  echo "Usage: bash $0 [--check]" >&2
  exit 2
fi
CHECK_ONLY=0
[[ "${1:-}" != --check ]] || CHECK_ONLY=1
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
[[ -d "$TFDS_DATA_DIR/cifar10" ]] || {
  echo "Missing prepared CIFAR-10: $TFDS_DATA_DIR/cifar10" >&2
  exit 1
}
for config in \
  configs/vp/cifar10_ddpmpp_deep_continuous_matched.py \
  configs/fox/cifar10_ddpmpp_deep_continuous_matern_3_2_kappa1000.py
do
  [[ -s "$config" ]] || { echo "Missing config: $config" >&2; exit 1; }
done
for workdir in "$VP_WORKDIR" "$FOX_WORKDIR"; do
  for required in run_manifest.json checkpoints/checkpoint_8.pth; do
    [[ -s "$workdir/$required" ]] || {
      echo "Missing DDPM++ training artifact: $workdir/$required" >&2
      exit 1
    }
  done
done

run_model() {
  local stage="$1" arm="$2" workdir other
  if [[ "$arm" == vp ]]; then workdir="$VP_WORKDIR"; other="$FOX_WORKDIR";
  else workdir="$FOX_WORKDIR"; other="$VP_WORKDIR"; fi
  python -u - "$stage" "$arm" "$workdir" "$other" <<'PY_CONTINUE'
import gc
import hashlib
import json
import logging
from pathlib import Path
import sys
import time

import torch
if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
    raise RuntimeError('Run inside the RTX container with exactly one visible CUDA GPU.')
torch.cuda.init()
import tensorflow as tf
tf.config.set_visible_devices([], 'GPU')

import run_lib
import losses
from models import utils as mutils
from models.ema import ExponentialMovingAverage
from configs.vp.cifar10_ddpmpp_deep_continuous_matched import get_config as get_vp
from configs.fox.cifar10_ddpmpp_deep_continuous_matern_3_2_kappa1000 import get_config as get_fox
from artifact_utils import atomic_write_bytes, build_training_manifest
from reproducibility import configure_reproducibility
from utils import latest_immutable_checkpoint, restore_checkpoint

SOURCE_STEP = 400001
TARGET_ITERS = 500000


def make_config(arm):
    config = get_vp() if arm == 'vp' else get_fox()
    config.training.n_iters = TARGET_ITERS
    config.training.batch_size = 128
    config.seed = 42
    if (config.training.snapshot_freq != 50000 or
            config.training.eval_freq != 100 or
            config.training.snapshot_sampling or
            config.model.num_res_blocks != 8 or
            config.model.fir or config.model.progressive_input != 'none' or
            config.training.noise_distribution != 'uniform_time' or
            config.model.noise_conditioning != 'time'):
        raise ValueError('Keep the original DDPM++ deep training config and checkpoint cadence.')
    config.lock()
    return config


def sha256(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def validate_state(state, config):
    step = int(state['step'])
    if not SOURCE_STEP <= step <= TARGET_ITERS + 1:
        raise ValueError('Resume state.step must be between 400001 and 500001.')
    rng = state.get('_rng_state_to_restore')
    if not rng or any(key not in rng for key in ('python', 'numpy', 'torch_cpu')):
        raise ValueError('Resume checkpoint has no complete saved CPU RNG state.')
    if len(rng.get('torch_cuda', [])) != 1:
        raise ValueError('Resume requires saved RNG for exactly one CUDA GPU.')
    # This pair was trained from scratch, unlike the imported NCSN++ pair.
    if int(state.get('data_batches_consumed', -1)) != step:
        raise ValueError('Training data counter must equal state.step for the fresh DDPM++ run.')
    expected_eval_batches = (step - 1) // config.training.eval_freq + 1
    if int(state.get('eval_batches_consumed', -1)) != expected_eval_batches:
        raise ValueError('Validation data counter differs from the original eval cadence.')
    parameters = [p for p in state['model'].parameters() if p.requires_grad]
    optimizer = state['optimizer']
    if len(optimizer.state) != len(parameters):
        raise ValueError('Missing Adam states for trainable parameters.')
    expected = dict(lr=config.optim.lr, betas=(config.optim.beta1, 0.999),
                    eps=config.optim.eps, weight_decay=config.optim.weight_decay)
    for group in optimizer.param_groups:
        for key, value in expected.items():
            if group[key] != value:
                raise ValueError(f'Checkpoint Adam {key} differs from the training config.')
    for parameter in parameters:
        moments = optimizer.state[parameter]
        if moments.get('step') is None or int(moments['step']) != step:
            raise ValueError('Adam step differs from state.step.')
        for key in ('exp_avg', 'exp_avg_sq'):
            tensor = moments.get(key)
            if tensor is None or tensor.shape != parameter.shape or not bool(torch.isfinite(tensor).all()):
                raise ValueError(f'Invalid Adam moment: {key}')
    if state['ema'].decay != config.model.ema_rate:
        raise ValueError('Checkpoint EMA decay differs from the training config.')
    for tensor in list(state['model'].state_dict().values()) + state['ema'].shadow_params:
        if not bool(torch.isfinite(tensor).all()):
            raise ValueError('Non-finite model/EMA tensor.')


def validate_final(path, manifest):
    saved = torch.load(str(path), map_location='cpu', weights_only=False)
    if (int(saved['step']) != TARGET_ITERS + 1 or
            saved.get('training_protocol_sha256') != manifest['training_protocol_sha256'] or
            saved.get('model_protocol_sha256') != manifest['model_manifest']['model_protocol_sha256']):
        raise ValueError('Final checkpoint step/provenance differs from this 500k continuation.')


def run(stage, arm, workdir, other):
    if stage not in ('check', 'train') or arm not in ('vp', 'fox'):
        raise ValueError('Invalid continuation stage/model.')
    workdir, other = Path(workdir).resolve(), Path(other).resolve()
    if not workdir.is_relative_to(Path.cwd().resolve() / 'workdirs'):
        raise ValueError('Continuation workdir must be inside repository workdirs/.')
    if workdir.is_relative_to(other) or other.is_relative_to(workdir):
        raise ValueError('VP and FOX workdirs must be distinct and non-overlapping.')
    config = make_config(arm)
    configure_reproducibility(config, tensorflow_module=tf)
    manifest = build_training_manifest(config)
    if json.loads((workdir / 'run_manifest.json').read_text()) != manifest:
        raise ValueError('Training code/config/environment changed; refusing implicit migration.')
    model = mutils.create_model(config)
    state = dict(model=model, optimizer=losses.get_optimizer(config, model.parameters()),
                 ema=ExponentialMovingAverage(model.parameters(), config.model.ema_rate),
                 step=0, data_batches_consumed=0, eval_batches_consumed=0,
                 model_protocol_sha256=manifest['model_manifest']['model_protocol_sha256'],
                 training_protocol_sha256=manifest['training_protocol_sha256'])
    parent = workdir / 'checkpoints' / 'checkpoint_8.pth'
    restore_checkpoint(str(parent), state, config.device, defer_rng=True)
    if int(state['step']) != SOURCE_STEP or state['checkpoint_id'].startswith('legacy-sha256-'):
        raise ValueError('Require the current-format 400k checkpoint_8 with state.step=400001.')
    validate_state(state, config)
    phase = dict(
        schema_version=1, model=arm, architecture='ddpmpp_deep', target_iters=TARGET_ITERS,
        source_checkpoint=str(parent), source_checkpoint_sha256=sha256(parent),
        source_checkpoint_id=state['checkpoint_id'], source_state_step=SOURCE_STEP,
        source_rng_state_available=True,
        source_data_batches_consumed=int(state['data_batches_consumed']),
        source_eval_batches_consumed=int(state['eval_batches_consumed']),
        resume_policy='Restore model/Adam/EMA/RNG/data from meta/latest immutable in the same workdir.',
        model_protocol_sha256=manifest['model_manifest']['model_protocol_sha256'],
        training_protocol_sha256=manifest['training_protocol_sha256'])
    phase_path = workdir / 'continuation_ddpmpp_deep_500k.json'
    if phase_path.exists() and json.loads(phase_path.read_text()) != phase:
        raise ValueError('The 500k continuation record belongs to a different 400k parent.')
    meta = workdir / 'checkpoints-meta' / 'checkpoint.pth'
    if meta.exists():
        restore_checkpoint(str(meta), state, config.device, defer_rng=True)
    immutable, immutable_step = latest_immutable_checkpoint(
        str(workdir / 'checkpoints'), config.training.snapshot_freq)
    if immutable_step is not None and immutable_step > int(state['step']):
        restore_checkpoint(immutable, state, config.device, defer_rng=True)
        if int(state['step']) != immutable_step:
            raise ValueError('Immutable checkpoint filename/state.step mismatch.')
    validate_state(state, config)
    initial_step = int(state['step'])
    logging.info('%s DDPM++ deep %s: parent step=400001; actual resume step=%d; target step=500001; batch=128',
                 stage, arm, initial_step)
    logging.info('Workdir: %s; parent SHA256: %s; GPU: %s',
                 workdir, phase['source_checkpoint_sha256'], torch.cuda.get_device_name(0))
    final = workdir / 'checkpoints' / 'checkpoint_10.pth'
    if initial_step == TARGET_ITERS + 1:
        if not final.is_file():
            raise ValueError('State reached 500k but checkpoint_10.pth is missing.')
        validate_final(final, manifest)
    if stage == 'check':
        logging.info('Preflight passed: model/Adam/EMA/RNG/data and manifest checked; no workdir writes.')
        return
    if not phase_path.exists():
        atomic_write_bytes(str(phase_path),
                           (json.dumps(phase, sort_keys=True, indent=2) + '\n').encode(), overwrite=False)
    if initial_step == TARGET_ITERS + 1:
        logging.info('%s already reached 500k; skipping updates.', arm)
        return
    del state, model
    gc.collect()
    torch.cuda.empty_cache()
    handler = logging.FileHandler(workdir / 'stdout.txt', mode='a')
    handler.setFormatter(logging.Formatter('%(asctime)s %(levelname)s %(message)s'))
    logger = logging.getLogger()
    logger.addHandler(handler)
    start = time.monotonic()
    try:
        logging.info('=== DDPM++ deep %s continuation from %d to 500001 started ===', arm, initial_step)
        run_lib.train(config, str(workdir))
        validate_final(final, manifest)
        logging.info('=== DDPM++ deep %s finished; checkpoint_10.pth; wall time %.1f seconds ===',
                     arm, time.monotonic() - start)
    except Exception:
        logging.exception('Continuation failed; rerun this script to resume.')
        raise
    finally:
        logger.removeHandler(handler)
        handler.close()


logging.basicConfig(level=logging.INFO, stream=sys.stdout,
                    format='%(asctime)s %(levelname)s %(message)s', force=True)
run(*sys.argv[1:])
PY_CONTINUE
}

# Validate both arms before either one advances.
for arm in vp fox; do run_model check "$arm"; done
if (( CHECK_ONLY )); then echo 'Preflight passed. No training started.'; exit 0; fi
for arm in vp fox; do
  echo "=== Continuing DDPM++ deep CIFAR $arm from 400k to 500k ==="
  run_model train "$arm"
done
echo '=== DDPM++ deep VP/FOX continuation to 500k finished ==='
