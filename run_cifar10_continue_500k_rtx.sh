#!/usr/bin/env bash
# Standalone continuation of the RTX CIFAR VP / real FOX pair, 400k -> 500k.
# Requires checkpoint_16.pth (state.step=400001), manifest and saved RNG/data.
# --check validates both models without training or workdir writes.
# CIFAR_CONTINUE_MODEL=vp|fox|all; optional CIFAR_CONTINUE_{VP,FOX}_WORKDIR.
set -euo pipefail
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"
if (( $# > 1 )) || { (( $# == 1 )) && [[ "$1" != --check ]]; }; then
  echo "Usage: bash $0 [--check]" >&2; exit 2
fi
CHECK_ONLY=0
[[ "${1:-}" != --check ]] || CHECK_ONLY=1
MODEL="${CIFAR_CONTINUE_MODEL:-all}"
case "$MODEL" in vp|fox|all) ;; *) echo "Invalid CIFAR_CONTINUE_MODEL: $MODEL" >&2; exit 2 ;; esac
VP_WORKDIR="${CIFAR_CONTINUE_VP_WORKDIR:-workdirs/cifar10_vp_200k_rtx_seed42}"
FOX_WORKDIR="${CIFAR_CONTINUE_FOX_WORKDIR:-workdirs/cifar10_fox_matern32_kappa1000_200k_rtx_seed42}"
export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-12.0}"
export TFDS_DATA_DIR="${TFDS_DATA_DIR:-$ROOT_DIR/.tfds}"
export TORCH_EXTENSIONS_DIR="${TORCH_EXTENSIONS_DIR:-$ROOT_DIR/.torch_extensions_rtx}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-$ROOT_DIR/.cache_rtx}"
export MAX_JOBS="${MAX_JOBS:-8}"
export TF_ENABLE_ONEDNN_OPTS=0
export TF_NUM_INTRAOP_THREADS="${TF_NUM_INTRAOP_THREADS:-8}"
export TF_NUM_INTEROP_THREADS="${TF_NUM_INTEROP_THREADS:-2}"
[[ -d "$TFDS_DATA_DIR/cifar10" ]] || { echo "Missing prepared CIFAR-10: $TFDS_DATA_DIR/cifar10" >&2; exit 1; }
MODELS=(vp fox)
[[ "$MODEL" == all ]] || MODELS=("$MODEL")
for arm in "${MODELS[@]}"; do
  if [[ "$arm" == vp ]]; then workdir="$VP_WORKDIR"; else workdir="$FOX_WORKDIR"; fi
  for required in run_manifest.json continuation.json checkpoints/checkpoint_16.pth; do
    [[ -s "$workdir/$required" ]] || { echo "Missing 400k training artifact: $workdir/$required" >&2; exit 1; }
  done
done

run_model() {
  local stage="$1" arm="$2" workdir other
  if [[ "$arm" == vp ]]; then workdir="$VP_WORKDIR"; other="$FOX_WORKDIR";
  else workdir="$FOX_WORKDIR"; other="$VP_WORKDIR"; fi
  python -u - "$stage" "$arm" "$workdir" "$other" <<'PY_500K'
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
from configs.vp.cifar10_ncsnpp_continuous import get_config as get_vp
from configs.fox.cifar10_ncsnpp_continuous_matern_3_2 import get_config as get_fox
from artifact_utils import atomic_write_bytes, build_training_manifest
from reproducibility import configure_reproducibility
from utils import latest_immutable_checkpoint, restore_checkpoint

TARGET_ITERS = 500000
SOURCE_STEP = 400001
LEGACY_IMPORT_STEP = 100001
LEGACY_PARENT_SHA256 = {
    'vp': '065b8f7288abff4789ccaab4e05228b8b896de9390cac0a702f9cc54ef19aaea',
    'fox': 'b0007e178bdec5cf949dd137a2ed3a2e2ff1e35e4c0f7da8aee059f526d3fa8d',
}


def make_config(arm):
    config = get_vp() if arm == 'vp' else get_fox()
    if arm == 'fox':
        config.model.fox_u = -5.0
        config.model.fox_drift_schedule = 'constant'
        config.model.fox_matern_length_scale = 346.4101615138
        config.model.fox_target_terminal_variance = 1.0
        if hasattr(config.model, 'fox_normalize_scale'):
            config.model.fox_normalize_scale = False
    config.training.batch_size = 128
    config.training.n_iters = TARGET_ITERS
    config.training.snapshot_freq = 25000
    config.training.snapshot_freq_for_preemption = 5000
    config.training.snapshot_sampling = False
    config.training.log_freq = 50
    config.training.eval_freq = 500
    config.seed = 42
    config.deterministic = False
    config.allow_tf32 = False
    config.cudnn_benchmark = False
    config.allow_fp16_reduced_precision_reduction = False
    if (getattr(config.model, 'noise_conditioning', 'time') != 'time' or
            getattr(config.training, 'noise_distribution', 'uniform_time') != 'uniform_time'):
        raise ValueError('Keep the historical time conditioning and uniform-t training loss.')
    if config.optim.lr != 2e-4 or config.model.ema_rate != 0.9999:
        raise ValueError('Historical learning rate/EMA changed.')
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
    if state.get('_rng_state_to_restore') is None:
        raise ValueError('Resume checkpoint has no saved RNG state.')
    if int(state.get('data_batches_consumed', 0)) != step - LEGACY_IMPORT_STEP:
        raise ValueError('Resume data position does not match the optimizer step.')
    parameters = [p for p in state['model'].parameters() if p.requires_grad]
    optimizer = state['optimizer']
    if len(optimizer.state) != len(parameters):
        raise ValueError('Missing Adam states for trainable parameters.')
    expected = dict(lr=config.optim.lr, betas=(config.optim.beta1, 0.999),
                    eps=config.optim.eps, weight_decay=config.optim.weight_decay)
    for group in optimizer.param_groups:
        for key, value in expected.items():
            if group[key] != value:
                raise ValueError('Checkpoint Adam %s differs from the historical setting.' % key)
    for parameter in parameters:
        moments = optimizer.state[parameter]
        if moments.get('step') is None or int(moments['step']) != step:
            raise ValueError('Adam step differs from state.step.')
        for key in ('exp_avg', 'exp_avg_sq'):
            tensor = moments.get(key)
            if tensor is None or tensor.shape != parameter.shape or not bool(torch.isfinite(tensor).all()):
                raise ValueError('Invalid Adam moment: %s' % key)
    if state['ema'].decay != config.model.ema_rate:
        raise ValueError('Checkpoint EMA decay changed.')
    for tensor in list(state['model'].state_dict().values()) + state['ema'].shadow_params:
        if not bool(torch.isfinite(tensor).all()):
            raise ValueError('Non-finite model/EMA tensor.')


def run(stage, arm, workdir, other):
    if stage not in ('check', 'train') or arm not in LEGACY_PARENT_SHA256:
        raise ValueError('Invalid continuation stage/model.')
    workdir = Path(workdir).resolve()
    other = Path(other).resolve()
    if not workdir.is_relative_to(Path.cwd().resolve() / 'workdirs'):
        raise ValueError('Continuation workdir must be inside repository workdirs/.')
    if workdir.is_relative_to(other) or other.is_relative_to(workdir):
        raise ValueError('VP and FOX workdirs must be distinct and non-overlapping.')
    config = make_config(arm)
    configure_reproducibility(config, tensorflow_module=tf)
    manifest = build_training_manifest(config)
    if json.loads((workdir / 'run_manifest.json').read_text()) != manifest:
        raise ValueError('Workdir training code/config/environment changed; refusing implicit migration.')
    lineage = json.loads((workdir / 'continuation.json').read_text())
    if (lineage.get('model') != arm or lineage.get('source_state_step') != LEGACY_IMPORT_STEP or
            lineage.get('source_checkpoint_sha256') != LEGACY_PARENT_SHA256[arm] or
            lineage.get('training_protocol_sha256') != manifest['training_protocol_sha256'] or
            lineage.get('model_protocol_sha256') != manifest['model_manifest']['model_protocol_sha256']):
        raise ValueError('Workdir is not a continuation of the evaluated CIFAR model.')
    model = mutils.create_model(config)
    state = dict(model=model, optimizer=losses.get_optimizer(config, model.parameters()),
                 ema=ExponentialMovingAverage(model.parameters(), config.model.ema_rate),
                 step=0, data_batches_consumed=0, eval_batches_consumed=0,
                 model_protocol_sha256=manifest['model_manifest']['model_protocol_sha256'],
                 training_protocol_sha256=manifest['training_protocol_sha256'])
    parent = workdir / 'checkpoints' / 'checkpoint_16.pth'
    restore_checkpoint(str(parent), state, config.device, defer_rng=True)
    if int(state['step']) != SOURCE_STEP or state['checkpoint_id'].startswith('legacy-sha256-'):
        raise ValueError('Require the current-format 400k checkpoint_16 with state.step=400001.')
    validate_state(state, config)
    phase = dict(
        schema_version=1, model=arm, target_iters=TARGET_ITERS,
        source_checkpoint=str(parent), source_checkpoint_sha256=sha256(parent),
        source_checkpoint_id=state['checkpoint_id'], source_state_step=SOURCE_STEP,
        source_rng_state_available=True,
        source_data_batches_consumed=int(state['data_batches_consumed']),
        source_eval_batches_consumed=int(state.get('eval_batches_consumed', 0)),
        resume_policy='Restore existing checkpoint RNG and data counters; continue from meta/latest immutable.',
        model_protocol_sha256=manifest['model_manifest']['model_protocol_sha256'],
        training_protocol_sha256=manifest['training_protocol_sha256'])
    phase_path = workdir / 'continuation_500k.json'
    if phase_path.exists() and json.loads(phase_path.read_text()) != phase:
        raise ValueError('The 500k record belongs to a different 400k parent.')
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
    logging.info('%s %s: parent step=400001; actual resume step=%d; target step=500001; training batch=128',
                 stage, arm, initial_step)
    logging.info('Workdir: %s; parent SHA256: %s; GPU: %s',
                 workdir, phase['source_checkpoint_sha256'], torch.cuda.get_device_name(0))
    if stage == 'check':
        logging.info('Preflight passed: model/Adam/EMA/RNG/data and manifest checked; no training/workdir writes.')
        return
    if not phase_path.exists():
        atomic_write_bytes(str(phase_path),
                           (json.dumps(phase, sort_keys=True, indent=2) + '\n').encode(), overwrite=False)
    final = workdir / 'checkpoints' / 'checkpoint_20.pth'
    if initial_step == TARGET_ITERS + 1:
        if not final.is_file():
            raise ValueError('State reached 500k but checkpoint_20.pth is missing.')
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
        logging.info('=== %s continuation from %d to 500001 started ===', arm, initial_step)
        # Standard training restores RNG/data and recovers absent/stale meta
        # from the newest immutable checkpoint in this workdir.
        run_lib.train(config, str(workdir))
        saved = torch.load(str(final), map_location='cpu', weights_only=False)
        if (int(saved['step']) != TARGET_ITERS + 1 or
                saved.get('training_protocol_sha256') != manifest['training_protocol_sha256'] or
                saved.get('model_protocol_sha256') != manifest['model_manifest']['model_protocol_sha256']):
            raise ValueError('Final checkpoint step/provenance differs from this 500k continuation.')
        del saved
        logging.info('=== %s finished; checkpoint_20.pth; wall time %.1f seconds ===', arm, time.monotonic() - start)
    except Exception:
        logging.exception('Continuation failed; rerun this script to resume.')
        raise
    finally:
        logger.removeHandler(handler)
        handler.close()


logging.basicConfig(level=logging.INFO, stream=sys.stdout,
                    format='%(asctime)s %(levelname)s %(message)s', force=True)
run(*sys.argv[1:])
PY_500K
}

for arm in "${MODELS[@]}"; do run_model check "$arm"; done
if (( CHECK_ONLY )); then echo 'Preflight passed. No training started.'; exit 0; fi
for arm in "${MODELS[@]}"; do
  echo "=== Continuing CIFAR $arm from 400k to 500k ==="
  run_model train "$arm"
done
echo '=== CIFAR continuation to 500k finished ==='
