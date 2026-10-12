#!/usr/bin/env bash
# Continue the evaluated CIFAR VP / real FOX pair from checkpoint 4 to 200k.
# Run inside score-sde-rtx with one visible GPU. Both models run sequentially.
# --check: validate both sources and destination identities, without training.
# CIFAR_CONTINUE_MODEL=vp|fox|all (default all).
# Destination overrides: CIFAR_CONTINUE_VP_WORKDIR / CIFAR_CONTINUE_FOX_WORKDIR.
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"
if (( $# > 1 )) || { (( $# == 1 )) && [[ "$1" != --check ]]; }; then
  echo "Usage: bash $0 [--check]" >&2
  exit 2
fi
CHECK_ONLY=0
[[ "${1:-}" != --check ]] || CHECK_ONLY=1
MODEL="${CIFAR_CONTINUE_MODEL:-all}"
case "$MODEL" in vp|fox|all) ;; *) echo "Invalid CIFAR_CONTINUE_MODEL: $MODEL" >&2; exit 2 ;; esac

VP_SOURCE=workdirs/cifar10_vp_kappa1_50k/checkpoints/checkpoint_4.pth
FOX_SOURCE=workdirs/cifar10_fox_matern32_kappa1000_100k/checkpoints/checkpoint_4.pth
VP_DEST="${CIFAR_CONTINUE_VP_WORKDIR:-workdirs/cifar10_vp_200k_rtx_seed42}"
FOX_DEST="${CIFAR_CONTINUE_FOX_WORKDIR:-workdirs/cifar10_fox_matern32_kappa1000_200k_rtx_seed42}"

export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-12.0}"
export TFDS_DATA_DIR="${TFDS_DATA_DIR:-$ROOT_DIR/.tfds}"
export TORCH_EXTENSIONS_DIR="${TORCH_EXTENSIONS_DIR:-$ROOT_DIR/.torch_extensions_rtx}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-$ROOT_DIR/.cache_rtx}"
export MAX_JOBS="${MAX_JOBS:-8}"
export TF_ENABLE_ONEDNN_OPTS=0
export TF_NUM_INTRAOP_THREADS="${TF_NUM_INTRAOP_THREADS:-8}"
export TF_NUM_INTEROP_THREADS="${TF_NUM_INTEROP_THREADS:-2}"
export CIFAR_CONTINUE_VP_WORKDIR="$VP_DEST"
export CIFAR_CONTINUE_FOX_WORKDIR="$FOX_DEST"
[[ -d "$TFDS_DATA_DIR/cifar10" ]] || { echo "Missing prepared CIFAR-10: $TFDS_DATA_DIR/cifar10" >&2; exit 1; }
[[ "$VP_DEST" != "$FOX_DEST" ]] || { echo 'VP and FOX need distinct destinations.' >&2; exit 2; }
MODELS=(vp fox)
[[ "$MODEL" == all ]] || MODELS=("$MODEL")
for arm in "${MODELS[@]}"; do
  if [[ "$arm" == vp ]]; then source="$VP_SOURCE"; else source="$FOX_SOURCE"; fi
  [[ -s "$source" ]] || { echo "Missing source checkpoint: $source" >&2; exit 1; }
done

run_model() {
  local stage="$1" arm="$2" source destination
  if [[ "$arm" == vp ]]; then
    source="$VP_SOURCE"; destination="$VP_DEST"
  else
    source="$FOX_SOURCE"; destination="$FOX_DEST"
  fi
  python -u - "$stage" "$arm" "$source" "$destination" <<'PY_CONTINUE'
import gc
import hashlib
import json
import logging
import os
from pathlib import Path
import sys
import time
import uuid

import torch
if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
    raise RuntimeError('Run this script inside the RTX container with exactly one CUDA GPU.')
torch.cuda.init()
import tensorflow as tf
tf.config.set_visible_devices([], 'GPU')

import run_lib
import losses
from models import utils as mutils
from models.ema import ExponentialMovingAverage
from configs.vp.cifar10_ncsnpp_continuous import get_config as get_vp
from configs.fox.cifar10_ncsnpp_continuous_matern_3_2 import get_config as get_fox
from artifact_utils import (atomic_write_bytes, build_training_manifest,
                            ensure_training_manifest)
from reproducibility import configure_reproducibility, seed_everything
from utils import (latest_immutable_checkpoint, restore_checkpoint,
                   save_checkpoint)

TARGET_ITERS = 200000
SOURCE_STEP = 100001
SOURCE_SHA256 = {
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
    # Keep the historical time conditioning and uniform-t training loss.
    # The common log-SNR endpoints and batch 512 belonged to evaluation only.
    if getattr(config.model, 'noise_conditioning', 'time') != 'time':
        raise ValueError('Legacy CIFAR checkpoints require time conditioning.')
    if getattr(config.training, 'noise_distribution', 'uniform_time') != 'uniform_time':
        raise ValueError('Do not change the historical training noise distribution.')
    if config.optim.lr != 2e-4 or config.model.ema_rate != 0.9999:
        raise ValueError('Historical LR/EMA configuration changed.')
    config.lock()
    return config


def sha256(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def new_state(config, manifest):
    model = mutils.create_model(config)
    return dict(model=model, optimizer=losses.get_optimizer(config, model.parameters()),
                ema=ExponentialMovingAverage(model.parameters(), config.model.ema_rate),
                step=0, data_batches_consumed=0, eval_batches_consumed=0,
                model_protocol_sha256=manifest['model_manifest']['model_protocol_sha256'],
                training_protocol_sha256=manifest['training_protocol_sha256'])


def validate_state(state, config):
    # restore_checkpoint already strictly checks model keys and EMA shapes.
    # Adam.load_state_dict alone does not validate its moment tensor shapes.
    parameters = [p for p in state['model'].parameters() if p.requires_grad]
    optimizer = state['optimizer']
    if len(optimizer.state) != len(parameters):
        raise ValueError('Missing Adam states for trainable model parameters.')
    for group in optimizer.param_groups:
        expected = dict(lr=config.optim.lr, betas=(config.optim.beta1, 0.999),
                        eps=config.optim.eps, weight_decay=config.optim.weight_decay)
        for key, value in expected.items():
            if group[key] != value:
                raise ValueError('Source Adam %s=%r, expected %r' % (key, group[key], value))
    for parameter in parameters:
        moments = optimizer.state[parameter]
        adam_step = moments.get('step')
        if adam_step is None or int(adam_step) != int(state['step']):
            raise ValueError('Adam step does not match checkpoint state.step.')
        for key in ('exp_avg', 'exp_avg_sq'):
            tensor = moments.get(key)
            if tensor is None or tensor.shape != parameter.shape or not bool(torch.isfinite(tensor).all()):
                raise ValueError('Invalid Adam moment: %s' % key)
    if state['ema'].decay != config.model.ema_rate:
        raise ValueError('Source EMA decay changed.')
    for tensor in list(state['model'].state_dict().values()) + state['ema'].shadow_params:
        if not bool(torch.isfinite(tensor).all()):
            raise ValueError('Non-finite model/EMA tensor in checkpoint.')


def check_destination(destination, sources, manifest, migration):
    root = Path.cwd().resolve()
    if not destination.is_relative_to(root / 'workdirs'):
        raise ValueError('Continuation destination must be inside repository workdirs/.')
    for source_workdir in sources:
        if (destination == source_workdir or destination.is_relative_to(source_workdir)
                or source_workdir.is_relative_to(destination)):
            raise ValueError('Continuation destination overlaps a source workdir.')
    manifest_path = destination / 'run_manifest.json'
    migration_path = destination / 'continuation.json'
    if manifest_path.exists():
        if json.loads(manifest_path.read_text()) != manifest:
            raise ValueError('Destination has a different training code/config/environment. Use a new destination.')
        if not migration_path.exists() and {p.name for p in destination.iterdir()} == {'run_manifest.json'}:
            # Recover an interrupted import before any checkpoint was written.
            return False
        if not migration_path.exists() or json.loads(migration_path.read_text()) != migration:
            raise ValueError('Destination has no matching explicit continuation provenance.')
    elif destination.exists() and any(destination.iterdir()):
        raise ValueError('Destination is nonempty without a training manifest. Refusing implicit migration.')
    return manifest_path.exists()


def atomic_copy(source, destination):
    temporary = destination.with_name(destination.name + '.tmp-' + uuid.uuid4().hex)
    try:
        tf.io.gfile.copy(str(source), str(temporary), overwrite=False)
        tf.io.gfile.rename(str(temporary), str(destination), overwrite=False)
    finally:
        if temporary.exists():
            temporary.unlink()


def run(stage, arm, source, destination):
    if stage not in ('check', 'train') or arm not in SOURCE_SHA256:
        raise ValueError('Invalid continuation stage/model.')
    source = Path(source).resolve()
    destination = Path(destination).resolve()
    vp_dest = Path(os.environ['CIFAR_CONTINUE_VP_WORKDIR']).resolve()
    fox_dest = Path(os.environ['CIFAR_CONTINUE_FOX_WORKDIR']).resolve()
    if vp_dest.is_relative_to(fox_dest) or fox_dest.is_relative_to(vp_dest):
        raise ValueError('VP and FOX destinations must be distinct, non-overlapping workdirs.')
    source_hash = sha256(source)
    if source_hash != SOURCE_SHA256[arm]:
        raise ValueError('Source checkpoint SHA256 differs from the evaluated 100k model: %s' % source)
    config = make_config(arm)
    configure_reproducibility(config, tensorflow_module=tf)
    manifest = build_training_manifest(config)
    migration = dict(
        schema_version=1, model=arm, source_checkpoint=str(source),
        source_checkpoint_sha256=source_hash, source_state_step=SOURCE_STEP,
        source_training_manifest_available=False,
        source_rng_state_available=False, source_data_position_known=False,
        source_sde_provenance='Historical run_cifar10_vp_matern32_kappa1000_100k.sh; legacy weights contain no config.',
        restored_components=['model', 'optimizer', 'ema', 'step'],
        continuation_seed=42, data_batches_consumed_at_import=0,
        eval_batches_consumed_at_import=0,
        rng_policy='New seed-42 RNG stream in RTX environment; legacy data position is reset.',
        model_protocol_sha256=manifest['model_manifest']['model_protocol_sha256'],
        training_protocol_sha256=manifest['training_protocol_sha256'])
    sources = [(Path('workdirs') / name).resolve() for name in
               ('cifar10_vp_kappa1_50k', 'cifar10_fox_matern32_kappa1000_100k')]
    existing = check_destination(destination, sources, manifest, migration)
    state = new_state(config, manifest)
    restore_checkpoint(str(source), state, config.device, restore_rng=False)
    if int(state['step']) != SOURCE_STEP:
        raise ValueError('Expected source state.step=100001, got %r' % state['step'])
    validate_state(state, config)
    meta_path = destination / 'checkpoints-meta' / 'checkpoint.pth'
    imported_path = destination / 'checkpoints' / 'checkpoint_4.pth'
    initial_step = SOURCE_STEP
    if existing:
        # Validate the actual resume state before either model starts training.
        immutable, immutable_step = latest_immutable_checkpoint(
            str(destination / 'checkpoints'), config.training.snapshot_freq)
        if meta_path.exists():
            restore_checkpoint(str(meta_path), state, config.device, restore_rng=False)
            initial_step = int(state['step'])
        if immutable_step is not None and immutable_step > initial_step:
            restore_checkpoint(immutable, state, config.device, restore_rng=False)
            initial_step = int(state['step'])
            if initial_step != immutable_step:
                raise ValueError('Resume checkpoint filename/state.step mismatch.')
        if not SOURCE_STEP <= initial_step <= TARGET_ITERS + 1:
            raise ValueError('Destination checkpoint step is outside the continuation range.')
        validate_state(state, config)
        if not imported_path.exists() and (meta_path.exists() or immutable is not None):
            raise ValueError('Destination has resume artifacts but no imported parent checkpoint.')
    logging.info('%s %s: source step=%d, resume step=%d, target state.step=%d, train batch=128, LR=2e-4, EMA=.9999',
                 stage, arm, SOURCE_STEP, initial_step, TARGET_ITERS + 1)
    logging.info('Destination: %s; source SHA256: %s; GPU: %s', destination, source_hash, torch.cuda.get_device_name(0))
    if stage == 'check':
        logging.info('Preflight passed: strict model/EMA load and Adam states checked; no training or workdir writes.')
        return
    final_path = destination / 'checkpoints' / 'checkpoint_8.pth'
    if initial_step == TARGET_ITERS + 1:
        if not final_path.exists():
            raise ValueError('Destination reached 200k but its final checkpoint_8.pth is missing.')
        logging.info('%s already reached 200k; keeping existing checkpoints and skipping training.', arm)
        return

    if not existing:
        # Write the current identity before importing any legacy state. The
        # historical workdirs are never passed to run_lib.train.
        ensure_training_manifest(str(destination), manifest)
        atomic_write_bytes(str(destination / 'continuation.json'),
                           (json.dumps(migration, sort_keys=True, indent=2) + '\n').encode(), overwrite=False)
    if not imported_path.exists():
        if meta_path.exists():
            raise ValueError('Destination has a resume meta but no imported parent checkpoint.')
        state['data_batches_consumed'] = 0
        state['eval_batches_consumed'] = 0
        seed_everything(config.seed, tensorflow_module=tf)
        save_checkpoint(str(imported_path), state, overwrite=False)
    if not meta_path.exists():
        meta_path.parent.mkdir(parents=True, exist_ok=True)
        atomic_copy(imported_path, meta_path)
    del state
    gc.collect()
    torch.cuda.empty_cache()
    handler = logging.FileHandler(destination / 'stdout.txt', mode='a')
    handler.setFormatter(logging.Formatter('%(asctime)s %(levelname)s %(message)s'))
    logger = logging.getLogger()
    logger.addHandler(handler)
    start = time.monotonic()
    try:
        logging.info('=== %s continuation to 200k started ===', arm)
        run_lib.train(config, str(destination))
        final_state = torch.load(str(final_path), map_location='cpu', weights_only=False)
        if (int(final_state['step']) != TARGET_ITERS + 1 or
                final_state.get('training_protocol_sha256') != manifest['training_protocol_sha256'] or
                final_state.get('model_protocol_sha256') != manifest['model_manifest']['model_protocol_sha256']):
            raise ValueError('Final checkpoint step/provenance does not match this 200k continuation.')
        del final_state
        logging.info('=== %s continuation finished; wall time %.1f seconds; final checkpoint_8.pth ===',
                     arm, time.monotonic() - start)
    except Exception:
        logging.exception('Continuation failed; rerun with the same workdir to resume.')
        raise
    finally:
        logger.removeHandler(handler)
        handler.close()


logging.basicConfig(level=logging.INFO, stream=sys.stdout,
                    format='%(asctime)s %(levelname)s %(message)s', force=True)
run(*sys.argv[1:])
PY_CONTINUE
}

# Catch a bad FOX checkpoint/config before spending time on the VP continuation.
for arm in "${MODELS[@]}"; do run_model check "$arm"; done
if (( CHECK_ONLY )); then
  echo 'Both selected preflight checks passed. No training started.'
  exit 0
fi
for arm in "${MODELS[@]}"; do
  echo "=== Continuing CIFAR $arm from 100k to 200k ==="
  run_model train "$arm"
done
echo '=== CIFAR continuation to 200k finished ==='
