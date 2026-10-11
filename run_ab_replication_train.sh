#!/usr/bin/env bash
# Independent A/B replication. Run against the verified frozen runtime;
# AB_REPLICATION_SOURCE_ROOT permits mounting this launcher separately.
# Default: audit only. AB_REPLICATION_MODE=train audits, then trains A and B.
set -euo pipefail

[[ -f /.dockerenv ]] || { echo 'Run this launcher inside Docker.' >&2; exit 2; }
LAUNCHER_PATH="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/$(basename "${BASH_SOURCE[0]}")"
ROOT_DIR="${AB_REPLICATION_SOURCE_ROOT:-$(dirname "$LAUNCHER_PATH")}"
cd "$ROOT_DIR"

MODE="${AB_REPLICATION_MODE:-preflight}"
SEED="${AB_REPLICATION_SEED:-43}"
case "$MODE" in preflight|train) ;; *) echo 'Mode must be preflight or train.' >&2; exit 2 ;; esac
case "$SEED" in 43|44) ;; *) echo 'Replication seed must be 43 or 44.' >&2; exit 2 ;; esac
[[ "${CUDA_VISIBLE_DEVICES:-}" == 0 ]] || {
  echo 'Expose one physical GPU through Docker, with CUDA_VISIBLE_DEVICES=0.' >&2; exit 2; }

export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-9.0}"
export TFDS_DATA_DIR="${TFDS_DATA_DIR:-/workspace/score_sde_pytorch/.tfds}"
export TORCH_EXTENSIONS_DIR="${TORCH_EXTENSIONS_DIR:-/workspace/score_sde_pytorch/.torch_extensions}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-/workspace/score_sde_pytorch/.cache}"
export MAX_JOBS="${MAX_JOBS:-8}"
export AB_REPLICATION_SEED="$SEED"
export AB_REPLICATION_LAUNCHER="$LAUNCHER_PATH"
export AB_REPLICATION_REFERENCE_ROOT="${AB_REPLICATION_REFERENCE_ROOT:-workdirs}"
export AB_REPLICATION_A_WORKDIR="workdirs/celeba_ab_replication_A_vp_pvp_seed${SEED}_200k"
export AB_REPLICATION_B_WORKDIR="workdirs/celeba_ab_replication_B_vp_pfox_seed${SEED}_200k"

for workdir in "$AB_REPLICATION_A_WORKDIR" "$AB_REPLICATION_B_WORKDIR"; do
  [[ ! -e "$workdir" && ! -L "$workdir" ]] || {
    echo "Fresh replication requires an absent workdir: $workdir" >&2
    echo 'Use a separately reviewed resume command for interrupted training.' >&2; exit 1; }
done
mkdir -p logs "$TFDS_DATA_DIR" "$TORCH_EXTENSIONS_DIR" "$XDG_CACHE_HOME"
RUN_ID="$(date -u +%Y%m%dT%H%M%SZ)_$$"
export AB_REPLICATION_AUDIT="logs/ab_replication_seed${SEED}_${RUN_ID}_preflight.json"
exec > >(tee -a "logs/ab_replication_seed${SEED}_${RUN_ID}_${MODE}.log") 2>&1

python -u - <<'PY'
import gc
import hashlib
import json
import os
from pathlib import Path

import torch
if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
    raise RuntimeError('The preflight requires exactly one visible CUDA GPU.')
torch.cuda.init()
import run_lib
from artifact_utils import _canonicalize, _environment_manifest, _source_fingerprint
from configs.factorial.celeba_a_vp_pvp import get_config as get_a
from configs.factorial.celeba_b_vp_pfox import get_config as get_b
from reproducibility import configure_reproducibility

EXPECTED_SOURCE = 'ddf4c6acbc51b67da7a3465f2b90678ff559a628f6925d47e991350b02ff54f7'
actual_source = _source_fingerprint(os.getcwd())
if actual_source != EXPECTED_SOURCE:
    raise RuntimeError('Training runtime differs from seed 42: '
                       f'expected {EXPECTED_SOURCE}, got {actual_source}. '
                       'Use the original frozen runtime; do not bypass this check.')

seed = int(os.environ['AB_REPLICATION_SEED'])
reference_root = Path(os.environ['AB_REPLICATION_REFERENCE_ROOT'])
references = {}
configs = {}
for arm, getter, old_name in (
    ('A', get_a, 'celeba_factorial_A_vp_pvp_50k'),
    ('B', get_b, 'celeba_factorial_B_vp_pfox_50k'),
):
    path = reference_root / old_name / 'run_manifest.json'
    reference = json.loads(path.read_text())
    if reference['source_sha256'] != EXPECTED_SOURCE or reference['config']['seed'] != 42:
        raise RuntimeError(f'Unexpected seed-42 reference: {path}')
    config = getter()
    config.seed = seed
    config.training.batch_size = 128
    config.training.n_iters = 200000
    config.training.snapshot_freq = 50000
    config.training.snapshot_freq_for_preemption = 5000
    config.training.snapshot_sampling = False
    config.training.log_freq = 50
    config.training.eval_freq = 500
    config.allow_tf32 = True
    config.cudnn_benchmark = True
    configs[arm] = config
    references[arm] = reference

def scientific_config(config):
    # Seed and snapshot cadence are intentionally changed; training itself is not.
    result = json.loads(json.dumps(_canonicalize(config)))
    result.pop('eval', None)
    result.pop('seed', None)
    for key in ('n_iters', 'log_freq', 'eval_freq', 'snapshot_freq',
                'snapshot_freq_for_preemption', 'snapshot_sampling'):
        result['training'].pop(key, None)
    return result

for arm in configs:
    if scientific_config(configs[arm]) != scientific_config(references[arm]['config']):
        raise RuntimeError(f'Arm {arm} config differs from its seed-42 reference.')

a = scientific_config(configs['A'])
b = scientific_config(configs['B'])
for config in (a, b):
    config['training'].pop('factorial_arm')
    config['training'].pop('noise_distribution')
if a != b:
    raise RuntimeError('A and B differ outside the arm name and allocation.')

environment = _environment_manifest()
for arm, reference in references.items():
    for key in ('python', 'numpy', 'tensorflow', 'torch', 'torch_cuda', 'cudnn', 'packages'):
        if environment[key] != reference['environment'][key]:
            raise RuntimeError(f'Arm {arm}: runtime environment differs in {key}.')

audit = dict(seed=seed, source_sha256=actual_source, environment=environment,
             launcher_sha256=hashlib.sha256(Path(os.environ['AB_REPLICATION_LAUNCHER']).read_bytes()).hexdigest(),
             target_n_iters=200000, expected_final_state_step=200001,
             snapshot_freq=50000, preemption_freq=5000,
             snapshot_rng_policy='snapshot_sampling=False', runs={})
for arm, config in configs.items():
    configure_reproducibility(config, tensorflow_module=run_lib.tf)
    model = run_lib.mutils.create_model(config)
    digest = hashlib.sha256()
    for name, tensor in sorted(model.state_dict().items()):
        array = tensor.detach().cpu().contiguous().numpy()
        digest.update(name.encode() + b'\0')
        digest.update(str(array.dtype).encode() + b'\0')
        digest.update(str(array.shape).encode() + b'\0')
        digest.update(array.tobytes())
    weights_hash = digest.hexdigest()
    del model
    gc.collect()
    torch.cuda.empty_cache()
    data_root = run_lib.datasets.get_local_celeba_root(config)
    if data_root is None:
        raise RuntimeError('Expected the same local CelebA loader as the seed-42 run.')
    data_hashes = {name: hashlib.sha256((Path(data_root) / name).read_bytes()).hexdigest()
                   for name in ('list_eval_partition.txt', 'list_attr_celeba.txt',
                                'list_landmarks_align_celeba.txt')}
    train_ds, _, _ = run_lib.datasets.get_dataset(
        config, uniform_dequantization=config.data.uniform_dequantization)
    iterator = iter(train_ds)
    batch_hashes = []
    for _ in range(2):
        array = next(iterator)['image']._numpy()
        if array.shape != (128, 64, 64, 3):
            raise RuntimeError(f'Unexpected training batch shape: {array.shape}')
        batch_hashes.append(hashlib.sha256(array.tobytes()).hexdigest())
    audit['runs'][arm] = dict(
        config=_canonicalize(config), initial_state_sha256=weights_hash,
        first_train_batch_sha256=batch_hashes, dataset_metadata_sha256=data_hashes,
        reference_training_protocol_sha256=references[arm]['training_protocol_sha256'],
        workdir=os.environ[f'AB_REPLICATION_{arm}_WORKDIR'])
    del iterator, train_ds
    gc.collect()

for key in ('initial_state_sha256', 'first_train_batch_sha256', 'dataset_metadata_sha256'):
    if audit['runs']['A'][key] != audit['runs']['B'][key]:
        raise RuntimeError(f'The A/B preflight is not matched in {key}.')
audit['status'] = 'passed'
Path(os.environ['AB_REPLICATION_AUDIT']).write_text(json.dumps(audit, indent=2) + '\n')
print(json.dumps(audit, indent=2), flush=True)
PY

if [[ "$MODE" == preflight ]]; then
  echo "Preflight passed: $AB_REPLICATION_AUDIT. No training started."
  exit 0
fi

# Reserve fresh directories. A concurrent/repeated invocation must not resume them.
mkdir -p workdirs
mkdir "$AB_REPLICATION_A_WORKDIR"
mkdir "$AB_REPLICATION_B_WORKDIR"
for arm in A B; do
  if [[ "$arm" == A ]]; then
    config=configs/factorial/celeba_a_vp_pvp.py
    workdir="$AB_REPLICATION_A_WORKDIR"
  else
    config=configs/factorial/celeba_b_vp_pfox.py
    workdir="$AB_REPLICATION_B_WORKDIR"
  fi
  echo "=== A/B replication $arm: seed=$SEED, n_iters=200000, workdir=$workdir ==="
  python -u main.py \
    --mode=train --config="$config" --workdir="$workdir" \
    --config.training.batch_size=128 \
    --config.training.n_iters=200000 \
    --config.training.snapshot_freq=50000 \
    --config.training.snapshot_freq_for_preemption=5000 \
    --config.training.snapshot_sampling=False \
    --config.training.log_freq=50 \
    --config.training.eval_freq=500 \
    --config.seed="$SEED" \
    --config.allow_tf32=True \
    --config.cudnn_benchmark=True
done
echo "A/B seed $SEED finished. Evaluation is a separate stage."
