#!/usr/bin/env bash
# Docker-only A/B eval: seed44 (default) or seed42, 200k then 100k/common/10k/seed0.
set -euo pipefail
[[ -f /.dockerenv && "${CUDA_VISIBLE_DEVICES:-}" == 0 ]] || {
  echo 'Use Docker with one physical GPU and CUDA_VISIBLE_DEVICES=0.' >&2; exit 2; }
cd /workspace/score_sde_pytorch
export AB_EVAL_TRAINING_SEED="${AB_EVAL_TRAINING_SEED:-44}"
case "$AB_EVAL_TRAINING_SEED" in 44) checkpoints='4 2' ;; 42) checkpoints='40 20' ;; *) echo 'Expected training seed 44 or 42.' >&2; exit 2 ;; esac
mkdir -p "$TFDS_DATA_DIR" "$TORCH_EXTENSIONS_DIR" "$XDG_CACHE_HOME"

python -u - <<'PY'
import gc
import hashlib
import json
import os
from pathlib import Path
import torch
if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
    raise RuntimeError('Eval requires exactly one visible CUDA GPU.')
torch.cuda.init()
from ml_collections import ConfigDict
from artifact_utils import _source_fingerprint, _environment_manifest, build_model_manifest
import evaluation

seed = int(os.environ['AB_EVAL_TRAINING_SEED'])
snapshot_freq = 50000 if seed == 44 else 5000
checkpoints = ((4, 200001), (2, 100001)) if seed == 44 else ((40, 200001), (20, 100001))
def workdir_name(name):
    return f'celeba_ab_replication_{name}_seed44_200k' if seed == 44 else f'celeba_factorial_{name}_50k'

source = _source_fingerprint(os.getcwd())
if source != os.environ['AB_REPLICATION_EVAL_SOURCE_SHA256']:
    raise RuntimeError('Eval Python sources changed after queue creation.')
environment = _environment_manifest()
reference = Path('assets/stats/celeba_64_0c2d89311192_stats.npz')
reference_digest = hashlib.sha256()
with reference.open('rb') as reference_file:
    for block in iter(lambda: reference_file.read(1024 * 1024), b''):
        reference_digest.update(block)
reference_hash = reference_digest.hexdigest()
if reference_hash != os.environ['AB_REPLICATION_EVAL_REFERENCE_SHA256']:
    raise RuntimeError('CelebA reference changed after queue creation.')
audit = dict(source_sha256=source, reference_sha256=reference_hash,
             environment=environment, sampling_seed=0, num_samples=10000,
             training_seed=seed, checkpoints=dict(checkpoints),
             endpoint='common', em_steps=1000, nfe=1001, runs={})
for arm, name, allocation in (
    ('A', 'A_vp_pvp', 'vp_time'), ('B', 'B_vp_pfox', 'normalized_fox_time')):
    root = Path('/workspace/training_workdirs') / workdir_name(name)
    manifest = json.loads((root / 'run_manifest.json').read_text())
    config = ConfigDict(manifest['config'])
    if (manifest['source_sha256'] != 'ddf4c6acbc51b67da7a3465f2b90678ff559a628f6925d47e991350b02ff54f7'
            or config.seed != seed
            or config.training.snapshot_freq != snapshot_freq
            or config.training.batch_size != 128
            or config.training.sde != 'vpsde'
            or config.training.factorial_arm != name
            or config.training.noise_distribution != allocation
            or config.model.noise_conditioning != 'logsnr'
            or config.model.logsnr_min != -10.0
            or config.model.logsnr_max != 9.115429865459795):
        raise RuntimeError(f'Unexpected training manifest for {arm}.')
    for key in ('python', 'numpy', 'tensorflow', 'torch', 'torch_cuda', 'cudnn', 'packages'):
        if environment[key] != manifest['environment'][key]:
            raise RuntimeError(f'Training/eval environment differs in {key}.')
    if evaluation._get_stats_filename(config) != str(reference):
        raise RuntimeError('Eval would select a different CelebA reference.')
    audit['runs'][arm] = {}
    for ckpt, expected_step in checkpoints:
        checkpoint = root / f'checkpoints/checkpoint_{ckpt}.pth'
        state = torch.load(checkpoint, map_location='cpu', weights_only=False)
        if (int(state['step']) != expected_step
                or state['training_protocol_sha256'] != manifest['training_protocol_sha256']
                or state['model_protocol_sha256'] != build_model_manifest(config)['model_protocol_sha256']):
            raise RuntimeError(f'Checkpoint identity/step mismatch for {arm}, checkpoint {ckpt}.')
        audit['runs'][arm][str(ckpt)] = dict(checkpoint_id=state['checkpoint_id'], step=int(state['step']),
                                 training_protocol_sha256=manifest['training_protocol_sha256'])
        del state
        gc.collect()
audit['status'] = 'passed'
Path(f'logs/ab_replication_seed{seed}_eval_seed0_preflight.json').write_text(json.dumps(audit, indent=2)+'\n')
print(json.dumps(audit, indent=2), flush=True)
PY

tmp_config="$(mktemp /tmp/ab_replication_eval.XXXXXX.py)"
trap 'rm -f "$tmp_config"' EXIT
cat > "$tmp_config" <<'PY'
import json
import os
from pathlib import Path
from ml_collections import ConfigDict
from configs.factorial.celeba_a_vp_pvp import get_config as get_a
from configs.factorial.celeba_b_vp_pfox import get_config as get_b

def get_config(arm):
    name = {'A': 'A_vp_pvp', 'B': 'B_vp_pfox'}[arm]
    seed = int(os.environ['AB_EVAL_TRAINING_SEED'])
    directory = f'celeba_ab_replication_{name}_seed44_200k' if seed == 44 else f'celeba_factorial_{name}_50k'
    root = Path('/workspace/training_workdirs') / directory
    config = ConfigDict(json.loads((root / 'run_manifest.json').read_text())['config'])
    # Training manifests deliberately omit eval and orchestration fields.
    config.eval = (get_a if arm == 'A' else get_b)().eval
    config.training.n_iters = 200000
    config.sampling.time_grid = 'uniform_logsnr'
    config.sampling.logsnr_min = -10.0
    config.sampling.logsnr_max = 9.115429865459795
    config.sampling.noise_removal = True
    config.eval.endpoint_denoise_variants = True
    config.eval.include_final_checkpoint = False
    return config
PY
for ckpt in $checkpoints; do
for arm in A B; do
  if [[ "$arm" == A ]]; then name=A_vp_pvp; else name=B_vp_pfox; fi
  if [[ "$AB_EVAL_TRAINING_SEED" == 44 ]]; then
    workdir="workdirs/celeba_ab_replication_${name}_seed44_200k"
  else
    workdir="workdirs/celeba_factorial_${name}_50k"
  fi
  folder="eval_ab_replication_seed${AB_EVAL_TRAINING_SEED}_${arm}_common_ckpt${ckpt}_nfe1000_10000samples_batch512_seed0"
  echo "=== A/B replication eval $arm/common: ckpt=$ckpt, 10k, sampling_seed=0, NFE=1001 ==="
  python -u main.py --mode=eval --config="$tmp_config:$arm" \
    --workdir="$workdir" --eval_folder="$folder" \
    --config.eval.begin_ckpt="$ckpt" --config.eval.end_ckpt="$ckpt" \
    --config.eval.batch_size=512 --config.eval.num_samples=10000 \
    --config.eval.enable_loss=False --config.eval.enable_sampling=True \
    --config.eval.enable_bpd=False --config.eval.sampling_num_scales=1000 \
    --config.eval.sampling_seed=0 --config.sampling.method=pc \
    --config.sampling.predictor=euler_maruyama --config.sampling.corrector=none \
    --config.allow_tf32=True --config.cudnn_benchmark=True
done
done
echo 'A/B replication primary eval completed; reports and features saved.'
