#!/usr/bin/env bash
# Docker-only primary eval: seed43 A/B, checkpoint4, common, 10k, sampling seed0.
set -euo pipefail
[[ -f /.dockerenv && "${CUDA_VISIBLE_DEVICES:-}" == 0 ]] || {
  echo 'Use Docker with one physical GPU and CUDA_VISIBLE_DEVICES=0.' >&2; exit 2; }
cd /workspace/score_sde_pytorch
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
             checkpoint=4, expected_checkpoint_step=200001,
             endpoint='common', em_steps=1000, nfe=1001, runs={})
for arm, name, allocation in (
    ('A', 'A_vp_pvp', 'vp_time'), ('B', 'B_vp_pfox', 'normalized_fox_time')):
    root = Path('/workspace/training_workdirs') / f'celeba_ab_replication_{name}_seed43_200k'
    manifest = json.loads((root / 'run_manifest.json').read_text())
    config = ConfigDict(manifest['config'])
    if (manifest['source_sha256'] != 'ddf4c6acbc51b67da7a3465f2b90678ff559a628f6925d47e991350b02ff54f7'
            or config.seed != 43 or config.training.n_iters != 200000
            or config.training.snapshot_freq != 50000
            or config.training.batch_size != 128
            or config.training.sde != 'vpsde'
            or config.training.factorial_arm != name
            or config.training.noise_distribution != allocation):
        raise RuntimeError(f'Unexpected training manifest for {arm}.')
    for key in ('python', 'numpy', 'tensorflow', 'torch', 'torch_cuda', 'cudnn', 'packages'):
        if environment[key] != manifest['environment'][key]:
            raise RuntimeError(f'Training/eval environment differs in {key}.')
    if evaluation._get_stats_filename(config) != str(reference):
        raise RuntimeError('Eval would select a different CelebA reference.')
    checkpoint = root / 'checkpoints/checkpoint_4.pth'
    state = torch.load(checkpoint, map_location='cpu', weights_only=False)
    if (int(state['step']) != 200001
            or state['training_protocol_sha256'] != manifest['training_protocol_sha256']
            or state['model_protocol_sha256'] != build_model_manifest(config)['model_protocol_sha256']):
        raise RuntimeError(f'Checkpoint identity/step mismatch for {arm}.')
    audit['runs'][arm] = dict(checkpoint_id=state['checkpoint_id'], step=int(state['step']),
                             training_protocol_sha256=manifest['training_protocol_sha256'])
    del state
    gc.collect()
audit['status'] = 'passed'
Path('logs/ab_replication_eval_seed0_preflight.json').write_text(json.dumps(audit, indent=2)+'\n')
print(json.dumps(audit, indent=2), flush=True)
PY

tmp_config="$(mktemp /tmp/ab_replication_eval.XXXXXX.py)"
trap 'rm -f "$tmp_config"' EXIT
cat > "$tmp_config" <<'PY'
import json
from pathlib import Path
from ml_collections import ConfigDict

def get_config(arm):
    name = {'A': 'A_vp_pvp', 'B': 'B_vp_pfox'}[arm]
    root = Path('/workspace/training_workdirs') / f'celeba_ab_replication_{name}_seed43_200k'
    config = ConfigDict(json.loads((root / 'run_manifest.json').read_text())['config'])
    config.sampling.time_grid = 'uniform_logsnr'
    config.sampling.logsnr_min = -10.0
    config.sampling.logsnr_max = 9.115429865459795
    config.sampling.noise_removal = True
    config.eval.endpoint_denoise_variants = True
    config.eval.include_final_checkpoint = False
    return config
PY
for arm in A B; do
  if [[ "$arm" == A ]]; then name=A_vp_pvp; else name=B_vp_pfox; fi
  workdir="workdirs/celeba_ab_replication_${name}_seed43_200k"
  folder="eval_ab_replication_seed43_${arm}_common_ckpt4_nfe1000_10000samples_batch512_seed0"
  echo "=== A/B replication eval $arm/common: ckpt=4, 10k, sampling_seed=0, NFE=1001 ==="
  python -u main.py --mode=eval --config="$tmp_config:$arm" \
    --workdir="$workdir" --eval_folder="$folder" \
    --config.eval.begin_ckpt=4 --config.eval.end_ckpt=4 \
    --config.eval.batch_size=512 --config.eval.num_samples=10000 \
    --config.eval.enable_loss=False --config.eval.enable_sampling=True \
    --config.eval.enable_bpd=False --config.eval.sampling_num_scales=1000 \
    --config.eval.sampling_seed=0 --config.sampling.method=pc \
    --config.sampling.predictor=euler_maruyama --config.sampling.corrector=none \
    --config.allow_tf32=True --config.cudnn_benchmark=True
done
echo 'A/B replication primary eval completed; reports and features saved.'
