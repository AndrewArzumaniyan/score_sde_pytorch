#!/usr/bin/env bash
# CIFAR DDPM++ deep E3b: fixed VP dynamics, native FOX epsilon via adapter.
# 7 new routes x sampling seeds 0/1 x 5000 images; reuse native VP seed0/1.
# Batch 512, 10 rounds per new eval. No generation for the VP baseline.
# Common lambda [-10, 9.115429865459795], 1000 EM + explicit Tweedie, EMA FP32.
# The final network follows the route (head -> VP; tail/full -> FOX).
# No training or changes to old scripts/configs. New .py config lives in /tmp.
# Native pre-Tweedie float states are saved alongside images/features.
set -euo pipefail
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"
if (( $# != 0 )); then
  echo "Usage: bash $0" >&2
  exit 2
fi

VP_WORKDIR=workdirs/cifar10_vp_ddpmpp_deep_200k_rtx_seed42
FOX_WORKDIR=workdirs/cifar10_fox_ddpmpp_deep_matern32_kappa1000_200k_rtx_seed42
# The existing native VP eval is the control; generate only interventions.
# tail:9.5 is outside the CIFAR common interval and is intentionally omitted.
RUNS=(full head:-3.3 tail:8 head:8 tail:-3.3 tail:0 tail:4)
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
export CIFAR_SPLICE_FOX_WORKDIR="$FOX_WORKDIR"
export CIFAR_SPLICE_DRIVER="$ROOT_DIR/$(basename "${BASH_SOURCE[0]}")"
for config in configs/vp/cifar10_ddpmpp_deep_continuous_matched.py \
              configs/fox/cifar10_ddpmpp_deep_continuous_matern_3_2_kappa1000.py; do
  [[ -s "$config" ]] || { echo "Missing config: $config" >&2; exit 1; }
done
[[ -s assets/stats/cifar10_stats.npz ]] || { echo 'Missing CIFAR reference statistics' >&2; exit 1; }
[[ -d "$TFDS_DATA_DIR/cifar10" ]] || { echo "Missing prepared CIFAR-10: $TFDS_DATA_DIR/cifar10" >&2; exit 1; }
for workdir in "$VP_WORKDIR" "$FOX_WORKDIR"; do
  for file in checkpoints/checkpoint_12.pth run_manifest.json; do
    [[ -s "$workdir/$file" ]] || { echo "Missing or empty: $workdir/$file" >&2; exit 1; }
  done
done

# Fail before any sampling if either parent is not the expected DDPM++ model.
python -u - "$VP_WORKDIR" "$FOX_WORKDIR" <<'PY_CHECK'
import hashlib
import json
from pathlib import Path
import sys
import numpy as np
import torch
# Same mixed-framework startup as the evaluation process, no inference here.
if torch.cuda.is_available():
    torch.cuda.init()
import tensorflow as tf
tf.config.set_visible_devices([], 'GPU')
from artifact_utils import build_eval_manifest, validate_npz_metadata
from reproducibility import derive_seed
from configs.vp.cifar10_ddpmpp_deep_continuous_matched import get_config as get_vp
from configs.fox.cifar10_ddpmpp_deep_continuous_matern_3_2_kappa1000 import get_config as get_fox

for workdir, factory in zip(sys.argv[1:], (get_vp, get_fox)):
    manifest = json.loads((Path(workdir) / 'run_manifest.json').read_text())
    payload = {k: v for k, v in manifest.items() if k != 'training_protocol_sha256'}
    digest = hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(',', ':'), ensure_ascii=True).encode()).hexdigest()
    if digest != manifest['training_protocol_sha256']:
        raise ValueError('Corrupt training manifest: ' + workdir)
    cfg = factory('eval')
    expected_model = json.loads(json.dumps(cfg.model.to_dict()))
    saved_model = manifest['model_manifest']
    if saved_model['model'] != expected_model or saved_model['training_sde'] != cfg.training.sde:
        raise ValueError('Checkpoint model/SDE differs from matched DDPM++ config: ' + workdir)
    for key in ('dataset', 'image_size', 'num_channels', 'centered'):
        if saved_model['data'][key] != getattr(cfg.data, key):
            raise ValueError('Checkpoint data config differs: ' + workdir)
    path = Path(workdir) / 'checkpoints' / 'checkpoint_12.pth'
    state = torch.load(path, map_location='cpu', weights_only=False)
    if int(state['step']) != 600001 or not state.get('checkpoint_id'):
        raise ValueError('Expected identified 600k checkpoint: ' + str(path))
    if state.get('model_protocol_sha256') != saved_model['model_protocol_sha256'] or state.get('training_protocol_sha256') != digest:
        raise ValueError('Checkpoint/manifest identity mismatch: ' + str(path))
    print(f'600k DDPM++ checkpoint checked: {path}; step={state["step"]}; id={state["checkpoint_id"]}', flush=True)
    if factory is get_vp:
        reference_path = Path('assets/stats/cifar10_stats.npz')
        reference_hash = hashlib.sha256(reference_path.read_bytes()).hexdigest()
        for seed in (0, 1):
            cfg.eval.sampling_seed = seed
            folder = Path(workdir) / (
                'eval_cifar_endpoint_denoise_ddpmpp_deep600k_screen_vp_common_ckpt12_'
                f'nfe1000_5000samples_batch512_seed{seed}')
            saved = json.loads((folder / 'manifest.json').read_text())
            if saved != build_eval_manifest(cfg):
                raise ValueError('Cached VP baseline has a different source/config/environment: ' + str(folder))
            common = dict(checkpoint_step=600001, training_protocol_sha256=digest)
            with np.load(folder / 'report_12.npz', allow_pickle=False) as z:
                validate_npz_metadata(z, str(folder), saved['protocol_sha256'], state['checkpoint_id'],
                                      required_arrays=('num_samples', 'nfe', 'dataset_stats_id',
                                                       'fid_x_mean', 'fid_x_t', 'fid_tweedie_xt'), **common)
                if int(z['num_samples']) != 5000 or int(z['nfe']) != 1001 or z['dataset_stats_id'].item() != reference_hash:
                    raise ValueError('Cached VP baseline count/NFE/reference mismatch: ' + str(folder))
                fid = float(z['fid_tweedie_xt'])
            statistics = folder / 'ckpt_12'
            expected = {statistics / f'statistics_{i}.npz' for i in range(10)}
            if set(statistics.glob('statistics_*.npz')) != expected:
                raise ValueError('Cached VP baseline needs all 10 feature batches: ' + str(folder))
            for i in range(10):
                stat = statistics / f'statistics_{i}.npz'
                with np.load(stat, allow_pickle=False) as z:
                    validate_npz_metadata(z, str(stat), saved['protocol_sha256'], state['checkpoint_id'],
                                          round_id=i, sampling_seed=derive_seed(seed, 'sampling-round', i),
                                          required_arrays=('pool_3_x_mean', 'pool_3_x_t', 'pool_3_tweedie_xt',
                                                           'num_samples', 'nfe'), **common)
                    if int(z['num_samples']) != 512 or int(z['nfe']) != 1001 or any(
                            z['pool_3_'+v].shape != (512, 2048) for v in ('x_mean', 'x_t', 'tweedie_xt')):
                        raise ValueError('Invalid cached VP feature batch: ' + str(stat))
            print(f'Reusing VP baseline seed={seed}: {folder}; Tweedie FID={fid:.6f}; no generation.', flush=True)
    del state
PY_CHECK

TMP_CONFIG="$(mktemp /tmp/cifar_ddpmpp_600k_splice.XXXXXX)"
trap 'rm -f "$TMP_CONFIG"' EXIT
cat > "$TMP_CONFIG" <<'PY_CONFIG'
import hashlib
import math
import os
from pathlib import Path
from configs.vp.cifar10_ddpmpp_deep_continuous_matched import get_config as get_vp
from configs.fox.cifar10_ddpmpp_deep_continuous_matern_3_2_kappa1000 import get_config as get_fox


def get_config(name):
    parts = name.split('_')
    mode = parts[0]
    if mode not in ('none', 'full', 'head', 'tail') or len(parts) != (2 if mode in ('head', 'tail') else 1):
        raise ValueError('Invalid splice route: ' + name)
    threshold = float(parts[1]) if len(parts) == 2 else None
    if threshold is not None and (not math.isfinite(threshold) or not -10.0 < threshold < 9.115429865459795):
        raise ValueError('Splice threshold must lie inside the common lambda interval.')
    vp, fox = get_vp('eval'), get_fox('eval')
    vp.eval.begin_ckpt = vp.eval.end_ckpt = 12
    vp.eval.splice_enabled = True
    vp.eval.splice_shared_logsnr_conditioning = False
    vp.eval.splice_mode = mode
    vp.eval.splice_lambda = threshold
    vp.eval.splice_alt = fox
    vp.eval.splice_alt_workdir = os.environ['CIFAR_SPLICE_FOX_WORKDIR']
    vp.eval.splice_alt_ckpt = 12
    # Bind cache identity to this standalone driver, including float capture.
    vp.eval.splice_capture_native_float = True
    vp.eval.splice_driver_sha256 = hashlib.sha256(Path(os.environ['CIFAR_SPLICE_DRIVER']).read_bytes()).hexdigest()
    return vp
PY_CONFIG

run_route() {
  local spec="$1" seed="$2" mode threshold config_name folder_mode folder log
  mode="${spec%%:*}"
  if [[ "$spec" == *:* ]]; then
    threshold="${spec#*:}"
    config_name="${mode}_${threshold}"
    folder_mode="${mode}${threshold}"
  else
    config_name="$mode"
    folder_mode="$mode"
  fi
  folder="eval_cifar_ddpmpp_deep600k_splice_v1_${folder_mode}_vpcommon_ckpt12_nfe1000_5000samples_batch512_seed${seed}"
  log="run_cifar10_ddpmpp_deep600k_splice_v1_${folder_mode}_5k_seed${seed}.log"
  echo "=== CIFAR DDPM++ E3b: route=$spec; sampling seed=$seed; VP/FOX ckpt12; 5000 images; batch512 ==="
  echo 'VP dynamics; common lambda [-10, 9.115429865459795]; 1000 EM + route-dependent Tweedie'
  python -u - \
    --mode=eval --config="$TMP_CONFIG:$config_name" \
    --workdir="$VP_WORKDIR" --eval_folder="$folder" \
    --config.eval.begin_ckpt=12 --config.eval.end_ckpt=12 \
    --config.eval.batch_size=512 --config.eval.num_samples=5000 \
    --config.eval.enable_loss=False --config.eval.enable_sampling=True \
    --config.eval.enable_bpd=False --config.eval.endpoint_denoise_variants=True \
    --config.eval.sampling_num_scales=1000 --config.eval.sampling_seed="$seed" \
    --config.sampling.method=pc --config.sampling.predictor=euler_maruyama \
    --config.sampling.corrector=none --config.sampling.time_grid=uniform_logsnr \
    --config.sampling.logsnr_min=-10.0 --config.sampling.logsnr_max=9.115429865459795 \
    --config.sampling.noise_removal=True --config.seed=42 \
    --config.allow_tf32=False --config.cudnn_benchmark=False \
    --config.allow_fp16_reduced_precision_reduction=False \
    <<'PY_EVAL' 2>&1 | tee -a "$log"
import math
from pathlib import Path
import runpy
import sys
import numpy as np
import torch
if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
    raise RuntimeError('Run in the RTX container with exactly one visible CUDA GPU.')
torch.cuda.init()
import tensorflow as tf
tf.config.set_visible_devices([], 'GPU')
import evaluation
evaluation.jax = None  # Same CPU Inception path as the native CIFAR eval.
import endpoint_denoise_eval
import sampling
import run_lib
from artifact_utils import atomic_savez, validate_npz_metadata
from reproducibility import derive_seed

original_evaluate = endpoint_denoise_eval.evaluate_checkpoint


def evaluate_with_float_cache(config, eval_dir, ckpt, score_model, sampling_fn,
                             inception_model, protocol_sha256, checkpoint_id,
                             checkpoint_step, training_protocol, artifact_identity,
                             extra_identity=None):
    """Preserve the real native float x_t; keep the original metric writer."""
    sde, eps = run_lib.get_sde(config, n_override=config.eval.sampling_num_scales)
    shape = (config.eval.batch_size, config.data.num_channels,
             config.data.image_size, config.data.image_size)
    native_sampler = sampling.get_sampling_fn(config, sde, shape, lambda x: x, eps)
    rounds = math.ceil(config.eval.num_samples / config.eval.batch_size)
    directory = Path(eval_dir) / f'ckpt_{ckpt}'
    missing = iter(i for i in range(rounds) if not (directory / f'samples_{i}.npz').exists())
    metadata = dict(artifact_identity)
    metadata.update({k: np.asarray(v) for k, v in score_model.expected_route_metadata.items()})
    end_t = sde.sampling_time_grid(eps, grid=config.sampling.time_grid,
                                   device=config.device, dtype=torch.float32,
                                   N=sde.N + 1)[-1]

    def validate_float(path, round_id):
        with np.load(path, allow_pickle=False) as z:
            validate_npz_metadata(z, str(path), protocol_sha256, checkpoint_id,
                                  round_id=round_id,
                                  sampling_seed=derive_seed(config.eval.sampling_seed, 'sampling-round', round_id),
                                  checkpoint_step=checkpoint_step,
                                  training_protocol_sha256=training_protocol,
                                  required_arrays=('x_t_native', 'representation', 't_endpoint', 'nfe'),
                                  extra_identity=extra_identity)
            x = z['x_t_native']
            if (x.dtype != np.float32 or x.shape != shape or not np.isfinite(x).all()
                    or str(z['representation'].item()) != 'native_centered_NCHW_float32_before_Tweedie'
                    or int(z['nfe']) != 1001 or float(z['t_endpoint']) != float(end_t)):
                raise ValueError('Invalid native float state: ' + str(path))
            return x.copy()

    for i in range(rounds):
        if (directory / f'samples_{i}.npz').exists():
            validate_float(directory / f'float_state_{i}.npz', i)

    def capture(model):
        round_id = next(missing)
        native, nfe = native_sampler(model)
        # This is called inside the original isolated per-round RNG context.
        model.checked_route_metadata()
        state = native['x_t'].detach().cpu().numpy()
        path = directory / f'float_state_{round_id}.npz'
        if path.exists():
            if not np.array_equal(validate_float(path, round_id), state):
                raise ValueError('Replayed float state differs from cache: ' + str(path))
        else:
            atomic_savez(path, overwrite=False, x_t_native=state,
                         representation=np.asarray('native_centered_NCHW_float32_before_Tweedie'),
                         t_endpoint=np.asarray(float(end_t)), nfe=np.asarray(nfe),
                         round_id=np.asarray(round_id),
                         sampling_seed=np.asarray(derive_seed(config.eval.sampling_seed, 'sampling-round', round_id)),
                         **metadata)
        # CIFAR centered-data inverse_scaler, applied after float capture.
        return {key: (value + 1.) / 2. for key, value in native.items()}, nfe

    original_evaluate(config, eval_dir, ckpt, score_model, capture, inception_model,
                      protocol_sha256, checkpoint_id, checkpoint_step,
                      training_protocol, artifact_identity, extra_identity=extra_identity)
    for i in range(rounds):
        validate_float(directory / f'float_state_{i}.npz', i)
    if config.eval.splice_mode == 'none':
        baseline = Path(eval_dir).parent / (
            'eval_cifar_endpoint_denoise_ddpmpp_deep600k_screen_vp_common_ckpt12_'
            f'nfe1000_5000samples_batch512_seed{config.eval.sampling_seed}')
        old_report = baseline / 'report_12.npz'
        if old_report.exists():
            with np.load(old_report, allow_pickle=False) as old, np.load(Path(eval_dir) / 'report_12.npz', allow_pickle=False) as new:
                if (old['checkpoint_id'].item() != checkpoint_id
                        or old['dataset_stats_id'].item() != new['dataset_stats_id'].item()):
                    raise ValueError('VP identity control compares different checkpoint/reference.')
                for variant in ('x_mean', 'x_t', 'tweedie_xt'):
                    if not np.isclose(old['fid_'+variant], new['fid_'+variant], rtol=1e-6, atol=1e-6):
                        raise ValueError('VP-only splice does not reproduce old FID: ' + variant)
            print('VP-only control reproduces native VP FIDs.', flush=True)
        else:
            print('No old VP report found; the new none route is the baseline.', flush=True)


endpoint_denoise_eval.evaluate_checkpoint = evaluate_with_float_cache
print('DDPM++ splice evaluation arguments:', ' '.join(sys.argv[1:]), flush=True)
sys.argv[0] = 'main.py'
runpy.run_path('main.py', run_name='__main__')
PY_EVAL
  echo "=== Finished route=$spec seed=$seed; report: $VP_WORKDIR/$folder/report_12.npz ==="
}

# Finish each route on both seeds before moving to the next intervention.
for spec in "${RUNS[@]}"; do
  for seed in 0 1; do
    run_route "$spec" "$seed"
  done
done

# Read-only compact result table; original reports contain the full provenance.
python -u - "$VP_WORKDIR" "${RUNS[@]}" <<'PY_SUMMARY'
from pathlib import Path
import sys
import numpy as np

workdir = Path(sys.argv[1])
print('\n=== FID after Tweedie: separate 5k seed evaluations, NOT pooled FID ===')
print('route\tseed0\tseed1\tmean\tmean_delta_vs_VP')
base_values, base_ids, base_references = [], [], []
for seed in (0, 1):
    folder = workdir / ('eval_cifar_endpoint_denoise_ddpmpp_deep600k_screen_vp_common_ckpt12_'
                        f'nfe1000_5000samples_batch512_seed{seed}')
    with np.load(folder / 'report_12.npz', allow_pickle=False) as z:
        if int(z['checkpoint_step']) != 600001 or int(z['num_samples']) != 5000 or int(z['nfe']) != 1001:
            raise ValueError('Invalid cached VP baseline: ' + str(folder))
        base_values.append(float(z['fid_tweedie_xt']))
        base_ids.append(z['checkpoint_id'].item())
        base_references.append(z['dataset_stats_id'].item())
baseline = float(np.mean(base_values))
print('VP (reused)\t%.6f\t%.6f\t%.6f\t%+.6f' % (*base_values, baseline, 0.))
for spec in sys.argv[2:]:
    values = []
    for seed in (0, 1):
        folder = workdir / ('eval_cifar_ddpmpp_deep600k_splice_v1_%s_vpcommon_ckpt12_'
                            'nfe1000_5000samples_batch512_seed%d' % (spec.replace(':', ''), seed))
        with np.load(folder / 'report_12.npz', allow_pickle=False) as z:
            if (int(z['checkpoint_step']) != 600001 or int(z['alt_checkpoint_step']) != 600001
                    or int(z['num_samples']) != 5000 or int(z['nfe']) != 1001
                    or z['checkpoint_id'].item() != base_ids[seed]
                    or z['dataset_stats_id'].item() != base_references[seed]):
                raise ValueError('Unexpected checkpoint/count: ' + str(folder))
            values.append(float(z['fid_tweedie_xt']))
    mean = float(np.mean(values))
    print('%s\t%.6f\t%.6f\t%.6f\t%+.6f' % (spec, *values, mean, mean-baseline))
print('Finished all 14 splice evaluations; reused 2 existing VP baselines. Native float states are saved for the new splice routes.')
PY_SUMMARY
