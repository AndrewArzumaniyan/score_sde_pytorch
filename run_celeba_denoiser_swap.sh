#!/usr/bin/env bash
# Host: bash run_celeba_denoiser_swap.sh GPU_ID
# CPU checks only: bash run_celeba_denoiser_swap.sh --check
# One new shell file; existing Python sources and their training hash stay intact.
set -euo pipefail

if [[ "${1:-}" != _container ]]; then
  [[ $# == 1 ]] || { echo 'Usage: bash run_celeba_denoiser_swap.sh GPU_ID|--check' >&2; exit 2; }
  main="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
  launcher="$(basename "${BASH_SOURCE[0]}")"
  root=/workspace/score_sde_pytorch
  image_id="$(docker image inspect --format '{{.Id}}' "${DENOISER_SWAP_IMAGE:-score-sde-h200}")"
  if [[ "$1" == --check ]]; then
    docker run --rm --network none --cpus 2 --memory 4g \
      --mount "type=bind,src=$main,dst=$root,readonly" --workdir "$root" \
      --env SWAP_SELF_TEST=1 --env PYTHONDONTWRITEBYTECODE=1 \
      --entrypoint bash "$image_id" "$launcher" _container
    exit
  fi
  gpu="$1"
  [[ "$gpu" =~ ^[0-9]+$ || "$gpu" == GPU-* ]] || { echo 'Expected a physical GPU index or UUID.' >&2; exit 2; }
  output="$(realpath -m "${DENOISER_SWAP_OUTPUT:-$(dirname "$main")/celeba_denoiser_swap_10k}")"
  name=celeba-denoiser-swap-10k-seeds01
  case "$output/" in "$main/"*|/) echo 'Output must be outside the source repository.' >&2; exit 2 ;; esac
  [[ ! -e "$output" ]] || { echo "Output exists: $output; inspect/resume the original container." >&2; exit 1; }
  if docker inspect "$name" >/dev/null 2>&1; then
    echo "Container exists: $name; inspect/resume it." >&2; exit 1
  fi
  tfds="${TFDS_DATA_DIR:-$root/.tfds}"
  extensions="${TORCH_EXTENSIONS_DIR:-$root/.torch_extensions}"
  cache="${XDG_CACHE_HOME:-$root/.cache}"
  for path in "$tfds" "$extensions" "$cache"; do
    [[ "$path" == /* && "$path" != *','* && "$path" != *'..'* ]] || { echo "Invalid cache path: $path" >&2; exit 2; }
    case "$path/" in
      "$root/.tfds/"|"$root/.torch_extensions/"|"$root/.cache/") ;;
      "$root/"*|/|/workspace/|/output/|/output/*|/opt/|/opt/*|/tmp/|/usr/*|/etc/*|/proc/*|/dev/*|/sys/*|/bin/*|/sbin/*|/lib/*|/lib64/*)
        echo "Cache overlaps runtime/system directories: $path" >&2; exit 2 ;;
    esac
  done
  for pair in "$tfds|$extensions" "$tfds|$cache" "$extensions|$cache"; do
    left="${pair%%|*}/"; right="${pair#*|}/"
    [[ "$left" != "$right"* && "$right" != "$left"* ]] || { echo 'Cache mounts must not overlap.' >&2; exit 2; }
  done
  cd "$main"
  reference=assets/stats/celeba_64_0c2d89311192_stats.npz
  inputs=("$reference"
    workdirs/celeba_vp_continuous_50k/checkpoints/checkpoint_260.pth
    workdirs/celeba_fox_matern32_kappa1000_50k/checkpoints/checkpoint_260.pth)
  for path in "${inputs[@]}" sampling.py splice_model.py "$launcher"; do
    [[ -s "$path" ]] || { echo "Missing input: $path" >&2; exit 1; }
  done
  printf '%s  %s\n' 66f3e831815ccebf90fe477348fcd0de92e5d966b7ed1eef77dc80fab72b6327 "$reference" | sha256sum -c -
  mkdir -p "$output/runtime" "$output/results" "$main/.tfds" "$main/.torch_extensions" "$main/.cache"
  git ls-files -z -- '*.py' '*.sh' '*.c' '*.cc' '*.cpp' '*.cu' '*.cuh' '*.h' Dockerfile requirements.txt \
    | tar --null -T - -cf - | tar -xf - -C "$output/runtime"
  # fetch + restore --worktree can deliver this launcher outside HEAD/index.
  # Freeze its actual on-disk contents explicitly, including in that case.
  cp -- "$launcher" "$output/runtime/$launcher"
  mkdir -p "$output/runtime/assets/stats" "$output/runtime/workdirs"
  cp "$reference" "$output/runtime/$reference"
  for arm in celeba_vp_continuous_50k celeba_fox_matern32_kappa1000_50k; do
    mkdir -p "$output/runtime/workdirs/$arm/checkpoints"
  done
  for path in "$tfds" "$extensions" "$cache"; do
    case "$path" in "$root"/*) mkdir -p "$output/runtime/${path#"$root"/}" ;; esac
  done
  (cd "$output/runtime"; find . -type f -print0 | sort -z | xargs -0 sha256sum) > "$output/runtime.sha256"
  sha256sum "${inputs[@]}" > "$output/inputs.sha256"
  git rev-parse HEAD > "$output/commit.txt"
  git status --porcelain > "$output/source_status.txt"
  printf 'image=%s\nphysical_gpu=%s\nTFDS_DATA_DIR=%s\nTORCH_EXTENSIONS_DIR=%s\nXDG_CACHE_HOME=%s\n' \
    "$image_id" "$gpu" "$tfds" "$extensions" "$cache" > "$output/docker.txt"
  args=(run -d --name "$name" --gpus "device=$gpu" --ipc=host
    --mount "type=bind,src=$output/runtime,dst=$root,readonly"
    --mount "type=bind,src=$output/results,dst=/output"
    --mount "type=bind,src=$output,dst=/opt/denoiser-swap,readonly"
    --mount "type=bind,src=$main/.tfds,dst=$tfds"
    --mount "type=bind,src=$main/.torch_extensions,dst=$extensions"
    --mount "type=bind,src=$main/.cache,dst=$cache"
    --workdir "$root" --env CUDA_VISIBLE_DEVICES=0 --env PYTHONDONTWRITEBYTECODE=1
    --env TFDS_DATA_DIR="$tfds" --env TORCH_EXTENSIONS_DIR="$extensions" --env XDG_CACHE_HOME="$cache"
    --env TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-9.0}" --env MAX_JOBS="${MAX_JOBS:-8}"
    --entrypoint bash)
  for arm in celeba_vp_continuous_50k celeba_fox_matern32_kappa1000_50k; do
    args+=(--mount "type=bind,src=$main/workdirs/$arm/checkpoints,dst=$root/workdirs/$arm/checkpoints,readonly")
  done
  docker "${args[@]}" "$image_id" "$launcher" _container
  docker inspect "$name" > "$output/docker_inspect_start.json"
  printf 'Started %s on GPU %s. Results: %s/results\n' "$name" "$gpu" "$output"
  exit
fi

if [[ "${SWAP_SELF_TEST:-0}" != 1 ]]; then
  [[ "${CUDA_VISIBLE_DEVICES:-}" == 0 ]] || { echo 'Expected one Docker-routed GPU, CVD=0.' >&2; exit 2; }
  exec > >(tee -a /output/queue.log) 2>&1
  trap 'code=$?; printf "failed exit=%s\n" "$code" > /output/queue.state' ERR
  sha256sum -c /opt/denoiser-swap/runtime.sha256
  sha256sum -c /opt/denoiser-swap/inputs.sha256
  printf 'running\n' > /output/queue.state
fi

python -u - <<'PY_DENOISER_SWAP'
"""VP/tail trajectories crossed with VP/FOX endpoint denoisers; Docker only."""
import gc
import hashlib
import json
import logging
import math
import os
from pathlib import Path
import subprocess
import tempfile

import torch
if torch.cuda.is_available():
    torch.cuda.init()  # Match main.py: initialize Torch before TensorFlow.
import numpy as np
import tensorflow as tf
import tensorflow_gan as tfgan

# CPU checks need the real SDE/sampler/adapter, but no NCSN++ CUDA extensions.
if os.environ.get('SWAP_SELF_TEST') != '1':
    import run_lib  # Registers the native model definitions for scientific eval.
import sampling
import sde_lib
import evaluation
import datasets
import losses
from models import utils as mutils
from models.ema import ExponentialMovingAverage
from splice_model import SplicedEpsModel
from configs.vp.celeba_ncsnpp_continuous import get_config as get_vp
from configs.fox.celeba_ncsnpp_continuous_matern_3_2 import get_config as get_fox
from artifact_utils import (atomic_savez, atomic_write_bytes, build_model_manifest,
                            build_eval_manifest, ensure_eval_manifest,
                            validate_npz_metadata)
from reproducibility import configure_reproducibility, derive_seed, isolated_torch_rng
from utils import restore_checkpoint

ROOT = Path('/workspace/score_sde_pytorch')
OUTPUT = Path('/output')
REFERENCE = 'assets/stats/celeba_64_0c2d89311192_stats.npz'
REFERENCE_SHA = '66f3e831815ccebf90fe477348fcd0de92e5d966b7ed1eef77dc80fab72b6327'
ARMS = {'vp': 'celeba_vp_continuous_50k', 'fox': 'celeba_fox_matern32_kappa1000_50k'}
VARIANTS = ('tweedie_vp', 'tweedie_fox')
COUNT, BATCH, STEPS = 10000, 512, 1000
ENDPOINT, THRESHOLD = 10.575749165617202, -3.3


def digest(path):
    result = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            result.update(block)
    return result.hexdigest()


def array_digest(array):
    array = np.ascontiguousarray(array)
    result = hashlib.sha256()
    result.update(str(array.dtype).encode('ascii'))
    result.update(str(array.shape).encode('ascii'))
    result.update(array.tobytes())
    return result.hexdigest()


def scalar(array):
    return np.asarray(array).item()


def read_archive(path, identity, required, linked=None):
    with np.load(str(path), allow_pickle=False) as archive:
        validate_npz_metadata(
            archive, str(path), str(identity['protocol_sha256']),
            str(identity['checkpoint_id']), required_arrays=required,
            checkpoint_step=1300002)
        for key, value in dict(identity, **(linked or {})).items():
            if key not in archive.files or not np.array_equal(archive[key], np.asarray(value)):
                raise ValueError(f'Cached artifact has wrong {key}: {path}')
        return {key: archive[key].copy() for key in archive.files}


def publish_json(path, value):
    atomic_write_bytes(str(path), (json.dumps(value, indent=2, sort_keys=True,
                                            allow_nan=False) + '\n').encode(), overwrite=True)


def make_config(seed, route):
    vp, fox = get_vp(), get_fox()
    fox.model.fox_u = -5.0
    fox.model.fox_drift_schedule = 'constant'
    fox.model.fox_normalize_scale = False
    fox.model.fox_matern_length_scale = 346.4101615138
    fox.model.fox_target_terminal_variance = 1.0
    vp.seed, vp.allow_tf32, vp.cudnn_benchmark = 42, True, True
    vp.deterministic = False
    vp.allow_fp16_reduced_precision_reduction = False
    vp.sampling.time_grid = 'uniform_logsnr'
    vp.sampling.logsnr_min, vp.sampling.logsnr_max = -10.0, ENDPOINT
    vp.sampling.noise_removal = False  # Return raw x at the actual endpoint.
    vp.sampling.probability_flow = False
    vp.eval.begin_ckpt = vp.eval.end_ckpt = 260
    vp.eval.num_samples, vp.eval.batch_size = COUNT, BATCH
    vp.eval.sampling_num_scales, vp.eval.sampling_seed = STEPS, seed
    vp.eval.enable_loss = vp.eval.enable_bpd = False
    vp.eval.enable_sampling = True
    vp.eval.endpoint_denoise_variants = False
    vp.eval.denoiser_swap_route = route
    vp.eval.denoiser_swap_threshold = THRESHOLD
    vp.eval.denoiser_swap_variants = VARIANTS
    vp.eval.denoiser_swap_alt_config = fox
    return vp, fox


def make_route(vp, fox, vp_sde, fox_sde, route, eps, device):
    return SplicedEpsModel(vp, fox, vp_sde, fox_sde,
                           mode='none' if route == 'vp' else 'tail',
                           threshold=None if route == 'vp' else THRESHOLD,
                           sampling_eps=eps, time_grid='uniform_logsnr',
                           device=device).eval()


def trajectory(sampler, model, seed):
    model.reset_route()
    with isolated_torch_rng(seed), torch.no_grad():
        state, nfe = sampler(model)
    # The existing splice class expects N predictor calls PLUS one final call.
    # Here final calls bypass the router and use both child models independently.
    expected = model.expected_routes[:-1]
    if nfe != len(expected) or tuple(model.route_history) != expected:
        raise RuntimeError('Trajectory used the wrong network route or NFE.')
    if state.dtype != torch.float32 or not torch.isfinite(state).all():
        raise RuntimeError('Invalid raw endpoint state.')
    return state.detach().cpu().numpy(), np.asarray(expected, dtype=np.bool_)


def denoise_pair(x, end_t, sde, vp_score, fox_score, inverse_scaler):
    """Apply both scores to the identical, unmodified raw state at the last node."""
    with torch.no_grad():
        original = x.clone()
        alpha, sigma = sde.marginal_prob(torch.ones_like(x[:, :1, :1, :1]), end_t)
        outputs = {}
        for name, score_fn in zip(VARIANTS, (vp_score, fox_score)):
            score = score_fn(x, end_t)
            if not torch.equal(x, original):
                raise RuntimeError('A denoiser modified its shared input state.')
            clean = (x + sigma[:, None, None, None].square() * score) / alpha
            image = inverse_scaler(clean)
            if not torch.isfinite(image).all():
                raise RuntimeError('Non-finite denoised output.')
            outputs[name] = image
        return outputs


def load_model(config, arm):
    path = ROOT / 'workdirs' / ARMS[arm] / 'checkpoints/checkpoint_260.pth'
    if not path.is_file():
        raise FileNotFoundError(path)
    model = mutils.create_model(config)
    optimizer = losses.get_optimizer(config, model.parameters())
    ema = ExponentialMovingAverage(model.parameters(), decay=config.model.ema_rate)
    state = dict(model=model, optimizer=optimizer, ema=ema, step=0,
                 model_protocol_sha256=build_model_manifest(config)['model_protocol_sha256'])
    restore_checkpoint(str(path), state, device=config.device, restore_rng=False)
    if int(state['step']) != 1300002:
        raise RuntimeError(f'Wrong final checkpoint step: {arm}')
    ema.copy_to(model.parameters())
    identity = {'checkpoint_id': state['checkpoint_id'], 'checkpoint_step': int(state['step']),
                'checkpoint_sha256': digest(path),
                'training_protocol_sha256': state.get('training_protocol_sha256') or 'legacy-unknown'}
    model.eval()
    del optimizer, ema, state
    gc.collect()
    torch.cuda.empty_cache()
    return model, identity


def check_snapshot():
    for filename in ('runtime.sha256', 'inputs.sha256'):
        subprocess.run(['sha256sum', '--status', '-c', '/opt/denoiser-swap/' + filename],
                       cwd=str(ROOT), check=True)


def evaluate_cell(seed, route, vp_model, fox_model, identities, inception, reference):
    check_snapshot()
    config, fox_config = make_config(seed, route)
    sde, eps = run_lib.get_sde(config, n_override=STEPS)
    fox_sde, _ = run_lib.get_sde(fox_config)
    model = make_route(vp_model, fox_model, sde, fox_sde, route, eps, config.device)
    grid = sde.sampling_time_grid(eps, grid='uniform_logsnr', device=config.device,
                                 dtype=torch.float32, N=STEPS + 1)
    grid_hash = array_digest(grid.cpu().numpy())
    expected_history = np.asarray(model.expected_routes[:-1], dtype=np.bool_)
    end_t = torch.full((BATCH,), grid[-1].item(), device=config.device, dtype=torch.float32)
    alpha, sigma = sde.marginal_prob(torch.ones((BATCH, 1, 1, 1), device=config.device), end_t)
    if not torch.allclose(sde.log_snr(end_t), torch.full_like(end_t, ENDPOINT), atol=1e-3, rtol=0):
        raise RuntimeError('Actual endpoint does not match the requested log-SNR.')
    folder = OUTPUT / f'{route}_seed{seed}'
    manifest = build_eval_manifest(config)
    manifest.pop('protocol_sha256')
    manifest['experiment'] = dict(
        name='celeba-denoiser-swap-v1', launcher_sha256=digest(ROOT / 'run_celeba_denoiser_swap.sh'),
        checkpoints=identities, reference_sha256=REFERENCE_SHA, grid_sha256=grid_hash,
        trajectory_nfe=STEPS, denoiser_calls=2, executed_nfe=STEPS + 2,
        nfe_per_output=STEPS + 1, paired_denoisers=True,
        between_route_pairing='not_assumed', raw_state_coordinates='centered_NCHW_float32',
        quantization='inverse_scaler; clip(image*255,0,255); uint8',
        route_vp_calls=int((~expected_history).sum()), route_fox_calls=int(expected_history.sum()))
    manifest['protocol_sha256'] = hashlib.sha256(json.dumps(
        manifest, sort_keys=True, separators=(',', ':'), ensure_ascii=True).encode()).hexdigest()
    protocol = ensure_eval_manifest(str(folder), manifest)
    base = dict(protocol_sha256=protocol, **identities['vp'],
                alt_checkpoint_id=identities['fox']['checkpoint_id'],
                alt_checkpoint_step=identities['fox']['checkpoint_step'],
                alt_training_protocol_sha256=identities['fox']['training_protocol_sha256'],
                grid_sha256=grid_hash, route=route, trajectory_nfe=STEPS,
                denoiser_calls=2, executed_nfe=STEPS + 2, nfe_per_output=STEPS + 1)
    sample_dir = folder / 'ckpt_260'
    sample_dir.mkdir(exist_ok=True)
    sampler = sampling.get_pc_sampler(
        sde, (BATCH, 3, 64, 64), sampling.EulerMaruyamaPredictor,
        sampling.NoneCorrector, inverse_scaler=lambda x: x,
        snr=config.sampling.snr, continuous=True, denoise=False,
        time_grid='uniform_logsnr', eps=eps, device=config.device)
    scores = (mutils.get_score_fn(sde, vp_model, train=False, continuous=True),
              mutils.get_score_fn(sde, model.fox_adapter, train=False, continuous=True))
    inverse_scaler = datasets.get_data_inverse_scaler(config)
    pools, logits = {v: [] for v in VARIANTS}, {v: [] for v in VARIANTS}
    statistics_hashes = []
    for round_id in range(math.ceil(COUNT / BATCH)):
        derived = derive_seed(seed, 'sampling-round', round_id)
        identity = dict(base, round_id=round_id, sampling_seed=derived,
                        sample_start=round_id * BATCH, num_samples=BATCH)
        state_path = sample_dir / f'state_{round_id}.npz'
        samples_path = sample_dir / f'samples_{round_id}.npz'
        stat_path = sample_dir / f'statistics_{round_id}.npz'
        state_keys = ('x_endpoint', 'route_history', 'endpoint_t', 'alpha', 'sigma', 'state_sha256')
        if state_path.exists():
            state = read_archive(state_path, identity, state_keys)
        else:
            if samples_path.exists() or stat_path.exists():
                raise RuntimeError(f'Cannot reconstruct a missing float state from cached uint8/features: {state_path}')
            x, history = trajectory(sampler, model, derived)
            state = dict(identity, x_endpoint=x, route_history=history,
                         endpoint_t=np.asarray(grid[-1].item()), alpha=alpha.cpu().numpy(),
                         sigma=sigma.cpu().numpy(), state_sha256=array_digest(x))
            atomic_savez(str(state_path), **state)
        x = state['x_endpoint']
        if (x.shape != (BATCH, 3, 64, 64) or x.dtype != np.float32 or
                not np.array_equal(state['route_history'], expected_history) or
                scalar(state['state_sha256']) != array_digest(x) or
                scalar(state['endpoint_t']) != grid[-1].item() or
                not np.array_equal(state['alpha'], alpha.cpu().numpy()) or
                not np.array_equal(state['sigma'], sigma.cpu().numpy())):
            raise RuntimeError(f'Invalid saved raw state or route: {state_path}')
        linked = dict(state_sha256=scalar(state['state_sha256']))
        image_keys = tuple(f'samples_{v}' for v in VARIANTS)
        if samples_path.exists():
            images = read_archive(samples_path, identity, image_keys, linked)
        else:
            outputs = denoise_pair(torch.from_numpy(x).to(config.device), end_t, sde,
                                   *scores, inverse_scaler=inverse_scaler)
            images = dict(identity, **linked)
            for variant in VARIANTS:
                images[f'samples_{variant}'] = np.clip(
                    outputs[variant].permute(0, 2, 3, 1).cpu().numpy() * 255., 0, 255).astype(np.uint8)
            atomic_savez(str(samples_path), **images)
            del outputs
        for key in image_keys:
            if images[key].shape != (BATCH, 64, 64, 3) or images[key].dtype != np.uint8:
                raise RuntimeError(f'Invalid denoised images: {samples_path}')
        linked['samples_sha256'] = digest(samples_path)
        stat_keys = tuple(k for v in VARIANTS for k in (f'pool_3_{v}', f'logits_{v}'))
        if stat_path.exists():
            stats = read_archive(stat_path, identity, stat_keys, linked)
        else:
            stats = dict(identity, **linked)
            for variant in VARIANTS:
                gc.collect()
                result = evaluation.run_inception_distributed(images[f'samples_{variant}'], inception)
                stats[f'pool_3_{variant}'] = result['pool_3'].numpy()
                stats[f'logits_{variant}'] = result['logits'].numpy()
            atomic_savez(str(stat_path), **stats)
        statistics_hashes.append(digest(stat_path))
        for variant in VARIANTS:
            pool, logit = stats[f'pool_3_{variant}'], stats[f'logits_{variant}']
            if pool.shape != (BATCH, 2048) or logit.ndim != 2 or logit.shape[0] != BATCH:
                raise RuntimeError(f'Wrong Inception batch shape: {stat_path}')
            if not np.isfinite(pool).all() or not np.isfinite(logit).all():
                raise RuntimeError(f'Non-finite Inception batch: {stat_path}')
            pools[variant].append(pool)
            logits[variant].append(logit)
        logging.info('Completed route=%s seed=%s batch=%s/20; both denoisers share state=%s',
                     route, seed, round_id + 1, linked['state_sha256'])
        del state, images, stats
    report_identity = dict(base, num_samples=COUNT, sampling_seed=seed, dataset_stats_id=REFERENCE_SHA,
                           statistics_sha256=hashlib.sha256(''.join(statistics_hashes).encode()).hexdigest())
    metric_keys = tuple(f'{metric}_{v}' for v in VARIANTS for metric in ('IS', 'fid', 'kid'))
    report_path = folder / 'report_260.npz'
    if report_path.exists():
        report = read_archive(report_path, report_identity, metric_keys)
    else:
        report = dict(report_identity)
        for variant in VARIANTS:
            pool = np.concatenate(pools[variant], axis=0)[:COUNT]
            logit = np.concatenate(logits[variant], axis=0)[:COUNT]
            report[f'IS_{variant}'] = np.asarray(tfgan.eval.classifier_score_from_logits(logit))
            report[f'fid_{variant}'] = np.asarray(tfgan.eval.frechet_classifier_distance_from_activations(reference, pool))
            tf.random.set_seed(derive_seed(seed, 'denoiser-swap-kid', route, variant))
            report[f'kid_{variant}'] = np.asarray(tfgan.eval.kernel_classifier_distance_from_activations(
                tf.convert_to_tensor(reference), tf.convert_to_tensor(pool)).numpy())
        if not all(np.isfinite(report[k]) for k in metric_keys):
            raise RuntimeError('Non-finite report metrics.')
        atomic_savez(str(report_path), **report)
    metrics = {k: float(report[k]) for k in metric_keys}
    publish_json(folder / 'result.json', dict(route=route, seed=seed, **metrics))
    for variant in VARIANTS:
        logging.info('SWAP route=%s seed=%s %s IS=%.6f FID=%.6f KID=%.9f', route, seed, variant,
                     metrics[f'IS_{variant}'], metrics[f'fid_{variant}'], metrics[f'kid_{variant}'])
    return metrics


def self_test():
    """CPU-only scientific checks: endpoint, pairing, route and stale-cache rejection."""
    config, fox_config = make_config(0, 'tail')
    config.device = fox_config.device = torch.device('cpu')
    sde = sde_lib.VPSDE(beta_min=config.model.beta_min, beta_max=config.model.beta_max, N=4)
    sde.noise_conditioning = 'time'
    sde.sampling_logsnr_min, sde.sampling_logsnr_max = -10.0, ENDPOINT
    fox_sde = sde_lib.FoxVPSDE(u=fox_config.model.fox_u, kernel='matern_3_2',
                              matern_length_scale=fox_config.model.fox_matern_length_scale,
                              target_terminal_variance=1.0, N=STEPS)
    eps = 1e-3
    class Constant(torch.nn.Module):
        def __init__(self, value):
            super().__init__()
            self.value = value
        def forward(self, x, labels):
            return torch.full_like(x, self.value)
    class Spy(torch.nn.Module):
        def __init__(self, value):
            super().__init__()
            self.value, self.inputs = value, []
        def forward(self, x, labels):
            self.inputs.append((x.clone(), labels.clone()))
            return torch.full_like(x, self.value)
    for route in ('vp', 'tail'):
        model = make_route(Constant(0.1), Constant(-0.2), sde, fox_sde, route, eps, 'cpu')
        sampler = sampling.get_pc_sampler(sde, (2, 3, 4, 4), sampling.EulerMaruyamaPredictor,
                    sampling.NoneCorrector, lambda x: x, snr=0.17, continuous=True,
                    denoise=False, time_grid='uniform_logsnr', eps=eps, device='cpu')
        x, history = trajectory(sampler, model, 17)
        assert len(history) == 4 and (not history.any() if route == 'vp' else history.any())
        assert len(model.route_history) == 4  # No hidden final network call.
        original_sampler = sampling.get_pc_sampler(
            sde, (2, 3, 4, 4), sampling.EulerMaruyamaPredictor, sampling.NoneCorrector,
            lambda y: y, snr=0.17, continuous=True, denoise=True,
            endpoint_variants=True, time_grid='uniform_logsnr', eps=eps, device='cpu')
        model.reset_route()
        with isolated_torch_rng(17):
            original_outputs, original_nfe = original_sampler(model)
        model.checked_route_metadata()
        assert original_nfe == 5
        assert np.array_equal(x, original_outputs['x_t'].numpy())
        row_scores = [mutils.get_score_fn(sde, m, train=False, continuous=True)
                      for m in (model.vp_model, model.fox_adapter)]
        row_grid = sde.sampling_time_grid(eps, grid='uniform_logsnr', device='cpu',
                                          dtype=torch.float32, N=5)
        row_outputs = denoise_pair(torch.from_numpy(x), row_grid[-1].repeat(2), sde,
                                   *row_scores, inverse_scaler=lambda y: y)
        variant = 'tweedie_vp' if route == 'vp' else 'tweedie_fox'
        torch.testing.assert_close(row_outputs[variant], original_outputs['tweedie_xt'])
    a, b = Spy(0.25), Spy(-0.5)
    scores = [mutils.get_score_fn(sde, m.eval(), train=False, continuous=True) for m in (a, b)]
    x = torch.from_numpy(x)
    grid = sde.sampling_time_grid(eps, grid='uniform_logsnr', device='cpu', dtype=torch.float32, N=5)
    t = grid[-1].repeat(2)
    original = x.clone()
    outputs = denoise_pair(x, t, sde, *scores, inverse_scaler=lambda y: y)
    alpha, sigma = sde.marginal_prob(torch.ones_like(x[:, :1, :1, :1]), t)
    for variant, eps_value in zip(VARIANTS, (0.25, -0.5)):
        torch.testing.assert_close(outputs[variant], (original - sigma[:, None, None, None] * eps_value) / alpha)
    assert torch.equal(x, original)
    assert torch.equal(a.inputs[0][0], b.inputs[0][0])
    torch.testing.assert_close(a.inputs[0][1], t * 999)
    identity = dict(protocol_sha256='test', checkpoint_id='vp', checkpoint_step=1300002,
                    alt_checkpoint_id='fox', round_id=0, sampling_seed=17)
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / 'state.npz'
        atomic_savez(str(path), **identity, x_endpoint=x.numpy())
        restored = read_archive(path, identity, ('x_endpoint',))
        assert np.array_equal(restored['x_endpoint'], x.numpy())
        for changed in (dict(identity, sampling_seed=18), dict(identity, alt_checkpoint_id='other')):
            try:
                read_archive(path, changed, ('x_endpoint',))
            except ValueError:
                pass
            else:
                raise AssertionError('A stale seed/alternate-checkpoint cache was accepted.')
    print('PASS: raw NFE/routes; original E4/E3b control equivalence; actual endpoint; paired inputs; analytic Tweedie; float persistence; cache identity.')


def main():
    if os.environ.get('SWAP_SELF_TEST') == '1':
        self_test()
        return
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError('Expected exactly one Docker-visible CUDA GPU.')
    check_snapshot()
    if digest(ROOT / REFERENCE) != REFERENCE_SHA:
        raise RuntimeError('Unexpected fixed CelebA reference.')
    config, fox_config = make_config(0, 'vp')
    configure_reproducibility(config, tensorflow_module=tf)
    models, identities = {}, {}
    for arm, cfg in (('vp', config), ('fox', fox_config)):
        models[arm], identities[arm] = load_model(cfg, arm)
    # Read the exact pinned reference directly: inference needs no CelebA loader,
    # and must never fall back to recomputing stats for another dataset path.
    with np.load(str(ROOT / REFERENCE), allow_pickle=False) as archive:
        reference = archive['pool_3'].copy()
    if (reference.ndim != 2 or reference.shape[1] != 2048 or
            reference.shape[0] < 2 or not np.isfinite(reference).all()):
        raise RuntimeError('Invalid fixed reference features.')
    inception = evaluation.get_inception_model(inceptionv3=False)
    results = {}
    # Finish the full 2x2 comparison on seed 0 before starting seed 1.
    for seed in (0, 1):
        cells = {}
        for route in ('vp', 'tail'):
            logging.info('=== Denoiser swap route=%s seed=%s samples=%s ===', route, seed, COUNT)
            cells[route] = evaluate_cell(seed, route, models['vp'], models['fox'], identities,
                                         inception, reference)
            gc.collect()
        a, b = (cells['vp'][f'fid_{v}'] for v in VARIANTS)
        c, d = (cells['tail'][f'fid_{v}'] for v in VARIANTS)
        results[str(seed)] = dict(cells=cells, fid_contrasts=dict(
            denoiser_on_vp=b-a, denoiser_on_tail=d-c, tail_with_vp_denoiser=c-a,
            total_tail=d-a, interaction=(d-c)-(b-a)))
        publish_json(OUTPUT / 'summary.json', dict(
            status='completed' if seed == 1 else 'seed0_completed', num_samples=COUNT,
            results=results, uncertainty='point estimates only; bootstrap not run'))
    check_snapshot()


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s', force=True)
    main()
PY_DENOISER_SWAP

if [[ "${SWAP_SELF_TEST:-0}" != 1 ]]; then
  printf 'completed\n' > /output/queue.state
  echo "Denoiser swap completed at $(date -Is)"
fi
