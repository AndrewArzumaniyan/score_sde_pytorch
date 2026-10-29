#!/usr/bin/env bash
# Read-only checkpoint diagnostics: common-lambda errors and final-denoiser swaps.
# No training. The final test replays the old sampler ONCE to cache FLOAT states:
# historical samples_*.npz contain uint8 images, not recoverable native states.
# bash "$0" --help; --stage profile avoids the expensive sampler replay.
# All new artifacts go into results/, never into the original eval/workdirs.
set -euo pipefail
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"
export TFDS_DATA_DIR="${TFDS_DATA_DIR:-$ROOT_DIR/.tfds}"
export TORCH_EXTENSIONS_DIR="${TORCH_EXTENSIONS_DIR:-$ROOT_DIR/.torch_extensions_rtx}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-$ROOT_DIR/.cache_rtx}"
export TFHUB_CACHE_DIR="${TFHUB_CACHE_DIR:-$ROOT_DIR/.cache/tfhub_smoke}"
export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-12.0}"
export MAX_JOBS="${MAX_JOBS:-8}"
export TF_ENABLE_ONEDNN_OPTS=0
export TF_NUM_INTRAOP_THREADS="${TF_NUM_INTRAOP_THREADS:-8}"
export TF_NUM_INTEROP_THREADS="${TF_NUM_INTEROP_THREADS:-2}"
export PYTHONDONTWRITEBYTECODE=1
export DDPM_DIAGNOSTICS_SCRIPT="$ROOT_DIR/$(basename "${BASH_SOURCE[0]}")"
exec "${PYTHON:-python}" -u - "$@" <<'PY_DIAGNOSTICS'
"""DDPM++ VP/FOX at 600k, EMA FP32, no learning or checkpoint writes.

Profile: same CIFAR test images and Gaussian draws for both networks at each
actual VP log-SNR. FOX receives its native scale/time via NativeEpsAdapter.
Errors are epsilon/x0 denoising MSE, not error against the exact marginal score.

Final: original 1000-EM paths at batch 512, seeds 0/1; cache native float32 x_t,
then apply VP/FOX denoisers to each identical state. Both 2x2 diagonal FIDs must
reproduce original reports before the experiment is marked complete.
"""
import argparse
from datetime import datetime, timezone
import csv
import hashlib
import io
import json
import logging
import math
import os
from pathlib import Path
import sys
import time

import numpy as np

LOWER = -10.0
UPPER = 9.115429865459795
ARMS = ('vp', 'fox')
CHECKPOINT = 12
STEP = 600001
COUNT = 5000
BATCH = 512


def require(condition, message):
    if not condition:
        raise ValueError(message)


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def manifest_hash(manifest, field):
    payload = {k: v for k, v in manifest.items() if k != field}
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(',', ':'),
                                     ensure_ascii=True).encode()).hexdigest()


def scalar(archive, key):
    value = np.asarray(archive[key])
    require(value.shape == (), 'Expected scalar metadata: ' + key)
    return value.item()


def quantize(x):
    # Exact ordering of the historical centered-data inverse scaler and uint8 conversion.
    return np.clip(((x + 1.) / 2.).permute(0, 2, 3, 1).cpu().numpy() * 255.,
                   0, 255).astype(np.uint8)


def final_tweedie(x, score, alpha, sigma):
    # Preserve the operation ordering used by sampling.get_pc_sampler.
    return (x + sigma ** 2 * score) / alpha


def profile_summary(eps_errors, x0_errors, grid_metadata):
    """SE clusters all noise repetitions for each clean test image."""
    rows = []
    # Arrays: noise repetition, network (VP/FOX), lambda, clean-image index.
    for j, item in enumerate(grid_metadata):
        row = dict(item)
        for name, errors in (('epsilon_mse', eps_errors), ('x0_mse', x0_errors)):
            vp, fox = errors[:, 0, j, :], errors[:, 1, j, :]
            delta_by_image = (fox - vp).mean(axis=0)
            row[name + '_vp'] = float(vp.mean())
            row[name + '_fox'] = float(fox.mean())
            row[name + '_fox_minus_vp'] = float(delta_by_image.mean())
            row[name + '_paired_se_across_images'] = float(
                delta_by_image.std(ddof=1) / math.sqrt(len(delta_by_image)))
        rows.append(row)
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage', choices=('all', 'profile', 'final'), default='all')
    parser.add_argument('--vp-workdir', default='workdirs/cifar10_vp_ddpmpp_deep_200k_rtx_seed42')
    parser.add_argument('--fox-workdir', default='workdirs/cifar10_fox_ddpmpp_deep_matern32_kappa1000_200k_rtx_seed42')
    parser.add_argument('--profile-count', type=int, default=5000)
    parser.add_argument('--profile-batch', type=int, default=128)
    parser.add_argument('--lambda-points', type=int, default=41)
    parser.add_argument('--noise-seeds', nargs='+', type=int, default=[0, 1])
    parser.add_argument('--output-root', default='results/cifar10_ddpmpp_deep_600k_score_denoiser_diagnostics')
    args = parser.parse_args()
    require(2 <= args.profile_count <= 10000 and 1 <= args.profile_batch <= 512,
            'Need 2..10000 profile images and batch 1..512.')
    require(args.profile_count <= (10000 // args.profile_batch) * args.profile_batch,
            'Profile count exceeds the full batches available in the test split.')
    require(args.lambda_points >= 2 and len(set(args.noise_seeds)) == len(args.noise_seeds)
            and all(0 <= s < 2**31 for s in args.noise_seeds), 'Invalid lambda grid/noise seeds.')

    # Same startup and numerical settings as the existing RTX eval, no JAX dataset patch.
    import torch
    require(torch.cuda.is_available() and torch.cuda.device_count() == 1,
            'Run in score-sde-rtx with exactly one visible CUDA GPU.')
    torch.cuda.init()
    import tensorflow as tf
    tf.config.set_visible_devices([], 'GPU')
    import tensorflow_gan as tfgan
    import run_lib
    import datasets
    import evaluation
    evaluation.jax = None  # Historical Inception runs on CPU.
    import losses
    import sampling
    from models import utils as mutils
    from models.ema import ExponentialMovingAverage
    from splice_model import NativeEpsAdapter
    from configs.vp.cifar10_ddpmpp_deep_continuous_matched import get_config as get_vp
    from configs.fox.cifar10_ddpmpp_deep_continuous_matern_3_2_kappa1000 import get_config as get_fox
    from artifact_utils import (atomic_write_bytes, atomic_savez, build_model_manifest,
                                build_eval_manifest, validate_npz_metadata)
    from reproducibility import configure_reproducibility, derive_seed, isolated_torch_rng
    from utils import restore_checkpoint

    logging.basicConfig(level=logging.INFO, stream=sys.stdout,
                        format='%(asctime)s %(levelname)s %(message)s', force=True)
    started = time.monotonic()
    output = Path(args.output_root)
    configs = {a: f('eval') for a, f in (('vp', get_vp), ('fox', get_fox))}
    workdirs = dict(vp=Path(args.vp_workdir), fox=Path(args.fox_workdir))
    reference_path = Path('assets/stats/cifar10_stats.npz')
    require(reference_path.is_file(), 'Missing reference: ' + str(reference_path))
    if args.stage in ('all', 'profile'):
        require((Path(os.environ['TFDS_DATA_DIR']) / 'cifar10').is_dir(), 'Missing prepared CIFAR-10.')
    reference_sha = file_hash(reference_path)
    for arm, cfg in configs.items():
        cfg.eval.begin_ckpt = cfg.eval.end_ckpt = CHECKPOINT
        require(cfg.model.num_res_blocks == 8 and not cfg.model.fir
                and cfg.model.progressive_input == 'none' and not cfg.model.scale_by_sigma
                and cfg.model.noise_conditioning == 'time', 'Expected native DDPM++ deep epsilon models.')
        require(cfg.training.sde == ('vpsde' if arm == 'vp' else 'foxvpsde'), 'Wrong SDE.')
    configure_reproducibility(configs['vp'], tensorflow_module=tf)

    models, identities, sdes, sampling_eps = {}, {}, {}, {}
    inputs = {}
    for arm in ARMS:
        cfg = configs[arm]
        path = workdirs[arm] / 'checkpoints' / 'checkpoint_12.pth'
        training_path = workdirs[arm] / 'run_manifest.json'
        require(path.is_file() and training_path.is_file(), 'Missing checkpoint/manifest: ' + str(workdirs[arm]))
        training_manifest = json.loads(training_path.read_text())
        require(manifest_hash(training_manifest, 'training_protocol_sha256') == training_manifest['training_protocol_sha256'],
                'Corrupt training manifest: ' + str(training_path))
        model_manifest = build_model_manifest(cfg)
        require(training_manifest['model_manifest'] == model_manifest, 'Checkpoint/config model mismatch: ' + arm)
        model = mutils.create_model(cfg)
        optimizer = losses.get_optimizer(cfg, model.parameters())
        ema = ExponentialMovingAverage(model.parameters(), cfg.model.ema_rate)
        state = dict(model=model, optimizer=optimizer, ema=ema, step=0,
                     model_protocol_sha256=model_manifest['model_protocol_sha256'],
                     training_protocol_sha256=training_manifest['training_protocol_sha256'])
        restore_checkpoint(str(path), state, cfg.device, restore_rng=False)
        require(int(state['step']) == STEP, 'Expected checkpoint state.step=600001: ' + str(path))
        require(ema.decay == cfg.model.ema_rate, 'EMA decay mismatch.')
        ema.copy_to(model.parameters())
        model.eval()
        for tensor in model.parameters():
            require(bool(torch.isfinite(tensor).all()), 'Non-finite EMA parameters: ' + arm)
        identities[arm] = dict(checkpoint_id=state['checkpoint_id'], checkpoint_step=STEP,
                               model_protocol_sha256=state['model_protocol_sha256'],
                               training_protocol_sha256=state['training_protocol_sha256'])
        inputs[str(path)] = file_hash(path)
        inputs[str(training_path)] = file_hash(training_path)
        models[arm] = model
        sdes[arm], sampling_eps[arm] = run_lib.get_sde(cfg, n_override=1000)
        del state, ema, optimizer
        torch.cuda.empty_cache()
        logging.info('Loaded %s checkpoint 12, EMA, step=%d, id=%s', arm, STEP, identities[arm]['checkpoint_id'])

    # Validate that we replay exactly the old eval protocol, not a new source/config.
    baselines = {}
    for seed in (0, 1):
        for arm in ARMS:
            cfg = configs[arm]
            cfg.eval.sampling_seed = seed
            folder = workdirs[arm] / (
                f'eval_cifar_endpoint_denoise_ddpmpp_deep600k_screen_{arm}_common_'
                f'ckpt12_nfe1000_5000samples_batch512_seed{seed}')
            manifest_path, report_path = folder / 'manifest.json', folder / 'report_12.npz'
            saved_manifest = json.loads(manifest_path.read_text())
            current_manifest = build_eval_manifest(cfg)
            require(saved_manifest == current_manifest,
                    'Eval source/config/environment differs from the saved baseline: ' + str(folder))
            with np.load(report_path, allow_pickle=False) as z:
                identity = identities[arm]
                validate_npz_metadata(z, str(report_path), saved_manifest['protocol_sha256'],
                                      identity['checkpoint_id'], checkpoint_step=STEP,
                                      training_protocol_sha256=identity['training_protocol_sha256'],
                                      required_arrays=('fid_x_mean', 'fid_x_t', 'fid_tweedie_xt'))
                require(scalar(z, 'dataset_stats_id') == reference_sha
                        and scalar(z, 'num_samples') == COUNT and scalar(z, 'nfe') == 1001,
                        'Baseline reference/count/NFE differs: ' + str(folder))
                report = {k: scalar(z, k) for k in ('fid_x_mean', 'fid_x_t', 'fid_tweedie_xt')}
            baselines[arm, seed] = dict(folder=folder, manifest=saved_manifest, report=report)
            inputs[str(manifest_path)] = file_hash(manifest_path)
            inputs[str(report_path)] = file_hash(report_path)
    # Sampling seed is set explicitly before each replay; restore preset for identity.
    for cfg in configs.values():
        cfg.eval.sampling_seed = 0
    settings = {k: v for k, v in vars(args).items() if k not in ('stage', 'output_root')}
    protocol = dict(schema_version=1, settings=settings, inputs=inputs,
                    checkpoint_identities=identities, reference_sha256=reference_sha,
                    script_sha256=file_hash(os.environ['DDPM_DIAGNOSTICS_SCRIPT']),
                    eval_source_sha256=baselines['vp', 0]['manifest']['source_sha256'],
                    environment=baselines['vp', 0]['manifest']['environment'],
                    final_protocol=dict(count=COUNT, batch=BATCH, nfe=1001, sampling_seeds=[0, 1],
                                        logsnr_min=LOWER, logsnr_max=UPPER))
    protocol['diagnostic_protocol_sha256'] = manifest_hash(protocol, 'diagnostic_protocol_sha256')
    output.mkdir(parents=True, exist_ok=True)
    manifest_path = output / 'manifest.json'
    if manifest_path.exists():
        require(json.loads(manifest_path.read_text()) == protocol,
                'Diagnostic output belongs to another protocol; use a new --output-root.')
    else:
        require(not any(output.iterdir()), 'Nonempty diagnostic directory has no manifest.')
        atomic_write_bytes(str(manifest_path), (json.dumps(protocol, indent=2) + '\n').encode(), overwrite=False)
    token = protocol['diagnostic_protocol_sha256']
    progress_path = output / 'summary.json'
    result = (json.loads(progress_path.read_text()) if progress_path.exists() else
              dict(schema_version=1, status='running', created_utc=datetime.now(timezone.utc).isoformat(),
                   diagnostic_protocol_sha256=token, profile=dict(status='pending'), final=dict(status='pending'),
                   limitations=['Denoising MSE is not exact marginal-score error and does not by itself explain FID.',
                                'Final swaps hold each native trajectory fixed; they do not test replacing its earlier score calls.',
                                'Fixed checkpoints and seeds; no training-seed uncertainty or causal attribution to training measure.']))
    require(result['diagnostic_protocol_sha256'] == token, 'Summary protocol mismatch.')
    result.pop('last_error', None)

    def save_progress():
        result['elapsed_seconds_this_invocation'] = time.monotonic() - started
        result['status'] = ('failed' if 'last_error' in result else
                            'complete' if all(result[k]['status'] == 'complete' for k in ('profile', 'final'))
                            else 'partial')
        atomic_write_bytes(str(progress_path), (json.dumps(result, indent=2, allow_nan=False) + '\n').encode(), overwrite=True)

    def save_npz(path, **arrays):
        atomic_savez(str(path), overwrite=False, diagnostic_protocol_sha256=np.asarray(token), **arrays)

    def read_npz(path):
        with np.load(path, allow_pickle=False) as z:
            require(scalar(z, 'diagnostic_protocol_sha256') == token, 'Diagnostic cache identity differs: ' + str(path))
            return {key: np.asarray(z[key]).copy() for key in z.files}

    networks = {(origin, target): (models[target] if origin == target else
                 NativeEpsAdapter(models[target], sdes[origin], sdes[target]).eval())
                for origin in ARMS for target in ARMS}

    def coefficients(arm, times):
        one = torch.ones((len(times), 1, 1, 1), device=times.device, dtype=times.dtype)
        alpha, _ = sdes[arm].marginal_prob(one, times)
        sigma = sdes[arm].marginal_prob(torch.zeros_like(one), times)[1]
        return alpha, sigma[:, None, None, None]

    def run_profile():
        logging.info('Starting common-lambda error profile: %d test images, %d levels, noise seeds=%s',
                     args.profile_count, args.lambda_points, args.noise_seeds)
        clean_path = output / 'profile' / 'clean_test_images.npz'
        if clean_path.exists():
            clean = read_npz(clean_path)['clean_images']
        else:
            data_cfg = get_vp('eval')
            data_cfg.eval.batch_size = args.profile_batch
            _, test_ds, _ = datasets.get_dataset(data_cfg, uniform_dequantization=False, evaluation=True)
            chunks = []
            for batch in test_ds:
                images = np.asarray(batch['image']._numpy(), dtype=np.float32)
                chunks.append((2. * images - 1.).transpose(0, 3, 1, 2))
                if sum(len(x) for x in chunks) >= args.profile_count:
                    break
            clean = np.concatenate(chunks)[:args.profile_count]
            save_npz(clean_path, clean_images=clean, split=np.asarray('CIFAR10/test'))
        require(clean.dtype == np.float32 and clean.shape == (args.profile_count, 3, 32, 32)
                and np.isfinite(clean).all() and clean.min() >= -1 and clean.max() <= 1,
                'Invalid cached clean test images.')
        clean_sha = file_hash(clean_path)
        lambdas = np.linspace(LOWER, UPPER, args.lambda_points)
        times = sdes['vp'].time_from_log_snr(torch.tensor(lambdas, device='cuda', dtype=torch.float32))
        alpha, sigma = coefficients('vp', times)
        actual_lambda = 2 * torch.log(alpha.flatten() / sigma.flatten())
        fox_times = sdes['fox'].time_from_log_snr(actual_lambda)
        metadata = [dict(lambda_requested=float(lambdas[j]), lambda_actual=float(actual_lambda[j]),
                         t_vp=float(times[j]), t_fox=float(fox_times[j])) for j in range(len(lambdas))]
        shape = (len(args.noise_seeds), 2, len(lambdas), args.profile_count)
        eps_errors, x0_errors = np.empty(shape, dtype=np.float64), np.empty(shape, dtype=np.float64)
        noise_hashes = []
        for k, seed in enumerate(args.noise_seeds):
            generator = torch.Generator(device='cpu').manual_seed(derive_seed(seed, 'ddpmpp600-profile-noise'))
            noise = torch.randn(clean.shape, generator=generator, dtype=torch.float32)
            noise_hashes.append(hashlib.sha256(noise.numpy().tobytes()).hexdigest())
            for batch_id, start in enumerate(range(0, args.profile_count, args.profile_batch)):
                end = min(start + args.profile_batch, args.profile_count)
                path = output / 'profile' / f'noise{seed}_batch{batch_id}.npz'
                if path.exists():
                    cached = read_npz(path)
                    require(scalar(cached, 'start') == start and scalar(cached, 'end') == end
                            and scalar(cached, 'noise_sha256') == noise_hashes[-1]
                            and scalar(cached, 'clean_images_sha256') == clean_sha, 'Profile chunk mismatch.')
                    errors = cached['epsilon_mse'], cached['x0_mse']
                else:
                    x0 = torch.from_numpy(clean[start:end]).to('cuda')
                    z = noise[start:end].to('cuda')
                    chunks = [np.empty((2, len(lambdas), end-start), dtype=np.float64) for _ in range(2)]
                    with torch.no_grad():
                        for j, t in enumerate(times):
                            x = alpha[j:j+1] * x0 + sigma[j:j+1] * z
                            labels = t.expand(len(x)) * 999
                            for m, target in enumerate(ARMS):
                                prediction = networks['vp', target](x, labels)
                                estimate = (x - sigma[j:j+1] * prediction) / alpha[j:j+1]
                                chunks[0][m, j] = (prediction-z).square().mean(dim=(1, 2, 3)).cpu().numpy()
                                chunks[1][m, j] = (estimate-x0).square().mean(dim=(1, 2, 3)).cpu().numpy()
                    errors = chunks
                    save_npz(path, epsilon_mse=chunks[0], x0_mse=chunks[1], start=np.asarray(start),
                             end=np.asarray(end), noise_sha256=np.asarray(noise_hashes[-1]),
                             clean_images_sha256=np.asarray(clean_sha))
                for dest, values in zip((eps_errors, x0_errors), errors):
                    require(values.shape == (2, len(lambdas), end-start) and np.isfinite(values).all(), 'Invalid profile errors.')
                    dest[k, :, :, start:end] = values
                logging.info('Profile noise seed %d: %d/%d images', seed, end, args.profile_count)
        rows = profile_summary(eps_errors, x0_errors, metadata)
        csv_path = output / 'profile.csv'
        buffer = io.StringIO()
        writer = csv.DictWriter(buffer, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
        atomic_write_bytes(str(csv_path), buffer.getvalue().encode(), overwrite=True)
        per_image = output / 'profile_errors.npz'
        if per_image.exists():
            cached = read_npz(per_image)
            require(scalar(cached, 'clean_images_sha256') == clean_sha
                    and np.array_equal(cached['epsilon_mse'], eps_errors)
                    and np.array_equal(cached['x0_mse'], x0_errors), 'Aggregate profile cache differs from its chunks.')
        else:
            save_npz(per_image, epsilon_mse=eps_errors, x0_mse=x0_errors,
                     lambda_requested=lambdas, lambda_actual=actual_lambda.cpu().numpy(),
                     noise_seeds=np.asarray(args.noise_seeds), clean_images_sha256=np.asarray(clean_sha))
        result['profile'] = dict(status='complete', count=args.profile_count, split='CIFAR10/test',
                                 noise_seeds=args.noise_seeds, noise_sha256=noise_hashes,
                                 csv=str(csv_path), per_image_errors=str(per_image), rows=rows,
                                 comparison='Same VP-coordinate noisy state and epsilon target; FOX adapted to native R/time.',
                                 uncertainty='Paired SE over clean images after averaging noise repetitions; pointwise, not simultaneous.')
        save_progress()

    def inception_metrics(pools, logits, reference):
        return dict(FID=float(tfgan.eval.frechet_classifier_distance_from_activations(reference, pools)),
                    KID=float(tfgan.eval.kernel_classifier_distance_from_activations(
                        tf.convert_to_tensor(reference), tf.convert_to_tensor(pools)).numpy()),
                    IS=float(tfgan.eval.classifier_score_from_logits(logits)))

    def run_final():
        logging.info('Starting final-only swap. Old eval has uint8 outputs; missing float caches require sampler replay.')
        inception = evaluation.get_inception_model(inceptionv3=False)
        with np.load(reference_path, allow_pickle=False) as z:
            reference = np.asarray(z['pool_3'])
        require(reference.ndim == 2 and reference.shape[1] == 2048 and np.isfinite(reference).all(), 'Invalid reference features.')
        rows = []
        for seed in (0, 1):
            for origin in ARMS:
                cfg = configs[origin]
                cfg.eval.sampling_seed = seed
                grid = sdes[origin].sampling_time_grid(sampling_eps[origin], grid='uniform_logsnr',
                                                     device='cuda', dtype=torch.float32, N=1001)
                t_end = grid[-1]
                sampler = sampling.get_sampling_fn(cfg, sdes[origin], (BATCH, 3, 32, 32),
                                                   lambda x: x, sampling_eps[origin])
                pools = {target: [] for target in ARMS}
                logits = {target: [] for target in ARMS}
                replay_checks = []
                for round_id in range(math.ceil(COUNT / BATCH)):
                    round_seed = derive_seed(seed, 'sampling-round', round_id)
                    state_path = output / 'float_states' / f'seed{seed}' / origin / f'state_{round_id}.npz'
                    if state_path.exists():
                        state = read_npz(state_path)
                    else:
                        logging.info('Replay %s seed %d round %d/10: 1000 EM + native final denoiser', origin, seed, round_id+1)
                        with isolated_torch_rng(round_seed):
                            native, nfe = sampler(models[origin])
                        require(nfe == 1001, 'Unexpected replay NFE.')
                        native_images = {v: quantize(native[v]) for v in ('x_mean', 'x_t', 'tweedie_xt')}
                        original_path = baselines[origin, seed]['folder'] / 'ckpt_12' / f'samples_{round_id}.npz'
                        comparison = {}
                        if original_path.exists():
                            with np.load(original_path, allow_pickle=False) as old:
                                identity = identities[origin]
                                validate_npz_metadata(old, str(original_path), baselines[origin, seed]['manifest']['protocol_sha256'],
                                                      identity['checkpoint_id'], round_id=round_id, sampling_seed=round_seed,
                                                      checkpoint_step=STEP, training_protocol_sha256=identity['training_protocol_sha256'],
                                                      required_arrays=tuple('samples_'+v for v in native_images))
                                for variant, image in native_images.items():
                                    old_image = np.asarray(old['samples_'+variant])
                                    require(old_image.shape == image.shape and old_image.dtype == np.uint8, 'Invalid old samples.')
                                    comparison[variant] = dict(exact_match=bool(np.array_equal(image, old_image)),
                                                              changed_pixel_fraction=float(np.mean(image != old_image)))
                            comparison['old_samples_sha256'] = file_hash(original_path)
                        else:
                            comparison['old_samples_available'] = False
                        # Never infer floats from clipped/rounded images.
                        save_npz(state_path, x_t_native=native['x_t'].cpu().numpy(),
                                 samples_native_tweedie=native_images['tweedie_xt'],
                                 samples_x_mean=native_images['x_mean'],
                                 representation=np.asarray('native_centered_NCHW_float32_before_final_Tweedie'),
                                 round_id=np.asarray(round_id), sampling_seed=np.asarray(round_seed),
                                 t_endpoint=np.asarray(float(t_end)), replay_comparison=np.asarray(json.dumps(comparison)))
                        del native
                        state = read_npz(state_path)
                    require(scalar(state, 'representation') == 'native_centered_NCHW_float32_before_final_Tweedie'
                            and scalar(state, 'round_id') == round_id and scalar(state, 'sampling_seed') == round_seed
                            and scalar(state, 't_endpoint') == float(t_end), 'Wrong float-state identity.')
                    x_float = state['x_t_native']
                    require(x_float.dtype == np.float32 and x_float.shape == (BATCH, 3, 32, 32)
                            and np.isfinite(x_float).all(), 'Expected genuine, finite float32 sampler states.')
                    require(state['samples_native_tweedie'].dtype == np.uint8
                            and state['samples_native_tweedie'].shape == (BATCH, 32, 32, 3), 'Invalid native diagonal image cache.')
                    replay_checks.append(json.loads(scalar(state, 'replay_comparison')))
                    stats_path = output / 'final_statistics' / f'seed{seed}' / origin / f'statistics_{round_id}.npz'
                    if stats_path.exists():
                        stats = read_npz(stats_path)
                        require(scalar(stats, 'float_state_sha256') == file_hash(state_path), 'Float state changed after denoising.')
                    else:
                        x = torch.from_numpy(x_float).to('cuda')
                        times = t_end.expand(BATCH)
                        alpha, sigma = coefficients(origin, times)
                        stats = dict(float_state_sha256=np.asarray(file_hash(state_path)))
                        with torch.no_grad():
                            for target in ARMS:
                                score_fn = mutils.get_score_fn(sdes[origin], networks[origin, target], train=False, continuous=True)
                                # Identical operation ordering to the old sampler's Tweedie.
                                denoised = final_tweedie(x, score_fn(x, times), alpha, sigma)
                                image = quantize(denoised)
                                if target == origin:
                                    stats['diagonal_changed_pixel_fraction'] = np.asarray(
                                        float(np.mean(image != state['samples_native_tweedie'])))
                                pred = evaluation.run_inception_distributed(image, inception)
                                stats['pool_3_'+target] = pred['pool_3'].numpy()
                                stats['logits_'+target] = pred['logits'].numpy()
                                stats['samples_'+target] = image
                        save_npz(stats_path, **stats)
                    for target in ARMS:
                        pool, logit = stats['pool_3_'+target], stats['logits_'+target]
                        require(pool.shape == (BATCH, 2048) and logit.shape[0] == BATCH
                                and np.isfinite(pool).all() and np.isfinite(logit).all(), 'Invalid final Inception features.')
                        pools[target].append(pool); logits[target].append(logit)
                    logging.info('Final swap %s trajectory, seed %d: round %d/10 complete', origin, seed, round_id+1)
                metrics = {target: inception_metrics(np.concatenate(pools[target])[:COUNT],
                                                     np.concatenate(logits[target])[:COUNT], reference) for target in ARMS}
                native_fid = baselines[origin, seed]['report']['fid_tweedie_xt']
                require(np.isclose(metrics[origin]['FID'], native_fid, atol=2e-3, rtol=1e-4),
                        'Native diagonal FID does not reproduce original eval; do not interpret the swap: '
                        f'{origin} seed {seed}: {metrics[origin]["FID"]} vs {native_fid}')
                row = dict(sampling_seed=seed, trajectory=origin, count=COUNT, metrics_by_denoiser=metrics,
                           final_fid_fox_minus_vp=metrics['fox']['FID']-metrics['vp']['FID'],
                           original_native_fid=native_fid, native_fid_reproduced=True,
                           replay_checks=replay_checks, note='Only the last network is changed; the cached float input is identical.')
                rows.append(row)
                result['final'] = dict(status='running', rows=rows)
                save_progress()
                logging.info('Final %s trajectory seed %d: VP-denoiser FID=%.6f; FOX-denoiser FID=%.6f',
                             origin, seed, metrics['vp']['FID'], metrics['fox']['FID'])
        means = {origin: {target: {metric: float(np.mean([r['metrics_by_denoiser'][target][metric]
                    for r in rows if r['trajectory'] == origin])) for metric in ('FID', 'KID', 'IS')}
                    for target in ARMS} for origin in ARMS}
        result['final'] = dict(status='complete', rows=rows, mean_of_separate_seed_metrics=means,
                               mean_final_fid_fox_minus_vp={origin: means[origin]['fox']['FID']-means[origin]['vp']['FID']
                                                          for origin in ARMS},
                               matrix_definition='Rows: fixed native VP/FOX trajectories. Columns: final VP/FOX denoiser. Not pooled FID.')
        save_progress()

    save_progress()
    try:
        if args.stage in ('all', 'profile'):
            run_profile()
        if args.stage in ('all', 'final'):
            run_final()
    except Exception as error:
        result['last_error'] = repr(error)
        save_progress()
        logging.exception('Diagnostics stopped. Valid caches are reusable by rerunning the same command.')
        raise
    result.pop('last_error', None)
    save_progress()
    logging.info('Requested diagnostics finished: %s', progress_path)
    print(json.dumps({key: result[key] for key in ('status', 'profile', 'final')}, indent=2), flush=True)


if __name__ == '__main__':
    main()
PY_DIAGNOSTICS
