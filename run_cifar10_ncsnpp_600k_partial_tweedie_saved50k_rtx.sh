#!/usr/bin/env bash
# Native NCSN++ VP/FOX 600k: damp ONLY the final Tweedie correction.
# Reads the completed DPM++ 2M 64-call, extended-endpoint, seed-2 50k eval.
# No trajectory sampling, training, dataset iterator, or original artifact writes.
# 0%/100% reuse original Inception features; only 25%/50% need new features.
# All four weights use the SAME saved float states within each arm.
set -euo pipefail
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"
export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-12.0}"
export TORCH_EXTENSIONS_DIR="${TORCH_EXTENSIONS_DIR:-$ROOT_DIR/.torch_extensions_rtx}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-$ROOT_DIR/.cache_rtx}"
export TFDS_DATA_DIR="${TFDS_DATA_DIR:-$ROOT_DIR/.tfds}"
export TFHUB_CACHE_DIR="${TFHUB_CACHE_DIR:-$ROOT_DIR/.cache/tfhub_smoke}"
export MAX_JOBS="${MAX_JOBS:-8}"
export TF_ENABLE_ONEDNN_OPTS=0
export TF_NUM_INTRAOP_THREADS="${TF_NUM_INTRAOP_THREADS:-8}"
export TF_NUM_INTEROP_THREADS="${TF_NUM_INTEROP_THREADS:-2}"
export PYTHONDONTWRITEBYTECODE=1
export CIFAR_PARTIAL_TWEEDIE_SCRIPT="$ROOT_DIR/$(basename "${BASH_SOURCE[0]}")"
exec "${PYTHON:-python}" -u - "$@" <<'PY_PARTIAL'
"""Final-state intervention, not a new sampler or a training experiment.

In inverse-scaled, UNCLIPPED [0,1] data coordinates, y_w = y + w*(T(y)-y).
T is the original normalized-native ODE Tweedie formula, including its scale
conversion. This damps the COMPLETE correction, not just the score term.
FP32 inverse scaling loses a few bits: check the reconstructed full correction
against saved uint8 output, allowing <=0.1% pixels to differ by at most 1/255.
All extrema metrics use the ORIGINAL features, not reconstructed approximations.
The experiment describes fixed checkpoints/states/seed; no causal claim about
training measure or temporal correlation, and no confidence intervals are made.
"""
import argparse
import csv
from datetime import datetime, timezone
import gc
import hashlib
import io
import json
import math
import os
from pathlib import Path
import sys
import time

import numpy as np

COUNT, BATCH, CHECKPOINT, STEP, SEED = 50000, 512, 24, 600001, 2
ROUNDS = math.ceil(COUNT / BATCH)
WEIGHTS = {'w000': 0.0, 'w025': 0.25, 'w050': 0.5, 'w100': 1.0}
REFERENCE_SHA = 'c464743bce16cb0f64408eb36bdc7a7ef0e7b4dfd618c38f8119d4e29ef0c9d5'
IDS = {'vp': 'db841db3b8cd45949ef96d1d8915c1f9',
       'fox': '8c08568db0d144b9a231763669af5319'}
CHECKPOINT_SHA = {
    'vp': '2609013532af432a85fd147da4b8aa4e676ffec860ba5bf6f5fa482d43fd12b1',
    'fox': '497681bd00c89cffb7170b115f6f8c04d303e554cd4597e23f3d7c85d2ea7be6'}
IDENTITY = ('protocol_sha256', 'checkpoint_id', 'checkpoint_step',
            'training_protocol_sha256')


def require(ok, message):
    if not ok:
        raise ValueError(message)


def file_hash(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def manifest_hash(manifest, field):
    return hashlib.sha256(json.dumps(
        {k: v for k, v in manifest.items() if k != field}, sort_keys=True,
        separators=(',', ':'), ensure_ascii=True).encode()).hexdigest()


def scalar(z, key):
    a = np.asarray(z[key])
    require(a.shape == (), 'Expected scalar: ' + key)
    return a.item()


def quantize(y):
    return np.clip(y.transpose(0, 2, 3, 1) * 255., 0, 255).astype(np.uint8)


def mix(y, clean, weight):
    # Keep the two endpoints exact, including FP32 operation order.
    if weight == 0:
        return y
    if weight == 1:
        return clean
    return y + np.float32(weight) * (clean - y)


def image_diagnostics(y, clean, outputs, original):
    d = (clean.astype(np.float64) - y.astype(np.float64)).reshape(len(y), -1)
    result = dict(correction_rms=np.sqrt(np.mean(d*d, axis=1)),
                  correction_mean=d.mean(axis=1))
    for name, a in outputs.items():
        result['outside_fraction_' + name] = ((a < 0) | (a > 1)).reshape(len(a), -1).mean(axis=1)
        result['changed_fraction_' + name] = (quantize(a) != original).reshape(len(a), -1).mean(axis=1)
    return result


def check_protocol(cfg, arm):
    m, s, e = cfg['model'], cfg['sampling'], cfg['eval']
    for key, value in dict(name='ncsnpp', num_res_blocks=4, nf=128,
                           ch_mult=[1, 2, 2, 2], fir=True,
                           progressive_input='residual', scale_by_sigma=False,
                           num_scales=1000, ema_rate=0.9999).items():
        require(m.get(key) == value, f'{arm}: unexpected model.{key}: {m.get(key)!r}')
    require(m.get('noise_conditioning', 'time') == 'time'
            and cfg['training']['continuous'] is True
            and cfg['training']['sde'] == ('vpsde' if arm == 'vp' else 'foxvpsde')
            and cfg['training'].get('noise_distribution', 'uniform_time') == 'uniform_time',
            arm + ': unexpected SDE/conditioning/training distribution')
    require(cfg['seed'] == 42 and cfg['data']['dataset'] == 'CIFAR10'
            and cfg['data']['image_size'] == 32 and cfg['data']['num_channels'] == 3
            and cfg['data']['centered'] is True, arm + ': unexpected training/data settings')
    for key in ('allow_tf32', 'cudnn_benchmark', 'deterministic',
                'allow_fp16_reduced_precision_reduction'):
        require(cfg.get(key) is False, arm + ': expected disabled flag ' + key)
    for key, value in dict(method='ode_dpmpp_2m', probability_flow=True,
                           time_grid='uniform_logsnr', logsnr_min=-10.0,
                           logsnr_max=10.575749165617202).items():
        require(s.get(key) == value, arm + ': unexpected sampling.' + key)
    for key, value in dict(num_samples=COUNT, batch_size=BATCH, sampling_seed=SEED,
                           sampling_num_scales=64, endpoint_denoise_variants=True,
                           save_endpoint_states=True).items():
        require(e.get(key) == value, arm + ': unexpected eval.' + key)
    # build_eval_manifest excludes orchestration-only begin_ckpt/end_ckpt.
    require(tuple(e.get('endpoint_denoise_outputs', ())) == ('x_t', 'tweedie_xt'),
            arm + ': expected saved x_t / Tweedie pair')
    if arm == 'fox':
        require(m.get('fox_normalize_scale', False) is False, 'Expected real, unnormalized FOX')
        for key, value in dict(fox_u=-5.0, fox_drift_schedule='constant',
                               fox_kernel='matern_3_2', fox_matern_length_scale=346.4101615138,
                               fox_target_terminal_variance=1.0, fox_schedule_grid_size=4096).items():
            require(m.get(key) == value, 'Unexpected FOX model.' + key)


def validate_identity(z, report, path, round_id=None, round_seed=None):
    for key in IDENTITY:
        require(scalar(z, key) == report[key], 'Identity mismatch ' + key + ': ' + str(path))
    require(int(scalar(z, 'nfe')) == 65, 'Expected original 64+1 NFE: ' + str(path))
    if round_id is not None:
        require(scalar(z, 'round_id') == round_id and scalar(z, 'sampling_seed') == round_seed,
                'Wrong source round/seed: ' + str(path))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--vp-workdir', default='workdirs/cifar10_vp_200k_rtx_seed42')
    parser.add_argument('--fox-workdir', default='workdirs/cifar10_fox_matern32_kappa1000_200k_rtx_seed42')
    parser.add_argument('--output-dir', help='Optional NEW directory; existing directories are never overwritten.')
    args = parser.parse_args()
    # Parse --help before importing GPU/ML packages.
    import torch
    require(torch.cuda.is_available() and torch.cuda.device_count() == 1,
            'Run in score-sde-rtx with exactly one visible GPU.')
    torch.cuda.init()
    import tensorflow as tf
    tf.config.set_visible_devices([], 'GPU')
    import tensorflow_gan as tfgan
    from ml_collections import ConfigDict
    import run_lib  # Registers the project's trained architectures.
    import evaluation
    evaluation.jax = None  # Same CPU Inception path as the source evaluation.
    from artifact_utils import (atomic_savez, atomic_write_bytes, build_model_manifest,
                                build_eval_manifest)
    from models import utils as mutils
    from models.ema import ExponentialMovingAverage
    from losses import get_optimizer
    from reproducibility import configure_reproducibility, derive_seed
    from utils import restore_checkpoint

    started = time.monotonic()
    reference_path = Path('assets/stats/cifar10_stats.npz')
    require(file_hash(reference_path) == REFERENCE_SHA, 'CIFAR reference differs from original eval.')
    with np.load(reference_path, allow_pickle=False) as z:
        reference = np.asarray(z['pool_3']).copy()
    require(reference.shape == (COUNT, 2048) and np.isfinite(reference).all(), 'Invalid CIFAR reference.')
    sources, inputs = {}, {str(reference_path): REFERENCE_SHA}
    for arm in ('vp', 'fox'):
        workdir = Path(getattr(args, arm + '_workdir'))
        folder = workdir / (f'eval_cifar_ncsnpp600k_dpmpp64_extended50k_v1_ode_dpmpp_2m_{arm}_'
                            'extended_ckpt24_nfe64_50000samples_batch512_seed2')
        print('Checking saved float states:', folder, flush=True)
        manifest_path, report_path = folder/'manifest.json', folder/'report_24.npz'
        manifest = json.loads(manifest_path.read_text())
        require(manifest_hash(manifest, 'protocol_sha256') == manifest['protocol_sha256'],
                'Corrupt eval manifest: ' + str(manifest_path))
        check_protocol(manifest['config'], arm)
        with np.load(report_path, allow_pickle=False) as z:
            report = {k: scalar(z, k) for k in IDENTITY}
            require(report['protocol_sha256'] == manifest['protocol_sha256']
                    and report['checkpoint_id'] == IDS[arm] and report['checkpoint_step'] == STEP,
                    'Wrong source checkpoint/protocol: ' + str(folder))
            require(scalar(z, 'num_samples') == COUNT and scalar(z, 'dataset_stats_id') == REFERENCE_SHA,
                    'Source count/reference differs: ' + str(folder))
            validate_identity(z, report, report_path)
            baseline = {v: {metric: float(scalar(z, metric+'_'+v)) for metric in ('fid', 'kid', 'IS')}
                        for v in ('x_t', 'tweedie_xt')}
        checkpoint_path, training_path = workdir/'checkpoints/checkpoint_24.pth', workdir/'run_manifest.json'
        require(file_hash(checkpoint_path) == CHECKPOINT_SHA[arm], 'Checkpoint SHA differs: ' + arm)
        training = json.loads(training_path.read_text())
        require(manifest_hash(training, 'training_protocol_sha256') == training['training_protocol_sha256']
                and training['training_protocol_sha256'] == report['training_protocol_sha256'],
                'Training manifest differs from the evaluated checkpoint: ' + arm)
        cfg = ConfigDict(manifest['config'])  # Reconstruct from actual saved eval, not today's base config.
        require(build_model_manifest(cfg) == training['model_manifest'], 'Saved model/config mismatch: ' + arm)
        for path in (manifest_path, report_path, training_path):
            inputs[str(path)] = file_hash(path)
        inputs[str(checkpoint_path)] = CHECKPOINT_SHA[arm]
        sample_dir = folder/'ckpt_24'
        for kind in ('samples', 'statistics'):
            expected = {sample_dir/f'{kind}_{i}.npz' for i in range(ROUNDS)}
            require(set(sample_dir.glob(kind+'_*.npz')) == expected,
                    'Missing or extra source rounds: ' + str(sample_dir) + '/' + kind)
        # Pin every input before publishing a new result; detect mutation while reading later.
        for i in range(ROUNDS):
            for kind in ('samples', 'statistics'):
                path = sample_dir/f'{kind}_{i}.npz'
                inputs[str(path)] = file_hash(path)
        sources[arm] = dict(folder=folder, cfg=cfg, manifest=manifest, report=report,
                            baseline=baseline, training=training, checkpoint=checkpoint_path)
    a, b = (sources[arm]['manifest'] for arm in ('vp', 'fox'))
    require(a['source_sha256'] == b['source_sha256'] and a['environment'] == b['environment'],
            'Source VP/FOX runtimes differ.')
    require(a['config']['data'] == b['config']['data']
            and a['config']['sampling'] == b['config']['sampling'], 'Source data/sampling protocols differ.')
    configure_reproducibility(sources['vp']['cfg'], tensorflow_module=tf)
    current = build_eval_manifest(sources['vp']['cfg'])
    # New unrelated repository files can change the broad source hash. Record it;
    # enforce versions/model identity and validate endpoint reproduction directly.
    for key in ('python', 'numpy', 'tensorflow', 'torch', 'torch_cuda', 'cudnn'):
        require(current['environment'][key] == a['environment'][key],
                'Runtime version changed since original eval: ' + key)
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S.%fZ')
    output = Path(args.output_dir or ('results/cifar10_ncsnpp_partial_tweedie/600k_seed2_50k_' + stamp))
    require(not output.exists(), 'Output directory already exists; choose a NEW --output-dir: ' + str(output))
    protocol = dict(schema_version=1, experiment='partial_native_final_tweedie',
                    count=COUNT, batch_size=BATCH, rounds=ROUNDS, sampling_seed=SEED,
                    checkpoint_step=STEP, weights=WEIGHTS, lambda_max=10.575749165617202,
                    interpolation='Unclipped inverse-scaled data y + w*(T(y)-y); endpoints kept exact.',
                    full_reconstruction_max_changed_fraction=0.001,
                    full_reconstruction_max_uint8_difference=1,
                    input_sha256=inputs, script_sha256=file_hash(os.environ['CIFAR_PARTIAL_TWEEDIE_SCRIPT']),
                    original_runtime_source_sha256=a['source_sha256'],
                    current_runtime_source_sha256=current['source_sha256'], environment=current['environment'])
    protocol['diagnostic_protocol_sha256'] = manifest_hash(protocol, 'diagnostic_protocol_sha256')
    token = protocol['diagnostic_protocol_sha256']
    output.mkdir(parents=True, exist_ok=False)
    atomic_write_bytes(str(output/'manifest.json'), (json.dumps(protocol, indent=2)+'\n').encode())
    result = dict(schema_version=1, status='running', created_utc=stamp,
                  diagnostic_protocol_sha256=token, output_dir=str(output), arms={},
                  limitations=['One trained pair and sampling seed 2; no training-seed uncertainty.',
                               'FID change is not evidence of a specific visual or training mechanism.',
                               '0%/100% use original features; only 25%/50% have new Inception features.',
                               'Float inverse scaling has rounding loss; reconstruction tolerance is recorded.'])

    def save_progress():
        result['elapsed_seconds'] = time.monotonic() - started
        atomic_write_bytes(str(output/'summary.json'),
                           (json.dumps(result, indent=2, allow_nan=False)+'\n').encode(), overwrite=True)

    def save_npz(path, **arrays):
        atomic_savez(str(path), overwrite=False,
                     diagnostic_protocol_sha256=np.asarray(token), **arrays)

    save_progress()
    print('Output:', output, flush=True)
    print('No trajectories. 98 final network calls per arm; new Inception only for 25% / 50%.', flush=True)
    try:
        inception = evaluation.get_inception_model(inceptionv3=False)
        rows = []
        for arm, src in sources.items():
            cfg = src['cfg']
            configure_reproducibility(cfg, tensorflow_module=tf)
            model = mutils.create_model(cfg)
            ema = ExponentialMovingAverage(model.parameters(), cfg.model.ema_rate)
            optimizer = get_optimizer(cfg, model.parameters())
            state = dict(model=model, ema=ema, optimizer=optimizer, step=0,
                         model_protocol_sha256=src['training']['model_manifest']['model_protocol_sha256'],
                         training_protocol_sha256=src['report']['training_protocol_sha256'])
            restore_checkpoint(str(src['checkpoint']), state, cfg.device, restore_rng=False)
            require(state['step'] == STEP and state['checkpoint_id'] == IDS[arm], 'Wrong loaded checkpoint: ' + arm)
            ema.copy_to(model.parameters())
            model.eval()
            require(all(bool(torch.isfinite(p).all()) for p in model.parameters()), 'Non-finite EMA: ' + arm)
            del state, optimizer, ema
            # Reproduce the ODE's coefficient/grid precision and native score scaling.
            sde, eps = run_lib.get_sde(cfg, n_override=64)
            grid64 = sde.sampling_time_grid(eps, grid='uniform_logsnr', device=cfg.device,
                                            dtype=torch.float64, N=65)
            # Validate the inverse using the stable schedule in float64. Near
            # clean VP, marginal_prob's FP32 1-exp(2*log_alpha) loses relative
            # precision: lambda reconstructed from those sampler coefficients
            # can differ by >2e-4 even though time inversion is correct.
            requested_lambda = float(cfg.sampling.logsnr_max)
            schedule_lambda = float(sde.log_snr(grid64[-1:])[0])
            require(math.isfinite(schedule_lambda) and abs(schedule_lambda - requested_lambda) < 2e-6,
                    f'Endpoint inversion mismatch: {arm}; requested={requested_lambda:.12g}, '
                    f'stable_float64={schedule_lambda:.12g}, time={float(grid64[-1]):.12g}')
            grid = grid64.to(torch.float32)
            rounded_time_lambda = float(sde.log_snr(grid[-1:].to(torch.float64))[0])
            ones = torch.ones((65, 1, 1, 1), device=cfg.device, dtype=torch.float32)
            native_alpha, native_sigma = sde.marginal_prob(ones, grid)
            native_alpha = native_alpha.flatten()
            scale = torch.sqrt(native_alpha.square() + native_sigma.square())
            alpha, sigma = native_alpha / scale, native_sigma / scale
            score_fn = mutils.get_score_fn(sde, model, train=False, continuous=True)
            # Keep these FP32 coefficients EXACTLY as in ode_sampling.py;
            # replacing them by more accurate coefficients would change the
            # historical full-Tweedie operator that this test must reproduce.
            actual_lambda = float(2*torch.log(alpha[-1] / sigma[-1]))
            require(math.isfinite(actual_lambda) and math.isfinite(rounded_time_lambda),
                    'Non-finite endpoint coefficients: ' + arm)
            print(f'{arm} endpoint: requested={requested_lambda:.9f}; '
                  f'stable_float64={schedule_lambda:.9f}; '
                  f'from_original_fp32_coefficients={actual_lambda:.9f}', flush=True)
            pools, logits = {k: [] for k in WEIGHTS}, {k: [] for k in WEIGHTS}
            diagnostics, reproduction = {}, []
            arm_progress = dict(status='running', completed_rounds=0,
                                endpoint=dict(native_time=float(grid[-1]), native_time_float64=float(grid64[-1]),
                                              lambda_requested=requested_lambda,
                                              lambda_actual=schedule_lambda,
                                              lambda_at_float32_time_stable=rounded_time_lambda,
                                              lambda_from_fp32_sampler_coefficients=actual_lambda,
                                              native_alpha=float(native_alpha[-1]),
                                              native_scale=float(scale[-1]), normalized_sigma=float(sigma[-1])))
            result['arms'][arm] = arm_progress
            arm_started = time.monotonic()
            for i in range(ROUNDS):
                valid = min(BATCH, COUNT - i*BATCH)
                seed = derive_seed(SEED, 'sampling-round', i)
                paths = {k: src['folder']/'ckpt_24'/f'{k}_{i}.npz' for k in ('samples', 'statistics')}
                for path in paths.values():
                    require(file_hash(path) == inputs[str(path)], 'Source file changed: ' + str(path))
                with np.load(paths['samples'], allow_pickle=False) as z:
                    validate_identity(z, src['report'], paths['samples'], i, seed)
                    y = np.asarray(z['float_x_t']).copy()
                    original = np.asarray(z['samples_x_t']).copy()
                    old_full = np.asarray(z['samples_tweedie_xt']).copy()
                require(y.dtype == np.float32 and y.shape == (BATCH, 3, 32, 32)
                        and np.isfinite(y).all(), 'Invalid saved float state: ' + str(paths['samples']))
                for image in (original, old_full):
                    require(image.dtype == np.uint8 and image.shape == (BATCH, 32, 32, 3), 'Invalid original images.')
                require(np.array_equal(quantize(y), original), 'float_x_t does not reproduce original raw images.')
                with torch.no_grad():
                    normalized = (torch.from_numpy(y).to(cfg.device) * 2. - 1.) / scale[-1]
                    times = torch.full((BATCH,), grid[-1].item(), device=cfg.device, dtype=torch.float32)
                    normalized_score = scale[-1] * score_fn(normalized * scale[-1], times)
                    clean = (normalized + sigma[-1].square()*normalized_score) / alpha[-1]
                    clean = ((clean + 1.) / 2.).cpu().numpy()
                require(clean.shape == y.shape and np.isfinite(clean).all(), 'Non-finite final correction.')
                diff = np.abs(quantize(clean).astype(np.int16) - old_full.astype(np.int16))
                changed, maximum = float(np.mean(diff != 0)), int(diff.max())
                require(changed <= 0.001 and maximum <= 1,
                        f'{arm} round {i}: full Tweedie reproduction failed: changed={changed:.6g}, max={maximum}. '
                        'Do not interpret partial metrics; check weights, state coordinates and conditioning.')
                reproduction.append(dict(round_id=i, changed_pixel_fraction=changed, max_uint8_difference=maximum))
                outputs = {name: mix(y, clean, w) for name, w in WEIGHTS.items()}
                per_image = image_diagnostics(y[:valid], clean[:valid],
                                              {k: a[:valid] for k, a in outputs.items()}, original[:valid])
                for key, values in per_image.items():
                    diagnostics.setdefault(key, []).append(values)
                with np.load(paths['statistics'], allow_pickle=False) as z:
                    validate_identity(z, src['report'], paths['statistics'], i, seed)
                    require(scalar(z, 'num_samples') == BATCH, 'Wrong original feature count.')
                    for name, variant in (('w000', 'x_t'), ('w100', 'tweedie_xt')):
                        pools[name].append(np.asarray(z['pool_3_'+variant]).copy())
                        logits[name].append(np.asarray(z['logits_'+variant]).copy())
                new = {}
                for name in ('w025', 'w050'):
                    pred = evaluation.run_inception_distributed(quantize(outputs[name]), inception)
                    new['pool_3_'+name] = pred['pool_3'].numpy()
                    new['logits_'+name] = pred['logits'].numpy()
                    pools[name].append(new['pool_3_'+name])
                    logits[name].append(new['logits_'+name])
                for name in WEIGHTS:
                    require(pools[name][-1].shape == (BATCH, 2048)
                            and logits[name][-1].ndim == 2 and logits[name][-1].shape[0] == BATCH
                            and np.isfinite(pools[name][-1]).all() and np.isfinite(logits[name][-1]).all(),
                            'Invalid Inception features: ' + arm + '/' + name)
                save_npz(output/arm/f'statistics_{i}.npz', **new,
                         source_sample_sha256=np.asarray(inputs[str(paths['samples'])]),
                         source_statistics_sha256=np.asarray(inputs[str(paths['statistics'])]),
                         source_protocol_sha256=np.asarray(src['report']['protocol_sha256']),
                         checkpoint_id=np.asarray(IDS[arm]), checkpoint_step=np.asarray(STEP),
                         round_id=np.asarray(i), sampling_seed=np.asarray(seed),
                         num_samples=np.asarray(BATCH), num_samples_used=np.asarray(valid))
                if i == 0:
                    save_npz(output/arm/'preview.npz', **{k: quantize(a[:32]) for k, a in outputs.items()},
                             original_full=old_full[:32], weights=np.asarray(list(WEIGHTS.values())))
                arm_progress['completed_rounds'] = i + 1
                arm_progress['full_reconstruction_max_changed_fraction'] = max(r['changed_pixel_fraction'] for r in reproduction)
                save_progress()
                elapsed = time.monotonic() - arm_started
                eta = elapsed/(i+1)*(ROUNDS-i-1)
                print(f'{arm}: {i+1}/{ROUNDS}, used={min((i+1)*BATCH, COUNT)}/{COUNT}, '
                      f'full-check changed={changed:.3g} max={maximum}; elapsed={elapsed:.1f}s ETA={eta:.1f}s', flush=True)
                gc.collect()
            diag = {k: np.concatenate(v) for k, v in diagnostics.items()}
            save_npz(output/arm/'per_image_diagnostics.npz', **diag)
            for name, weight in WEIGHTS.items():
                p, l = np.concatenate(pools[name])[:COUNT], np.concatenate(logits[name])[:COUNT]
                metrics = dict(fid=float(tfgan.eval.frechet_classifier_distance_from_activations(reference, p)),
                               kid=float(tfgan.eval.kernel_classifier_distance_from_activations(
                                   tf.convert_to_tensor(reference), tf.convert_to_tensor(p)).numpy()),
                               IS=float(tfgan.eval.classifier_score_from_logits(l)))
                require(all(np.isfinite(v) for v in metrics.values()), 'Non-finite metric.')
                if name in ('w000', 'w100'):
                    variant = 'x_t' if name == 'w000' else 'tweedie_xt'
                    require(np.isclose(metrics['fid'], src['baseline'][variant]['fid'], atol=2e-3, rtol=1e-4),
                            f'{arm}/{name}: reused features do not reproduce original FID.')
                rms = float(np.sqrt(np.mean(diag['correction_rms']**2))) * weight
                row = dict(arm=arm, weight=weight, percent=int(weight*100), count=COUNT, sampling_seed=SEED,
                           **metrics, fid_minus_no_tweedie=metrics['fid'] - src['baseline']['x_t']['fid'],
                           correction_rms_data=rms, correction_rms_uint8_units=rms*255,
                           mean_outside_fraction=float(diag['outside_fraction_'+name].mean()),
                           mean_changed_uint8_fraction=float(diag['changed_fraction_'+name].mean()),
                           features='original' if name in ('w000', 'w100') else 'new')
                rows.append(row)
                print(f'{arm} Tweedie {weight:4.0%}: FID={metrics["fid"]:.6f}, '
                      f'delta_vs_0={row["fid_minus_no_tweedie"]:+.6f}, '
                      f'correction RMS={rms*255:.4f}/255, outside={row["mean_outside_fraction"]:.6%}', flush=True)
            arm_progress.update(status='complete', rows=[r for r in rows if r['arm'] == arm],
                                reconstruction_checks=reproduction,
                                full_correction_image_rms_quantiles_uint8_units={str(q): float(np.quantile(diag['correction_rms'], q)*255)
                                                                              for q in (0.5, 0.9, 0.99)},
                                full_correction_signed_mean_data=float(diag['correction_mean'].mean()))
            save_progress()
            del model, sde, score_fn, pools, logits, normalized, normalized_score, clean, p, l
            gc.collect()
            torch.cuda.empty_cache()
        buffer = io.StringIO()
        writer = csv.DictWriter(buffer, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
        atomic_write_bytes(str(output/'metrics.csv'), buffer.getvalue().encode())
        result['rows'] = rows
        result['fox_minus_vp_fid_by_weight'] = {name: next(r['fid'] for r in rows if r['arm']=='fox' and r['weight']==w)
                                              - next(r['fid'] for r in rows if r['arm']=='vp' and r['weight']==w)
                                              for name, w in WEIGHTS.items()}
        result['status'] = 'complete'
        save_progress()
        print('\nFinished partial Tweedie, VP and FOX. Summary:', output/'summary.json', flush=True)
        print('weight\tVP FID\tFOX FID\tFOX-VP', flush=True)
        for name, w in WEIGHTS.items():
            vp = next(r['fid'] for r in rows if r['arm']=='vp' and r['weight']==w)
            fox = next(r['fid'] for r in rows if r['arm']=='fox' and r['weight']==w)
            print(f'{w:.0%}\t{vp:.6f}\t{fox:.6f}\t{fox-vp:+.6f}', flush=True)
    except BaseException as error:
        result['status'] = 'failed'
        result['error'] = repr(error)
        save_progress()
        raise


if __name__ == '__main__':
    main()
PY_PARTIAL
