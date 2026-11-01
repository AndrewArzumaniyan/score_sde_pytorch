"""Controlled A/B ODE queue profile; metrics and weight loading stay in run_lib.

The Docker/snapshot/cache shell is run_celeba_low_nfe_overnight.sh --ode.
This module provides its plan, configs, input validation and report intake.
"""
import gc
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

from e3b_final50k import copy_runtime, digest, verify_runtime, write_json, REFERENCE, REFERENCE_SHA
from configs.factorial.celeba_common import LOGSNR_MIN, LOGSNR_MAX

ROOT = Path('/workspace/score_sde_pytorch')
OUTPUT = Path('/output')
PIN = Path('/opt/low-nfe')
INPUT_REPO = Path('/inputs/repo')
VARIANTS = ('x_t', 'tweedie_xt')
METHODS = ('ode_dpmpp_2m', 'ode_heun')
# Complete matched A/B pairs first; no selection based on the observed FID sign.
CELLS = [(n, seed, arm, method) for n in (128, 256) for seed in (42, 43, 44)
         for method in METHODS for arm in ('A', 'B')]


def model_name(seed, arm):
    label = 'A_vp_pvp' if arm == 'A' else 'B_vp_pfox'
    if seed == 42:
        return 'celeba_factorial_{}_50k'.format(label)
    return 'celeba_ab_replication_{}_seed{}_200k'.format(label, seed)


def checkpoint(seed):
    return 40 if seed == 42 else 4


def training_root(seed):
    return INPUT_REPO if seed == 42 else Path('/inputs/seed{}'.format(seed))


def get_config(name):
    from ml_collections import ConfigDict
    from configs.factorial.celeba_a_vp_pvp import get_config as get_a
    from configs.factorial.celeba_b_vp_pfox import get_config as get_b
    method, budget, seed, arm = name.rsplit('_', 3)
    budget, seed = int(budget), int(seed)
    if (budget, seed, arm, method) not in CELLS:
        raise ValueError('Unknown ODE cell: ' + name)
    manifest = json.loads((ROOT / 'training_inputs/{}_{}.json'.format(seed, arm)).read_text())
    cfg = ConfigDict(manifest['config'])
    cfg.eval = (get_a if arm == 'A' else get_b)().eval
    cfg.device = 'cuda:0'
    cfg.sampling.method = method
    cfg.sampling.probability_flow = True
    cfg.sampling.time_grid = 'uniform_logsnr'
    cfg.sampling.logsnr_min = LOGSNR_MIN
    cfg.sampling.logsnr_max = LOGSNR_MAX
    cfg.sampling.noise_removal = True
    cfg.sampling.predictor = 'none'  # The ODE method owns its stages; no PC predictor.
    cfg.sampling.corrector = 'none'
    cfg.eval.endpoint_denoise_variants = True
    cfg.eval.endpoint_denoise_outputs = VARIANTS
    cfg.eval.save_endpoint_states = True
    cfg.eval.sampling_num_scales = budget  # Trajectory calls, not Heun steps.
    cfg.eval.sampling_seed = 0
    cfg.eval.num_samples = 10000
    cfg.eval.batch_size = 512
    cfg.eval.enable_loss = False
    cfg.eval.enable_bpd = False
    cfg.eval.enable_sampling = True
    cfg.eval.begin_ckpt = cfg.eval.end_ckpt = checkpoint(seed)
    cfg.eval.include_final_checkpoint = False
    cfg.allow_tf32 = True
    cfg.cudnn_benchmark = True
    cfg.allow_fp16_reduced_precision_reduction = False
    return cfg


def snapshot(source, output):
    source, output = Path(source), Path(output)
    runtime = output / 'runtime'
    source_hash = copy_runtime(source, runtime)
    for folder in ('logs', 'workdirs', 'celeba', 'assets/stats', 'training_inputs'):
        (runtime / folder).mkdir(parents=True, exist_ok=True)
    for folder in ('logs', 'workdirs'):
        (output / folder).mkdir()
    ref = INPUT_REPO / REFERENCE
    if digest(ref) != REFERENCE_SHA:
        raise RuntimeError('Unexpected CelebA reference.')
    shutil.copy2(str(ref), str(runtime / REFERENCE))
    inputs = {REFERENCE: REFERENCE_SHA}
    for seed in (42, 43, 44):
        for arm in ('A', 'B'):
            relative = 'workdirs/{}/checkpoints/checkpoint_{}.pth'.format(model_name(seed, arm), checkpoint(seed))
            inputs[relative] = digest(training_root(seed) / relative)
            origin = training_root(seed) / 'workdirs' / model_name(seed, arm) / 'run_manifest.json'
            target = runtime / 'training_inputs/{}_{}.json'.format(seed, arm)
            shutil.copy2(str(origin), str(target))
            (output / 'workdirs' / model_name(seed, arm) / 'checkpoints').mkdir(parents=True)
    write_json(output / 'snapshot.json', dict(profile='controlled_ode', source_sha256=source_hash,
        runtime_files={str(p.relative_to(runtime)): digest(p) for p in sorted(runtime.rglob('*')) if p.is_file()},
        inputs=inputs, cells=CELLS, num_samples=10000, batch_size=512, sampling_seed=0,
        logsnr_min=LOGSNR_MIN, logsnr_max=LOGSNR_MAX, outputs=VARIANTS,
        extra_endpoint_calls=1, dynamic_thresholding=False,
        implicit_denoise_to_zero=False, explicit_tweedie=True))


def check():
    saved = json.loads((PIN / 'snapshot.json').read_text())
    verify_runtime(ROOT, saved['source_sha256'], saved['runtime_files'], saved['inputs'])
    return saved


def preflight():
    import torch
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1 or os.environ.get('CUDA_VISIBLE_DEVICES') != '0':
        raise RuntimeError('Expected exactly one Docker-routed GPU, CVD=0.')
    torch.cuda.init()
    from artifact_utils import build_model_manifest, _environment_manifest
    saved = check()
    identities = {}
    env = _environment_manifest()
    for seed in (42, 43, 44):
        pair = []
        for arm in ('A', 'B'):
            cfg = get_config('ode_heun_128_{}_{}'.format(seed, arm))
            if (cfg.seed != seed or cfg.training.sde != 'vpsde' or
                cfg.model.noise_conditioning != 'logsnr' or
                cfg.model.logsnr_min != LOGSNR_MIN or cfg.model.logsnr_max != LOGSNR_MAX or
                cfg.data.dataset != 'CELEBA' or cfg.data.image_size != 64 or not cfg.data.centered or
                cfg.training.noise_distribution != ('vp_time' if arm == 'A' else 'normalized_fox_time')):
                raise RuntimeError('Unexpected controlled training protocol.')
            manifest = json.loads((ROOT / 'training_inputs/{}_{}.json'.format(seed, arm)).read_text())
            path = ROOT / 'workdirs' / model_name(seed, arm) / 'checkpoints/checkpoint_{}.pth'.format(checkpoint(seed))
            # Use the same strict loader and EMA implementation as the real evaluator.
            from models import utils as mutils, ncsnpp
            from models.ema import ExponentialMovingAverage
            from losses import get_optimizer
            from utils import restore_checkpoint
            model = mutils.create_model(cfg)
            ema = ExponentialMovingAverage(model.parameters(), decay=cfg.model.ema_rate)
            state = dict(model=model, optimizer=get_optimizer(cfg, model.parameters()), ema=ema,
                         step=0, model_protocol_sha256=build_model_manifest(cfg)['model_protocol_sha256'],
                         training_protocol_sha256=manifest['training_protocol_sha256'])
            state = restore_checkpoint(str(path), state, cfg.device, restore_rng=False)
            if int(state['step']) != 200001:
                raise RuntimeError('Expected 200k checkpoint: ' + str(path))
            identities['{}_{}'.format(seed, arm)] = state['checkpoint_id']
            ema.copy_to(model.parameters())
            from run_lib import get_sde
            sde, eps = get_sde(cfg, n_override=4)
            from ode_sampling import get_vp_ode_sampler
            # Same initial states for both smoke solvers, isolated from real evaluation RNG.
            z = torch.zeros((2, 3, 64, 64), device=cfg.device)
            for method in METHODS:
                get_vp_ode_sampler(sde, z.shape, lambda x: x, method, eps, cfg.device)(model, z=z)
            pair.append(dict(model=cfg.model.to_dict(), data=cfg.data.to_dict()))
            del state, model, ema, z
            gc.collect()
            torch.cuda.empty_cache()
        if pair[0] != pair[1]:
            raise RuntimeError('A/B model or data configurations differ within a training pair.')
    write_json(OUTPUT / 'logs/preflight.json', dict(status='passed', checkpoint_ids=identities,
        source_sha256=saved['source_sha256'], environment=env, started_unix=time.time()))
    print('ODE preflight passed: six checkpoint/config/EMA loads and 12 finite solver smokes.', flush=True)


def eval_folder(n, seed, arm, method):
    return 'eval_controlled_ode_{}_nfe{}_10k_train{}_{}_seed0'.format(method, n, seed, arm)


def run(n, seed, arm, method):
    subprocess.run([sys.executable, '-u', 'main.py', '--mode=eval',
        '--config=celeba_ode_eval.py:{}_{}_{}_{}'.format(method, n, seed, arm),
        '--workdir=workdirs/' + model_name(seed, arm),
        '--eval_folder=' + eval_folder(n, seed, arm, method)], check=True)


def report(n, seed, arm, method):
    import numpy as np
    cfg = get_config('{}_{}_{}_{}'.format(method, n, seed, arm))
    directory = ROOT / 'workdirs' / model_name(seed, arm) / eval_folder(n, seed, arm, method)
    manifest = json.loads((directory / 'manifest.json').read_text())
    actual = manifest['config']
    # Compare the complete scientifically relevant protocol, not just output filenames.
    for section in ('model', 'data', 'sampling', 'eval'):
        from artifact_utils import _canonicalize
        expected = _canonicalize(cfg[section])
        if section == 'eval':
            expected.pop('begin_ckpt', None)
            expected.pop('end_ckpt', None)
        if _canonicalize(actual[section]) != expected:
            raise RuntimeError('Unexpected ODE report config: ' + section)
    preflight_data = json.loads((OUTPUT / 'logs/preflight.json').read_text())
    if manifest['source_sha256'] != preflight_data['source_sha256']:
        raise RuntimeError('ODE report uses different sources.')
    identity = preflight_data['checkpoint_ids']['{}_{}'.format(seed, arm)]
    with np.load(str(directory / 'report_{}.npz'.format(checkpoint(seed)))) as result:
        for key, value in dict(num_samples=10000, nfe=n + 1, checkpoint_step=200001,
                               checkpoint_id=identity, dataset_stats_id=REFERENCE_SHA).items():
            if result[key].item() != value:
                raise RuntimeError('Unexpected ODE report identity: ' + key)
        metrics = {metric + '_' + variant: float(result[metric + '_' + variant])
                   for metric in ('IS', 'fid', 'kid') for variant in VARIANTS}
        if not all(math.isfinite(x) for x in metrics.values()):
            raise RuntimeError('Non-finite ODE metric.')
    if len(list((directory / 'ckpt_{}'.format(checkpoint(seed))).glob('statistics_*.npz'))) != 20:
        raise RuntimeError('Expected 20 statistics shards.')
    write_json(OUTPUT / 'logs/{}_{}_{}_{}_result.json'.format(method, n, seed, arm),
        dict(nfe=n + 1, trajectory_nfe=n, training_seed=seed, sampling_seed=0, arm=arm, method=method, **metrics))
    cells = [json.loads(p.read_text()) for p in sorted((OUTPUT / 'logs').glob('*_result.json'))]
    contrasts = []
    for budget, training_seed, _, solver in CELLS[::2]:
        pair = {c['arm']: c for c in cells if
                (c['trajectory_nfe'], c['training_seed'], c['method']) == (budget, training_seed, solver)}
        if set(pair) == {'A', 'B'}:
            contrasts.append(dict(trajectory_nfe=budget, training_seed=training_seed, method=solver,
                B_minus_A=pair['B']['fid_tweedie_xt'] - pair['A']['fid_tweedie_xt']))
    summary = dict(status='completed' if len(cells) == 24 else 'running', completed_cells=len(cells),
                   total_cells=24, cells=cells, point_contrasts=contrasts, confidence_intervals_computed=False)
    write_json(OUTPUT / 'summary.json', summary)
    print('ODE accepted {}/24: {} training_seed={} arm={} NFE={} FID={:.6f}'.format(
        len(cells), method, seed, arm, n + 1, metrics['fid_tweedie_xt']), flush=True)


if __name__ == '__main__':
    mode, args = sys.argv[1], sys.argv[2:]
    if mode == 'snapshot':
        snapshot(*args)
    elif mode == 'check':
        check()
    elif mode == 'preflight':
        preflight()
    elif mode == 'plan':
        for cell in CELLS:
            print(*cell)
    elif mode in ('run', 'report'):
        n, seed, arm, method = args
        cell = (int(n), int(seed), arm, method)
        if cell not in CELLS:
            raise ValueError('Unknown cell.')
        (run if mode == 'run' else report)(*cell)
    else:
        raise ValueError(mode)
