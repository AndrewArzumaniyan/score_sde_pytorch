"""Final historical VP/FOX fixed-budget ODE evaluation profile."""
import gc
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

from e3b_final50k import copy_runtime, digest, verify_runtime, write_json, REFERENCE, REFERENCE_SHA, VP, FOX

ROOT = Path('/workspace/score_sde_pytorch')
INPUT = Path('/inputs/repo')
OUTPUT = Path('/output')
PIN = Path('/opt/low-nfe')
VARIANTS = ('x_t', 'tweedie_xt')
METHODS = ('ode_dpmpp_2m', 'ode_heun')
ARMS = {'vp': VP, 'fox': FOX}
CELLS = [(256, count, arm, method) for count in (10000, 50000)
         for method in METHODS for arm in ('vp', 'fox')]


def get_config(name):
    import torch
    from configs.vp.celeba_ncsnpp_continuous import get_config as get_vp
    from configs.fox.celeba_ncsnpp_continuous_matern_3_2 import get_config as get_fox
    method, count, arm = name.rsplit('_', 2)
    count = int(count)
    if (256, count, arm, method) not in CELLS:
        raise ValueError('Unknown final ODE cell')
    cfg = get_vp() if arm == 'vp' else get_fox()
    if arm == 'fox':
        cfg.model.fox_u = -5.0
        cfg.model.fox_matern_length_scale = 346.4101615138
        cfg.model.fox_target_terminal_variance = 1.0
    cfg.device = torch.device('cuda:0')
    cfg.sampling.method = method
    cfg.sampling.probability_flow = True
    cfg.sampling.time_grid = 'uniform_logsnr'
    cfg.sampling.logsnr_min = -10.0
    cfg.sampling.logsnr_max = 10.575749165617202
    cfg.sampling.noise_removal = True
    cfg.sampling.predictor = 'none'
    cfg.sampling.corrector = 'none'
    cfg.eval.endpoint_denoise_variants = True
    cfg.eval.endpoint_denoise_outputs = VARIANTS
    cfg.eval.save_endpoint_states = True
    cfg.eval.sampling_num_scales = 256
    cfg.eval.sampling_seed = 0
    cfg.eval.num_samples = count
    cfg.eval.batch_size = 512
    cfg.eval.enable_loss = cfg.eval.enable_bpd = False
    cfg.eval.enable_sampling = True
    cfg.eval.begin_ckpt = cfg.eval.end_ckpt = 260
    cfg.eval.include_final_checkpoint = False
    cfg.allow_tf32 = cfg.cudnn_benchmark = True
    cfg.allow_fp16_reduced_precision_reduction = False
    return cfg


def snapshot(source, output):
    source, output = Path(source), Path(output)
    runtime = output / 'runtime'
    source_hash = copy_runtime(source, runtime)
    for name in ('logs', 'workdirs', 'celeba', 'assets/stats'):
        (runtime / name).mkdir(parents=True, exist_ok=True)
    for name in ('logs', 'workdirs'):
        (output / name).mkdir()
    if digest(INPUT / REFERENCE) != REFERENCE_SHA:
        raise RuntimeError('Unexpected CelebA reference')
    shutil.copy2(str(INPUT / REFERENCE), str(runtime / REFERENCE))
    inputs = {REFERENCE: REFERENCE_SHA}
    for arm in ARMS.values():
        relative = 'workdirs/{}/checkpoints/checkpoint_260.pth'.format(arm)
        inputs[relative] = digest(INPUT / relative)
        (output / 'workdirs' / arm / 'checkpoints').mkdir(parents=True)
    files = {str(p.relative_to(runtime)): digest(p) for p in sorted(runtime.rglob('*')) if p.is_file()}
    write_json(output / 'snapshot.json', dict(profile='final_ode', source_sha256=source_hash,
        runtime_files=files, inputs=inputs, cells=CELLS, reference=REFERENCE_SHA,
        sampling_seed=0, endpoint=[-10.0, 10.575749165617202], outputs=VARIANTS))


def check():
    saved = json.loads((PIN / 'snapshot.json').read_text())
    verify_runtime(ROOT, saved['source_sha256'], saved['runtime_files'], saved['inputs'])
    return saved


def preflight():
    import torch
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1 or os.environ.get('CUDA_VISIBLE_DEVICES') != '0':
        raise RuntimeError('Expected one Docker-routed GPU, CVD=0')
    from artifact_utils import build_model_manifest, _environment_manifest
    from models import utils as mutils, ncsnpp
    from models.ema import ExponentialMovingAverage
    from losses import get_optimizer
    from utils import restore_checkpoint
    from run_lib import get_sde
    from ode_sampling import get_vp_ode_sampler
    saved = check()
    identities = {}
    for arm, work in ARMS.items():
        cfg = get_config('ode_heun_10000_' + arm)
        model = mutils.create_model(cfg)
        ema = ExponentialMovingAverage(model.parameters(), decay=cfg.model.ema_rate)
        state = dict(model=model, optimizer=get_optimizer(cfg, model.parameters()), ema=ema, step=0,
                     model_protocol_sha256=build_model_manifest(cfg)['model_protocol_sha256'])
        state = restore_checkpoint(str(ROOT / 'workdirs' / work / 'checkpoints/checkpoint_260.pth'), state, cfg.device, restore_rng=False)
        if int(state['step']) != 1300002:
            raise RuntimeError('Unexpected checkpoint step: ' + arm)
        identities[arm] = state['checkpoint_id']
        ema.copy_to(model.parameters())
        sde, eps = get_sde(cfg, n_override=4)
        z = torch.zeros((2, 3, 64, 64), device=cfg.device)
        for method in METHODS:
            outputs, nfe = get_vp_ode_sampler(sde, z.shape, lambda x: x, method, eps, cfg.device)(model, z=z)
            if nfe != 5 or not all(torch.isfinite(x).all() for x in outputs.values()):
                raise RuntimeError('Failed real-weights smoke')
        del model, ema, state, z
        gc.collect()
        torch.cuda.empty_cache()
    write_json(OUTPUT / 'logs/preflight.json', dict(status='passed', checkpoint_ids=identities,
        source_sha256=saved['source_sha256'], environment=_environment_manifest(), started_unix=time.time()))


def folder(count, arm, method):
    return 'eval_final_ode_{}_{}_{}_seed0'.format(method, arm, count)


def run(n, count, arm, method):
    subprocess.run([sys.executable, '-u', 'main.py', '--mode=eval',
        '--config=celeba_final_ode_eval.py:{}_{}_{}'.format(method, count, arm),
        '--workdir=workdirs/' + ARMS[arm], '--eval_folder=' + folder(count, arm, method)], check=True)


def report(n, count, arm, method):
    import numpy as np
    from artifact_utils import _canonicalize
    directory = ROOT / 'workdirs' / ARMS[arm] / folder(count, arm, method)
    manifest = json.loads((directory / 'manifest.json').read_text())
    cfg = get_config('{}_{}_{}'.format(method, count, arm))
    for section in ('model', 'data', 'sampling', 'eval'):
        expected = _canonicalize(cfg[section])
        if section == 'eval':
            expected.pop('begin_ckpt', None)
            expected.pop('end_ckpt', None)
        if _canonicalize(manifest['config'][section]) != expected:
            raise RuntimeError('Unexpected report config: ' + section)
    pre = json.loads((OUTPUT / 'logs/preflight.json').read_text())
    if manifest['source_sha256'] != pre['source_sha256']:
        raise RuntimeError('Unexpected report source')
    with np.load(str(directory / 'report_260.npz')) as result:
        expected = dict(num_samples=count, nfe=257, checkpoint_step=1300002,
                        checkpoint_id=pre['checkpoint_ids'][arm], dataset_stats_id=REFERENCE_SHA)
        if any(result[k].item() != v for k, v in expected.items()):
            raise RuntimeError('Unexpected report identity')
        metrics = {metric + '_' + variant: float(result[metric + '_' + variant])
                   for metric in ('IS', 'fid', 'kid') for variant in VARIANTS}
        if not all(math.isfinite(x) for x in metrics.values()):
            raise RuntimeError('Non-finite metric')
    if len(list((directory / 'ckpt_260').glob('statistics_*.npz'))) != (count + 511) // 512:
        raise RuntimeError('Unexpected number of statistics shards')
    write_json(OUTPUT / 'logs/{}_{}_{}_result.json'.format(method, count, arm),
               dict(trajectory_nfe=n, nfe=n + 1, num_samples=count, arm=arm, method=method, **metrics))
    cells = [json.loads(p.read_text()) for p in sorted((OUTPUT / 'logs').glob('*_result.json'))]
    write_json(OUTPUT / 'summary.json', dict(status='completed' if len(cells) == len(CELLS) else 'running',
        completed_cells=len(cells), total_cells=len(CELLS), cells=cells))


if __name__ == '__main__':
    mode, args = sys.argv[1], sys.argv[2:]
    if mode == 'snapshot': snapshot(*args)
    elif mode == 'check': check()
    elif mode == 'preflight': preflight()
    elif mode == 'plan':
        for cell in CELLS: print(*cell)
    elif mode in ('run', 'report'):
        n, count, arm, method = args
        cell = (int(n), int(count), arm, method)
        if cell not in CELLS: raise ValueError('Unknown cell')
        (run if mode == 'run' else report)(*cell)
    else: raise ValueError(mode)
