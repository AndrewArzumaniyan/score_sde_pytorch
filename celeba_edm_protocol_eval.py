"""EDM-protocol CelebA queue: EDM ckpt-2, VP/FOX and controlled A/B at 35 calls.

The Docker/snapshot/cache shell is run_celeba_low_nfe_overnight.sh --edm-protocol.
EDM ckpt-2 uses its own Algorithm 2 sampler. Score models use ode_edm_heun:
Karras rho=7 nodes, 17 Heun steps and the final denoise (34 + 1 calls).
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

from e3b_final50k import copy_runtime, digest, verify_runtime, write_json, REFERENCE, REFERENCE_SHA, VP, FOX
from configs.factorial.celeba_common import LOGSNR_MIN, LOGSNR_MAX

ROOT = Path('/workspace/score_sde_pytorch')
INPUT_REPO = Path('/inputs/repo')
OUTPUT = Path('/output')
PIN = Path('/opt/low-nfe')
EDM_OFFICIAL = Path('/opt/edm')
VARIANTS = ('x_t', 'tweedie_xt')
EDM = 'celeba_edm_canonical_50k'
NFE = 35
WINDOWS = {
    'edm': (-2 * math.log(80.0), -2 * math.log(0.002)),
    'common': (-10.0, 10.575749165617202),
    'ab': (LOGSNR_MIN, LOGSNR_MAX),
}
STEPS = {2: 25001, 20: 100001, 260: 1300002, 40: 200001, 4: 200001}
# (num_samples, checkpoint, arm, window). The matched-images trio first, then
# complete A/B pairs, then the long final cells; order does not depend on FIDs.
CELLS = ([(10000, 2, 'edm', 'native'), (10000, 20, 'vp', 'edm'), (10000, 20, 'fox', 'edm')] +
         [(10000, 40 if seed == 42 else 4, arm + str(seed), 'ab')
          for seed in (42, 43, 44) for arm in ('A', 'B')] +
         [(50000, 260, arm, window) for window in ('edm', 'common') for arm in ('vp', 'fox')])


def model_name(arm):
    if arm == 'edm':
        return EDM
    if arm in ('vp', 'fox'):
        return VP if arm == 'vp' else FOX
    from celeba_ode_eval import model_name as ab_name
    return ab_name(int(arm[1:]), arm[0])


def origin(arm):
    if arm[1:] in ('43', '44'):
        return Path('/inputs/seed' + arm[1:])
    return INPUT_REPO


def cell_name(count, ckpt, arm, window):
    return '{}_{}_{}_{}'.format(window, count, ckpt, arm)


def get_config(name):
    import torch
    window, count, ckpt, arm = name.split('_')
    cell = (int(count), int(ckpt), arm, window)
    if cell not in CELLS:
        raise ValueError('Unknown EDM-protocol cell: ' + name)
    count, ckpt = cell[:2]
    if arm == 'edm':
        from configs.edm.celeba_canonical import get_config as get_edm
        cfg = get_edm()
        if cfg.sampling.method != 'edm' or cfg.sampling.edm_num_steps != 18:
            raise ValueError('Unexpected canonical EDM sampler.')
        cfg.device = torch.device('cuda:0')
        cfg.allow_tf32 = False  # As trained; canonical EDM disables TF32.
        cfg.cudnn_benchmark = True
    else:
        if arm in ('vp', 'fox'):
            from celeba_final_ode_eval import get_config as final_config
            cfg = final_config('ode_heun_10000_' + arm)
        else:
            from celeba_ode_eval import get_config as ab_config
            cfg = ab_config('ode_heun_128_{}_{}'.format(arm[1:], arm[0]))
        cfg.sampling.method = 'ode_edm_heun'
        cfg.sampling.time_grid = 'karras'
        cfg.sampling.logsnr_min, cfg.sampling.logsnr_max = WINDOWS[window]
        cfg.eval.sampling_num_scales = NFE - 1  # Trajectory calls; Tweedie adds one.
    cfg.eval.num_samples = count
    cfg.eval.batch_size = 512
    cfg.eval.sampling_seed = 0
    cfg.eval.enable_sampling = True
    cfg.eval.enable_loss = cfg.eval.enable_bpd = False
    cfg.eval.begin_ckpt = cfg.eval.end_ckpt = ckpt
    cfg.eval.include_final_checkpoint = False
    return cfg


def weights():
    """Distinct (arm, checkpoint) pairs in queue order."""
    seen = []
    for _, ckpt, arm, _ in CELLS:
        if (arm, ckpt) not in seen:
            seen.append((arm, ckpt))
    return seen


def checkpoint_path(arm, ckpt):
    return 'workdirs/{}/checkpoints/checkpoint_{}.pth'.format(model_name(arm), ckpt)


def official_sources(root):
    root = Path(root)
    extensions = ('.py', '.c', '.cc', '.cpp', '.cu', '.cuh', '.h')
    return {str(p.relative_to(root)): digest(p) for p in sorted(root.rglob('*'))
            if p.is_file() and p.suffix in extensions and '.git' not in p.relative_to(root).parts
            and '__pycache__' not in p.relative_to(root).parts}


def snapshot(source, output):
    source, output = Path(source), Path(output)
    runtime = output / 'runtime'
    source_hash = copy_runtime(source, runtime)
    if not (runtime / 'third_party/edm/training/networks.py').is_file():
        raise RuntimeError('The source checkout has no official EDM sources in third_party/edm.')
    for folder in ('logs', 'workdirs', 'celeba', 'assets/stats', 'training_inputs'):
        (runtime / folder).mkdir(parents=True, exist_ok=True)
    for folder in ('logs', 'workdirs'):
        (output / folder).mkdir()
    if digest(INPUT_REPO / REFERENCE) != REFERENCE_SHA:
        raise RuntimeError('Unexpected CelebA reference.')
    shutil.copy2(str(INPUT_REPO / REFERENCE), str(runtime / REFERENCE))
    inputs = {REFERENCE: REFERENCE_SHA}
    for arm, ckpt in weights():
        relative = checkpoint_path(arm, ckpt)
        inputs[relative] = digest(origin(arm) / relative)
        (output / 'workdirs' / model_name(arm) / 'checkpoints').mkdir(parents=True, exist_ok=True)
        if arm[0] in 'AB' and arm[1:].isdigit():
            shutil.copy2(str(origin(arm) / 'workdirs' / model_name(arm) / 'run_manifest.json'),
                         str(runtime / 'training_inputs/{}_{}.json'.format(arm[1:], arm[0])))
    write_json(output / 'snapshot.json', dict(profile='edm_protocol', source_sha256=source_hash,
        runtime_files={str(p.relative_to(runtime)): digest(p) for p in sorted(runtime.rglob('*')) if p.is_file()},
        inputs=inputs, cells=CELLS, windows=WINDOWS, checkpoint_steps=STEPS, nfe=NFE, karras_rho=7.0,
        sampling_seed=0, outputs=VARIANTS, edm_official_sources=official_sources(runtime / 'third_party/edm')))


def check():
    saved = json.loads((PIN / 'snapshot.json').read_text())
    verify_runtime(ROOT, saved['source_sha256'], saved['runtime_files'], saved['inputs'])
    return saved


def preflight():
    import torch
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1 or os.environ.get('CUDA_VISIBLE_DEVICES') != '0':
        raise RuntimeError('Expected exactly one Docker-routed GPU, CVD=0.')
    saved = check()
    # The adapter verifies the pinned commit through Git; the frozen runtime has
    # no .git, so the mounted checkout must hold byte-identical sources.
    if Path(os.environ.get('EDM_OFFICIAL_ROOT', '')) != EDM_OFFICIAL:
        raise RuntimeError('EDM_OFFICIAL_ROOT must point to the read-only official checkout.')
    if official_sources(EDM_OFFICIAL) != saved['edm_official_sources']:
        raise RuntimeError('Mounted official EDM sources differ from the frozen runtime copy.')
    from artifact_utils import build_model_manifest, _environment_manifest
    from models import utils as mutils, ncsnpp, edm_canonical  # noqa: F401 (registration)
    from models.ema import ExponentialMovingAverage
    from losses import get_optimizer
    from utils import restore_checkpoint
    from run_lib import get_sde
    from ode_sampling import get_vp_ode_sampler
    from sampling import get_edm_sampler
    identities = {}
    for arm, ckpt in weights():
        count, _, _, window = next(c for c in CELLS if c[1:3] == (ckpt, arm))
        cfg = get_config(cell_name(count, ckpt, arm, window))
        model = mutils.create_model(cfg)
        ema = ExponentialMovingAverage(model.parameters(), decay=cfg.model.ema_rate)
        state = dict(model=model, optimizer=get_optimizer(cfg, model.parameters()), ema=ema, step=0,
                     model_protocol_sha256=build_model_manifest(cfg)['model_protocol_sha256'])
        if arm[0] in 'AB' and arm[1:].isdigit():
            manifest = json.loads((ROOT / 'training_inputs/{}_{}.json'.format(arm[1:], arm[0])).read_text())
            state['training_protocol_sha256'] = manifest['training_protocol_sha256']
        state = restore_checkpoint(str(ROOT / checkpoint_path(arm, ckpt)), state, cfg.device, restore_rng=False)
        if int(state['step']) != STEPS[ckpt]:
            raise RuntimeError('Unexpected checkpoint step for {} ckpt-{}: {}'.format(arm, ckpt, state['step']))
        identities['{}_{}'.format(arm, ckpt)] = state['checkpoint_id']
        ema.copy_to(model.parameters())
        z = torch.zeros((2, 3, 64, 64), device=cfg.device)
        if arm == 'edm':
            sde, _ = get_sde(cfg, n_override=2)
            out, nfe = get_edm_sampler(sde, z.shape, lambda x: x, device=cfg.device)(model)
            outputs, expected = [out], 3
        else:
            sde, eps = get_sde(cfg, n_override=4)
            result, nfe = get_vp_ode_sampler(sde, z.shape, lambda x: x, 'ode_edm_heun', eps, cfg.device)(model, z=z)
            outputs, expected = list(result.values()), 5
        if nfe != expected or not all(torch.isfinite(x).all() for x in outputs):
            raise RuntimeError('Failed real-weights smoke: {} ckpt-{}'.format(arm, ckpt))
        del model, ema, state, z, outputs
        gc.collect()
        torch.cuda.empty_cache()
    write_json(OUTPUT / 'logs/preflight.json', dict(status='passed', checkpoint_ids=identities,
        source_sha256=saved['source_sha256'], environment=_environment_manifest(), started_unix=time.time()))
    print('EDM-protocol preflight passed: {} checkpoints, official EDM parity, finite smokes.'.format(
        len(identities)), flush=True)


def folder(count, ckpt, arm, window):
    return 'eval_edm_protocol_{}_nfe{}_{}samples_ckpt{}_seed0'.format(window, NFE, count, ckpt)


def run(count, ckpt, arm, window):
    subprocess.run([sys.executable, '-u', 'main.py', '--mode=eval',
        '--config=celeba_edm_protocol_eval.py:' + cell_name(count, ckpt, arm, window),
        '--workdir=workdirs/' + model_name(arm), '--eval_folder=' + folder(count, ckpt, arm, window)], check=True)


def report(count, ckpt, arm, window):
    import numpy as np
    from artifact_utils import _canonicalize
    directory = ROOT / 'workdirs' / model_name(arm) / folder(count, ckpt, arm, window)
    manifest = json.loads((directory / 'manifest.json').read_text())
    cfg = get_config(cell_name(count, ckpt, arm, window))
    for section in ('model', 'data', 'sampling', 'eval'):
        expected = _canonicalize(cfg[section])
        if section == 'eval':
            expected.pop('begin_ckpt', None)
            expected.pop('end_ckpt', None)
        if _canonicalize(manifest['config'][section]) != expected:
            raise RuntimeError('Unexpected report config: ' + section)
    pre = json.loads((OUTPUT / 'logs/preflight.json').read_text())
    if manifest['source_sha256'] != pre['source_sha256']:
        raise RuntimeError('Report uses different sources.')
    keys = ('IS', 'fid', 'kid') if arm == 'edm' else tuple(
        m + '_' + v for m in ('IS', 'fid', 'kid') for v in VARIANTS)
    with np.load(str(directory / 'report_{}.npz'.format(ckpt))) as result:
        expected = dict(num_samples=count, checkpoint_step=STEPS[ckpt],
                        checkpoint_id=pre['checkpoint_ids']['{}_{}'.format(arm, ckpt)],
                        dataset_stats_id=REFERENCE_SHA)
        if arm != 'edm':
            expected['nfe'] = NFE
        if any(result[k].item() != v for k, v in expected.items()):
            raise RuntimeError('Unexpected report identity.')
        metrics = {k: float(result[k]) for k in keys}
    if not all(math.isfinite(x) for x in metrics.values()):
        raise RuntimeError('Non-finite metric.')
    shards = directory / 'ckpt_{}'.format(ckpt)
    if arm == 'edm':
        # The canonical EDM report has no NFE field; samples record the actual calls.
        with np.load(str(shards / 'samples_0.npz')) as samples:
            if int(samples['nfe'].item()) != NFE:
                raise RuntimeError('Unexpected EDM sampler NFE.')
    if len(list(shards.glob('statistics_*.npz'))) != (count + 511) // 512:
        raise RuntimeError('Unexpected number of statistics shards.')
    primary = metrics['fid'] if arm == 'edm' else metrics['fid_tweedie_xt']
    write_json(OUTPUT / 'logs/{}_result.json'.format(cell_name(count, ckpt, arm, window)),
               dict(arm=arm, checkpoint=ckpt, checkpoint_step=STEPS[ckpt], window=window,
                    logsnr=None if arm == 'edm' else list(WINDOWS[window]), num_samples=count, nfe=NFE,
                    sampler='edm_algorithm2' if arm == 'edm' else 'ode_edm_heun',
                    primary_output='edm_denoised' if arm == 'edm' else 'tweedie_xt',
                    primary_fid=primary, **metrics))
    cells = [json.loads(p.read_text()) for p in sorted((OUTPUT / 'logs').glob('*_result.json'))]
    write_json(OUTPUT / 'summary.json', dict(status='completed' if len(cells) == len(CELLS) else 'running',
        completed_cells=len(cells), total_cells=len(CELLS), cells=cells, confidence_intervals_computed=False))
    print('EDM protocol accepted {}/{}: {} ckpt-{} window={} n={} FID={:.6f}'.format(
        len(cells), len(CELLS), arm, ckpt, window, count, primary), flush=True)


if __name__ == '__main__':
    mode, args = sys.argv[1], sys.argv[2:]
    if mode == 'snapshot': snapshot(*args)
    elif mode == 'check': check()
    elif mode == 'preflight': preflight()
    elif mode == 'plan':
        for cell in CELLS: print(*cell)
    elif mode in ('run', 'report'):
        count, ckpt, arm, window = args
        cell = (int(count), int(ckpt), arm, window)
        if cell not in CELLS: raise ValueError('Unknown cell')
        (run if mode == 'run' else report)(*cell)
    else: raise ValueError(mode)
