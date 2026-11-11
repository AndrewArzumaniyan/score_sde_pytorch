"""AFHQv2-64: FOX, A and VP over training steps, the screen behind AFHQV2_STEP_RULE.md.

Stage 0 protocol of CelebA: DPM++ 2M, 64 trajectory calls plus Tweedie, 10k images, sampling
seed 0, window [-10; 9.115429865459795] for every model, since A was trained on that
support. Reference statistics afhqv2_64_cd017afab009, all 15803 images.

Twelve cells in a fixed order: equal steps first (200k for FOX, A and VP), then 100k, then
FOX at 400k and 600k, then FOX, A and VP at 200k with sampling seed 1, then FOX at 800k. The
seed 1 cells only measure the noise of the sampling seed; they take no part in the choice
of step. Each cell waits for its snapshot; the training writes snapshots
by rename, so an existing file is complete. FOX at 800k arrives a day after the others,
so the queue waits for it on the same GPU without holding the card's memory.

There is no reference value in this protocol for AFHQv2, so there is no control cell.
Preflight checks the statistics file, the step and the training protocol of every model,
the grid and a short real run. One training run per model, one sampling seed, 10k
images: a screen without intervals. The choice of step follows AFHQV2_STEP_RULE.md.

Run through run_afhqv2_curve_rtx.sh from a separate worktree.
"""
import csv
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time

import afhqv2_rtx_train as afhq
import celeba_50k_lo_eval as lo
import celeba_followup_eval as followup
import celeba_stage0_eval as stage0
from celeba_stage0_eval import LOGSNR_MIN, WINDOWS, set_protocol, write_json

ROOT = Path(__file__).resolve().parent
LOGS = ROOT / 'logs'
RUNS = ROOT / 'runs'  # The training workdirs, mounted read-only as runs/<arm>.
WINDOW = 'lo'
CALLS = stage0.BUDGET
COUNT = stage0.COUNT
ATTEMPTS = 2
SNAPSHOT_FREQ = afhq.SNAPSHOT_FREQ
POLL_SECONDS = int(os.environ.get('AFHQ_CURVE_POLL_SECONDS', '60'))
WAIT_SECONDS = int(os.environ.get('AFHQ_CURVE_WAIT_SECONDS', str(72 * 3600)))  # For the whole queue.
LABELS = dict(fox='FOX', a='A', vp='VP')
# The order is fixed in advance and does not depend on any observed FID.
ORDER = (('fox', 4, 0), ('a', 4, 0), ('vp', 4, 0), ('fox', 2, 0), ('a', 2, 0), ('vp', 2, 0), ('fox', 8, 0),
         ('fox', 12, 0), ('fox', 4, 1), ('a', 4, 1), ('vp', 4, 1), ('fox', 16, 0))


def work(arm):
    return Path(afhq.WORK[arm]).name


def cell_name(arm, ckpt, seed=0):
    return '{}_c{}_lo'.format(arm, ckpt) + ('_seed{}'.format(seed) if seed else '')


CELLS = {cell_name(arm, ckpt, seed): dict(arm=arm, model=LABELS[arm], ckpt=ckpt, step=SNAPSHOT_FREQ * ckpt + 1,
                                          window=WINDOW, work=work(arm), seed=seed)
         for arm, ckpt, seed in ORDER}
PAIRS = (('fox', 'a'), ('fox', 'vp'), ('a', 'vp'))
TIE = 0.2  # AFHQV2_STEP_RULE.md, point 2.


def select(fids):
    """{step: FID} of one model: the latest step within TIE of the minimum."""
    best = min(fids.values())
    return max(step for step, fid in fids.items() if fid <= best + TIE + 1e-9)  # The rule includes equality.


def selection(cells):
    """Point 2 of the rule, once every candidate of a model is in."""
    chosen = {}
    for arm in LABELS:
        names = [name for name, spec in CELLS.items() if spec['arm'] == arm and spec['seed'] == 0]
        if all(name in cells for name in names):
            fids = {CELLS[name]['step'] - 1: cells[name]['fid_tweedie_xt'] for name in names}
            step = select(fids)
            chosen[LABELS[arm]] = dict(step=step, fid=fids[step])
    result = dict(rule='latest candidate with FID <= minimum + {}'.format(TIE), selected=chosen)
    if len(chosen) == 3:
        baseline = min(('A', 'VP'), key=lambda label: chosen[label]['fid'])
        result.update(baseline=baseline, fox_minus_best_baseline=chosen['FOX']['fid'] - chosen[baseline]['fid'])
    return result


def manifest(arm):
    return json.loads((RUNS / arm / 'run_manifest.json').read_text())


def get_config(name):
    from ml_collections import ConfigDict
    if name not in CELLS:
        raise ValueError('Unknown AFHQv2 curve cell: ' + name)
    spec = CELLS[name]
    config = ConfigDict(manifest(spec['arm'])['config'])
    if (config.seed, config.data.dataset, config.data.centered) != (afhq.SEED, 'AFHQV2', True):
        raise ValueError('Unexpected run behind ' + spec['model'])
    config.eval = afhq.base_config(spec['arm']).eval  # Manifests exclude evaluation settings.
    set_protocol(config, spec['window'], spec['ckpt'])
    config.eval.sampling_seed = spec['seed']
    return config


def checkpoint_path(spec):
    return ROOT / 'workdirs' / spec['work'] / 'checkpoints/checkpoint_{}.pth'.format(spec['ckpt'])


def check_identity(arm, step, expected_step, training_protocol):
    if step != expected_step:
        raise RuntimeError('Expected step {}, found {}'.format(expected_step, step))
    if training_protocol != manifest(arm)['training_protocol_sha256']:
        raise RuntimeError('The checkpoint does not belong to the run manifest of ' + arm)


def load_model(config, spec):
    """stage0.load_model assumes snapshots every 5k; these come every 50k."""
    from artifact_utils import build_model_manifest
    from losses import get_optimizer
    from models import utils as mutils, ncsnpp  # noqa: F401, registers the model.
    from models.ema import ExponentialMovingAverage
    from utils import restore_checkpoint
    model = mutils.create_model(config)
    ema = ExponentialMovingAverage(model.parameters(), decay=config.model.ema_rate)
    state = dict(model=model, optimizer=get_optimizer(config, model.parameters()), ema=ema, step=0,
                 model_protocol_sha256=build_model_manifest(config)['model_protocol_sha256'])
    path = checkpoint_path(spec)
    if not path.is_file():
        raise RuntimeError('Missing checkpoint: ' + str(path))
    state = restore_checkpoint(str(path), state, config.device, restore_rng=False)
    check_identity(spec['arm'], int(state['step']), spec['step'], state['training_protocol_sha256'])
    ema.copy_to(model.parameters())
    model.eval()
    return model, dict(checkpoint_id=state['checkpoint_id'], step=int(state['step']))


def preflight():
    """Statistics, then each model at its first cell in the order: grid, weights, short run."""
    import torch
    import evaluation
    from run_lib import get_sde
    path = LOGS / 'preflight.json'
    if path.exists():
        path.unlink()
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError('Expected one Docker-routed GPU.')
    arch = '{}.{}'.format(*torch.cuda.get_device_capability(0))
    if arch not in os.environ.get('TORCH_CUDA_ARCH_LIST', '').replace(';', ' ').split():
        raise RuntimeError('TORCH_CUDA_ARCH_LIST does not include sm ' + arch)
    lo.check_inception()
    from artifact_utils import _environment_manifest, _source_fingerprint
    first = {}
    for name, spec in CELLS.items():
        first.setdefault(spec['arm'], name)
    config = get_config(first['fox'])
    afhq.check_stats(config)
    stats_id = evaluation.dataset_stats_fingerprint(config)
    identities, failures = {}, {}
    for arm, name in first.items():
        spec = CELLS[name]
        try:
            config = get_config(name)
            sde, eps = get_sde(config, n_override=CALLS)
            stage0.check_grid(config, sde, eps)
            model, identities[name] = load_model(config, spec)
            followup.smoke(config, model, stage0.METHOD, eps)
            del model
            torch.cuda.empty_cache()
            print('Preflight', name, identities[name], flush=True)
        except Exception as error:  # A truncated checkpoint raises whatever torch.load does.
            failures[arm] = repr(error)
            print('Preflight failed: {}: {!r}'.format(name, error), flush=True)
    write_json(path, dict(status='failed' if failures else 'passed', checkpoints=identities, failures=failures,
        dataset_stats_id=stats_id, source_sha256=_source_fingerprint(str(ROOT)), environment=_environment_manifest(),
        started_unix=time.time()))
    print('Preflight', 'failed: {}'.format(failures) if failures else 'passed', flush=True)
    return 1 if failures else 0


def eval_folder(name):
    return 'eval_afhqv2_curve_' + name


def result_path(name):
    return LOGS / (name + '_result.json')


def failure_path(name):
    return LOGS / (name + '_failed.json')


def run(name):
    began, clock = time.time(), time.monotonic()
    result = subprocess.run([sys.executable, '-u', 'main.py', '--mode=eval', '--config=afhqv2_curve_eval.py:' + name,
                             '--workdir=workdirs/' + CELLS[name]['work'], '--eval_folder=' + eval_folder(name)],
                            cwd=str(ROOT))
    write_json(LOGS / (name + '_timing.json'), dict(started_unix=began, finished_unix=time.time(),
        elapsed_seconds=time.monotonic() - clock, returncode=result.returncode))
    result.check_returncode()


def report(name):
    import numpy as np
    from artifact_utils import _canonicalize
    spec = CELLS[name]
    config = get_config(name)
    directory = ROOT / 'workdirs' / spec['work'] / eval_folder(name)
    saved = json.loads((directory / 'manifest.json').read_text())
    for section in ('model', 'data', 'sampling', 'eval'):
        expected = _canonicalize(config[section])
        if section == 'eval':
            expected.pop('begin_ckpt', None)
            expected.pop('end_ckpt', None)
        if _canonicalize(saved['config'][section]) != expected:
            raise RuntimeError('Unexpected report config: ' + section)
    pre = json.loads((LOGS / 'preflight.json').read_text())
    nfe = CALLS + 1
    with np.load(str(directory / 'report_{}.npz'.format(spec['ckpt'])), allow_pickle=False) as data:
        expected = dict(num_samples=COUNT, nfe=nfe, checkpoint_step=spec['step'], dataset_stats_id=pre['dataset_stats_id'],
                        training_protocol_sha256=manifest(spec['arm'])['training_protocol_sha256'])
        for key, value in expected.items():
            if data[key].item() != value:
                raise RuntimeError('Unexpected report identity: ' + key)
        checkpoint_id = str(data['checkpoint_id'].item())
        metrics = {metric + '_' + variant: float(data[metric + '_' + variant])
                   for metric in ('IS', 'fid', 'kid') for variant in stage0.VARIANTS}
    if not all(math.isfinite(value) for value in metrics.values()):
        raise RuntimeError('Non-finite metric.')
    timing = json.loads((LOGS / (name + '_timing.json')).read_text())
    write_json(result_path(name), dict(spec, cell=name, checkpoint_id=checkpoint_id,
        lambda_window=[LOGSNR_MIN, WINDOWS[WINDOW]], nfe=nfe, num_samples=COUNT, sampling_seed=spec['seed'],
        elapsed_seconds=timing['elapsed_seconds'], **metrics))
    write_summary()
    print('AFHQv2 {}: FID after Tweedie {:.4f}, {:.1f} min'.format(
        name, metrics['fid_tweedie_xt'], timing['elapsed_seconds'] / 60), flush=True)


def write_summary(failed=()):
    cells = {name: json.loads(result_path(name).read_text()) for name in CELLS if result_path(name).exists()}
    point = []
    for ckpt in (2, 4):
        for left, right in PAIRS:
            a, b = cell_name(left, ckpt), cell_name(right, ckpt)
            if a in cells and b in cells:
                point.append(dict(contrast='{}_minus_{}_{}k'.format(LABELS[left], LABELS[right], ckpt * 50), left=a,
                                  right=b, value=cells[a]['fid_tweedie_xt'] - cells[b]['fid_tweedie_xt']))
    noise = {}
    for arm in LABELS:
        first, second = cell_name(arm, 4), cell_name(arm, 4, 1)
        if first in cells and second in cells:
            value = cells[second]['fid_tweedie_xt'] - cells[first]['fid_tweedie_xt']
            noise[LABELS[arm] + '_200k'] = dict(seed1_minus_seed0=value, absolute=abs(value))
    status = 'completed' if len(cells) == len(CELLS) else 'failed' if failed else 'running'
    write_json(LOGS / 'summary.json', dict(sampling_seed_noise=noise, status=status, completed_cells=len(cells), total_cells=len(CELLS),
        failed_cells=list(failed), cells=list(cells.values()), equal_step_contrasts=point,
        step_selection=selection(cells),
        rule='AFHQV2_STEP_RULE.md', scope='One training run per model, one sampling seed at 10k; no intervals'))
    with (LOGS / 'summary.csv').open('w', newline='') as stream:
        writer = csv.writer(stream)
        writer.writerow(('cell', 'model', 'step', 'sampling_seed', 'lambda_max', 'fid_tweedie', 'fid_x_t', 'kid_tweedie',
                         'is_tweedie'))
        for name, value in cells.items():
            writer.writerow((name, value['model'], value['step'], value['sampling_seed'], value['lambda_window'][1],
                             value['fid_tweedie_xt'],
                             value['fid_x_t'], value['kid_tweedie_xt'], value['IS_tweedie_xt']))


def wait_for(spec, deadline):
    """True once the snapshot exists; False when the queue runs out of time."""
    path = checkpoint_path(spec)
    printed = False
    while not path.is_file():
        if time.monotonic() > deadline:
            return False
        if not printed:
            print('Waiting for ' + str(path), flush=True)
            printed = True
        time.sleep(POLL_SECONDS)
    return True


def evaluate(name):
    for attempt in range(1, ATTEMPTS + 1):
        try:
            run(name)
            report(name)
            if failure_path(name).exists():
                failure_path(name).unlink()
            return True
        except (subprocess.CalledProcessError, RuntimeError, OSError, KeyError, ValueError) as error:
            write_json(failure_path(name), dict(error=repr(error), attempt=attempt, time=time.time()))
            print('AFHQv2 cell failed: {}: {!r}'.format(name, error), flush=True)
    return False


def queue():
    """Each cell as its snapshot appears. A failed cell does not stop the next; a missing one stops the rest."""
    pre = LOGS / 'preflight.json'
    if not pre.exists():
        print('No preflight.json: the queue stops.', flush=True)
        return 1
    blocked = json.loads(pre.read_text())['failures']
    deadline = time.monotonic() + WAIT_SECONDS
    failed = []
    names = list(CELLS)
    for index, name in enumerate(names):
        spec = CELLS[name]
        if result_path(name).exists():
            continue
        if spec['arm'] in blocked:
            failed.append(name)
            continue
        if not wait_for(spec, deadline):
            failed += [rest for rest in names[index:] if not result_path(rest).exists()]
            print('No snapshot before the deadline: ' + name, flush=True)
            break
        if not evaluate(name):
            failed.append(name)
    write_summary(failed)
    print('AFHQv2 curve finished, failed cells: {}'.format(failed or 'none'), flush=True)
    return 1 if failed else 0


if __name__ == '__main__':
    mode = sys.argv[1] if len(sys.argv) > 1 else ''
    if mode == 'preflight':
        sys.exit(preflight())
    elif mode == 'queue':
        sys.exit(queue())
    elif mode == 'plan':
        for cell_label, cell_spec in CELLS.items():
            print(cell_label, cell_spec)
    else:
        raise SystemExit('Usage: afhqv2_curve_eval.py preflight|queue|plan')
