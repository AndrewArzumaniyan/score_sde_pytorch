"""CelebA: the curve of arm B over training steps, evaluated while celeba_b_continue.py trains.

Stage 0 protocol: DPM++ 2M, 64 trajectory calls plus Tweedie, 10k images, sampling seed 0,
window [-10; 9.115429865459795]. The first cell is the control: B at 200k from the original
workdir against the stage 0 value 8.3244. Stage 0 ran on an RTX PRO 6000 and this runs on
the H200 in the same image, and FID moved by 0.6% between the two cards in the October smoke
test. So a deviation up to 0.02 counts as a match, up to 0.1 as a match within the change of
card, and only beyond 0.1 does the queue stop with exit code 3 before evaluating anything
new. The deviation and its verdict go to summary.json.

Then 300k to 800k every 100k, in that order, each as soon as its snapshot exists; the
training writes snapshots by rename, so an existing file is complete. One training run of
B, one sampling seed, 10k images: a screen without intervals. Native FOX values on the same
protocol are kept for reading only; their windows are given with them.

Run through run_celeba_b_continue_800k_h200.sh, on the GPU of the training.
"""
import csv
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time

import celeba_50k_lo_eval as lo
import celeba_followup_eval as followup
import celeba_stage0_eval as stage0
from celeba_stage0_eval import LOGSNR_MIN, REFERENCE_SHA, WINDOWS, set_protocol, write_json

ROOT = Path(__file__).resolve().parent
LOGS = ROOT / 'logs'
TRAINING_INPUTS = ROOT / 'workdirs/_training_inputs'
CONTINUATION = ROOT / 'continuation'  # The training workdir, mounted read-only.
SOURCE_WORK = 'celeba_factorial_B_vp_pfox_50k'
WORK = 'celeba_factorial_B_vp_pfox_800k_h200_seed42'
SOURCE_ID = '70f0b3d68a294f5282f860a7bb292825'
CONTROL_CKPT = 40
CONTROL = 8.324407577514648  # Stage 0, B_c40_lo.
CONTROL_TOLERANCE = 0.02
CARD_TOLERANCE = 0.1  # RTX PRO 6000 to H200; beyond it the queue stops.
GATE_EXIT = 3
CKPTS = (60, 80, 100, 120, 140, 160)
WINDOW = 'lo'
CALLS = stage0.BUDGET
COUNT = stage0.COUNT
BATCH = stage0.BATCH
ATTEMPTS = 2
POLL_SECONDS = int(os.environ.get('BCURVE_POLL_SECONDS', '60'))
WAIT_SECONDS = int(os.environ.get('BCURVE_WAIT_SECONDS', str(120 * 3600)))  # For the whole queue.
NATIVE_FOX = {
    40: dict(value=9.07, window='[-10; 9.115]'),
    80: dict(value=4.48, window='[-10; 10.576]'),
    160: dict(value=2.46, window='[-10; 10.576]'),
}


def cell_name(ckpt):
    return 'B_c{}_lo'.format(ckpt)


# The order is fixed in advance and does not depend on any observed FID.
CELLS = {cell_name(ckpt): dict(model='B', ckpt=ckpt, window=WINDOW, step=5000 * ckpt + 1,
                               work=SOURCE_WORK if ckpt == CONTROL_CKPT else WORK)
         for ckpt in (CONTROL_CKPT,) + CKPTS}
CONTROL_CELL = cell_name(CONTROL_CKPT)


def manifest(spec):
    path = (TRAINING_INPUTS / (SOURCE_WORK + '.json') if spec['ckpt'] == CONTROL_CKPT
            else CONTINUATION / 'run_manifest.json')
    return json.loads(path.read_text())


def get_config(name):
    from ml_collections import ConfigDict
    from configs.factorial.celeba_b_vp_pfox import get_config as get_b
    if name not in CELLS:
        raise ValueError('Unknown B curve cell: ' + name)
    spec = CELLS[name]
    config = ConfigDict(manifest(spec)['config'])
    if config.training.factorial_arm != 'B_vp_pfox' or config.seed != 42:
        raise ValueError('Unexpected run behind B')
    config.eval = get_b().eval  # Manifests exclude evaluation settings.
    set_protocol(config, spec['window'], spec['ckpt'])
    return config


def checkpoint_path(spec):
    return ROOT / 'workdirs' / spec['work'] / 'checkpoints/checkpoint_{}.pth'.format(spec['ckpt'])


def preflight():
    """Real weights of the control, its grid and a short real run. Returns 1 on failure."""
    import torch
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
    config = get_config(CONTROL_CELL)
    sde, eps = get_sde(config, n_override=CALLS)
    stage0.check_grid(config, sde, eps)
    model, identity = stage0.load_model(config, SOURCE_WORK, CONTROL_CKPT)
    failures = {}
    if identity != dict(checkpoint_id=SOURCE_ID, step=200001):
        failures[CONTROL_CELL] = 'Unexpected control checkpoint: {}'.format(identity)
    else:
        followup.smoke(config, model, stage0.METHOD, eps)
    write_json(path, dict(status='failed' if failures else 'passed', control=identity, failures=failures,
        source_sha256=_source_fingerprint(str(ROOT)), environment=_environment_manifest(), started_unix=time.time()))
    print('Preflight', 'failed: {}'.format(failures) if failures else 'passed', flush=True)
    return 1 if failures else 0


def eval_folder(name):
    return 'eval_bcurve_' + name


def result_path(name):
    return LOGS / (name + '_result.json')


def failure_path(name):
    return LOGS / (name + '_failed.json')


def run(name):
    began, clock = time.time(), time.monotonic()
    result = subprocess.run([sys.executable, '-u', 'main.py', '--mode=eval', '--config=celeba_b_curve_eval.py:' + name,
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
    nfe = CALLS + 1
    training = manifest(spec)['training_protocol_sha256']
    with np.load(str(directory / 'report_{}.npz'.format(spec['ckpt'])), allow_pickle=False) as data:
        expected = dict(num_samples=COUNT, nfe=nfe, checkpoint_step=spec['step'], dataset_stats_id=REFERENCE_SHA,
                        training_protocol_sha256=training)
        if spec['ckpt'] == CONTROL_CKPT:
            expected['checkpoint_id'] = SOURCE_ID
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
        lambda_window=[LOGSNR_MIN, WINDOWS[WINDOW]], nfe=nfe, num_samples=COUNT, sampling_seed=stage0.SEED,
        elapsed_seconds=timing['elapsed_seconds'], **metrics))
    write_summary()
    print('B curve {}: FID after Tweedie {:.4f}, {:.1f} min'.format(
        name, metrics['fid_tweedie_xt'], timing['elapsed_seconds'] / 60), flush=True)


def control_verdict(value):
    deviation = abs(value - CONTROL)
    if deviation <= CONTROL_TOLERANCE:
        return 'matched'
    if deviation <= CARD_TOLERANCE:
        return 'matched_within_card_tolerance'
    return 'failed'


def write_summary(failed=(), gate=None):
    cells = {name: json.loads(result_path(name).read_text()) for name in CELLS if result_path(name).exists()}
    control = cells.get(CONTROL_CELL)
    status = ('gate_failed' if gate is False else 'completed' if len(cells) == len(CELLS)
              else 'failed' if failed else 'running')
    write_json(LOGS / 'summary.json', dict(status=status, completed_cells=len(cells), total_cells=len(CELLS),
        failed_cells=list(failed), cells=list(cells.values()),
        control=dict(reference=CONTROL, tolerance=CONTROL_TOLERANCE, card_tolerance=CARD_TOLERANCE,
                     value=control['fid_tweedie_xt'] if control else None,
                     difference=control['fid_tweedie_xt'] - CONTROL if control else None,
                     verdict=control_verdict(control['fid_tweedie_xt']) if control else None),
        native_fox_for_reading=NATIVE_FOX,
        scope='One training run of B and one sampling seed at 10k; no intervals'))
    with (LOGS / 'summary.csv').open('w', newline='') as stream:
        writer = csv.writer(stream)
        writer.writerow(('cell', 'model', 'step', 'lambda_max', 'fid_tweedie', 'fid_x_t', 'kid_tweedie', 'is_tweedie'))
        for name, value in cells.items():
            writer.writerow((name, value['model'], value['step'], value['lambda_window'][1], value['fid_tweedie_xt'],
                             value['fid_x_t'], value['kid_tweedie_xt'], value['IS_tweedie_xt']))


def wait_for(spec, deadline):
    """True once the snapshot and the manifest exist; False when the queue runs out of time."""
    path = checkpoint_path(spec)
    printed = False
    while not (path.is_file() and (CONTINUATION / 'run_manifest.json').is_file()):
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
            print('B curve cell failed: {}: {!r}'.format(name, error), flush=True)
    return False


def queue():
    """Control, gate, then each snapshot as it appears. A failed cell does not stop the next."""
    pre = LOGS / 'preflight.json'
    if not pre.exists() or json.loads(pre.read_text())['status'] != 'passed':
        print('Preflight did not pass: the queue stops.', flush=True)
        return 1
    if not result_path(CONTROL_CELL).exists() and not evaluate(CONTROL_CELL):
        write_summary([CONTROL_CELL])
        return 1
    value = json.loads(result_path(CONTROL_CELL).read_text())['fid_tweedie_xt']
    verdict = control_verdict(value)
    if verdict == 'failed':
        write_summary(gate=False)
        print('Control {:.4f} differs from {:.4f} by more than {}: the queue stops.'.format(
            value, CONTROL, CARD_TOLERANCE), flush=True)
        return GATE_EXIT
    print('Control {:.4f}, stage 0 {:.4f}, deviation {:+.4f}: {}'.format(value, CONTROL, value - CONTROL, verdict),
          flush=True)
    deadline = time.monotonic() + WAIT_SECONDS
    failed = []
    for ckpt in CKPTS:
        name = cell_name(ckpt)
        if result_path(name).exists():
            continue
        if not wait_for(CELLS[name], deadline):
            failed += [cell_name(rest) for rest in CKPTS[CKPTS.index(ckpt):] if not result_path(cell_name(rest)).exists()]
            print('No snapshot before the deadline: ' + name, flush=True)
            break
        if not evaluate(name):
            failed.append(name)
    write_summary(failed)
    print('B curve finished, failed cells: {}'.format(failed or 'none'), flush=True)
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
        raise SystemExit('Usage: celeba_b_curve_eval.py preflight|queue|plan')
