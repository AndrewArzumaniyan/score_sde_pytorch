"""CelebA on the lower window: native FOX over training steps and a 256-call point for the pair.

Stage 0 protocol: DPM++ 2M, 64 trajectory calls plus Tweedie, 10k images, sampling seed 0,
window [-10; 9.115429865459795]. Ten cells in a fixed order:

1. Two controls that repeat finished RTX cells on the same weights and protocol:
   native FOX at 1.3M (stage 0, fox_c260_lo) and A at 400k (GPU 2 queue, A_c80_lo).
   A deviation up to 0.02 counts as a match, up to 0.1 as a match within the change of
   image or card; beyond 0.1 the queue stops with exit code 3 before anything new.
2. Native FOX at 300k to 800k every 100k on the lower window. The B curve
   (celeba_b_curve_eval.py) evaluates B at the same steps; its native FOX values at 400k and
   800k come from the upper window, so this row is what B is read against. B runs on the
   H200 and this on an RTX PRO 6000: read the two rows as curves, not as exact differences.
3. FOX at 800k and A at 400k with 256 trajectory calls, the pair of the main comparison.
   Compared with the 64-call cells of the same weights at the same 10k images; 256 calls
   is a denser grid, not a proven limit.

The order is fixed in advance and does not depend on any observed FID. One training run
per model, one sampling seed, 10k images: a screen without intervals.

LOCURVE_STOP_AT (Unix time, optional) keeps the card free for a queue that starts later on
the same GPU: a cell that would not finish before it, at a generous duration, is not started.

Run through run_celeba_lo_curve_rtx.sh from a separate worktree.
"""
import csv
import gc
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
from celeba_stage0_eval import LOGSNR_MIN, REFERENCE_SHA, WINDOWS, check_grid, set_protocol, write_json

ROOT = Path(__file__).resolve().parent
LOGS = ROOT / 'logs'
WINDOW = 'lo'
METHOD = stage0.METHOD
COUNT = stage0.COUNT
BATCH = stage0.BATCH
SEED = stage0.SEED
CURVE_CALLS = stage0.BUDGET
LIMIT_CALLS = 256
ATTEMPTS = 2
GATE_EXIT = 3
MATCH_TOLERANCE = 0.02
STOP_TOLERANCE = 0.1
# Generous upper bounds on an RTX PRO 6000; measured cells took about 13 and 40 minutes.
CELL_SECONDS = {CURVE_CALLS: 25 * 60, LIMIT_CALLS: 60 * 60}
WORKDIRS = {'fox': stage0.FOX, 'A': followup.WORKDIRS['A']}

# FID after Tweedie of finished cells with the same weights, protocol and image, on an RTX.
CONTROLS = {
    'fox_c260_lo': dict(value=2.4948720932006836, source='stage 0, fox_c260_lo'),
    'A_c80_lo': dict(value=2.57734751701355, source='GPU 2 queue, A_c80_lo'),
}
CURVE_CKPTS = (60, 80, 100, 120, 140, 160)
LIMIT_CELLS = (('fox', 160), ('A', 80))


def cell_name(model, ckpt, calls=CURVE_CALLS):
    return '{}_c{}_lo'.format(model, ckpt) + ('' if calls == CURVE_CALLS else '_n{}'.format(calls))


def cell(model, ckpt, calls=CURVE_CALLS):
    return cell_name(model, ckpt, calls), dict(model=model, ckpt=ckpt, window=WINDOW, calls=calls)


CELLS = dict([cell('fox', 260), cell('A', 80)] +
             [cell('fox', ckpt) for ckpt in CURVE_CKPTS] +
             [cell(model, ckpt, LIMIT_CALLS) for model, ckpt in LIMIT_CELLS])

# Each contrast is left minus right, FID after Tweedie.
CONTRASTS = (
    ('fox800_n256_minus_n64', cell_name('fox', 160, LIMIT_CALLS), cell_name('fox', 160), 'Denser grid, FOX at 800k'),
    ('A400_n256_minus_n64', cell_name('A', 80, LIMIT_CALLS), cell_name('A', 80), 'Denser grid, A at 400k'),
    ('fox800_minus_A400_n64', cell_name('fox', 160), cell_name('A', 80), 'Main pair at 64 calls, 10k'),
    ('fox800_minus_A400_n256', cell_name('fox', 160, LIMIT_CALLS), cell_name('A', 80, LIMIT_CALLS),
     'Main pair at 256 calls, 10k'),
)


def identity_key(spec):
    return '{}_{}'.format(spec['model'], spec['ckpt'])


def get_config(name):
    if name not in CELLS:
        raise ValueError('Unknown lower-window curve cell: ' + name)
    spec = CELLS[name]
    if spec['model'] == 'fox':
        config = stage0.native_config('fox')
        set_protocol(config, spec['window'], spec['ckpt'])
    else:
        config = followup.a_config(spec)  # Stage 0 protocol on the trained A, as in the GPU 2 queue.
    config.eval.sampling_num_scales = spec['calls']  # Trajectory calls.
    return config


def stop_at():
    value = os.environ.get('LOCURVE_STOP_AT', '').strip()
    return float(value) if value else None


def fits_before_stop(spec, now=None):
    limit = stop_at()
    return limit is None or (now if now is not None else time.time()) + CELL_SECONDS[spec['calls']] <= limit


def preflight():
    """Grids, real weights and a short real run for every cell. Returns 1 on failure."""
    import torch
    from artifact_utils import _environment_manifest, _source_fingerprint
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
    identities, failures = {}, {}
    for name, spec in CELLS.items():
        key = identity_key(spec)
        try:
            config = get_config(name)
            sde, eps = get_sde(config, n_override=spec['calls'])
            check_grid(config, sde, eps, spec['calls'])
            if key in identities:
                continue
            model, identities[key] = stage0.load_model(config, WORKDIRS[spec['model']], spec['ckpt'])
            followup.smoke(config, model, METHOD, eps)
            del model
            gc.collect()
            torch.cuda.empty_cache()
            print('Preflight', key, identities[key]['step'], flush=True)
        except (RuntimeError, OSError, KeyError, ValueError) as error:
            failures[name] = repr(error)
    write_json(path, dict(status='failed' if failures else 'passed', checkpoints=identities, failures=failures,
        source_sha256=_source_fingerprint(str(ROOT)), environment=_environment_manifest(), started_unix=time.time()))
    print('Preflight', 'failed: {}'.format(failures) if failures else 'passed', flush=True)
    return 1 if failures else 0


def eval_folder(name):
    return 'eval_locurve_{}_{}_seed{}'.format(name, COUNT, SEED)


def cell_directory(name):
    return ROOT / 'workdirs' / WORKDIRS[CELLS[name]['model']] / eval_folder(name)


def result_path(name):
    return LOGS / (name + '_result.json')


def failure_path(name):
    return LOGS / (name + '_failed.json')


def run(name):
    began, clock = time.time(), time.monotonic()
    result = subprocess.run([sys.executable, '-u', 'main.py', '--mode=eval', '--config=celeba_lo_curve_eval.py:' + name,
                             '--workdir=workdirs/' + WORKDIRS[CELLS[name]['model']], '--eval_folder=' + eval_folder(name)],
                            cwd=str(ROOT))
    write_json(LOGS / (name + '_timing.json'), dict(started_unix=began, finished_unix=time.time(),
        elapsed_seconds=time.monotonic() - clock, returncode=result.returncode))
    result.check_returncode()


def report(name):
    import numpy as np
    from artifact_utils import _canonicalize
    spec = CELLS[name]
    config = get_config(name)
    directory = cell_directory(name)
    manifest = json.loads((directory / 'manifest.json').read_text())
    for section in ('model', 'data', 'sampling', 'eval'):
        expected = _canonicalize(config[section])
        if section == 'eval':
            expected.pop('begin_ckpt', None)
            expected.pop('end_ckpt', None)
        if _canonicalize(manifest['config'][section]) != expected:
            raise RuntimeError('Unexpected report config: ' + section)
    pre = json.loads((LOGS / 'preflight.json').read_text())
    if manifest['source_sha256'] != pre['source_sha256']:
        raise RuntimeError('Report comes from different sources.')
    identity = pre['checkpoints'][identity_key(spec)]
    nfe = spec['calls'] + 1
    with np.load(str(directory / 'report_{}.npz'.format(spec['ckpt'])), allow_pickle=False) as data:
        expected = dict(num_samples=COUNT, nfe=nfe, checkpoint_step=identity['step'],
                        checkpoint_id=identity['checkpoint_id'], dataset_stats_id=REFERENCE_SHA)
        for key, value in expected.items():
            if data[key].item() != value:
                raise RuntimeError('Unexpected report identity: ' + key)
        metrics = {metric + '_' + variant: float(data[metric + '_' + variant])
                   for metric in ('IS', 'fid', 'kid') for variant in stage0.VARIANTS}
    if not all(math.isfinite(value) for value in metrics.values()):
        raise RuntimeError('Non-finite metric.')
    shards = len(list((directory / 'ckpt_{}'.format(spec['ckpt'])).glob('statistics_*.npz')))
    if shards != (COUNT + BATCH - 1) // BATCH:
        raise RuntimeError('Unexpected number of statistics shards.')
    timing = json.loads((LOGS / (name + '_timing.json')).read_text())
    write_json(result_path(name), dict(spec, cell=name, checkpoint_step=identity['step'],
        checkpoint_id=identity['checkpoint_id'], lambda_window=[LOGSNR_MIN, WINDOWS[WINDOW]], nfe=nfe,
        num_samples=COUNT, sampling_seed=SEED, elapsed_seconds=timing['elapsed_seconds'], **metrics))
    write_summary()
    print('Lower-window curve {}: FID after Tweedie {:.6f}, {:.1f} min'.format(
        name, metrics['fid_tweedie_xt'], timing['elapsed_seconds'] / 60), flush=True)


def control_verdict(name, value):
    deviation = abs(value - CONTROLS[name]['value'])
    if deviation <= MATCH_TOLERANCE:
        return 'matched'
    if deviation <= STOP_TOLERANCE:
        return 'matched_within_card_tolerance'
    return 'failed'


def write_summary(failed=(), gate=None, skipped=()):
    cells = {name: json.loads(result_path(name).read_text()) for name in CELLS if result_path(name).exists()}
    controls = {}
    for name, reference in CONTROLS.items():
        value = cells[name]['fid_tweedie_xt'] if name in cells else None
        controls[name] = dict(reference, value=value, difference=None if value is None else value - reference['value'],
                              verdict=None if value is None else control_verdict(name, value))
    contrasts = [dict(contrast=label, left=left, right=right, question=question,
                      value=cells[left]['fid_tweedie_xt'] - cells[right]['fid_tweedie_xt'])
                 for label, left, right, question in CONTRASTS if left in cells and right in cells]
    status = ('gate_failed' if gate is False else 'completed' if len(cells) == len(CELLS)
              else 'failed' if failed or skipped else 'running')
    write_json(LOGS / 'summary.json', dict(status=status, completed_cells=len(cells), total_cells=len(CELLS),
        failed_cells=list(failed), skipped_before_stop=list(skipped), cells=list(cells.values()),
        controls=dict(cells=controls, match_tolerance=MATCH_TOLERANCE, stop_tolerance=STOP_TOLERANCE),
        point_contrasts=contrasts, confidence_intervals_computed=False,
        scope='10k screen, sampling seed 0, one training run per model'))
    with (LOGS / 'summary.csv').open('w', newline='') as stream:
        writer = csv.writer(stream)
        writer.writerow(('cell', 'model', 'step', 'calls', 'lambda_max', 'fid_tweedie', 'fid_x_t', 'kid_tweedie',
                         'is_tweedie'))
        for name, value in cells.items():
            writer.writerow((name, value['model'], value['checkpoint_step'], value['calls'], value['lambda_window'][1],
                             value['fid_tweedie_xt'], value['fid_x_t'], value['kid_tweedie_xt'], value['IS_tweedie_xt']))


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
            print('Lower-window curve cell failed: {}: {!r}'.format(name, error), flush=True)
    return False


def queue():
    """Controls and gate, then the other cells in order. A failed cell does not stop the next."""
    pre = LOGS / 'preflight.json'
    if not pre.exists() or json.loads(pre.read_text())['status'] != 'passed':
        print('Preflight did not pass: the queue stops.', flush=True)
        return 1
    names = list(CELLS)  # The controls come first; the plan test checks it.
    for name in CONTROLS:
        if not result_path(name).exists():
            if not fits_before_stop(CELLS[name]):
                skipped = [rest for rest in names if not result_path(rest).exists()]
                write_summary(skipped=skipped)
                print('No time before LOCURVE_STOP_AT: nothing evaluated.', flush=True)
                return 1
            if not evaluate(name):
                write_summary([name], gate=False)
                print('Control failed to evaluate: {}: the queue stops.'.format(name), flush=True)
                return GATE_EXIT
        value = json.loads(result_path(name).read_text())['fid_tweedie_xt']
        verdict = control_verdict(name, value)
        print('Control {} {:.6f}, reference {:.6f}, deviation {:+.6f}: {}'.format(
            name, value, CONTROLS[name]['value'], value - CONTROLS[name]['value'], verdict), flush=True)
        if verdict == 'failed':
            write_summary(gate=False)
            print('Control outside {}: the queue stops.'.format(STOP_TOLERANCE), flush=True)
            return GATE_EXIT
    failed, skipped = [], []
    for name in names[len(CONTROLS):]:
        if result_path(name).exists():
            continue
        if skipped or not fits_before_stop(CELLS[name]):
            skipped.append(name)
            continue
        if not evaluate(name):
            failed.append(name)
    write_summary(failed, skipped=skipped)
    print('Lower-window curve finished, failed cells: {}, not started before the stop time: {}'.format(
        failed or 'none', skipped or 'none'), flush=True)
    return 1 if failed or skipped else 0


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
        raise SystemExit('Usage: celeba_lo_curve_eval.py preflight|queue|plan')
