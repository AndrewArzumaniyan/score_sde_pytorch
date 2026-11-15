"""AFHQv2 A and VP at 200k, 50k matched35: a robustness check of the +0.2 tie rule.

Recorded in AFHQV2_STEP_RULE.md, addendum of 15 November. The main rule does not change: A and
VP stay at their selected 250k. Here both are also evaluated at 200k, the snapshot with the
lowest 10k FID, on the protocol of the afhq_a_cbest and afhq_vp_cbest cells: 34 trajectory
calls plus Tweedie, DPM++ 2M and EDM Heun, sampling seeds 0 and 1, window
[-10; 9.115429865459795], statistics afhqv2_64_cd017afab009. Configs come from
gpu2_queue_eval.config_of, the same path as those cells.

Two lanes, one container each: lane a (A at 200k) and lane vp (VP at 200k). 200k is
checkpoint_4 of the continuation runs, the copy that gpu2_queue_eval.resolve and
gpu1_vp_tail_eval.resolve already use for 200k; step 200001 and the training protocol of the
continuation manifest are checked on load and in every report.

Preflight is short: grid, weights, a short real run, and 512 images through
evaluate_checkpoint with DPM++ 2M; both solvers already ran on these models at 250k.
A failed cell does not stop the next one.

Run through run_afhq200k_rtx.sh from a separate worktree.
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
import gpu2_queue_eval as q

ROOT = q.ROOT
LOGS = q.LOGS
PREVIOUS = ROOT / 'previous'  # Logs of the first AFHQv2 curve, mounted read-only.
ATTEMPTS = q.ATTEMPTS
CKPT = 4  # 200k.
LANES = {'a': 'afhq_a', 'vp': 'afhq_vp'}
TEN_K = {'a': ('a_c4_lo', 'afhq_a'), 'vp': ('vp_c4_lo', 'afhq_vp')}  # The 10k cell of 200k on the first curve.

# The order is fixed in advance and does not depend on any observed FID.
CELLS = dict(q.m35('afhq_a', CKPT) + q.m35('afhq_vp', CKPT))
for _name, _spec in CELLS.items():
    _spec['dataset'] = 'AFHQV2'
    _spec['lane'] = 'a' if _spec['source'] == 'afhq_a' else 'vp'


def lane():
    value = os.environ.get('AFHQ200K_LANE', '')
    if value not in LANES:
        raise ValueError('AFHQ200K_LANE must be a or vp, not {!r}'.format(value))
    return value


def lane_cells(which):
    return [name for name, spec in CELLS.items() if spec['lane'] == which]


def resolve(name):
    if name not in CELLS:
        raise ValueError('Unknown AFHQv2 200k cell: ' + name)
    spec = dict(CELLS[name])
    spec['work'] = spec['source']
    spec['step'] = q.SOURCES[spec['source']]['freq'] * spec['ckpt'] + 1
    return spec


def get_config(name):
    return q.config_of(resolve(name))


write_json = q.write_json


def preflight():
    """The lane's model through its real path. Returns 1 on failure."""
    import torch
    import evaluation
    import celeba_followup_eval as followup
    from run_lib import get_sde
    which = lane()
    path = LOGS / 'preflight.json'
    if path.exists():
        path.unlink()
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError('Expected one Docker-routed GPU.')
    arch = '{}.{}'.format(*torch.cuda.get_device_capability(0))
    if arch not in os.environ.get('TORCH_CUDA_ARCH_LIST', '').replace(';', ' ').split():
        raise RuntimeError('TORCH_CUDA_ARCH_LIST does not include sm ' + arch)
    try:  # Inception must not take the whole card.
        import tensorflow as tf
        for device in tf.config.list_physical_devices('GPU'):
            tf.config.experimental.set_memory_growth(device, True)
    except (RuntimeError, ValueError, AttributeError, TypeError):
        pass
    q.lo.check_inception()
    from artifact_utils import _environment_manifest, _source_fingerprint
    first = lane_cells(which)[0]
    config = get_config(first)
    afhq.check_stats(config)
    stats = dict(AFHQV2=evaluation.dataset_stats_fingerprint(config))
    group = LANES[which] + '_m35'
    identities, failures, smokes = {}, {}, {}
    try:
        spec = resolve(first)
        sde, eps = get_sde(config, n_override=spec['calls'])
        followup.check_cell_grid(config, sde, eps, spec)
        model, identities[group] = q.load_model(config, spec)
        followup.smoke(config, model, spec['method'], eps)
        smokes['{}_{}'.format(group, spec['method'])] = q.endpoint_smoke(config, model, spec,
                                                                         LOGS / 'preflight_endpoint')
        del model
        print('Preflight', group, identities[group], flush=True)
    except Exception as error:  # A truncated checkpoint raises whatever torch.load does.
        failures[group] = repr(error)
        print('Preflight failed: {}: {!r}'.format(group, error), flush=True)
    write_json(path, dict(status='failed' if failures else 'passed', lane=which, checkpoints=identities,
        failures=failures, endpoint_smoke_fid_512=smokes, dataset_stats_id=stats,
        source_sha256=_source_fingerprint(str(ROOT)), environment=_environment_manifest(), started_unix=time.time()))
    print('Preflight', 'failed: {}'.format(failures) if failures else 'passed', flush=True)
    return 1 if failures else 0


def eval_folder(name):
    return 'eval_afhq200k_' + name


def result_path(name):
    return LOGS / (name + '_result.json')


def failure_path(name):
    return LOGS / (name + '_failed.json')


def run(name):
    spec = resolve(name)
    began, clock = time.time(), time.monotonic()
    result = subprocess.run([sys.executable, '-u', 'main.py', '--mode=eval', '--config=afhq200k_eval.py:' + name,
                             '--workdir=workdirs/' + spec['work'], '--eval_folder=' + eval_folder(name)],
                            cwd=str(ROOT))
    write_json(LOGS / (name + '_timing.json'), dict(started_unix=began, finished_unix=time.time(),
        elapsed_seconds=time.monotonic() - clock, returncode=result.returncode))
    result.check_returncode()


def report(name):
    """The checks of gpu2_queue_eval.report, on the cells of this queue."""
    import numpy as np
    from artifact_utils import _canonicalize
    spec = resolve(name)
    config = q.config_of(spec)
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
    nfe = spec['calls'] + 1
    with np.load(str(directory / 'report_{}.npz'.format(spec['ckpt'])), allow_pickle=False) as data:
        expected = dict(num_samples=spec['count'], nfe=nfe, checkpoint_step=spec['step'],
                        dataset_stats_id=pre['dataset_stats_id']['AFHQV2'],
                        training_protocol_sha256=q.training_protocol(spec['source']))
        for key, value in expected.items():
            if data[key].item() != value:
                raise RuntimeError('Unexpected report identity: ' + key)
        checkpoint_id = str(data['checkpoint_id'].item())
        metrics = {metric + '_' + variant: float(data[metric + '_' + variant])
                   for metric in ('IS', 'fid', 'kid') for variant in q.stage0.VARIANTS}
    if checkpoint_id != pre['checkpoints'][spec['group']]['checkpoint_id']:
        raise RuntimeError('The report and the preflight saw different checkpoints.')
    if not all(math.isfinite(value) for value in metrics.values()):
        raise RuntimeError('Non-finite metric.')
    rounds = directory / 'ckpt_{}'.format(spec['ckpt'])
    if len(list(rounds.glob('statistics_*.npz'))) != (spec['count'] + q.BATCH - 1) // q.BATCH:
        raise RuntimeError('Unexpected number of statistics shards.')
    with np.load(str(rounds / 'statistics_0.npz'), allow_pickle=False) as data:
        if q.FEATURES not in data.files or data[q.FEATURES].shape != (q.BATCH, 2048):
            raise RuntimeError('Shards lack the bootstrap features.')
    timing = json.loads((LOGS / (name + '_timing.json')).read_text())
    write_json(result_path(name), dict(spec, cell=name, checkpoint_id=checkpoint_id,
        lambda_window=[q.LOGSNR_MIN, q.WINDOWS[q.WINDOW]], nfe=nfe, num_samples=spec['count'],
        sampling_seed=spec['seed'], elapsed_seconds=timing['elapsed_seconds'], **metrics))
    write_summary()
    print('AFHQv2 200k {}: FID after Tweedie {:.4f}, {:.1f} min'.format(
        name, metrics['fid_tweedie_xt'], timing['elapsed_seconds'] / 60), flush=True)


def ten_k(which):
    """The 10k FID of 200k on the first curve, for the record; it selects nothing here."""
    name, _ = TEN_K[which]
    path = PREVIOUS / (name + '_result.json')
    return dict(cell=name, fid_tweedie_xt=json.loads(path.read_text())['fid_tweedie_xt'] if path.exists() else None)


def write_summary(failed=()):
    which = lane()
    names = lane_cells(which)
    cells = {name: json.loads(result_path(name).read_text()) for name in names if result_path(name).exists()}
    status = 'completed' if len(cells) == len(names) else 'failed' if failed else 'running'
    write_json(LOGS / 'summary.json', dict(status=status, lane=which, completed_cells=len(cells),
        total_cells=len(names), failed_cells=list(failed), cells=list(cells.values()), ten_k_200k=ten_k(which),
        rule='AFHQV2_STEP_RULE.md, addendum of 15 November: a robustness check, the selected steps stay',
        scope='One training run; matched35 at 50k; intervals come from the bootstrap'))
    with (LOGS / 'summary.csv').open('w', newline='') as stream:
        writer = csv.writer(stream)
        writer.writerow(('cell', 'source', 'step', 'method', 'sampling_seed', 'num_samples', 'nfe',
                         'fid_tweedie', 'fid_x_t', 'kid_tweedie', 'is_tweedie'))
        for name, value in cells.items():
            writer.writerow((name, value['source'], value['step'], value['method'], value['sampling_seed'],
                             value['num_samples'], value['nfe'], value['fid_tweedie_xt'], value['fid_x_t'],
                             value['kid_tweedie_xt'], value['IS_tweedie_xt']))


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
            print('AFHQv2 200k cell failed: {}: {!r}'.format(name, error), flush=True)
    return False


def queue():
    pre = LOGS / 'preflight.json'
    if not pre.exists():
        print('No preflight.json: the queue stops.', flush=True)
        return 1
    blocked = json.loads(pre.read_text())['failures']
    failed = []
    for name in lane_cells(lane()):
        if result_path(name).exists():
            continue
        if CELLS[name]['group'] in blocked or not evaluate(name):
            failed.append(name)
    write_summary(failed)
    print('AFHQv2 200k lane finished, failed cells: {}'.format(failed or 'none'), flush=True)
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
        raise SystemExit('Usage: AFHQ200K_LANE=a|vp afhq200k_eval.py preflight|queue|plan')
