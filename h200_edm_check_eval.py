"""Two evaluation queues for one H200 each; the rule below is fixed on 15 November before any of these cells.

Rule. Steps were selected by 10k DPM++ curves; EDM Heun took no part. The EDM curve applies the
same rule to DIAL, B and A: the minimum FID among a model's candidates, then the latest candidate
within the minimum plus 0.2. It is a robustness check: the selected steps and the table stay.
If the EDM selection of DIAL or B is not 800k, a 50k EDM Heun cell at that step is planned in a
separate record, and the text says that the EDM Heun sign of DIAL - B depends on the checkpoint.
Differences below 0.2 at 10k are not interpreted. AFHQv2 DIAL stays selected at 200k; DIAL at
250k is reported next to A and VP at 250k as an equal-step check.

Lane curve: CelebA 10k curves under EDM Heun at 34 trajectory calls plus Tweedie (Karras grid),
sampling seed 0, window [-10; 9.115429865459795]: DIAL and B at 600k-800k, A at 400k-800k.
The configs are those of the 50k matched35 cells of each model, with 10k images:
DIAL as celeba_fox800_lo_eval (lo.native_config), B as gpu2_queue_eval.config_of,
A as gpu2_a50k_grids.config_of. The summary applies the +0.2 rule to the EDM curve; it
selects nothing for the paper.

Lane afhq: AFHQv2 DIAL at 250k, 50k matched35 with DPM++ 2M and EDM Heun, seeds 0 and 1,
configured as the afhq_fox_c4 cells of gpu2_queue_eval.

Preflight passes the first cell of every checkpoint source through its real path; a failed
check blocks only that source's cells, a failed cell does not stop the next one.

Run through run_h200_edm_check.sh from a separate worktree; H200_EDM_LANE=curve|afhq.
"""
import csv
import json
import math
import os
import subprocess
import sys
import time

import afhqv2_rtx_train as afhq
import gpu2_queue_eval as q
import gpu2_a50k_grids as grids  # Registers the CelebA A sources in q.SOURCES.

lo = q.lo
ROOT = q.ROOT
LOGS = q.LOGS
ATTEMPTS = q.ATTEMPTS
HEUN = 'ode_edm_heun'
CURVE_COUNT = 10000
LANES = ('curve', 'afhq')
A_FIRST = grids.A_FIRST  # 400k: the RTX run of table 4.
CANDIDATES = {'DIAL': ('celeba_fox', (120, 140, 160)), 'B': ('celeba_b', (120, 140, 160)),
              'A': (None, (80, 100, 120, 140, 160))}
write_json = q.write_json


def a_source(ckpt):
    return grids.source_of(ckpt)


def curve_cell(model, ckpt):
    source = CANDIDATES[model][0] or a_source(ckpt)
    return '{}_c{}_edm10k'.format(model.lower(), ckpt), dict(
        kind='edm10k', lane='curve', model=model, source=source, group=source + '_edm10k', ckpt=ckpt,
        method=HEUN, seed=0, count=CURVE_COUNT, calls=q.M35_CALLS, dataset='CELEBA')


def afhq_cells():
    cells = q.m35('afhq_fox', 5)
    for _, spec in cells:
        spec.update(lane='afhq', model='DIAL', dataset='AFHQV2')
    return cells


# The order is fixed in advance and does not depend on any observed FID.
CELLS = dict([curve_cell(m, c) for c in (160, 140, 120) for m in ('DIAL', 'B')] +
             [curve_cell('A', c) for c in CANDIDATES['A'][1]] + afhq_cells())


def lane():
    value = os.environ.get('H200_EDM_LANE', '')
    if value not in LANES:
        raise ValueError('H200_EDM_LANE must be curve or afhq, not {!r}'.format(value))
    return value


def lane_cells(which):
    return [name for name, spec in CELLS.items() if spec['lane'] == which]


def resolve(name):
    if name not in CELLS:
        raise ValueError('Unknown H200 EDM check cell: ' + name)
    spec = dict(CELLS[name])
    spec['work'] = spec['source']
    spec['step'] = q.SOURCES[spec['source']]['freq'] * spec['ckpt'] + 1
    return spec


def config_of(spec):
    if spec['source'] == 'celeba_fox':  # As celeba_fox800_lo_eval: lo.native_config of the 50k cell.
        config = lo.native_config(dict(calls=spec['calls'], count=lo.COUNT, model='fox', method=spec['method'],
                                       seed=spec['seed'], ckpt=spec['ckpt'], window=q.WINDOW))
        config.eval.num_samples = spec['count']
        return config
    m35 = dict(spec, kind='m35')
    if spec['source'] in ('celeba_a_first', 'celeba_a'):
        return grids.config_of(m35)
    return q.config_of(m35)


def get_config(name):
    return config_of(resolve(name))


def preflight():
    """The first cell of every source of the lane through its real path. Returns 1 on failure."""
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
    try:  # Several models share this process; Inception must not take the whole card.
        import tensorflow as tf
        for device in tf.config.list_physical_devices('GPU'):
            tf.config.experimental.set_memory_growth(device, True)
    except (RuntimeError, ValueError, AttributeError, TypeError):
        pass
    lo.check_inception()
    from artifact_utils import _environment_manifest, _source_fingerprint
    names = lane_cells(which)
    first = get_config(names[0])
    if which == 'afhq':
        afhq.check_stats(first)
    stats = {CELLS[names[0]]['dataset']: evaluation.dataset_stats_fingerprint(first)}
    if which == 'curve' and stats['CELEBA'] != q.REFERENCE_SHA:
        raise RuntimeError('Unexpected CelebA statistics.')
    firsts = {}
    for name in names:
        firsts.setdefault(CELLS[name]['group'], name)
    identities, failures, smokes = {}, {}, {}
    for group, name in firsts.items():
        try:
            spec = resolve(name)
            config = config_of(spec)
            sde, eps = get_sde(config, n_override=spec['calls'])
            followup.check_cell_grid(config, sde, eps, spec)
            model, identities[group] = q.load_model(config, spec)
            followup.smoke(config, model, spec['method'], eps)
            smokes[group] = q.endpoint_smoke(config, model, spec, LOGS / 'preflight_endpoint')
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
    return 'eval_h200edm_' + name


def result_path(name):
    return LOGS / (name + '_result.json')


def failure_path(name):
    return LOGS / (name + '_failed.json')


def run(name):
    spec = resolve(name)
    began, clock = time.time(), time.monotonic()
    result = subprocess.run([sys.executable, '-u', 'main.py', '--mode=eval', '--config=h200_edm_check_eval.py:' + name,
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
    config = config_of(spec)
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
                        dataset_stats_id=pre['dataset_stats_id'][spec['dataset']],
                        training_protocol_sha256=q.training_protocol(spec['source']))
        for key, value in expected.items():
            if data[key].item() != value:
                raise RuntimeError('Unexpected report identity: ' + key)
        checkpoint_id = str(data['checkpoint_id'].item())
        metrics = {metric + '_' + variant: float(data[metric + '_' + variant])
                   for metric in ('IS', 'fid', 'kid') for variant in q.stage0.VARIANTS}
    q.check_identity(spec, spec['step'], q.training_protocol(spec['source']), checkpoint_id)
    if not all(math.isfinite(value) for value in metrics.values()):
        raise RuntimeError('Non-finite metric.')
    rounds = directory / 'ckpt_{}'.format(spec['ckpt'])
    if len(list(rounds.glob('statistics_*.npz'))) != (spec['count'] + q.BATCH - 1) // q.BATCH:
        raise RuntimeError('Unexpected number of statistics shards.')
    if spec['kind'] == 'm35':
        with np.load(str(rounds / 'statistics_0.npz'), allow_pickle=False) as data:
            if q.FEATURES not in data.files or data[q.FEATURES].shape != (q.BATCH, 2048):
                raise RuntimeError('Shards lack the bootstrap features.')
    timing = json.loads((LOGS / (name + '_timing.json')).read_text())
    write_json(result_path(name), dict(spec, cell=name, checkpoint_id=checkpoint_id,
        lambda_window=[q.LOGSNR_MIN, q.WINDOWS[q.WINDOW]], nfe=nfe, num_samples=spec['count'],
        sampling_seed=spec['seed'], elapsed_seconds=timing['elapsed_seconds'], **metrics))
    write_summary()
    print('H200 EDM {}: FID after Tweedie {:.4f}, {:.1f} min'.format(
        name, metrics['fid_tweedie_xt'], timing['elapsed_seconds'] / 60), flush=True)


def edm_selection(cells):
    """The +0.2 rule on the EDM curve of each CelebA model; None for a model with missing candidates."""
    out = {}
    for model, (_, ckpts) in CANDIDATES.items():
        fids = {}
        for ckpt in ckpts:
            value = cells.get(curve_cell(model, ckpt)[0])
            if value is not None:
                fids[value['step'] - 1] = value['fid_tweedie_xt']
        if len(fids) != len(ckpts):
            out[model] = None
            continue
        step = q.select(fids)
        out[model] = dict(fids=fids, minimum=min(fids.values()), selected_step=step, selected_fid=fids[step])
    return out


def write_summary(failed=()):
    which = lane()
    names = lane_cells(which)
    cells = {name: json.loads(result_path(name).read_text()) for name in names if result_path(name).exists()}
    status = 'completed' if len(cells) == len(names) else 'failed' if failed else 'running'
    extra = dict(edm_selection=edm_selection(cells)) if which == 'curve' else {}
    write_json(LOGS / 'summary.json', dict(status=status, lane=which, completed_cells=len(cells),
        total_cells=len(names), failed_cells=list(failed), cells=list(cells.values()),
        rule='h200_edm_check_eval.py docstring (15 November): a robustness check, the selected steps stay',
        scope='One training run per model; 10k curves are a screen, matched35 at 50k has no intervals here',
        **extra))
    with (LOGS / 'summary.csv').open('w', newline='') as stream:
        writer = csv.writer(stream)
        writer.writerow(('cell', 'model', 'source', 'step', 'method', 'sampling_seed', 'num_samples', 'nfe',
                         'fid_tweedie', 'fid_x_t', 'kid_tweedie', 'is_tweedie'))
        for name, value in cells.items():
            writer.writerow((name, value['model'], value['source'], value['step'], value['method'],
                             value['sampling_seed'], value['num_samples'], value['nfe'], value['fid_tweedie_xt'],
                             value['fid_x_t'], value['kid_tweedie_xt'], value['IS_tweedie_xt']))


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
            print('H200 EDM cell failed: {}: {!r}'.format(name, error), flush=True)
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
    print('H200 EDM lane finished, failed cells: {}'.format(failed or 'none'), flush=True)
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
        raise SystemExit('Usage: H200_EDM_LANE=curve|afhq h200_edm_check_eval.py preflight|queue|plan')
