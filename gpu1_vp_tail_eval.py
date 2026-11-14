"""The tail of AFHQv2 VP on RTX GPU 1, after VP reaches 400k; built on gpu2_queue_eval.py.

1. VP at 350k and 400k on the 10k curve: the cells afhq_vp_c5_lo and afhq_vp_c6_lo of the
   GPU 2 queue with another checkpoint, through the same config_of.
2. The step of VP by AFHQV2_STEP_RULE.md (addendum of 14 November): candidates 100k and 200k
   from the first curve (previous/), 250k and 300k from the GPU 2 queue (gpu2/), 350k and
   400k from part 1, without 150k. The minimum FID, then the latest candidate within the
   minimum plus 0.2. Every input and the choice go to logs/vp_selection.json and summary.json.
   If a candidate is missing, the 50k cells are not run and count as failed.
3. VP at that step, 50k matched35: the protocol of the afhq_fox and afhq_a matched35 cells of
   the GPU 2 queue, DPM++ 2M and EDM Heun, sampling seeds 0 and 1. 100k and 200k come from
   the first run, 250k and later from the continuation.

Preflight passes 512 images through evaluate_checkpoint on VP weights for the curve and for
matched35 with both solvers; VP with EDM Heun on AFHQv2 had not run before. A failed cell
does not stop the next one.

Run through run_gpu1_vp_tail_rtx.sh from a separate worktree.
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
PREVIOUS = q.PREVIOUS  # Logs of the first AFHQv2 curve, mounted read-only.
GPU2 = ROOT / 'gpu2'  # Logs of the GPU 2 queue, mounted read-only.
SELECTION = 'vp_selection.json'
ATTEMPTS = q.ATTEMPTS

# The first VP run holds 100k and 200k; the continuation holds 200k to 400k.
q.SOURCES.setdefault('afhq_vp_first', dict(dataset='AFHQV2', arm='vp', manifest='afhq_vp', freq=afhq.SNAPSHOT_FREQ))

# The order is fixed in advance and does not depend on any observed FID.
CELLS = dict([q.curve('afhq_vp', 7), q.curve('afhq_vp', 8)] + q.m35('afhq_vp', None, 'afhq_vp'))
for _name, _spec in CELLS.items():
    _spec['dataset'] = 'AFHQV2'
VP_CANDIDATES = (('previous', 'vp_c2_lo', 2), ('previous', 'vp_c4_lo', 4), ('gpu2', 'afhq_vp_c5_lo', 5),
                 ('gpu2', 'afhq_vp_c6_lo', 6), ('local', 'afhq_vp_c7_lo', 7), ('local', 'afhq_vp_c8_lo', 8))
ORIGINS = {'previous': lambda: PREVIOUS, 'gpu2': lambda: GPU2, 'local': lambda: LOGS}


def vp_selection():
    """Point 2 of the rule for AFHQv2 VP over 100k-400k; incomplete until every candidate is in."""
    fids, inputs = {}, []
    for origin, name, ckpt in VP_CANDIDATES:
        path = ORIGINS[origin]() / (name + '_result.json')
        value = json.loads(path.read_text())['fid_tweedie_xt'] if path.exists() else None
        inputs.append(dict(cell=name, origin=origin, step=afhq.SNAPSHOT_FREQ * ckpt, fid_tweedie_xt=value))
        if value is not None:
            fids[afhq.SNAPSHOT_FREQ * ckpt] = value
    result = dict(rule='latest candidate with FID <= minimum + {} (AFHQV2_STEP_RULE.md, 14 November)'.format(q.TIE),
                  inputs=inputs, complete=len(fids) == len(VP_CANDIDATES))
    if result['complete']:
        step = q.select(fids)
        result.update(minimum=min(fids.values()), selected_step=step, selected_fid=fids[step],
                      selected_ckpt=step // afhq.SNAPSHOT_FREQ)
    return result


def resolve(name):
    if name not in CELLS:
        raise ValueError('Unknown VP tail cell: ' + name)
    spec = dict(CELLS[name])
    if spec['ckpt'] is None:
        path = LOGS / SELECTION
        if not path.exists():
            raise ValueError('No step selected for AFHQv2 VP yet.')
        selected = json.loads(path.read_text())
        if not selected.get('complete'):
            raise ValueError('The AFHQv2 VP selection is incomplete.')
        spec['ckpt'] = int(selected['selected_ckpt'])
        spec['source'] = 'afhq_vp' if spec['ckpt'] >= 4 else 'afhq_vp_first'  # Both runs hold 200k.
    spec['work'] = spec['source']
    spec['step'] = q.SOURCES[spec['source']]['freq'] * spec['ckpt'] + 1
    return spec


def get_config(name):
    return q.config_of(resolve(name))


def preflight():
    """The curve on VP at 350k and matched35 with both solvers on VP at 400k. Returns 1 on failure."""
    import torch
    import evaluation
    import celeba_followup_eval as followup
    from run_lib import get_sde
    path = LOGS / 'preflight.json'
    if path.exists():
        path.unlink()
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError('Expected one Docker-routed GPU.')
    arch = '{}.{}'.format(*torch.cuda.get_device_capability(0))
    if arch not in os.environ.get('TORCH_CUDA_ARCH_LIST', '').replace(';', ' ').split():
        raise RuntimeError('TORCH_CUDA_ARCH_LIST does not include sm ' + arch)
    try:  # Two models share this process; Inception must not take the whole card.
        import tensorflow as tf
        for device in tf.config.list_physical_devices('GPU'):
            tf.config.experimental.set_memory_growth(device, True)
    except (RuntimeError, ValueError, AttributeError, TypeError):
        pass
    q.lo.check_inception()
    from artifact_utils import _environment_manifest, _source_fingerprint
    config = get_config('afhq_vp_c7_lo')
    afhq.check_stats(config)
    stats = dict(AFHQV2=evaluation.dataset_stats_fingerprint(config))
    curve_spec = resolve('afhq_vp_c7_lo')
    m35_spec = dict(CELLS['afhq_vp_cbest_m35_ode_dpmpp_2m_seed0'], ckpt=8, source='afhq_vp', work='afhq_vp',
                    step=q.SOURCES['afhq_vp']['freq'] * 8 + 1)
    checks = {'afhq_vp_curve': (curve_spec, (curve_spec['method'],)), 'afhq_vp_m35': (m35_spec, q.METHODS)}
    identities, failures, smokes = {}, {}, {}
    for group, (spec, methods) in checks.items():
        try:
            config = q.config_of(spec)
            sde, eps = get_sde(config, n_override=spec['calls'])
            followup.check_cell_grid(config, sde, eps, spec)
            model, identities[group] = q.load_model(config, spec)
            followup.smoke(config, model, spec['method'], eps)
            for method in methods:
                other = dict(spec, method=method)
                smokes['{}_{}'.format(group, method)] = q.endpoint_smoke(q.config_of(other), model, other,
                                                                          LOGS / 'preflight_endpoint')
            del model
            print('Preflight', group, identities[group], flush=True)
        except Exception as error:  # A truncated checkpoint raises whatever torch.load does.
            failures[group] = repr(error)
            print('Preflight failed: {}: {!r}'.format(group, error), flush=True)
    write_json(path, dict(status='failed' if failures else 'passed', checkpoints=identities, failures=failures,
        endpoint_smoke_fid_512=smokes, dataset_stats_id=stats, source_sha256=_source_fingerprint(str(ROOT)),
        environment=_environment_manifest(), started_unix=time.time()))
    print('Preflight', 'failed: {}'.format(failures) if failures else 'passed', flush=True)
    return 1 if failures else 0


write_json = q.write_json


def eval_folder(name):
    return 'eval_gpu1vp_' + name


def result_path(name):
    return LOGS / (name + '_result.json')


def failure_path(name):
    return LOGS / (name + '_failed.json')


def run(name):
    spec = resolve(name)
    began, clock = time.time(), time.monotonic()
    result = subprocess.run([sys.executable, '-u', 'main.py', '--mode=eval', '--config=gpu1_vp_tail_eval.py:' + name,
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
    print('VP tail {}: FID after Tweedie {:.4f}, {:.1f} min'.format(
        name, metrics['fid_tweedie_xt'], timing['elapsed_seconds'] / 60), flush=True)


def write_summary(failed=()):
    cells = {name: json.loads(result_path(name).read_text()) for name in CELLS if result_path(name).exists()}
    path = LOGS / SELECTION
    status = 'completed' if len(cells) == len(CELLS) else 'failed' if failed else 'running'
    write_json(LOGS / 'summary.json', dict(status=status, completed_cells=len(cells), total_cells=len(CELLS),
        failed_cells=list(failed), cells=list(cells.values()),
        vp_selection=json.loads(path.read_text()) if path.exists() else vp_selection(),
        rule='AFHQV2_STEP_RULE.md, addendum of 14 November',
        scope='One training run; the 10k curve is a screen, matched35 at 50k has no intervals here'))
    with (LOGS / 'summary.csv').open('w', newline='') as stream:
        writer = csv.writer(stream)
        writer.writerow(('cell', 'kind', 'source', 'step', 'method', 'sampling_seed', 'num_samples', 'nfe',
                         'fid_tweedie', 'fid_x_t', 'kid_tweedie', 'is_tweedie'))
        for name, value in cells.items():
            writer.writerow((name, value['kind'], value['source'], value['step'], value['method'],
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
            print('VP tail cell failed: {}: {!r}'.format(name, error), flush=True)
    return False


def queue():
    """The curve, the selection, then matched35. Without a complete selection nothing at 50k runs."""
    pre = LOGS / 'preflight.json'
    if not pre.exists():
        print('No preflight.json: the queue stops.', flush=True)
        return 1
    blocked = json.loads(pre.read_text())['failures']
    failed = []
    for name, spec in CELLS.items():
        if result_path(name).exists():
            continue
        if spec['group'] in blocked:
            failed.append(name)
            continue
        if spec['ckpt'] is None:
            if not ((LOGS / SELECTION).exists() and json.loads((LOGS / SELECTION).read_text())['complete']):
                selected = vp_selection()  # A complete selection is never recomputed, so a rerun keeps its step.
                write_json(LOGS / SELECTION, selected)
                print('AFHQv2 VP selection: {}'.format(selected), flush=True)
            if not json.loads((LOGS / SELECTION).read_text())['complete']:
                failed.append(name)
                print('AFHQv2 VP selection is incomplete: ' + name, flush=True)
                continue
        if not evaluate(name):
            failed.append(name)
    write_summary(failed)
    print('VP tail finished, failed cells: {}'.format(failed or 'none'), flush=True)
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
        raise SystemExit('Usage: gpu1_vp_tail_eval.py preflight|queue|plan')
