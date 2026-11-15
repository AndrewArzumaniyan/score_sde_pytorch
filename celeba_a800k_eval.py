"""CelebA A at 800k, 50k matched35: the cells of A 500k at the step of DIAL 800k and B 800k.

In case a strictly equal step is asked for. The main comparison does not change: A stays at
its selected step (CELEBA_A800K_STEP_RULE.md). Model: A (VP-lambda), the training run behind
A 500k in gpu2_a50k_grids.py, celeba_factorial_A_vp_pvp_800k_h200_seed42; 800k is
checkpoint_160 (snapshots every 5k), step 800001. Protocol of the A 500k cells: 34 trajectory
calls plus Tweedie, DPM++ 2M and EDM Heun, sampling seeds 0 and 1, 50 000 images, batch 512,
window [-10; 9.115429865459795]; FID, KID and IS on x_t and Tweedie(x_t); the shards keep
pool_3_tweedie_xt for the bootstrap. Configs come from gpu2_a50k_grids.config_of, the path of
the A 500k cells.

Two lanes, one container and one RTX each, sharing one output: lane s0 holds the sampling
seed 0 cells, lane s1 the seed 1 cells; each runs DPM++ 2M, then EDM Heun. Eval writes only to
its own eval folder, so the lanes do not touch each other's files; per-lane files carry the
lane in their name, and the shared summary is written under a unique temporary name.

Preflight: grid, weights, a short real run, and 512 images through evaluate_checkpoint with
both solvers. The checkpoint must load at step 800001, carry the training protocol of the
run manifest, and have the checkpoint_id of the 800k cell of the A curve
(celeba_a_800k_h200_20261113/eval/logs/summary.json, mounted read-only); every report is
checked against the preflight id. A failed cell does not stop the next one.

Run through run_celeba_a800k_rtx.sh from a separate worktree.
"""
import csv
import io
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time

import celeba_followup_eval as followup
import gpu2_a50k_grids as g
import gpu2_queue_eval as q

ROOT = q.ROOT
LOGS = q.LOGS
ACURVE = g.ACURVE  # Logs of the CelebA A curve, mounted read-only.
ATTEMPTS = q.ATTEMPTS
SOURCE = 'celeba_a'  # Registered by gpu2_a50k_grids: the 800k run of A.
CKPT = 160  # 800k.
STEP = g.A_FREQ * CKPT + 1
LANES = {'s0': 0, 's1': 1}  # Lane: sampling seed.
write_json = q.write_json

# The order is fixed in advance and does not depend on any observed FID.
CELLS = dict(q.m35(SOURCE, CKPT))
for _spec in CELLS.values():
    _spec['dataset'] = 'CELEBA'
    _spec['lane'] = 's{}'.format(_spec['seed'])


def lane():
    value = os.environ.get('A800K_LANE', '')
    if value not in LANES:
        raise ValueError('A800K_LANE must be s0 or s1, not {!r}'.format(value))
    return value


def lane_cells(which):
    return [name for name, spec in CELLS.items() if spec['lane'] == which]


def resolve(name):
    if name not in CELLS:
        raise ValueError('Unknown A 800k cell: ' + name)
    return g.with_step(dict(CELLS[name]))


def get_config(name):
    return g.config_of(resolve(name))


def curve_identity():
    """checkpoint_id and step of the 800k cell of the A curve."""
    path = ACURVE / 'summary.json'
    if not path.exists():
        raise RuntimeError('No A curve summary: ' + str(path))
    cells = [cell for cell in json.loads(path.read_text()).get('cells', []) if cell.get('ckpt') == CKPT]
    if len(cells) != 1 or cells[0].get('step') != STEP or not cells[0].get('checkpoint_id'):
        raise RuntimeError('The A curve has no single 800k cell with a checkpoint id.')
    return dict(cell=cells[0]['cell'], checkpoint_id=cells[0]['checkpoint_id'], step=cells[0]['step'],
                fid_tweedie_xt_10k=cells[0].get('fid_tweedie_xt'))


def preflight_path(which):
    return LOGS / 'preflight_{}.json'.format(which)


def preflight():
    """The weights of A at 800k through the real path, both solvers. Returns 1 on failure."""
    import torch
    import evaluation
    from run_lib import get_sde
    which = lane()
    path = preflight_path(which)
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
    names = lane_cells(which)
    spec = resolve(names[0])
    config = g.config_of(spec)
    stats = evaluation.dataset_stats_fingerprint(config)
    if stats != q.REFERENCE_SHA:
        raise RuntimeError('Unexpected CelebA statistics.')
    identity, curve, failures, smokes = None, None, {}, {}
    try:
        curve = curve_identity()
        sde, eps = get_sde(config, n_override=spec['calls'])
        followup.check_cell_grid(config, sde, eps, spec)
        model, identity = q.load_model(config, spec)  # Checks step 800001 and the training protocol.
        if identity != dict(checkpoint_id=curve['checkpoint_id'], step=STEP):
            raise RuntimeError('The checkpoint is not the one of the A curve at 800k: {} vs {}'.format(identity, curve))
        followup.smoke(config, model, spec['method'], eps)
        for name in names:
            other = resolve(name)
            smokes[other['method']] = q.endpoint_smoke(g.config_of(other), model, other,
                                                       LOGS / 'preflight_endpoint_{}'.format(which))
        del model
    except Exception as error:  # A truncated checkpoint raises whatever torch.load does.
        failures['celeba_a_c160_m35'] = repr(error)
        print('Preflight failed: {!r}'.format(error), flush=True)
    write_json(path, dict(status='failed' if failures else 'passed', lane=which, ckpt=CKPT, source=SOURCE, step=STEP,
        checkpoint=identity, curve=curve, failures=failures, endpoint_smoke_fid_512=smokes,
        dataset_stats_id=dict(CELEBA=stats), source_sha256=_source_fingerprint(str(ROOT)),
        environment=_environment_manifest(), started_unix=time.time()))
    print('Preflight A at {}, lane {}:'.format(STEP, which), 'failed: {}'.format(failures) if failures else 'passed',
          flush=True)
    return 1 if failures else 0


def eval_folder(name):
    return 'eval_a800k_' + name


def result_path(name):
    return LOGS / (name + '_result.json')


def failure_path(name):
    return LOGS / (name + '_failed.json')


def run(name):
    spec = resolve(name)
    began, clock = time.time(), time.monotonic()
    result = subprocess.run([sys.executable, '-u', 'main.py', '--mode=eval', '--config=celeba_a800k_eval.py:' + name,
                             '--workdir=workdirs/' + spec['work'], '--eval_folder=' + eval_folder(name)],
                            cwd=str(ROOT))
    write_json(LOGS / (name + '_timing.json'), dict(started_unix=began, finished_unix=time.time(),
        elapsed_seconds=time.monotonic() - clock, returncode=result.returncode))
    result.check_returncode()


def report(name):
    """The checks of gpu2_a50k_grids.report, plus the checkpoint id of the preflight."""
    import numpy as np
    from artifact_utils import _canonicalize
    spec = resolve(name)
    config = g.config_of(spec)
    directory = ROOT / 'workdirs' / spec['work'] / eval_folder(name)
    saved = json.loads((directory / 'manifest.json').read_text())
    for section in ('model', 'data', 'sampling', 'eval'):
        expected = _canonicalize(config[section])
        if section == 'eval':
            expected.pop('begin_ckpt', None)
            expected.pop('end_ckpt', None)
        if _canonicalize(saved['config'][section]) != expected:
            raise RuntimeError('Unexpected report config: ' + section)
    pre = json.loads(preflight_path(spec['lane']).read_text())
    nfe = spec['calls'] + 1
    protocol = q.training_protocol(spec['source'])
    with np.load(str(directory / 'report_{}.npz'.format(spec['ckpt'])), allow_pickle=False) as data:
        expected = dict(num_samples=spec['count'], nfe=nfe, checkpoint_step=spec['step'],
                        dataset_stats_id=pre['dataset_stats_id']['CELEBA'], training_protocol_sha256=protocol)
        for key, value in expected.items():
            if data[key].item() != value:
                raise RuntimeError('Unexpected report identity: ' + key)
        checkpoint_id = str(data['checkpoint_id'].item())
        metrics = {metric + '_' + variant: float(data[metric + '_' + variant])
                   for metric in ('IS', 'fid', 'kid') for variant in q.stage0.VARIANTS}
    q.check_identity(spec, spec['step'], protocol, checkpoint_id)
    if checkpoint_id != pre['checkpoint']['checkpoint_id']:
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
    print('A 800k {}: FID after Tweedie {:.4f}, {:.1f} min'.format(
        name, metrics['fid_tweedie_xt'], timing['elapsed_seconds'] / 60), flush=True)


def write_shared(path, text):
    """Atomic under a unique temporary name: both lanes write the shared summary."""
    fd, tmp = tempfile.mkstemp(dir=str(path.parent), prefix='.' + path.name + '.', suffix='.tmp')
    try:
        with os.fdopen(fd, 'w', newline='') as stream:
            stream.write(text)
        os.chmod(tmp, 0o644)
        os.replace(tmp, str(path))
    except BaseException:
        if os.path.exists(tmp):
            os.unlink(tmp)
        raise


def write_summary(failed=()):
    """Per lane: summary_LANE.json with its failures. Shared: summary.{json,csv} over all four cells."""
    which = lane()
    names = lane_cells(which)
    mine = {name: json.loads(result_path(name).read_text()) for name in names if result_path(name).exists()}
    status = 'completed' if len(mine) == len(names) else 'failed' if failed else 'running'
    write_json(LOGS / 'summary_{}.json'.format(which), dict(status=status, lane=which, completed_cells=len(mine),
        total_cells=len(names), failed_cells=list(failed), cells=list(mine.values())))
    cells = {name: json.loads(result_path(name).read_text()) for name in CELLS if result_path(name).exists()}
    lanes = {}
    for other in LANES:
        path = LOGS / 'summary_{}.json'.format(other)
        lanes[other] = json.loads(path.read_text())['status'] if path.exists() else 'not started'
    overall = ('completed' if len(cells) == len(CELLS) else
               'failed' if 'failed' in lanes.values() and 'running' not in lanes.values() else 'running')
    curve = None
    if preflight_path(which).exists():
        curve = json.loads(preflight_path(which).read_text()).get('curve')
    write_shared(LOGS / 'summary.json', json.dumps(dict(status=overall, lanes=lanes, completed_cells=len(cells),
        total_cells=len(CELLS), cells=list(cells.values()), curve_800k=curve, step=STEP, ckpt=CKPT,
        rule='CELEBA_A800K_STEP_RULE.md: A stays at its selected step; 800k is for a strictly equal step',
        protocol='DPM++ 2M and EDM Heun, 34 trajectory calls plus Tweedie, window [-10; 9.115429865459795], '
                 '50k, sampling seeds 0 and 1; as the A 500k cells of gpu2_a50k_grids.py',
        scope='One training run; matched35 at 50k; intervals come from celeba_a800k_bootstrap.py'), indent=2) + '\n')
    stream = io.StringIO()
    writer = csv.writer(stream)
    writer.writerow(('cell', 'source', 'step', 'method', 'sampling_seed', 'num_samples', 'nfe', 'fid_tweedie',
                     'fid_x_t', 'kid_tweedie', 'is_tweedie'))
    for name, value in cells.items():
        writer.writerow((name, value['source'], value['step'], value['method'], value['sampling_seed'],
                         value['num_samples'], value['nfe'], value['fid_tweedie_xt'], value['fid_x_t'],
                         value['kid_tweedie_xt'], value['IS_tweedie_xt']))
    write_shared(LOGS / 'summary.csv', stream.getvalue())


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
            print('A 800k cell failed: {}: {!r}'.format(name, error), flush=True)
    return False


def queue():
    pre = preflight_path(lane())
    if not pre.exists():
        print('No preflight for this lane: the queue stops.', flush=True)
        return 1
    blocked = json.loads(pre.read_text())['failures']
    failed = []
    for name in lane_cells(lane()):
        if result_path(name).exists():
            continue
        if blocked or not evaluate(name):
            failed.append(name)
    write_summary(failed)
    print('A 800k lane {} finished, failed cells: {}'.format(lane(), failed or 'none'), flush=True)
    return 1 if failed else 0


if __name__ == '__main__':
    mode = sys.argv[1] if len(sys.argv) > 1 else ''
    if mode == 'preflight':
        sys.exit(preflight())
    elif mode == 'queue':
        sys.exit(queue())
    elif mode == 'plan':
        for cell_label, cell_spec in CELLS.items():
            print(cell_label, resolve(cell_label))
    else:
        raise SystemExit('Usage: A800K_LANE=s0|s1 celeba_a800k_eval.py preflight|queue|plan')
