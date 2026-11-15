"""The NFE limit of the models of the main table: three queues, one card each; the rule is fixed on 16 November before any of these cells.

Question. At 35 NFE on CelebA the sign of DIAL@800k - B@800k depends on the solver (-0.10 under DPM++ 2M,
+0.10 under EDM Heun, 50k). These cells check whether that is a difference between the models or a
discretisation error at 35 calls. They are a discretisation control: rows of the tables and selected steps
do not change.

Rule (NIGHT_NFE_LIMIT_PLAN_2026-11-16.md, section 2).
- Solver convergence. For each model, |FID(DPM++) - FID(EDM Heun)| at 257 NFE, 10k, common initial noise.
  At most 0.05: the solvers have converged and 257 NFE is called the reference ODE solution. Otherwise both
  numbers are reported and 257 is not called a reference.
- Main contrast. DIAL@800k - B@800k at 50k, DPM++ 2M with 257 NFE, sampling seed 0. |delta| < 0.05: the
  models are indistinguishable; otherwise the sign and size are reported. No interval by the morning;
  the noise scale is the spread of seeds at 50k (0.02-0.04).
- Second contrast. DIAL@800k - VP-lambda@400k under the same conditions and threshold.
- 10k curves. Differences between models below 0.2 are not interpreted. Within one model, across NFE and
  solver, only the shape of the curve is described.
- AFHQv2 (10k, 257 NFE). Sign only: whether DIAL - VP-lambda keeps, under both solvers, the sign it has at
  35 NFE (DIAL better). The size is not claimed.
- A negative result is reported. One training run per model, one sampling seed.

Cells. Window [-10; 9.115429865459795], Tweedie, sampling seed 0. Trajectory calls 256, 128, 64, 14
(NFE 257, 129, 65, 15); 35 NFE already exists. Within a curve the order is fixed: from 257 down, DPM++ before
EDM Heun. Lane dial (RTX): DIAL@800k curve, 8 cells of 10k, then DIAL@800k 50k DPM++-257. Lane b (RTX): the
same for B@800k. Lane a (H200): VP-lambda@400k curve, VP-lambda@400k 50k DPM++-257, then AFHQv2 DIAL@200k and
VP-lambda@200k at 10k and 257 NFE, both solvers. 31 cells.

Configs are those of the 50k matched35 cells of each model, built at 34 calls by the same functions
(celeba_fox800_lo_eval: lo.native_config; B and AFHQv2: gpu2_queue_eval.config_of; CelebA VP-lambda:
gpu2_a50k_grids.config_of); then only eval.sampling_num_scales and eval.num_samples change. The 50k cells
keep the Inception features for the bootstrap, as the matched35 cells do.

Preflight checks the grid of every cell, then for every checkpoint source: weights, a batch of two through
the real sampler at every (solver, calls) of the lane, and 512 images through evaluate_checkpoint on the
first cell. A failed check blocks only that source's cells; a failed cell does not stop the next one.
The card type goes to preflight.json and to every result.

Run through run_nfe_limit.sh from a separate worktree; NFE_LIMIT_LANE=dial|b|a.
`nfe_limit_eval.py combine DIAL_LOGS B_LOGS A_LOGS` prints the contrasts of the rule from three summaries.
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
DPM, HEUN = 'ode_dpmpp_2m', 'ode_edm_heun'
SHORT = {DPM: 'dpm', HEUN: 'heun'}
LANES = ('dial', 'b', 'a')
SEED = 0
TEN_K, FIFTY_K = 10000, 50000
TOP = 256  # Trajectory calls of the limit; NFE 257.
CURVE = [(calls, method) for calls in (TOP, 128, 64, 14) for method in (DPM, HEUN)]
CONVERGED = 0.05  # Solver gap at 257 NFE, 10k.
INDISTINGUISHABLE = 0.05  # Model contrast at 50k, DPM++-257.
CURVE_THRESHOLD = 0.2  # Model differences at 10k.
LANE_CARD = {'dial': 'RTX', 'b': 'RTX', 'a': 'H200'}
MODELS = {
    'DIAL': dict(source='celeba_fox', ckpt=160, dataset='CELEBA', lane='dial', label='DIAL@800k'),
    'B': dict(source='celeba_b', ckpt=160, dataset='CELEBA', lane='b', label='B@800k'),
    'VPL': dict(source='celeba_a_first', ckpt=grids.A_FIRST, dataset='CELEBA', lane='a', label='VP-lambda@400k'),
    'AFHQ_DIAL': dict(source='afhq_fox', ckpt=4, dataset='AFHQV2', lane='a', label='AFHQv2 DIAL@200k'),
    'AFHQ_VPL': dict(source='afhq_a', ckpt=4, dataset='AFHQV2', lane='a', label='AFHQv2 VP-lambda@200k'),
}
write_json = q.write_json


def cell(model, calls, method, count):
    name = '{}_nfe{}_{}_{}k'.format(model.lower(), calls + 1, SHORT[method], count // 1000)
    info = MODELS[model]
    return name, dict(model=model, source=info['source'], group=info['source'], ckpt=info['ckpt'],
                      dataset=info['dataset'], lane=info['lane'], method=method, seed=SEED, count=count, calls=calls)


def curve(model):
    return [cell(model, calls, method, TEN_K) for calls, method in CURVE]


# The order is fixed in advance and does not depend on any observed FID.
CELLS = dict(curve('DIAL') + [cell('DIAL', TOP, DPM, FIFTY_K)] +
             curve('B') + [cell('B', TOP, DPM, FIFTY_K)] +
             curve('VPL') + [cell('VPL', TOP, DPM, FIFTY_K)] +
             [cell(model, TOP, method, TEN_K) for method in (DPM, HEUN) for model in ('AFHQ_DIAL', 'AFHQ_VPL')])


def lane():
    value = os.environ.get('NFE_LIMIT_LANE', '')
    if value not in LANES:
        raise ValueError('NFE_LIMIT_LANE must be dial, b or a, not {!r}'.format(value))
    return value


def lane_cells(which):
    return [name for name, spec in CELLS.items() if spec['lane'] == which]


def resolve(name):
    if name not in CELLS:
        raise ValueError('Unknown NFE limit cell: ' + name)
    spec = dict(CELLS[name])
    spec['work'] = spec['source']
    spec['step'] = q.SOURCES[spec['source']]['freq'] * spec['ckpt'] + 1
    return spec


def matched35_config(spec):
    """The config of the model's 50k matched35 cell with this solver: 34 calls, 50k, sampling seed 0."""
    base = dict(spec, kind='m35', calls=q.M35_CALLS, count=FIFTY_K)
    if spec['source'] == 'celeba_fox':  # As celeba_fox800_lo_eval.
        return lo.native_config(dict(calls=q.M35_CALLS, count=lo.COUNT, model='fox', method=spec['method'],
                                     seed=spec['seed'], ckpt=spec['ckpt'], window=q.WINDOW))
    if spec['source'] in ('celeba_a_first', 'celeba_a'):
        return grids.config_of(base)
    return q.config_of(base)


def config_of(spec):
    config = matched35_config(spec)
    config.eval.sampling_num_scales = spec['calls']
    config.eval.num_samples = spec['count']
    return config


def get_config(name):
    return config_of(resolve(name))


def budget_smoke(config, model, source, which):
    """A batch of two through the real sampler at every (solver, calls) of this source in the lane."""
    import torch
    from ode_sampling import get_vp_ode_sampler
    from run_lib import get_sde
    budgets = sorted({(CELLS[n]['method'], CELLS[n]['calls']) for n in lane_cells(which)
                      if CELLS[n]['source'] == source})
    shape = (2, config.data.num_channels, config.data.image_size, config.data.image_size)
    for method, calls in budgets:
        torch.manual_seed(0)
        sde, eps = get_sde(config, n_override=calls)
        outputs, used = get_vp_ode_sampler(sde, shape, lambda x: x, method, eps, config.device)(model)
        if used != calls + 1 or not all(bool(torch.isfinite(x).all()) for x in outputs.values()):
            raise RuntimeError('Failed budget smoke: {} at {} calls'.format(method, calls))
    return ['{}_{}'.format(SHORT[m], c + 1) for m, c in budgets]


def preflight():
    """Grids of all cells, then every source of the lane through its real path. Returns 1 on failure."""
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
    card = torch.cuda.get_device_name(0)
    if LANE_CARD[which] not in card:
        raise RuntimeError('Lane {} expects an {} card, found {}'.format(which, LANE_CARD[which], card))
    try:  # Several models share this process; Inception must not take the whole card.
        import tensorflow as tf
        for device in tf.config.list_physical_devices('GPU'):
            tf.config.experimental.set_memory_growth(device, True)
    except (RuntimeError, ValueError, AttributeError, TypeError):
        pass
    lo.check_inception()
    from artifact_utils import _environment_manifest, _source_fingerprint
    names = lane_cells(which)
    stats = {}
    for name in names:
        dataset = CELLS[name]['dataset']
        if dataset not in stats:
            config = get_config(name)
            if dataset == 'AFHQV2':
                afhq.check_stats(config)
            stats[dataset] = evaluation.dataset_stats_fingerprint(config)
    if stats.get('CELEBA', q.REFERENCE_SHA) != q.REFERENCE_SHA:
        raise RuntimeError('Unexpected CelebA statistics.')
    firsts = {}
    for name in names:
        firsts.setdefault(CELLS[name]['group'], name)
    identities, failures, smokes, budgets = {}, {}, {}, {}
    for group, first in firsts.items():
        try:
            for name in names:  # Grids before weights: a wrong window costs no GPU time.
                if CELLS[name]['group'] == group:
                    spec = resolve(name)
                    config = config_of(spec)
                    sde, eps = get_sde(config, n_override=spec['calls'])
                    followup.check_cell_grid(config, sde, eps, spec)
            spec = resolve(first)
            config = config_of(spec)
            sde, eps = get_sde(config, n_override=spec['calls'])
            model, identities[group] = q.load_model(config, spec)
            followup.smoke(config, model, spec['method'], eps)
            budgets[group] = budget_smoke(config, model, spec['source'], which)
            smokes[group] = q.endpoint_smoke(config, model, spec, LOGS / 'preflight_endpoint')
            del model
            torch.cuda.empty_cache()
            print('Preflight', group, identities[group], flush=True)
        except Exception as error:  # A truncated checkpoint raises whatever torch.load does.
            failures[group] = repr(error)
            print('Preflight failed: {}: {!r}'.format(group, error), flush=True)
    write_json(path, dict(status='failed' if failures else 'passed', lane=which, gpu_name=card, gpu_arch=arch,
        checkpoints=identities, failures=failures, budget_smoke=budgets, endpoint_smoke_fid_512=smokes,
        dataset_stats_id=stats, source_sha256=_source_fingerprint(str(ROOT)), environment=_environment_manifest(),
        started_unix=time.time()))
    print('Preflight', 'failed: {}'.format(failures) if failures else 'passed', flush=True)
    return 1 if failures else 0


def eval_folder(name):
    return 'eval_nfelimit_' + name


def result_path(name):
    return LOGS / (name + '_result.json')


def failure_path(name):
    return LOGS / (name + '_failed.json')


def run(name):
    spec = resolve(name)
    began, clock = time.time(), time.monotonic()
    result = subprocess.run([sys.executable, '-u', 'main.py', '--mode=eval', '--config=nfe_limit_eval.py:' + name,
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
    with np.load(str(rounds / 'statistics_0.npz'), allow_pickle=False) as data:
        if q.FEATURES not in data.files or data[q.FEATURES].shape != (q.BATCH, 2048):
            raise RuntimeError('Shards lack the bootstrap features.')
    timing = json.loads((LOGS / (name + '_timing.json')).read_text())
    write_json(result_path(name), dict(spec, cell=name, label=MODELS[spec['model']]['label'],
        checkpoint_id=checkpoint_id, lambda_window=[q.LOGSNR_MIN, q.WINDOWS[q.WINDOW]], nfe=nfe,
        num_samples=spec['count'], sampling_seed=spec['seed'], gpu_name=pre['gpu_name'],
        elapsed_seconds=timing['elapsed_seconds'], **metrics))
    write_summary()
    print('NFE limit {}: FID after Tweedie {:.4f}, {:.1f} min'.format(
        name, metrics['fid_tweedie_xt'], timing['elapsed_seconds'] / 60), flush=True)


def fid(cells, model, calls, method, count):
    value = cells.get(cell(model, calls, method, count)[0])
    return None if value is None else value['fid_tweedie_xt']


def solver_gaps(cells):
    """Section 2: |DPM++ - EDM Heun| at 257 NFE, 10k, per model with both cells in."""
    out = {}
    for model in MODELS:
        dpm, heun = fid(cells, model, TOP, DPM, TEN_K), fid(cells, model, TOP, HEUN, TEN_K)
        if dpm is not None and heun is not None:
            gap = dpm - heun
            out[model] = dict(dpm=dpm, heun=heun, dpm_minus_heun=gap, converged=abs(gap) <= CONVERGED + 1e-9)
    return out


def write_summary(failed=()):
    which = lane()
    names = lane_cells(which)
    cells = {name: json.loads(result_path(name).read_text()) for name in names if result_path(name).exists()}
    status = 'completed' if len(cells) == len(names) else 'failed' if failed else 'running'
    write_json(LOGS / 'summary.json', dict(status=status, lane=which, completed_cells=len(cells),
        total_cells=len(names), failed_cells=list(failed), cells=list(cells.values()),
        solver_gap_257=solver_gaps(cells),
        rule='nfe_limit_eval.py docstring (16 November): a discretisation control, rows and selected steps stay',
        scope='One training run per model, one sampling seed; no intervals here'))
    with (LOGS / 'summary.csv').open('w', newline='') as stream:
        writer = csv.writer(stream)
        writer.writerow(('cell', 'model', 'source', 'step', 'method', 'sampling_seed', 'num_samples', 'nfe',
                         'fid_tweedie', 'fid_x_t', 'kid_tweedie', 'is_tweedie', 'gpu', 'minutes'))
        for name, value in cells.items():
            writer.writerow((name, value['model'], value['source'], value['step'], value['method'],
                             value['sampling_seed'], value['num_samples'], value['nfe'], value['fid_tweedie_xt'],
                             value['fid_x_t'], value['kid_tweedie_xt'], value['IS_tweedie_xt'], value['gpu_name'],
                             round(value['elapsed_seconds'] / 60, 1)))


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
            print('NFE limit cell failed: {}: {!r}'.format(name, error), flush=True)
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
    print('NFE limit lane finished, failed cells: {}'.format(failed or 'none'), flush=True)
    return 1 if failed else 0


def contrast(cells, first, second, calls, method, count, threshold):
    a, b = fid(cells, first, calls, method, count), fid(cells, second, calls, method, count)
    if a is None or b is None:
        return dict(first=first, second=second, nfe=calls + 1, method=method, num_samples=count, status='missing',
                    first_fid=a, second_fid=b)
    delta = a - b
    return dict(first=first, second=second, nfe=calls + 1, method=method, num_samples=count, first_fid=a,
                second_fid=b, difference=delta, threshold=threshold,
                status='below threshold' if abs(delta) < threshold else
                ('{} better'.format(first) if delta < 0 else '{} better'.format(second)))


def combine(directories):
    """The contrasts of section 2 from the summaries of the three lanes; reads only, prints JSON."""
    from pathlib import Path
    cells, cards, statuses = {}, {}, {}
    for directory in directories:
        path = Path(directory) / 'summary.json'
        if not path.exists():
            continue
        summary = json.loads(path.read_text())
        statuses[summary['lane']] = '{} ({}/{})'.format(summary['status'], summary['completed_cells'],
                                                       summary['total_cells'])
        for value in summary['cells']:
            cells[value['cell']] = value
            cards.setdefault(summary['lane'], value['gpu_name'])
    out = dict(lanes=statuses, cards=cards, solver_gap_257=solver_gaps(cells), rule='nfe_limit_eval.py docstring',
               main_contrast=contrast(cells, 'DIAL', 'B', TOP, DPM, FIFTY_K, INDISTINGUISHABLE),
               second_contrast=contrast(cells, 'DIAL', 'VPL', TOP, DPM, FIFTY_K, INDISTINGUISHABLE),
               curves_10k=[contrast(cells, 'DIAL', other, calls, method, TEN_K, CURVE_THRESHOLD)
                           for other in ('B', 'VPL') for calls, method in CURVE],
               afhq_sign_257=[dict(contrast(cells, 'AFHQ_DIAL', 'AFHQ_VPL', TOP, method, TEN_K, 0.0),
                                   sign_at_35='DIAL better under both solvers (10k: -0.16 / -0.23)')
                              for method in (DPM, HEUN)])
    for item in out['afhq_sign_257']:  # Sign only: no threshold, no size claimed.
        if 'difference' in item:
            item['status'] = 'sign kept' if item['difference'] < 0 else 'sign changed'
    out['models'] = {key: value['label'] for key, value in MODELS.items()}
    print(json.dumps(out, indent=2))
    return 0


if __name__ == '__main__':
    mode = sys.argv[1] if len(sys.argv) > 1 else ''
    if mode == 'preflight':
        sys.exit(preflight())
    elif mode == 'queue':
        sys.exit(queue())
    elif mode == 'plan':
        for cell_label, cell_spec in CELLS.items():
            print(cell_label, cell_spec)
    elif mode == 'combine':
        sys.exit(combine(sys.argv[2:]))
    else:
        raise SystemExit('Usage: NFE_LIMIT_LANE=dial|b|a nfe_limit_eval.py preflight|queue|plan; '
                         'nfe_limit_eval.py combine DIAL_LOGS B_LOGS A_LOGS')
