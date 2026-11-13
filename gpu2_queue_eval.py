"""One evaluation queue for RTX GPU 2 after AFHQv2 A reaches 400k; cells in a fixed order.

1. AFHQv2 curve, AFHQV2_STEP_RULE.md (addendum of 14 November): A at 250k-400k and VP at
   250k and 300k. Stage 0 protocol: DPM++ 2M, 64 trajectory calls plus Tweedie, 10k,
   sampling seed 0, window [-10; 9.115429865459795], statistics afhqv2_64_cd017afab009.
2. CelebA native FOX (DIAL) at 700k and 800k on the same protocol, configured exactly as
   celeba_lo_curve_eval.py of 4744d55 does: stage0.native_config('fox') and set_protocol.
3. CelebA B at 800k, 50k matched35: 34 trajectory calls plus Tweedie, DPM++ 2M and EDM Heun,
   sampling seeds 0 and 1, the same window, configured as the A cells of
   celeba_50k_lo_eval.py, so the numbers sit next to that table.
4. AFHQv2 FOX at 200k, 50k matched35, the same protocol on the AFHQv2 statistics.
5. AFHQv2 A at its selected step, 50k matched35. The queue selects the step itself before
   these cells: candidates 100k and 200k from the first curve (previous/), 250k-400k from
   part 1; the minimum FID, then the latest candidate within the minimum plus 0.2. The
   selection and every input go to logs/a_selection.json and summary.json.

Preflight runs every kind of cell through its real path: grid, weights, a short real run,
and one batch of 512 images through evaluate_checkpoint for the AFHQv2 curve, the CelebA
curve, matched35 on CelebA and matched35 on AFHQv2 with both solvers. A failed check
blocks only the cells of its group; a failed cell does not stop the next one.

Run through run_gpu2_queue_rtx.sh from a separate worktree.
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
from celeba_stage0_eval import LOGSNR_MIN, REFERENCE_SHA, WINDOWS, set_protocol, write_json

ROOT = Path(__file__).resolve().parent
LOGS = ROOT / 'logs'
INPUTS = ROOT / 'workdirs/_training_inputs'
PREVIOUS = ROOT / 'previous'  # Logs of the first AFHQv2 curve, mounted read-only.
WINDOW = 'lo'
ATTEMPTS = 2
TIE = 0.2  # AFHQV2_STEP_RULE.md, point 2.
CURVE_CALLS = stage0.BUDGET
M35_CALLS = lo.CALLS
M35_COUNT = lo.COUNT
METHODS = lo.METHODS
BATCH = stage0.BATCH
FEATURES = lo.FEATURES
SELECTION = 'a_selection.json'

# Eval workdirs, each with the checkpoints of one run mounted read-only.
SOURCES = {
    'afhq_a_first': dict(dataset='AFHQV2', arm='a', manifest='afhq_a', freq=afhq.SNAPSHOT_FREQ),
    'afhq_a': dict(dataset='AFHQV2', arm='a', manifest='afhq_a', freq=afhq.SNAPSHOT_FREQ),
    'afhq_vp': dict(dataset='AFHQV2', arm='vp', manifest='afhq_vp', freq=afhq.SNAPSHOT_FREQ),
    'afhq_fox': dict(dataset='AFHQV2', arm='fox', manifest='afhq_fox', freq=afhq.SNAPSHOT_FREQ),
    'celeba_fox': dict(dataset='CELEBA', arm='fox', manifest=None, freq=5000),
    'celeba_b': dict(dataset='CELEBA', arm='B', manifest='celeba_b', freq=5000),
}
LEGACY = 'legacy-unknown'  # What reports carry for checkpoints without a training fingerprint.
KNOWN_IDS = {('celeba_fox', 160): 'legacy-sha256-7874545fa077e31812c4ca6cddf196d6102ec91e4ce88423a6ef8f43bc35b148'}
A_CANDIDATES = (('previous', 'a_c2_lo', 2), ('previous', 'a_c4_lo', 4), ('queue', 'afhq_a_c5_lo', 5),
                ('queue', 'afhq_a_c6_lo', 6), ('queue', 'afhq_a_c7_lo', 7), ('queue', 'afhq_a_c8_lo', 8))


def curve(source, ckpt):
    return '{}_c{}_lo'.format(source, ckpt), dict(kind='curve', source=source, group=source + '_curve', ckpt=ckpt,
                                                  method=stage0.METHOD, seed=stage0.SEED, count=stage0.COUNT,
                                                  calls=CURVE_CALLS)


def m35(source, ckpt, label=None):
    return [('{}_c{}_m35_{}_seed{}'.format(label or source, ckpt if ckpt else 'best', method, seed),
             dict(kind='m35', source=source, group=(label or source) + '_m35', ckpt=ckpt, method=method, seed=seed,
                  count=M35_COUNT, calls=M35_CALLS))
            for seed in (0, 1) for method in METHODS]


# The order is fixed in advance and does not depend on any observed FID.
CELLS = dict([curve('afhq_a', c) for c in (5, 6, 7, 8)] + [curve('afhq_vp', c) for c in (5, 6)] +
             [curve('celeba_fox', c) for c in (140, 160)] + m35('celeba_b', 160) + m35('afhq_fox', 4) +
             m35('afhq_a', None, 'afhq_a'))
for _name, _spec in CELLS.items():
    _spec['dataset'] = SOURCES[_spec['source']]['dataset']


def select(fids):
    """{step: FID}: the latest step within TIE of the minimum."""
    best = min(fids.values())
    return max(step for step, fid in fids.items() if fid <= best + TIE + 1e-9)  # The rule includes equality.


def a_selection():
    """Point 2 of the rule for AFHQv2 A over 100k-400k; None until every candidate is in."""
    fids, inputs = {}, []
    for origin, name, ckpt in A_CANDIDATES:
        path = (PREVIOUS if origin == 'previous' else LOGS) / (name + '_result.json')
        value = json.loads(path.read_text())['fid_tweedie_xt'] if path.exists() else None
        inputs.append(dict(cell=name, origin=origin, step=afhq.SNAPSHOT_FREQ * ckpt, fid_tweedie_xt=value))
        if value is not None:
            fids[afhq.SNAPSHOT_FREQ * ckpt] = value
    result = dict(rule='latest candidate with FID <= minimum + {} (AFHQV2_STEP_RULE.md, 14 November)'.format(TIE),
                  inputs=inputs, complete=len(fids) == len(A_CANDIDATES))
    if result['complete']:
        step = select(fids)
        result.update(minimum=min(fids.values()), selected_step=step, selected_fid=fids[step],
                      selected_ckpt=step // afhq.SNAPSHOT_FREQ)
    return result


def resolve(name):
    """The spec of a cell, with the selected checkpoint filled in for the best A."""
    if name not in CELLS:
        raise ValueError('Unknown GPU 2 queue cell: ' + name)
    spec = dict(CELLS[name])
    if spec['ckpt'] is None:
        path = LOGS / SELECTION
        if not path.exists():
            raise ValueError('No step selected for AFHQv2 A yet.')
        selected = json.loads(path.read_text())
        if not selected.get('complete'):
            raise ValueError('The AFHQv2 A selection is incomplete.')
        spec['ckpt'] = int(selected['selected_ckpt'])
        spec['source'] = 'afhq_a' if spec['ckpt'] >= 4 else 'afhq_a_first'  # Both runs hold 200k.
    spec['work'] = spec['source']
    spec['step'] = SOURCES[spec['source']]['freq'] * spec['ckpt'] + 1
    return spec


def manifest(source):
    key = SOURCES[source]['manifest']
    return json.loads((INPUTS / (key + '.json')).read_text()) if key else None


def training_protocol(source):
    saved = manifest(source)
    return saved['training_protocol_sha256'] if saved else LEGACY


def base_config(spec):
    from ml_collections import ConfigDict
    from configs.factorial.celeba_b_vp_pfox import get_config as get_b
    source = SOURCES[spec['source']]
    if spec['source'] == 'celeba_fox':
        return stage0.native_config('fox')  # As celeba_lo_curve_eval.py of 4744d55.
    config = ConfigDict(manifest(spec['source'])['config'])
    if source['dataset'] == 'AFHQV2':
        if (config.seed, config.data.dataset, config.data.centered) != (afhq.SEED, 'AFHQV2', True):
            raise ValueError('Unexpected AFHQv2 run behind ' + spec['source'])
        config.eval = afhq.base_config(source['arm']).eval  # Manifests exclude evaluation settings.
    else:
        if (config.training.factorial_arm, config.seed) != ('B_vp_pfox', 42):
            raise ValueError('Unexpected run behind CelebA B')
        config.eval = get_b().eval
    return config


def config_of(spec):
    config = base_config(spec)
    set_protocol(config, WINDOW, spec['ckpt'])
    if spec['kind'] == 'm35':  # As celeba_50k_lo_eval.a_config.
        config.sampling.method = spec['method']
        config.sampling.time_grid = 'karras' if spec['method'] == 'ode_edm_heun' else 'uniform_logsnr'
        config.eval.sampling_num_scales = spec['calls']
        config.eval.sampling_seed = spec['seed']
        config.eval.num_samples = spec['count']
    return config


def get_config(name):
    return config_of(resolve(name))


def checkpoint_path(spec):
    return ROOT / 'workdirs' / spec['work'] / 'checkpoints/checkpoint_{}.pth'.format(spec['ckpt'])


def check_identity(spec, step, protocol, checkpoint_id=None):
    if step != spec['step']:
        raise RuntimeError('Expected step {}, found {}'.format(spec['step'], step))
    expected = training_protocol(spec['source'])
    if (protocol or LEGACY) != expected:
        raise RuntimeError('The checkpoint does not belong to the run of ' + spec['source'])
    known = KNOWN_IDS.get((spec['source'], spec['ckpt']))
    if known and checkpoint_id is not None and checkpoint_id != known:
        raise RuntimeError('Unexpected checkpoint id: ' + checkpoint_id)


def load_model(config, spec):
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
    check_identity(spec, int(state['step']), state.get('training_protocol_sha256'), state['checkpoint_id'])
    ema.copy_to(model.parameters())
    model.eval()
    return model, dict(checkpoint_id=state['checkpoint_id'], step=int(state['step']))


def endpoint_smoke(config, model, spec, directory):
    """One batch through evaluate_checkpoint: sampler, Inception, statistics, FID."""
    import shutil
    import numpy as np
    import datasets
    import endpoint_denoise_eval
    import evaluation
    import sampling
    from run_lib import get_sde
    shutil.rmtree(str(directory), ignore_errors=True)
    config = config.copy_and_resolve_references()
    config.eval.num_samples = config.eval.batch_size
    sde, eps = get_sde(config, n_override=spec['calls'])
    shape = (config.eval.batch_size, config.data.num_channels, config.data.image_size, config.data.image_size)
    sampling_fn = sampling.get_sampling_fn(config, sde, shape, datasets.get_data_inverse_scaler(config), eps)
    identity = dict(protocol_sha256=np.asarray('preflight'), checkpoint_id=np.asarray('preflight'),
                    checkpoint_step=np.asarray(spec['step']), training_protocol_sha256=np.asarray('preflight'))
    endpoint_denoise_eval.evaluate_checkpoint(
        config, str(directory), spec['ckpt'], model, sampling_fn, evaluation.get_inception_model(), 'preflight',
        'preflight', spec['step'], 'preflight', identity)
    with np.load(str(Path(directory) / 'report_{}.npz'.format(spec['ckpt'])), allow_pickle=False) as data:
        value = float(data['fid_tweedie_xt'])
    shutil.rmtree(str(directory), ignore_errors=True)
    if not math.isfinite(value):
        raise RuntimeError('Non-finite FID in the endpoint smoke.')
    return value


# One check per group: (first cell of the group, solvers to pass through evaluate_checkpoint).
SMOKES = {'afhq_a_curve': ('afhq_a_c5_lo', (stage0.METHOD,)), 'afhq_vp_curve': ('afhq_vp_c5_lo', ()),
          'celeba_fox_curve': ('celeba_fox_c140_lo', (stage0.METHOD,)),
          'celeba_b_m35': ('celeba_b_c160_m35_ode_dpmpp_2m_seed0', ('ode_dpmpp_2m',)),
          'afhq_fox_m35': ('afhq_fox_c4_m35_ode_dpmpp_2m_seed0', METHODS)}


def preflight():
    """Every kind of cell through its real path. Returns 1 if a group failed."""
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
    try:  # Several models share this process; Inception must not take the whole card.
        import tensorflow as tf
        for device in tf.config.list_physical_devices('GPU'):
            tf.config.experimental.set_memory_growth(device, True)
    except (RuntimeError, ValueError, AttributeError, TypeError):
        pass
    lo.check_inception()
    from artifact_utils import _environment_manifest, _source_fingerprint
    afhq_config = get_config('afhq_a_c5_lo')
    afhq.check_stats(afhq_config)
    stats = dict(AFHQV2=evaluation.dataset_stats_fingerprint(afhq_config),
                 CELEBA=evaluation.dataset_stats_fingerprint(get_config('celeba_fox_c140_lo')))
    if stats['CELEBA'] != REFERENCE_SHA:
        raise RuntimeError('Unexpected CelebA statistics.')
    identities, failures, smokes = {}, {}, {}
    for group, (name, methods) in SMOKES.items():
        try:
            spec = resolve(name)
            config = config_of(spec)
            sde, eps = get_sde(config, n_override=spec['calls'])
            followup.check_cell_grid(config, sde, eps, spec)
            model, identities[group] = load_model(config, spec)
            followup.smoke(config, model, spec['method'], eps)
            for method in methods:
                other = dict(spec, method=method)
                smoke_config = config_of(other)
                smokes['{}_{}'.format(group, method)] = endpoint_smoke(smoke_config, model, other,
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


def eval_folder(name):
    return 'eval_gpu2q_' + name


def result_path(name):
    return LOGS / (name + '_result.json')


def failure_path(name):
    return LOGS / (name + '_failed.json')


def run(name):
    spec = resolve(name)
    began, clock = time.time(), time.monotonic()
    result = subprocess.run([sys.executable, '-u', 'main.py', '--mode=eval', '--config=gpu2_queue_eval.py:' + name,
                             '--workdir=workdirs/' + spec['work'], '--eval_folder=' + eval_folder(name)],
                            cwd=str(ROOT))
    write_json(LOGS / (name + '_timing.json'), dict(started_unix=began, finished_unix=time.time(),
        elapsed_seconds=time.monotonic() - clock, returncode=result.returncode))
    result.check_returncode()


def report(name):
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
                        training_protocol_sha256=training_protocol(spec['source']))
        for key, value in expected.items():
            if data[key].item() != value:
                raise RuntimeError('Unexpected report identity: ' + key)
        checkpoint_id = str(data['checkpoint_id'].item())
        metrics = {metric + '_' + variant: float(data[metric + '_' + variant])
                   for metric in ('IS', 'fid', 'kid') for variant in stage0.VARIANTS}
    check_identity(spec, spec['step'], training_protocol(spec['source']), checkpoint_id)
    if not all(math.isfinite(value) for value in metrics.values()):
        raise RuntimeError('Non-finite metric.')
    rounds = directory / 'ckpt_{}'.format(spec['ckpt'])
    if len(list(rounds.glob('statistics_*.npz'))) != (spec['count'] + BATCH - 1) // BATCH:
        raise RuntimeError('Unexpected number of statistics shards.')
    if spec['kind'] == 'm35':
        with np.load(str(rounds / 'statistics_0.npz'), allow_pickle=False) as data:
            if FEATURES not in data.files or data[FEATURES].shape != (BATCH, 2048):
                raise RuntimeError('Shards lack the bootstrap features.')
    timing = json.loads((LOGS / (name + '_timing.json')).read_text())
    write_json(result_path(name), dict(spec, cell=name, checkpoint_id=checkpoint_id,
        lambda_window=[LOGSNR_MIN, WINDOWS[WINDOW]], nfe=nfe, num_samples=spec['count'],
        sampling_seed=spec['seed'], elapsed_seconds=timing['elapsed_seconds'], **metrics))
    write_summary()
    print('GPU 2 {}: FID after Tweedie {:.4f}, {:.1f} min'.format(
        name, metrics['fid_tweedie_xt'], timing['elapsed_seconds'] / 60), flush=True)


def write_summary(failed=()):
    cells = {name: json.loads(result_path(name).read_text()) for name in CELLS if result_path(name).exists()}
    selection_path = LOGS / SELECTION
    status = 'completed' if len(cells) == len(CELLS) else 'failed' if failed else 'running'
    write_json(LOGS / 'summary.json', dict(status=status, completed_cells=len(cells), total_cells=len(CELLS),
        failed_cells=list(failed), cells=list(cells.values()),
        afhq_a_selection=json.loads(selection_path.read_text()) if selection_path.exists() else a_selection(),
        afhq_fox_selection=dict(step=200000, note='Fixed on 13 November by the full FOX curve; not revised'),
        rule='AFHQV2_STEP_RULE.md and CELEBA_A800K_STEP_RULE.md',
        scope='One training run per model; 10k curves are a screen, matched35 at 50k has no intervals here'))
    with (LOGS / 'summary.csv').open('w', newline='') as stream:
        writer = csv.writer(stream)
        writer.writerow(('cell', 'dataset', 'kind', 'source', 'step', 'method', 'sampling_seed', 'num_samples', 'nfe',
                         'fid_tweedie', 'fid_x_t', 'kid_tweedie', 'is_tweedie'))
        for name, value in cells.items():
            writer.writerow((name, value['dataset'], value['kind'], value['source'], value['step'], value['method'],
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
            print('GPU 2 cell failed: {}: {!r}'.format(name, error), flush=True)
    return False


def queue():
    """The cells in order. A failed cell does not stop the next; a failed preflight group blocks its cells."""
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
        if spec['ckpt'] is None and not ((LOGS / SELECTION).exists() and
                                         json.loads((LOGS / SELECTION).read_text())['complete']):
            selected = a_selection()  # A complete selection is never recomputed, so a rerun keeps its step.
            write_json(LOGS / SELECTION, selected)
            print('AFHQv2 A selection: {}'.format(selected), flush=True)
        if spec['ckpt'] is None and not json.loads((LOGS / SELECTION).read_text())['complete']:
            failed.append(name)
            print('AFHQv2 A selection is incomplete: ' + name, flush=True)
            continue
        if not evaluate(name):
            failed.append(name)
    write_summary(failed)
    print('GPU 2 queue finished, failed cells: {}'.format(failed or 'none'), flush=True)
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
        raise SystemExit('Usage: gpu2_queue_eval.py preflight|queue|plan')
