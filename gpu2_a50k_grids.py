"""RTX GPU 2: sample grids for the figure, then CelebA A at its selected step, 50k matched35.

Phase 1, at once (modes preflight, then grids):
  Grids of the models selected by the step rules. CelebA: VP 200k (table 4), B 800k, DIAL 800k.
  AFHQv2: VP 250k, A 250k, DIAL 200k. Protocol of the 50k matched35 cells: DPM++ 2M, 34
  trajectory calls plus Tweedie, window [-10; 9.115429865459795], sampling seed 0, the same
  config as the seed 0 DPM++ 50k cell of each model. A grid is round 0 of that cell: batch
  512, the same round seed, so every model starts from the same white noise; the first 64
  images are kept. Both outputs are saved: x_t, the last ODE state without Tweedie (the ODE
  sampler has no separate x_mean), and Tweedie(x_t). Where that 50k cell is finished, its
  first batch is mounted read-only and the grid is compared with it.
  Preflight passes 512 images through evaluate_checkpoint on A weights of the 800k run (500k,
  an H200 snapshot, on RTX) with both solvers.
Phase 2, after celeba-a-curve-h200 ends by itself with code 0 (modes select, preflight-a, a):
  The A step from the curve summary, CELEBA_A800K_STEP_RULE.md, checked against the rule; if
  the curve did not finish cleanly, nothing runs. Then the A grid, then 50k matched35 for A:
  DPM++ 2M and EDM Heun, sampling seeds 0 and 1, configured as the A cells of table 4
  (celeba_50k_lo_eval.py) and celeba_b_c160_m35 of the GPU 2 queue. If 400k is selected, its
  50k is in table 4 (celeba_50k_lo_20261107/gpu1): not recomputed, the summary copies it.

Outputs: grids/ (PNG 8x8 per model and output, npz uint8, one sheet per dataset), logs/.
Run through run_gpu2_a50k_grids_rtx.sh from a separate worktree.
"""
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time

import celeba_50k_lo_eval as lo
import celeba_followup_eval as followup
import gpu2_queue_eval as q

ROOT = q.ROOT
LOGS = q.LOGS
GRIDS_DIR = ROOT / 'grids'
ACURVE = ROOT / 'acurve'  # Logs of the CelebA A curve, mounted read-only.
TABLE4 = ROOT / 'table4'  # Logs of celeba_50k_lo_20261107/gpu1, mounted read-only.
REFS = ROOT / 'refs'  # Round 0 of finished seed 0 DPM++ 50k cells, mounted read-only.
SELECTION = 'celeba_a_selection.json'
ATTEMPTS = q.ATTEMPTS
write_json = q.write_json

METHOD = 'ode_dpmpp_2m'
SEED = 0
SHOW = 64  # Images kept per model.
NROW = 8
SHEET_COLUMNS = 8
PAD = 2
VARIANTS = ('x_t', 'tweedie_xt')
A_FREQ = 5000
A_CANDIDATES = (80, 100, 120, 140, 160)  # 400k to 800k, CELEBA_A800K_STEP_RULE.md.
A_FIRST = 80  # 400k comes from the RTX run of table 4.
A_SMOKE = 100  # Phase 1 preflight: an H200 snapshot of the 800k run.

q.SOURCES.setdefault('celeba_vp', dict(dataset='CELEBA', arm='vp', manifest=None, freq=5000))
q.SOURCES.setdefault('celeba_a_first', dict(dataset='CELEBA', arm='A', manifest='celeba_a_first', freq=A_FREQ))
q.SOURCES.setdefault('celeba_a', dict(dataset='CELEBA', arm='A', manifest='celeba_a', freq=A_FREQ))
q.KNOWN_IDS.setdefault(('celeba_vp', lo.MODELS['vp']['ckpt']), lo.MODELS['vp']['checkpoint_id'])
q.KNOWN_IDS.setdefault(('celeba_a_first', A_FIRST), lo.MODELS['A']['checkpoint_id'])

GRIDS = {
    'celeba_vp': dict(dataset='CELEBA', label='VP', source='celeba_vp', ckpt=lo.MODELS['vp']['ckpt']),
    'celeba_a': dict(dataset='CELEBA', label='A', source=None, ckpt=None),
    'celeba_b': dict(dataset='CELEBA', label='B', source='celeba_b', ckpt=160),
    'celeba_dial': dict(dataset='CELEBA', label='DIAL', source='celeba_fox', ckpt=160),
    'afhq_vp': dict(dataset='AFHQV2', label='VP', source='afhq_vp', ckpt=5),
    'afhq_a': dict(dataset='AFHQV2', label='A', source='afhq_a', ckpt=5),
    'afhq_dial': dict(dataset='AFHQV2', label='DIAL', source='afhq_fox', ckpt=4),
}
KNOWN = tuple(name for name, grid in GRIDS.items() if grid['ckpt'] is not None)
SHEETS = {'celeba': tuple(n for n in GRIDS if GRIDS[n]['dataset'] == 'CELEBA'),
          'afhqv2': tuple(n for n in GRIDS if GRIDS[n]['dataset'] == 'AFHQV2')}
# Grids whose seed 0 DPM++ 50k cell is finished elsewhere; refs/<name>/samples_0.npz.
REFERENCES = ('celeba_vp', 'celeba_b', 'afhq_vp', 'afhq_a', 'afhq_dial')

# The order is fixed in advance and does not depend on any observed FID.
CELLS = dict(q.m35('celeba_a', None, 'celeba_a'))
for _spec in CELLS.values():
    _spec['dataset'] = 'CELEBA'
TABLE4_CELLS = {name: 'A_c80_lo_{}_seed{}'.format(spec['method'], spec['seed']) for name, spec in CELLS.items()}


def source_of(ckpt):
    return 'celeba_a_first' if ckpt == A_FIRST else 'celeba_a'


def a_selection():
    """CELEBA_A800K_STEP_RULE.md for A from the curve summary; incomplete unless the curve finished cleanly."""
    result = dict(rule='latest candidate with FID <= minimum + {} (CELEBA_A800K_STEP_RULE.md)'.format(q.TIE),
                  complete=False)
    path = ACURVE / 'summary.json'
    if not path.exists():
        result['reason'] = 'No curve summary.'
        return result
    saved = json.loads(path.read_text())
    cells = {cell['ckpt']: cell for cell in saved.get('cells', [])}
    chosen = ((saved.get('step_selection') or {}).get('selected') or {}).get('A')
    verdict = (saved.get('control') or {}).get('verdict')
    result.update(inputs=[dict(cell=cells[c]['cell'], step=A_FREQ * c, fid_tweedie_xt=cells[c]['fid_tweedie_xt'])
                          for c in A_CANDIDATES if c in cells], control=verdict, curve_selection=chosen)
    problems = []
    if saved.get('status') != 'completed':
        problems.append('curve status {}'.format(saved.get('status')))
    if saved.get('failed_cells'):
        problems.append('failed curve cells {}'.format(saved['failed_cells']))
    if sorted(cells) != list(A_CANDIDATES) or any(cells[c]['step'] != A_FREQ * c + 1 for c in cells):
        problems.append('candidates {}'.format(sorted(cells)))
    if verdict not in ('matched', 'matched_within_card_tolerance'):
        problems.append('control {}'.format(verdict))
    if not chosen:
        problems.append('no selected step')
    if not problems:
        fids = {A_FREQ * c: cells[c]['fid_tweedie_xt'] for c in A_CANDIDATES}
        step = q.select(fids)
        if step != chosen['step']:
            problems.append('the curve selected {}, the rule gives {}'.format(chosen['step'], step))
    if problems:
        result['reason'] = '; '.join(problems)
        return result
    ckpt = step // A_FREQ
    result.update(complete=True, minimum=min(fids.values()), selected_step=step, selected_fid=fids[step],
                  selected_ckpt=ckpt, source=source_of(ckpt))
    return result


def selected():
    path = LOGS / SELECTION
    if not path.exists():
        raise ValueError('No step selected for CelebA A yet.')
    saved = json.loads(path.read_text())
    if not saved.get('complete'):
        raise ValueError('The CelebA A selection is incomplete.')
    return saved


def with_step(spec):
    spec['work'] = spec['source']
    spec['step'] = q.SOURCES[spec['source']]['freq'] * spec['ckpt'] + 1
    return spec


def resolve(name):
    if name not in CELLS:
        raise ValueError('Unknown A 50k cell: ' + name)
    spec = dict(CELLS[name])
    spec['ckpt'] = int(selected()['selected_ckpt'])
    spec['source'] = source_of(spec['ckpt'])
    return with_step(spec)


def grid_spec(name):
    grid = GRIDS[name]
    spec = dict(kind='m35', group='grid', dataset=grid['dataset'], label=grid['label'], source=grid['source'],
                ckpt=grid['ckpt'], method=METHOD, seed=SEED, count=q.M35_COUNT, calls=q.M35_CALLS)
    if spec['ckpt'] is None:
        spec['ckpt'] = int(selected()['selected_ckpt'])
        spec['source'] = source_of(spec['ckpt'])
    return with_step(spec)


def config_of(spec):
    """The config of the 50k matched35 cell of the model behind spec."""
    if spec['source'] == 'celeba_vp':  # The VP cells of table 4.
        return lo.native_config(dict(calls=spec['calls'], count=spec['count'], model='vp', method=spec['method'],
                                     seed=spec['seed'], ckpt=spec['ckpt'], window=q.WINDOW))
    if spec['source'] not in ('celeba_a_first', 'celeba_a'):
        return q.config_of(spec)
    from ml_collections import ConfigDict
    from configs.factorial.celeba_a_vp_pvp import get_config as get_a
    config = ConfigDict(q.manifest(spec['source'])['config'])
    if (config.training.factorial_arm, config.seed) != ('A_vp_pvp', 42):
        raise ValueError('Unexpected run behind CelebA A: ' + spec['source'])
    config.eval = get_a().eval  # Manifests exclude evaluation settings.
    q.set_protocol(config, q.WINDOW, spec['ckpt'])
    config.sampling.method = spec['method']  # As celeba_50k_lo_eval.a_config and gpu2_queue_eval.config_of.
    config.sampling.time_grid = 'karras' if spec['method'] == 'ode_edm_heun' else 'uniform_logsnr'
    config.eval.sampling_num_scales = spec['calls']
    config.eval.sampling_seed = spec['seed']
    config.eval.num_samples = spec['count']
    return config


def get_config(name):
    return config_of(resolve(name))


# Grids.

def tile(images, nrow=NROW):
    """N x H x W x C uint8 to one image, white padding."""
    import numpy as np
    count, height, width, channels = images.shape
    rows = (count + nrow - 1) // nrow
    canvas = np.full((PAD + rows * (height + PAD), PAD + nrow * (width + PAD), channels), 255, np.uint8)
    for index, image in enumerate(images):
        row, column = divmod(index, nrow)
        y, x = PAD + row * (height + PAD), PAD + column * (width + PAD)
        canvas[y:y + height, x:x + width] = image
    return canvas


def save_png(image, path):
    from PIL import Image
    if not isinstance(image, Image.Image):
        image = Image.fromarray(image)
    temporary = Path(str(path) + '.tmp')
    image.save(str(temporary), format='PNG')
    os.replace(str(temporary), str(path))


def save_npz(path, **arrays):
    import numpy as np
    temporary = Path(str(path) + '.tmp.npz')
    np.savez(str(temporary), **arrays)
    os.replace(str(temporary), str(path))


def grid_files(name):
    return dict(npz=GRIDS_DIR / (name + '.npz'), x_t=GRIDS_DIR / (name + '_x_t.png'),
                tweedie_xt=GRIDS_DIR / (name + '_tweedie.png'))


def grid_record_path(name):
    return LOGS / (name + '_grid.json')


def grid_failure_path(name):
    return LOGS / (name + '_grid_failed.json')


def sample_grid(name):
    """Round 0 of the seed 0 DPM++ 50k cell of the model; keeps the first SHOW images."""
    import numpy as np
    import torch
    import datasets
    import sampling
    from reproducibility import derive_seed, isolated_torch_rng
    from run_lib import get_sde
    spec = grid_spec(name)
    config = config_of(spec)
    sde, eps = get_sde(config, n_override=spec['calls'])
    followup.check_cell_grid(config, sde, eps, spec)
    model, identity = q.load_model(config, spec)
    shape = (config.eval.batch_size, config.data.num_channels, config.data.image_size, config.data.image_size)
    sampling_fn = sampling.get_sampling_fn(config, sde, shape, datasets.get_data_inverse_scaler(config), eps)
    seed = derive_seed(config.eval.sampling_seed, 'sampling-round', 0)  # As endpoint_denoise_eval, round 0.
    with isolated_torch_rng(seed):
        white = torch.randn(*shape)
    drawn, prior = [], sde.prior_sampling

    def capture(prior_shape):
        value = prior(prior_shape)
        drawn.append(value.detach().cpu().clone())
        return value

    sde.prior_sampling = capture  # The sampler draws its start through this object.
    with isolated_torch_rng(seed):
        outputs, nfe = sampling_fn(model)
    del model
    scale = float(getattr(sde, '_prior_std', 1.0))
    if len(drawn) != 1 or not torch.allclose(drawn[0], white * scale, rtol=1e-6, atol=0):
        raise RuntimeError('The sampler did not start from the recorded white noise.')
    if int(nfe) != spec['calls'] + 1 or set(outputs) != set(VARIANTS):
        raise RuntimeError('Unexpected sampler output: {} calls, {}'.format(nfe, sorted(outputs)))
    images = {variant: np.clip(outputs[variant].permute(0, 2, 3, 1).cpu().numpy() * 255., 0, 255)
              .astype(np.uint8)[:SHOW] for variant in VARIANTS}  # The conversion of endpoint_denoise_eval.
    noise = white[:SHOW].numpy().astype(np.float32)
    record = dict(grid=name, dataset=spec['dataset'], label=spec['label'], source=spec['source'], ckpt=spec['ckpt'],
                  step=spec['step'], checkpoint_id=identity['checkpoint_id'], method=spec['method'],
                  trajectory_calls=spec['calls'], nfe=int(nfe), sampling_seed=spec['seed'], round_id=0, round_seed=seed,
                  lambda_window=[q.LOGSNR_MIN, q.WINDOWS[q.WINDOW]], batch=shape[0], shown=SHOW,
                  noise_sha256=hashlib.sha256(noise.tobytes()).hexdigest(),
                  files={key: str(path.relative_to(ROOT)) for key, path in grid_files(name).items()})
    GRIDS_DIR.mkdir(parents=True, exist_ok=True)
    files = grid_files(name)
    save_npz(files['npz'], x_t=images['x_t'], tweedie_xt=images['tweedie_xt'], white_noise=noise,
             checkpoint_id=np.asarray(identity['checkpoint_id']), checkpoint_step=np.asarray(spec['step']),
             round_seed=np.asarray(seed), nfe=np.asarray(int(nfe)), method=np.asarray(spec['method']))
    for variant in VARIANTS:
        save_png(tile(images[variant]), files[variant])
    write_json(grid_record_path(name), record)
    return record


def write_sheet(sheet):
    """Rows: models. Columns: the first SHEET_COLUMNS samples, x_t, then Tweedie(x_t)."""
    import numpy as np
    from PIL import Image, ImageDraw
    rows = []
    for name in SHEETS[sheet]:
        if grid_record_path(name).exists() and grid_files(name)['npz'].exists():
            with np.load(str(grid_files(name)['npz']), allow_pickle=False) as data:
                rows.append((json.loads(grid_record_path(name).read_text()), {v: data[v][:SHEET_COLUMNS] for v in VARIANTS}))
    if not rows:
        return None
    size = rows[0][1]['x_t'].shape[1]
    label_width, header, gap = 76, 16, 12
    block = SHEET_COLUMNS * (size + PAD)
    canvas = Image.new('RGB', (label_width + 2 * block + gap, header + PAD + len(rows) * (size + PAD)), 'white')
    draw = ImageDraw.Draw(canvas)
    for index, title in enumerate(('x_t, without Tweedie', 'Tweedie(x_t)')):
        draw.text((label_width + index * (block + gap), 2), title, fill='black')
    for row, (record, images) in enumerate(rows):
        y = header + PAD + row * (size + PAD)
        draw.text((4, y + size // 2 - 6), '{} {}k'.format(record['label'], (record['step'] - 1) // 1000), fill='black')
        for index, variant in enumerate(VARIANTS):
            for column, image in enumerate(images[variant]):
                canvas.paste(Image.fromarray(image), (label_width + index * (block + gap) + column * (size + PAD), y))
    path = GRIDS_DIR / 'sheet_{}.png'.format(sheet)
    save_png(canvas, path)
    return path


def reference_path(name, record):
    if name in REFERENCES:
        return REFS / name / 'samples_0.npz'
    if name == 'celeba_a':
        if record['ckpt'] == A_FIRST:
            return REFS / 'celeba_a_400k' / 'samples_0.npz'
        cell = 'celeba_a_cbest_m35_{}_seed{}'.format(METHOD, SEED)
        return ROOT / 'workdirs' / record['source'] / eval_folder(cell) / 'ckpt_{}'.format(record['ckpt']) / 'samples_0.npz'
    return None


def compare(name, record):
    """The grid against the first images of round 0 of the 50k cell, if that cell is finished."""
    import numpy as np
    path = reference_path(name, record)
    if path is None or not path.exists():
        return None
    with np.load(str(grid_files(name)['npz']), allow_pickle=False) as mine, \
            np.load(str(path), allow_pickle=False) as theirs:
        identity = dict(step=int(theirs['checkpoint_step']), round_id=int(theirs['round_id']),
                        round_seed=int(theirs['sampling_seed']))
        result = dict(reference=str(path.relative_to(ROOT)), reference_identity=identity,
                      identity_matches=identity == dict(step=record['step'], round_id=0, round_seed=record['round_seed']))
        for variant in VARIANTS:
            difference = np.abs(mine[variant].astype(np.int16) - theirs['samples_' + variant][:SHOW].astype(np.int16))
            result[variant] = dict(max_abs_difference=int(difference.max()), equal_fraction=float((difference == 0).mean()),
                                   mean_abs_difference=float(difference.mean()))
    return result


def run_grid(name):
    import torch
    for attempt in range(1, ATTEMPTS + 1):
        try:
            record = sample_grid(name)
            for sheet, names in SHEETS.items():
                if name in names:
                    write_sheet(sheet)
            if grid_failure_path(name).exists():
                grid_failure_path(name).unlink()
            print('Grid {}: {} {} at {}, noise {}'.format(name, record['label'], record['dataset'], record['step'],
                                                         record['noise_sha256'][:12]), flush=True)
            return True
        except Exception as error:  # A truncated checkpoint raises whatever torch.load does.
            write_json(grid_failure_path(name), dict(error=repr(error), attempt=attempt, time=time.time()))
            print('Grid failed: {}: {!r}'.format(name, error), flush=True)
        finally:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    return False


def grids():
    """Phase 1: the grids of the known models, in order. A failed grid does not stop the next.
    Grids need neither Inception nor statistics, so they do not depend on the preflight of A."""
    failed = [name for name in KNOWN if not grid_record_path(name).exists() and not run_grid(name)]
    write_summary(['grid ' + name for name in failed])
    print('Grids finished, failed: {}'.format(failed or 'none'), flush=True)
    return 1 if failed else 0


# Preflight and the A cells.

def preflight(ckpt, filename):
    """Weights of A at ckpt through the real path: grid, a short run, 512 images through evaluate_checkpoint
    with both solvers. Returns 1 on failure."""
    import torch
    import evaluation
    from run_lib import get_sde
    path = LOGS / filename
    if path.exists():
        path.unlink()
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError('Expected one Docker-routed GPU.')
    arch = '{}.{}'.format(*torch.cuda.get_device_capability(0))
    if arch not in os.environ.get('TORCH_CUDA_ARCH_LIST', '').replace(';', ' ').split():
        raise RuntimeError('TORCH_CUDA_ARCH_LIST does not include sm ' + arch)
    try:  # The model and Inception share this process; Inception must not take the whole card.
        import tensorflow as tf
        for device in tf.config.list_physical_devices('GPU'):
            tf.config.experimental.set_memory_growth(device, True)
    except (RuntimeError, ValueError, AttributeError, TypeError):
        pass
    lo.check_inception()
    from artifact_utils import _environment_manifest, _source_fingerprint
    first = next(iter(CELLS))
    spec = with_step(dict(CELLS[first], ckpt=ckpt, source=source_of(ckpt)))
    stats = evaluation.dataset_stats_fingerprint(config_of(spec))
    if stats != q.REFERENCE_SHA:
        raise RuntimeError('Unexpected CelebA statistics.')
    identity, failures, smokes = None, {}, {}
    try:
        config = config_of(spec)
        sde, eps = get_sde(config, n_override=spec['calls'])
        followup.check_cell_grid(config, sde, eps, spec)
        model, identity = q.load_model(config, spec)
        followup.smoke(config, model, spec['method'], eps)
        for method in q.METHODS:
            other = dict(spec, method=method)
            smokes[method] = q.endpoint_smoke(config_of(other), model, other, LOGS / 'preflight_endpoint')
        del model
    except Exception as error:  # A truncated checkpoint raises whatever torch.load does.
        failures['celeba_a_m35'] = repr(error)
        print('Preflight failed: {!r}'.format(error), flush=True)
    write_json(path, dict(status='failed' if failures else 'passed', ckpt=ckpt, source=spec['source'], step=spec['step'],
        checkpoint=identity, failures=failures, endpoint_smoke_fid_512=smokes, dataset_stats_id=dict(CELEBA=stats),
        source_sha256=_source_fingerprint(str(ROOT)), environment=_environment_manifest(), started_unix=time.time()))
    print('Preflight A at {}:'.format(spec['step']), 'failed: {}'.format(failures) if failures else 'passed', flush=True)
    return 1 if failures else 0


def select_step():
    """Phase 2: write the A selection once; a complete selection is never recomputed. 0 if complete."""
    path = LOGS / SELECTION
    if not (path.exists() and json.loads(path.read_text()).get('complete')):
        write_json(path, a_selection())
    saved = json.loads(path.read_text())
    print('CelebA A selection: {}'.format(saved), flush=True)
    if not saved['complete']:
        write_summary(['selection'])
        return 1
    return 0


def preflight_a():
    ckpt = int(selected()['selected_ckpt'])
    if ckpt == A_FIRST:
        write_json(LOGS / 'preflight_a.json', dict(status='passed', ckpt=ckpt, failures={},
                                                   note='400k: the 50k cells are in table 4, nothing to evaluate'))
        return 0
    return preflight(ckpt, 'preflight_a.json')


def eval_folder(name):
    return 'eval_gpu2a_' + name


def result_path(name):
    return LOGS / (name + '_result.json')


def failure_path(name):
    return LOGS / (name + '_failed.json')


def run(name):
    spec = resolve(name)
    began, clock = time.time(), time.monotonic()
    result = subprocess.run([sys.executable, '-u', 'main.py', '--mode=eval', '--config=gpu2_a50k_grids.py:' + name,
                             '--workdir=workdirs/' + spec['work'], '--eval_folder=' + eval_folder(name)],
                            cwd=str(ROOT))
    write_json(LOGS / (name + '_timing.json'), dict(started_unix=began, finished_unix=time.time(),
        elapsed_seconds=time.monotonic() - clock, returncode=result.returncode))
    result.check_returncode()


def report(name):
    """The checks of gpu2_queue_eval.report on the A cells."""
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
    pre = json.loads((LOGS / 'preflight_a.json').read_text())
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
    print('A 50k {}: FID after Tweedie {:.4f}, {:.1f} min'.format(
        name, metrics['fid_tweedie_xt'], timing['elapsed_seconds'] / 60), flush=True)


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
            print('A 50k cell failed: {}: {!r}'.format(name, error), flush=True)
    return False


def a_queue():
    """Phase 2 after select and preflight-a: the A grid, then the 50k cells unless 400k is selected."""
    ckpt = int(selected()['selected_ckpt'])
    failed = []
    if not grid_record_path('celeba_a').exists() and not run_grid('celeba_a'):
        failed.append('grid celeba_a')  # The grid is not a gate for the 50k cells.
    if ckpt == A_FIRST:
        print('A at 400k: its 50k cells are in table 4, not recomputed.', flush=True)
    else:
        pre = LOGS / 'preflight_a.json'
        blocked = json.loads(pre.read_text())['failures'] if pre.exists() else dict(preflight='missing')
        for name in CELLS:
            if result_path(name).exists():
                continue
            if blocked:
                failed.append(name)
                continue
            if not evaluate(name):
                failed.append(name)
    write_summary(failed)
    print('A phase finished, failed: {}'.format(failed or 'none'), flush=True)
    return 1 if failed else 0


def write_summary(failed=()):
    grids_done = {}
    for name in GRIDS:
        if grid_record_path(name).exists():
            record = json.loads(grid_record_path(name).read_text())
            record['comparison_with_50k_round_0'] = compare(name, record)
            grids_done[name] = record
    noise = {}
    for sheet, names in SHEETS.items():
        hashes = sorted({grids_done[n]['noise_sha256'] for n in names if n in grids_done})
        noise[sheet] = dict(noise_sha256=hashes, shared=len(hashes) <= 1)
    path = LOGS / SELECTION
    selection = json.loads(path.read_text()) if path.exists() else None
    cells = {name: json.loads(result_path(name).read_text()) for name in CELLS if result_path(name).exists()}
    table4 = None
    if selection and selection.get('complete') and selection['selected_ckpt'] == A_FIRST:
        table4 = dict(note='A at 400k: 50k matched35 from table 4 (celeba_50k_lo_20261107/gpu1), not recomputed',
                      cells={name: json.loads((TABLE4 / (theirs + '_result.json')).read_text())
                             for name, theirs in TABLE4_CELLS.items() if (TABLE4 / (theirs + '_result.json')).exists()})
    write_json(LOGS / 'summary.json', dict(
        grids=grids_done, grids_missing=[n for n in GRIDS if n not in grids_done], shared_noise=noise,
        a_selection=selection, a50k_cells=list(cells.values()), a50k_completed=len(cells), a50k_total=len(CELLS),
        a50k_table4=table4, failed=list(failed),
        protocol='DPM++ 2M and EDM Heun, 34 trajectory calls plus Tweedie, window [-10; 9.115429865459795]; grids: '
                 'DPM++ 2M, sampling seed 0, round 0 of the 50k cell, first {} images'.format(SHOW),
        rule='CELEBA_A800K_STEP_RULE.md and AFHQV2_STEP_RULE.md',
        scope='One training run per model; matched35 at 50k has no intervals here'))
    with (LOGS / 'summary.csv').open('w', newline='') as stream:
        writer = csv.writer(stream)
        writer.writerow(('cell', 'source', 'step', 'method', 'sampling_seed', 'num_samples', 'nfe', 'fid_tweedie',
                         'fid_x_t', 'kid_tweedie', 'is_tweedie'))
        for name, value in cells.items():
            writer.writerow((name, value['source'], value['step'], value['method'], value['sampling_seed'],
                             value['num_samples'], value['nfe'], value['fid_tweedie_xt'], value['fid_x_t'],
                             value['kid_tweedie_xt'], value['IS_tweedie_xt']))


if __name__ == '__main__':
    mode = sys.argv[1] if len(sys.argv) > 1 else ''
    if mode == 'preflight':
        sys.exit(preflight(A_SMOKE, 'preflight.json'))
    elif mode == 'grids':
        sys.exit(grids())
    elif mode == 'select':
        sys.exit(select_step())
    elif mode == 'preflight-a':
        sys.exit(preflight_a())
    elif mode == 'a':
        sys.exit(a_queue())
    elif mode == 'plan':
        for grid_label in GRIDS:
            print('grid', grid_label, GRIDS[grid_label])
        for cell_label, cell_spec in CELLS.items():
            print(cell_label, cell_spec)
    else:
        raise SystemExit('Usage: gpu2_a50k_grids.py preflight|grids|select|preflight-a|a|plan')
