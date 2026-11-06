"""Stage 0 for CelebA: one DPM++ protocol over training steps and A-G models.

Every cell is DPM++ 2M, 64 trajectory calls plus Tweedie, 10k images, sampling
seed 0, on an explicit log-SNR window. Results are a screen: a candidate for a
main table needs its own 50k confirmation. One training run per cell, so these
points do not measure variation between training seeds.

Run through run_celeba_stage0_rtx.sh, from a checkout separate from the one the
running trainings use: adding this file there would change their source
fingerprint and block their resume.

Windows are numeric: `hi` is [-10, 10.575749...], `lo` is [-10, 9.115429...].
A-G were trained on the `lo` support, so they are never evaluated on `hi`.
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

from e3b_final50k import REFERENCE_SHA, VP, FOX

ROOT = Path(__file__).resolve().parent
LOGS = ROOT / 'logs'
TRAINING_INPUTS = ROOT / 'workdirs/_training_inputs'
DEVICE = 'cuda:0'

METHOD = 'ode_dpmpp_2m'
BUDGET = 64
COUNT = 10000
BATCH = 512
SEED = 0
VARIANTS = ('x_t', 'tweedie_xt')
LOGSNR_MIN = -10.0
WINDOWS = {'hi': 10.575749165617202, 'lo': 9.115429865459795}
SPLICE_LAMBDA = 0.0

WORKDIRS = {
    'vp': VP,
    'fox': FOX,
    'A': 'celeba_factorial_A_vp_pvp_50k',
    'B': 'celeba_factorial_B_vp_pfox_50k',
    'C': 'celeba_factorial_C_normfox_pvp_50k',
    'D': 'celeba_factorial_D_normfox_pfox_50k',
    'E': 'celeba_factorial_E_realfox_pfox_50k',
    'F': 'celeba_wave2_F_vp_pvp_vptime_common_100k',
    'G': 'celeba_wave2_G_normfox_pfox_foxtime_common_100k',
}
TRAINED_ON_LO = tuple('ABCDEFG')
ROUTES = ('head', 'tail', 'full')

# Same weights and protocol as final_ode_lownfe, seed 0: the driver must reproduce them.
GATE = {'vp_c260_hi': 3.740010, 'fox_c260_hi': 2.443862}
GATE_TOLERANCE = 0.02
GATE_EXIT = 3  # The launcher skips the energy pass after this code.


def cell(model, ckpt, window, route=None):
    prefix = 'splice_' + route if route else model
    return '{}_c{}_{}'.format(prefix, ckpt, window), dict(
        model=model, ckpt=ckpt, window=window, route=route)


# The order is fixed in advance and does not depend on any observed FID.
CELLS = dict(
    [cell(m, 260, 'hi') for m in ('vp', 'fox')] +
    [cell(m, 40, 'lo') for m in ('A', 'B', 'vp', 'fox')] +
    [cell(m, 260, 'lo') for m in ('vp', 'fox')] +
    [cell(m, 20, 'lo') for m in ('A', 'F', 'vp', 'fox')] +
    [cell(m, c, 'hi') for c in (20, 40, 80, 160, 250) for m in ('vp', 'fox')] +
    [cell(m, 10, 'lo') for m in 'ABCDEFG'] +
    [cell('vp', 260, 'lo', route) for route in ROUTES] +
    [cell('B', 20, 'lo')])

# Each contrast is left minus right, FID after Tweedie, negative favours left.
CONTRASTS = (
    ('fox_vs_vp_1.3M_hi', 'fox_c260_hi', 'vp_c260_hi', 'Native gap, reference value'),
    ('fox_vs_vp_1.3M_lo', 'fox_c260_lo', 'vp_c260_lo', 'Native gap without the cleanest band'),
    ('fox_vs_vp_200k_lo', 'fox_c40_lo', 'vp_c40_lo', 'Native gap at 200k'),
    ('fox_vs_vp_100k_lo', 'fox_c20_lo', 'vp_c20_lo', 'Native gap at 100k'),
    ('A_vs_vp_200k_lo', 'A_c40_lo', 'vp_c40_lo', 'Is A stronger than native VP at equal steps?'),
    ('A_vs_fox_200k_lo', 'A_c40_lo', 'fox_c40_lo', 'A against native FOX at equal steps'),
    ('A200k_vs_vp1.3M_lo', 'A_c40_lo', 'vp_c260_lo', 'Baseline choice only, unequal training'),
    ('A200k_vs_fox1.3M_lo', 'A_c40_lo', 'fox_c260_lo', 'Baseline choice only, unequal training'),
    ('B_vs_A_200k_lo', 'B_c40_lo', 'A_c40_lo', 'Allocation under ODE; the known reversal'),
    ('B_vs_A_100k_lo', 'B_c20_lo', 'A_c20_lo', 'Allocation under ODE at 100k'),
    ('F_vs_A_100k_lo', 'F_c20_lo', 'A_c20_lo', 'Label type, VP, common support'),
    ('vp_vs_F_100k_lo', 'vp_c20_lo', 'F_c20_lo', 'Different training runs; supports differ by only 0.35% of t'),
    ('F_vs_A_50k_lo', 'F_c10_lo', 'A_c10_lo', 'Label type, VP'),
    ('G_vs_D_50k_lo', 'G_c10_lo', 'D_c10_lo', 'Label type, normalized FOX'),
    ('E_vs_D_50k_lo', 'E_c10_lo', 'D_c10_lo', 'Input scale, FOX allocation'),
    ('B_vs_A_50k_lo', 'B_c10_lo', 'A_c10_lo', 'Allocation on VP; is the reversal visible at 50k?'),
    ('D_vs_C_50k_lo', 'D_c10_lo', 'C_c10_lo', 'Allocation on normalized FOX'),
    ('C_vs_A_50k_lo', 'C_c10_lo', 'A_c10_lo', 'Clock at VP allocation; expected near zero'),
    ('D_vs_B_50k_lo', 'D_c10_lo', 'B_c10_lo', 'Clock at FOX allocation; expected near zero'),
    ('head_vs_vp_lo', 'splice_head_c260_lo', 'vp_c260_lo', 'FOX network for lambda<0 under ODE'),
    ('tail_vs_vp_lo', 'splice_tail_c260_lo', 'vp_c260_lo', 'FOX network for lambda>=0 under ODE'),
    ('full_vs_fox_lo', 'splice_full_c260_lo', 'fox_c260_lo',
     'Adapter check: FOX network on the VP route must match native FOX'),
)
# A-G share a support, so none of these isolates the wider support of native FOX.

ENERGY_CELLS = tuple(list(CELLS)[:12])


def write_json(path, value):
    """Atomic: a killed container must not leave a truncated file for the resume."""
    temporary = Path(str(path) + '.tmp')
    temporary.write_text(json.dumps(value, indent=2) + '\n')
    os.replace(str(temporary), str(path))


def set_device(config):
    import torch
    config.device = DEVICE if isinstance(config.device, str) else torch.device(DEVICE)


def native_config(model):
    from configs.vp.celeba_ncsnpp_continuous import get_config as get_vp
    from configs.fox.celeba_ncsnpp_continuous_matern_3_2 import get_config as get_fox
    config = get_vp() if model == 'vp' else get_fox()
    if model == 'fox':
        config.model.fox_u = -5.0
        config.model.fox_matern_length_scale = 346.4101615138
        config.model.fox_target_terminal_variance = 1.0
    return config


def training_config(model):
    from ml_collections import ConfigDict
    from configs.factorial.celeba_a_vp_pvp import get_config as get_a
    manifest = json.loads((TRAINING_INPUTS / (WORKDIRS[model] + '.json')).read_text())
    config = ConfigDict(manifest['config'])
    config.eval = get_a().eval  # Manifests exclude evaluation settings.
    return config


def set_protocol(config, window, ckpt):
    """The stage 0 sampling protocol, shared with celeba_kappa_eval.py."""
    set_device(config)
    config.sampling.method = METHOD
    config.sampling.probability_flow = True
    config.sampling.time_grid = 'uniform_logsnr'
    config.sampling.logsnr_min = LOGSNR_MIN
    config.sampling.logsnr_max = WINDOWS[window]
    config.sampling.noise_removal = True
    config.sampling.predictor = 'none'  # The ODE method owns its stages.
    config.sampling.corrector = 'none'
    config.eval.endpoint_denoise_variants = True
    config.eval.endpoint_denoise_outputs = VARIANTS
    config.eval.save_endpoint_states = True
    config.eval.sampling_num_scales = BUDGET  # Trajectory calls.
    config.eval.sampling_seed = SEED
    config.eval.num_samples = COUNT
    config.eval.batch_size = BATCH
    config.eval.enable_loss = False
    config.eval.enable_bpd = False
    config.eval.enable_sampling = True
    config.eval.begin_ckpt = config.eval.end_ckpt = ckpt
    config.eval.include_final_checkpoint = False
    config.allow_tf32 = True
    config.cudnn_benchmark = True
    config.allow_fp16_reduced_precision_reduction = False


def get_config(name):
    if name not in CELLS:
        raise ValueError('Unknown stage 0 cell: ' + name)
    spec = CELLS[name]
    model = spec['model']
    if model in TRAINED_ON_LO and spec['window'] != 'lo':
        raise ValueError('A-G are evaluated only inside their training support.')
    config = native_config(model) if model in ('vp', 'fox') else training_config(model)
    set_protocol(config, spec['window'], spec['ckpt'])
    if spec['route']:
        alternate = native_config('fox')
        set_device(alternate)
        config.eval.splice_enabled = True
        config.eval.splice_allow_ode = True
        config.eval.splice_mode = spec['route']
        config.eval.splice_lambda = None if spec['route'] == 'full' else SPLICE_LAMBDA
        config.eval.splice_alt = alternate
        config.eval.splice_alt_workdir = 'workdirs/' + FOX
        config.eval.splice_alt_ckpt = spec['ckpt']
    return config


def check_grid(config, sde, eps, trajectory_calls=BUDGET):
    """The solver must span the requested window, inside the training support."""
    import torch
    grid = sde.sampling_time_grid(eps, grid='uniform_logsnr', device=config.device,
                                  dtype=torch.float64, N=trajectory_calls + 1)
    levels = sde.log_snr(grid)
    upper = float(config.sampling.logsnr_max)
    if not bool(torch.all(levels[1:] > levels[:-1])):
        raise RuntimeError('The ODE grid is not increasing in log-SNR.')
    if abs(float(levels[0]) - LOGSNR_MIN) > 1e-3 or abs(float(levels[-1]) - upper) > 1e-3:
        raise RuntimeError('The ODE grid misses the requested window.')
    support = getattr(config.model, 'logsnr_max', None)
    if support is not None and upper > float(support) + 1e-9:
        raise RuntimeError('The window exceeds the training support.')
    return grid, levels


def load_model(config, work, ckpt):
    from artifact_utils import build_model_manifest
    from losses import get_optimizer
    from models import utils as mutils, ncsnpp  # noqa: F401, registers the model.
    from models.ema import ExponentialMovingAverage
    from utils import restore_checkpoint
    model = mutils.create_model(config)
    ema = ExponentialMovingAverage(model.parameters(), decay=config.model.ema_rate)
    state = dict(model=model, optimizer=get_optimizer(config, model.parameters()), ema=ema, step=0,
                 model_protocol_sha256=build_model_manifest(config)['model_protocol_sha256'])
    path = ROOT / 'workdirs' / work / 'checkpoints/checkpoint_{}.pth'.format(ckpt)
    if not path.is_file():
        raise RuntimeError('Missing checkpoint: ' + str(path))
    state = restore_checkpoint(str(path), state, config.device, restore_rng=False)
    step = int(state['step'])
    if not 5000 * ckpt <= step <= 5000 * ckpt + 2:
        raise RuntimeError('Unexpected step {} in {}'.format(step, path))
    ema.copy_to(model.parameters())
    model.eval()
    return model, dict(checkpoint_id=state['checkpoint_id'], step=step)


def preflight():
    import torch
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError('Expected one Docker-routed GPU.')
    arch = '{}.{}'.format(*torch.cuda.get_device_capability(0))
    if arch not in os.environ.get('TORCH_CUDA_ARCH_LIST', '').replace(';', ' ').split():
        raise RuntimeError('TORCH_CUDA_ARCH_LIST does not include sm ' + arch)
    from artifact_utils import _environment_manifest, _source_fingerprint
    from ode_sampling import get_vp_ode_sampler
    from run_lib import get_sde
    identities = {}
    for name, spec in CELLS.items():
        key = '{}_{}'.format(spec['model'], spec['ckpt'])
        config = get_config(name)
        sde, eps = get_sde(config, n_override=BUDGET)
        check_grid(config, sde, eps)
        if key in identities:
            continue
        model, identities[key] = load_model(config, WORKDIRS[spec['model']], spec['ckpt'])
        smoke_sde, _ = get_sde(config, n_override=4)
        z = torch.zeros((2, 3, 64, 64), device=config.device)
        outputs, calls = get_vp_ode_sampler(smoke_sde, z.shape, lambda x: x, METHOD, eps, config.device)(model, z=z)
        if calls != 5 or not all(bool(torch.isfinite(x).all()) for x in outputs.values()):
            raise RuntimeError('Failed real-weights smoke: ' + key)
        print('Preflight', key, identities[key]['step'], flush=True)
        del model
        gc.collect()
        torch.cuda.empty_cache()
    write_json(LOGS / 'preflight.json', dict(status='passed', checkpoints=identities,
        source_sha256=_source_fingerprint(str(ROOT)), environment=_environment_manifest(),
        started_unix=time.time()))
    print('Stage 0 preflight passed: {} checkpoints, {} grids.'.format(len(identities), len(CELLS)), flush=True)


def eval_folder(name):
    return 'eval_stage0_{}_n{}_{}_seed{}'.format(name, BUDGET, COUNT, SEED)


def cell_directory(name):
    return ROOT / 'workdirs' / WORKDIRS[CELLS[name]['model']] / eval_folder(name)


def result_path(name):
    return LOGS / (name + '_result.json')


def failure_path(name):
    return LOGS / (name + '_failed.json')


def run(name):
    began, clock = time.time(), time.monotonic()
    result = subprocess.run([sys.executable, '-u', 'main.py', '--mode=eval',
        '--config=celeba_stage0_eval.py:' + name, '--workdir=workdirs/' + WORKDIRS[CELLS[name]['model']],
        '--eval_folder=' + eval_folder(name)])
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
    identity = pre['checkpoints']['{}_{}'.format(spec['model'], spec['ckpt'])]
    with np.load(str(directory / 'report_{}.npz'.format(spec['ckpt'])), allow_pickle=False) as data:
        expected = dict(num_samples=COUNT, nfe=BUDGET + 1, checkpoint_step=identity['step'],
                        checkpoint_id=identity['checkpoint_id'], dataset_stats_id=REFERENCE_SHA)
        for key, value in expected.items():
            if data[key].item() != value:
                raise RuntimeError('Unexpected report identity: ' + key)
        metrics = {metric + '_' + variant: float(data[metric + '_' + variant])
                   for metric in ('IS', 'fid', 'kid') for variant in VARIANTS}
    if not all(math.isfinite(value) for value in metrics.values()):
        raise RuntimeError('Non-finite metric.')
    shards = len(list((directory / 'ckpt_{}'.format(spec['ckpt'])).glob('statistics_*.npz')))
    if shards != (COUNT + BATCH - 1) // BATCH:
        raise RuntimeError('Unexpected number of statistics shards.')
    timing = json.loads((LOGS / (name + '_timing.json')).read_text())
    write_json(result_path(name), dict(spec, cell=name, checkpoint_step=identity['step'],
        lambda_window=[LOGSNR_MIN, WINDOWS[spec['window']]], nfe=BUDGET + 1, num_samples=COUNT,
        sampling_seed=SEED, elapsed_seconds=timing['elapsed_seconds'], **metrics))
    write_summary()
    print('Stage 0 {}: FID after Tweedie {:.6f}, {:.1f} min'.format(
        name, metrics['fid_tweedie_xt'], timing['elapsed_seconds'] / 60), flush=True)


def write_summary():
    cells = {name: json.loads(result_path(name).read_text()) for name in CELLS if result_path(name).exists()}
    contrasts = []
    for label, left, right, question in CONTRASTS:
        if left in cells and right in cells:
            contrasts.append(dict(contrast=label, left=left, right=right, question=question,
                value=cells[left]['fid_tweedie_xt'] - cells[right]['fid_tweedie_xt']))
    write_json(LOGS / 'summary.json', dict(status='completed' if len(cells) == len(CELLS) else 'running',
        completed_cells=len(cells), total_cells=len(CELLS), cells=list(cells.values()),
        point_contrasts=contrasts, confidence_intervals_computed=False,
        scope='10k screen, sampling seed 0, one training run per cell'))
    with (LOGS / 'summary.csv').open('w', newline='') as stream:
        writer = csv.writer(stream)
        writer.writerow(('cell', 'model', 'route', 'step', 'lambda_max', 'fid_tweedie', 'fid_x_t',
                         'kid_tweedie', 'is_tweedie'))
        for name, value in cells.items():
            writer.writerow((name, value['model'], value['route'] or '', value['checkpoint_step'],
                             value['lambda_window'][1], value['fid_tweedie_xt'], value['fid_x_t'],
                             value['kid_tweedie_xt'], value['IS_tweedie_xt']))


def check_gate():
    """Stops the queue when the driver does not reproduce the known 65-call cells."""
    for name, expected in GATE.items():
        actual = json.loads(result_path(name).read_text())['fid_tweedie_xt']
        if abs(actual - expected) > GATE_TOLERANCE:
            print('Gate failed: {} gives {:.6f}, expected {:.6f}.'.format(name, actual, expected), flush=True)
            raise SystemExit(GATE_EXIT)
    print('Gate passed: both reference cells reproduced.', flush=True)


def queue():
    failed = []
    for name in CELLS:
        if not result_path(name).exists():
            try:
                run(name)
                report(name)
                if failure_path(name).exists():
                    failure_path(name).unlink()
            except (subprocess.CalledProcessError, RuntimeError, OSError, KeyError, ValueError) as error:
                write_json(failure_path(name), dict(error=repr(error), time=time.time()))
                print('Stage 0 cell failed: {}: {!r}'.format(name, error), flush=True)
                if name in GATE:
                    raise SystemExit(GATE_EXIT)
                failed.append(name)
                continue
        if name == list(GATE)[-1]:
            check_gate()
    print('Stage 0 queue finished, failed cells: {}'.format(failed or 'none'), flush=True)
    return 1 if failed else 0


def energy_probe(model):
    """Records state energy per network call; statistics only, outputs unchanged."""
    import torch

    class Probe(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.model = model
            self.rows = []

        def forward(self, x, labels):
            output = self.model(x, labels)
            self.rows.append(dict(state_sq=float(x.square().mean()),
                                  eps_sq=float(output.square().mean())))
            return output

    return Probe()


def energy_rows(sde, config, eps, rows):
    """Adds log-SNR and the state energy in VP-normalized coordinates."""
    import torch
    grid, levels = check_grid(config, sde, eps, len(rows) - 1)
    alpha, sigma = sde.marginal_prob(torch.ones(len(grid), 1, 1, 1, device=config.device, dtype=grid.dtype), grid)
    scale_sq = alpha.flatten().square() + sigma.square()
    return [dict(row, lam=float(level), scale_sq=float(value), normalized_state_sq=row['state_sq'] / float(value))
            for row, level, value in zip(rows, levels, scale_sq)]


def energy():
    """One batch per cell with the seed of round 0 of the same cell."""
    import torch
    from ode_sampling import get_vp_ode_sampler
    from reproducibility import derive_seed, isolated_torch_rng
    from run_lib import get_sde
    for name in ENERGY_CELLS:
        path = LOGS / (name + '_energy.json')
        if path.exists():
            continue
        spec = CELLS[name]
        config = get_config(name)
        model, identity = load_model(config, WORKDIRS[spec['model']], spec['ckpt'])
        sde, eps = get_sde(config, n_override=BUDGET)
        probe = energy_probe(model)
        sampler = get_vp_ode_sampler(sde, (BATCH, 3, 64, 64), lambda x: x, METHOD, eps, config.device)
        with torch.no_grad(), isolated_torch_rng(derive_seed(SEED, 'sampling-round', 0)):
            sampler(probe)
        write_json(path, dict(cell=name, batch=BATCH, checkpoint_id=identity['checkpoint_id'],
                              rows=energy_rows(sde, config, eps, probe.rows)))
        print('Energy', name, flush=True)
        del model, probe
        gc.collect()
        torch.cuda.empty_cache()


if __name__ == '__main__':
    mode = sys.argv[1] if len(sys.argv) > 1 else ''
    if mode == 'preflight':
        preflight()
    elif mode == 'queue':
        sys.exit(queue())
    elif mode == 'energy':
        energy()
    elif mode == 'plan':
        for cell_name, cell_spec in CELLS.items():
            print(cell_name, cell_spec)
    else:
        raise SystemExit('Usage: celeba_stage0_eval.py preflight|queue|energy|plan')
