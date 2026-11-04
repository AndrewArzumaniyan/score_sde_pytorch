"""Why controlled B@200k loses under PF-ODE while beating A under EM (CelebA, one GPU).

Three diagnostics on one A/B training pair, without changing training, samplers or queues:
  wrapper     score wrapper vs an independent label/sigma formula; ODE grid scale and labels;
  loss        paired epsilon-MSE versus log-SNR on CelebA test images;
  splice      DPM++ 2M, 256+1 calls from identical initial states: A, B and B only inside one
              log-SNR band (A elsewhere); FID after Tweedie on a small screening sample, plus
              per-call statistics and A/B disagreement along the pure A and pure B trajectories.
Screening FIDs use fewer images than the 10k queue and are only comparable within this run.
"""
import argparse
import hashlib
import json
import math
import time
from pathlib import Path

import numpy as np
import torch

import run_lib  # Initializes CUDA before TensorFlow and registers the models.
from run_lib import get_sde
import datasets
import evaluation
import e2_noise_level_error as e2
from artifact_utils import build_model_manifest
from configs.factorial.celeba_common import LOGSNR_MIN, LOGSNR_MAX
from losses import get_optimizer
from models import utils as mutils
from models.ema import ExponentialMovingAverage
from ode_sampling import get_vp_ode_sampler
from sampling import get_initial_sample
from utils import restore_checkpoint

REFERENCE = 'assets/stats/celeba_64_0c2d89311192_stats.npz'
REFERENCE_SHA = '66f3e831815ccebf90fe477348fcd0de92e5d966b7ed1eef77dc80fab72b6327'
METHOD, BUDGET = 'ode_dpmpp_2m', 256
# B inside [lo, hi) of log-SNR, A elsewhere; None means no B / all B.
BANDS = {'A': None, 'B': (-math.inf, math.inf), 'B_noisy_-10_0': (-math.inf, 0.0),
         'B_mid_0_5': (0.0, 5.0), 'B_clean_5_9.1': (5.0, math.inf)}


def model_name(seed, arm):
    label = 'A_vp_pvp' if arm == 'A' else 'B_vp_pfox'
    return 'celeba_factorial_{}_50k'.format(label) if seed == 42 else \
        'celeba_ab_replication_{}_seed{}_200k'.format(label, seed)


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1048576), b''):
            h.update(block)
    return h.hexdigest()


def save(path, obj):
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(obj, indent=2) + '\n')
    tmp.replace(path)


def load_arm(root, seed, arm, device):
    from ml_collections import ConfigDict
    work = root / 'workdirs' / model_name(seed, arm)
    manifest = json.loads((work / 'run_manifest.json').read_text())
    cfg = ConfigDict(manifest['config'])
    cfg.device = device
    if (cfg.seed != seed or cfg.training.sde != 'vpsde' or cfg.model.noise_conditioning != 'logsnr' or
            (cfg.model.logsnr_min, cfg.model.logsnr_max) != (LOGSNR_MIN, LOGSNR_MAX) or
            cfg.training.noise_distribution != ('vp_time' if arm == 'A' else 'normalized_fox_time')):
        raise RuntimeError('Unexpected controlled training protocol: ' + arm)
    cfg.sampling.time_grid = 'uniform_logsnr'
    cfg.sampling.logsnr_min, cfg.sampling.logsnr_max = LOGSNR_MIN, LOGSNR_MAX
    ckpt = 40 if seed == 42 else 4
    path = work / 'checkpoints' / 'checkpoint_{}.pth'.format(ckpt)
    model = mutils.create_model(cfg)
    ema = ExponentialMovingAverage(model.parameters(), decay=cfg.model.ema_rate)
    state = dict(model=model, optimizer=get_optimizer(cfg, model.parameters()), ema=ema, step=0,
                 model_protocol_sha256=build_model_manifest(cfg)['model_protocol_sha256'],
                 training_protocol_sha256=manifest['training_protocol_sha256'])
    state = restore_checkpoint(str(path), state, device, restore_rng=False)
    if int(state['step']) != 200001:
        raise RuntimeError('Expected the 200k checkpoint: ' + str(path))
    ema.copy_to(model.parameters())
    model.eval()
    return cfg, model, dict(checkpoint=str(path), checkpoint_id=state['checkpoint_id'], step=int(state['step']))


def label_of(lam):
    return (LOGSNR_MAX - lam) / (LOGSNR_MAX - LOGSNR_MIN) * 999.0


def lambda_of(label):
    return LOGSNR_MAX - label / 999.0 * (LOGSNR_MAX - LOGSNR_MIN)


def wrapper_check(sde, models, device):
    """Independent label/sigma/score vs the project wrapper on identical inputs."""
    out = dict(levels=[])
    torch.manual_seed(0)
    x = torch.randn(4, 3, 64, 64, device=device)
    for lam in np.linspace(LOGSNR_MIN, LOGSNR_MAX, 9):
        t = sde.time_from_log_snr(torch.tensor([lam], device=device, dtype=torch.float64)).float()
        actual = float(sde.log_snr(t.double()).item())  # level the float32 time actually encodes
        sigma = math.sqrt(1.0 / (1.0 + math.exp(actual)))  # VP: sigma^2 = sigmoid(-lambda)
        alpha, std = sde.marginal_prob(torch.ones(1, 1, 1, 1, device=device), t)
        row = dict(lam=float(lam), lambda_from_t=actual, sigma_independent=sigma, sigma_sde=float(std.item()),
                   scale=float(math.hypot(float(alpha.item()), float(std.item()))))
        for arm, model in models.items():
            wrapped = mutils.get_score_fn(sde, model, train=False, continuous=True)(x, t.expand(4))
            direct = -model(x, torch.full((4,), label_of(actual), device=device)) / sigma
            row[arm + '_max_rel_diff'] = float(((wrapped - direct).abs().max() / direct.abs().max()).item())
        out['levels'].append(row)
    grid = sde.sampling_time_grid(1e-3, grid='uniform_logsnr', device=device, dtype=torch.float64, N=BUDGET + 1)
    lam_grid = sde.log_snr(grid)
    alpha, std = sde.marginal_prob(torch.ones(len(grid), 1, 1, 1, device=device, dtype=torch.float64), grid)
    out['grid'] = dict(lambda_first=float(lam_grid[0]), lambda_last=float(lam_grid[-1]),
                       label_first=float(label_of(lam_grid[0])), label_last=float(label_of(lam_grid[-1])),
                       max_scale_error=float((torch.sqrt(alpha.flatten() ** 2 + std ** 2) - 1).abs().max()))
    worst = max(max(r['A_max_rel_diff'], r['B_max_rel_diff']) for r in out['levels'])
    out['passed'] = bool(worst < 1e-3 and out['grid']['max_scale_error'] < 1e-6 and
                         all(abs(r['lam'] - r['lambda_from_t']) < 3e-3 and
                             abs(r['sigma_sde'] - r['sigma_independent']) < 1e-5 for r in out['levels']))
    return out


def loss_profile(cfg, sde, models, images, levels, batch, device):
    scaler = datasets.get_data_scaler(cfg)
    errors = {arm: np.empty((len(levels), len(images)), np.float32) for arm in models}
    with torch.inference_mode():
        for i, lam in enumerate(levels):
            t = sde.time_from_log_snr(torch.tensor([lam], device=device, dtype=torch.float64)).float()
            alpha, sigma = (float(v.item()) for v in sde.marginal_prob(torch.ones(1, 1, 1, 1, device=device), t))
            for s in range(0, len(images), batch):
                clean = scaler(torch.from_numpy(images[s:s+batch]).to(device))
                noise = torch.from_numpy(e2.paired_noise(tuple(clean.shape), 20261101, i, s)).to(device)
                noisy = alpha * clean + sigma * noise
                for arm, model in models.items():
                    eps = model(noisy, torch.full((len(clean),), label_of(lam), device=device))
                    errors[arm][i, s:s+batch] = (eps - noise).square().flatten(1).mean(1).cpu().numpy()
            print('loss lambda=%+.3f A=%.6f B=%.6f' % (lam, errors['A'][i].mean(), errors['B'][i].mean()), flush=True)
    return errors


class Routed(torch.nn.Module):
    """Use B inside a log-SNR band and A elsewhere; optionally log both outputs per call."""

    def __init__(self, a, b, band):
        super().__init__()
        self.a, self.b, self.band, self.log = a, b, band, None

    def forward(self, x, labels):
        lam = float(lambda_of(labels[0]).item())
        use_b = self.band is not None and self.band[0] <= lam < self.band[1]
        out = (self.b if use_b else self.a)(x, labels)
        if self.log is not None:
            other = (self.a if use_b else self.b)(x, labels)
            self.log.append(dict(lam=lam, used='B' if use_b else 'A',
                                 state_sq=float(x.square().mean()), eps_sq=float(out.square().mean()),
                                 other_eps_sq=float(other.square().mean()),
                                 disagreement_sq=float((out - other).square().mean())))
        return out


def frechet(pool, ref_mean, ref_cov):
    from scipy import linalg
    mean, cov = pool.mean(0), np.cov(pool, rowvar=False)
    root = linalg.sqrtm(cov @ ref_cov, disp=False)[0].real
    return float(np.square(mean - ref_mean).sum() + np.trace(cov + ref_cov - 2 * root))


def splice_screen(cfg, sde, models, samples, batch, out, device):
    inverse = datasets.get_data_inverse_scaler(cfg)
    inception = evaluation.get_inception_model()
    with np.load(REFERENCE) as z:
        ref = z['pool_3']
    ref_mean = np.zeros(ref.shape[1])
    for s in range(0, len(ref), 8192):
        ref_mean += ref[s:s+8192].astype(np.float64).sum(0)
    ref_mean /= len(ref)
    ref_cov = np.zeros((ref.shape[1], ref.shape[1]))
    for s in range(0, len(ref), 8192):
        block = ref[s:s+8192].astype(np.float64) - ref_mean
        ref_cov += block.T @ block
    ref_cov /= len(ref) - 1
    del ref
    sampler = get_vp_ode_sampler(sde, (batch, 3, 64, 64), inverse, METHOD, 1e-3, device)
    grid = sde.sampling_time_grid(1e-3, grid='uniform_logsnr', device=device, dtype=torch.float64, N=BUDGET + 1)
    results = {}
    for name, band in BANDS.items():
        path = out / 'splice_{}.npz'.format(name)
        if path.exists():
            with np.load(path) as z:
                results[name] = dict(fid_tweedie_xt=float(z['fid_tweedie_xt']), fid_x_t=float(z['fid_x_t']))
            continue
        began = time.monotonic()
        routed = Routed(models['A'], models['B'], band).eval()
        pools, logs, first = {'x_t': [], 'tweedie_xt': []}, [], None
        for r in range(samples // batch):
            torch.manual_seed(20261101 + r)  # identical initial states for every configuration
            z = get_initial_sample(sde, (batch, 3, 64, 64), grid, device)
            routed.log = [] if r == 0 and name in ('A', 'B') else None
            outputs, calls = sampler(routed, z=z)
            if routed.log is not None:
                logs = routed.log
            for variant, images in outputs.items():
                images = np.clip(images.permute(0, 2, 3, 1).cpu().numpy() * 255., 0, 255).astype(np.uint8)
                if r == 0 and variant == 'tweedie_xt':
                    first = images[:64]
                pools[variant].append(evaluation.run_inception_distributed(images, inception)['pool_3'].numpy())
        pools = {k: np.concatenate(v).astype(np.float64) for k, v in pools.items()}
        fids = {'fid_' + k: frechet(v, ref_mean, ref_cov) for k, v in pools.items()}
        np.savez(path, pool_tweedie_xt=pools['tweedie_xt'].astype(np.float32), images=first,
                 calls=calls, seconds=time.monotonic() - began, **fids)
        if logs:
            save(out / 'trajectory_{}.json'.format(name), logs)
        results[name] = fids
        print('splice {} band={} FID tweedie={:.3f} raw={:.3f} ({:.0f}s)'.format(
            name, band, fids['fid_tweedie_xt'], fids['fid_x_t'], time.monotonic() - began), flush=True)
    return results


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--root', type=Path, required=True, help='directory containing workdirs/<A,B model>')
    ap.add_argument('--seed', type=int, choices=(42, 43, 44), default=42)
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--images', type=int, default=1024)
    ap.add_argument('--samples', type=int, default=2048)
    ap.add_argument('--batch', type=int, default=512)
    ap.add_argument('--pilot', action='store_true', help='tiny sizes for a smoke run')
    a = ap.parse_args()
    if a.pilot:
        a.images, a.samples, a.batch = 64, 64, 64
    if a.samples % a.batch or digest(REFERENCE) != REFERENCE_SHA:
        raise ValueError('samples must be a multiple of batch; reference must match')
    if not torch.cuda.is_available():
        raise RuntimeError('Requires one GPU')
    device = 'cuda:0'
    torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = True
    a.out.mkdir(parents=True, exist_ok=True)
    cfg_a, model_a, id_a = load_arm(a.root, a.seed, 'A', device)
    cfg_b, model_b, id_b = load_arm(a.root, a.seed, 'B', device)
    if cfg_a.model.to_dict() != cfg_b.model.to_dict() or cfg_a.data.to_dict() != cfg_b.data.to_dict():
        raise RuntimeError('A/B model or data configs differ')
    models = {'A': model_a, 'B': model_b}
    sde, _ = get_sde(cfg_a, n_override=BUDGET)
    save(a.out / 'manifest.json', dict(seed=a.seed, A=id_a, B=id_b, images=a.images, samples=a.samples,
         batch=a.batch, method=METHOD, budget=BUDGET, bands={k: v and list(v) for k, v in BANDS.items()},
         reference_sha256=REFERENCE_SHA, script_sha256=digest(__file__), device=torch.cuda.get_device_name(0)))
    with torch.inference_mode():
        check = wrapper_check(sde, models, device)
    save(a.out / 'wrapper.json', check)
    print('wrapper check passed' if check['passed'] else 'WRAPPER CHECK FAILED', flush=True)
    if not check['passed']:
        raise SystemExit(1)
    levels = np.linspace(LOGSNR_MIN, LOGSNR_MAX, 24)
    images, source = e2.load_test_images(cfg_a, a.images)
    errors = loss_profile(cfg_a, sde, models, images, levels, a.batch, device)
    diff = errors['B'].astype(np.float64) - errors['A']
    np.savez(a.out / 'loss.npz', levels=levels, A=errors['A'], B=errors['B'], source=source)
    np.savetxt(a.out / 'loss.csv', np.column_stack((levels, errors['A'].mean(1), errors['B'].mean(1), diff.mean(1),
               diff.std(1, ddof=1) / math.sqrt(len(images)))), delimiter=',', comments='',
               header='lambda,A_eps_mse,B_eps_mse,B_minus_A,paired_se')
    results = splice_screen(cfg_a, sde, models, a.samples, a.batch, a.out, device)
    save(a.out / 'summary.json', dict(wrapper_passed=check['passed'], splice=results,
         note='screening FID on {} samples; compare only within this run'.format(a.samples)))
    print('done', flush=True)


if __name__ == '__main__':
    main()
