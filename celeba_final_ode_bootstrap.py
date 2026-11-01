"""Fixed-reference FID bootstrap for the final VP/native FOX ODE queue (run in Docker, CPU only).

Inputs: the completed celeba_final_ode_20261031 output (50k cells) and, optionally,
the historical extended EM 50k seed-0 evals for the EM comparison. NumPy/SciPy only,
no project imports, models or image generation.

Resampling: the same row multiset for every ODE feature set (rows share derived round
seeds across arms and methods, so (VP_i, FOX_i, ...) are i.i.d. tuples); the EM pair gets
its own independent multiset. Scope: fixed checkpoints and fixed empirical reference;
finite generated-sample uncertainty only, not training-seed or sampling-seed variation.
"""
import argparse
import hashlib
import json
import resource
import time
import zipfile
from pathlib import Path

import numpy as np
from scipy import linalg

N, BATCH, DIM = 50000, 512, 2048
ROUNDS = (N + BATCH - 1) // BATCH
REFERENCE = 'assets/stats/celeba_64_0c2d89311192_stats.npz'
REFERENCE_SHA = '66f3e831815ccebf90fe477348fcd0de92e5d966b7ed1eef77dc80fab72b6327'
WORK = {'vp': 'celeba_vp_continuous_50k', 'fox': 'celeba_fox_matern32_kappa1000_50k'}
CHECKPOINT_IDS = {
    'vp': 'legacy-sha256-51571fa866db54207c7c74d8b58e9991241ec50f9fac40f6cf39f4d22440833d',
    'fox': 'legacy-sha256-86ab70f2ce55dde46453eca08073ade220693e866f722e0e64fe5d1ce0e7f7e2'}
METHODS = ('ode_dpmpp_2m', 'ode_heun')
VARIANTS = ('x_t', 'tweedie_xt')
BOOT_SEED = 20261101
TOLERANCE = 0.001


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


def scalar(z, key):
    return np.asarray(z[key]).item()


def round_seed(base, r):
    """Same as reproducibility.derive_seed(base, 'sampling-round', r)."""
    payload = '\0'.join((str(base), 'sampling-round', str(r)))
    return int.from_bytes(hashlib.sha256(payload.encode()).digest()[:8], 'big') % (2**31 - 1)


def moments(x, weights=None, block=2048):
    count = len(x) if weights is None else int(weights.sum())
    total = np.zeros(x.shape[1])
    for s in range(0, len(x), block):
        a = np.asarray(x[s:s+block], dtype=np.float64)
        total += a.sum(0) if weights is None else weights[s:s+len(a)] @ a
    mean = total / count
    cov = np.zeros((x.shape[1], x.shape[1]))
    for s in range(0, len(x), block):
        a = np.asarray(x[s:s+block], dtype=np.float64) - mean
        if weights is not None:
            a *= np.sqrt(weights[s:s+len(a)])[:, None]
        cov += a.T @ a
    return mean, cov / (count - 1)


def fid(x, ref, weights=None):
    real_mean, real_cov, real_sqrt = ref
    mean, cov = moments(x, weights)
    cross = real_sqrt @ cov @ real_sqrt
    eig = linalg.eigvalsh((cross + cross.T) / 2, check_finite=False, driver='evr')
    return float(np.square(mean - real_mean).sum() + np.trace(real_cov) + np.trace(cov)
                 - 2 * np.sqrt(np.maximum(eig, 0)).sum())


def feature_sets(with_em):
    sets = [dict(name=f'{m}_{a}_{v}', kind='ode', method=m, arm=a, variant=v)
            for m in METHODS for a in ('vp', 'fox') for v in VARIANTS]
    if with_em:
        sets += [dict(name=f'em_{a}_tweedie_xt', kind='em', arm=a, variant='tweedie_xt') for a in ('vp', 'fox')]
    return sets


def eval_dir(spec, ode_root, em_root):
    if spec['kind'] == 'ode':
        d = ode_root / 'workdirs' / WORK[spec['arm']] / 'eval_final_ode_{}_{}_{}_seed0'.format(spec['method'], spec['arm'], N)
        if not d.is_dir():
            raise ValueError(f'Missing ODE eval: {d}')
        return d
    pattern = f"eval_e4_endpoint_denoise_*_{spec['arm']}_extended_ckpt260_nfe1000_{N}samples_batch512_seed0"
    found = sorted((em_root / 'workdirs' / WORK[spec['arm']]).glob(pattern))
    if len(found) != 1:
        raise ValueError(f'Expected one EM eval {pattern}; found {found}')
    return found[0]


def prepare(ode_root, em_root, reference, out):
    if digest(reference) != REFERENCE_SHA:
        raise ValueError('Reference SHA256 mismatch')
    if (ode_root / 'logs/queue.state').read_text().strip() != 'completed':
        raise ValueError('ODE queue is not completed')
    summary = json.loads((ode_root / 'summary.json').read_text())
    pre = json.loads((ode_root / 'logs/preflight.json').read_text())
    if summary['status'] != 'completed' or pre['status'] != 'passed' or pre['checkpoint_ids'] != CHECKPOINT_IDS:
        raise ValueError('Unexpected ODE summary/preflight')
    audit = dict(reference_sha256=REFERENCE_SHA, ode_root=str(ode_root), em_root=str(em_root) if em_root else None,
                 ode_source_sha256=pre['source_sha256'], sets=[])
    for spec in feature_sets(em_root is not None):
        directory = eval_dir(spec, ode_root, em_root)
        nfe = 257 if spec['kind'] == 'ode' else 1001
        manifest = json.loads((directory / 'manifest.json').read_text())
        ev = manifest['config']['eval']
        if (ev['sampling_seed'], ev['num_samples'], ev['batch_size']) != (0, N, BATCH):
            raise ValueError(f'Unexpected eval config: {directory}')
        if spec['kind'] == 'ode' and manifest['source_sha256'] != pre['source_sha256']:
            raise ValueError(f'Unexpected ODE source: {directory}')
        with np.load(directory / 'report_260.npz', allow_pickle=False) as z:
            expected = dict(checkpoint_id=CHECKPOINT_IDS[spec['arm']], checkpoint_step=1300002,
                            dataset_stats_id=REFERENCE_SHA, num_samples=N, nfe=nfe)
            for key, value in expected.items():
                if scalar(z, key) != value:
                    raise ValueError(f'Report {key} mismatch: {directory}')
            report_fid = float(scalar(z, 'fid_' + spec['variant']))
        files = [directory / 'ckpt_260' / f'statistics_{i}.npz' for i in range(ROUNDS)]
        if sorted(p.name for p in (directory / 'ckpt_260').glob('statistics_*.npz')) != sorted(p.name for p in files):
            raise ValueError(f'Unexpected statistics shards: {directory}')
        x = np.lib.format.open_memmap(out / f"{spec['name']}.npy", mode='w+', dtype=np.float32, shape=(N, DIM))
        for i, path in enumerate(files):
            with np.load(path, allow_pickle=False) as z:
                if (scalar(z, 'round_id'), scalar(z, 'sampling_seed'), scalar(z, 'num_samples'), scalar(z, 'nfe')) != \
                        (i, round_seed(0, i), BATCH, nfe):
                    raise ValueError(f'Shard metadata mismatch: {path}')
                if scalar(z, 'checkpoint_id') != CHECKPOINT_IDS[spec['arm']]:
                    raise ValueError(f'Shard checkpoint mismatch: {path}')
                a = z['pool_3_' + spec['variant']]
                if a.shape != (BATCH, DIM) or a.dtype != np.float32 or not np.isfinite(a).all():
                    raise ValueError(f'Bad features: {path}')
                take = min(BATCH, N - i * BATCH)
                x[i*BATCH:i*BATCH+take] = a[:take]
        x.flush()
        del x
        audit['sets'].append(dict(spec, directory=str(directory), report_fid=report_fid))
        print('audited', spec['name'], flush=True)
    save(out / 'audit.json', audit)
    raw = out / 'reference_pool.npy'
    with zipfile.ZipFile(reference) as z, z.open('pool_3.npy') as src, raw.open('wb') as dst:
        for block in iter(lambda: src.read(1048576), b''):
            dst.write(block)
    mean, cov = moments(np.load(raw, mmap_mode='r'))
    raw.unlink()
    vals, vecs = linalg.eigh(cov, check_finite=False)
    np.savez(out / 'reference_moments.npz', mean=mean, cov=cov, sqrt=(vecs * np.sqrt(np.maximum(vals, 0))) @ vecs.T)
    ref = (mean, cov, (vecs * np.sqrt(np.maximum(vals, 0))) @ vecs.T)
    checks = []
    for s in audit['sets']:
        value = fid(np.load(out / f"{s['name']}.npy", mmap_mode='r'), ref)
        checks.append(dict(name=s['name'], recomputed_fid=value, report_fid=s['report_fid'], error=value - s['report_fid']))
        print(checks[-1], flush=True)
        if abs(checks[-1]['error']) > TOLERANCE:
            raise ValueError(f'Point FID mismatch exceeds predeclared tolerance {TOLERANCE}')
    save(out / 'point_checks.json', checks)


def contrasts(f):
    """Named FID differences from a dict of set name -> FID."""
    c = {}
    for m in METHODS:
        for v in VARIANTS:
            c[f'fox_minus_vp_{m}_{v}'] = f[f'{m}_fox_{v}'] - f[f'{m}_vp_{v}']
    for a in ('vp', 'fox'):
        for v in VARIANTS:
            c[f'heun_minus_dpmpp_{a}_{v}'] = f[f'ode_heun_{a}_{v}'] - f[f'ode_dpmpp_2m_{a}_{v}']
        c[f'tweedie_minus_raw_dpmpp_{a}'] = f[f'ode_dpmpp_2m_{a}_tweedie_xt'] - f[f'ode_dpmpp_2m_{a}_x_t']
    if 'em_vp_tweedie_xt' in f:
        c['fox_minus_vp_em_tweedie_xt'] = f['em_fox_tweedie_xt'] - f['em_vp_tweedie_xt']
        for m in METHODS:
            c[f'gap_change_{m}_tweedie_xt'] = c[f'fox_minus_vp_{m}_tweedie_xt'] - c['fox_minus_vp_em_tweedie_xt']
            for a in ('vp', 'fox'):
                c[f'ode_minus_em_{m}_{a}_tweedie_xt'] = f[f'{m}_{a}_tweedie_xt'] - f[f'em_{a}_tweedie_xt']
    return c


def bootstrap(out, reps):
    audit = json.loads((out / 'audit.json').read_text())
    point = {c['name']: c['recomputed_fid'] for c in json.loads((out / 'point_checks.json').read_text())}
    with np.load(out / 'reference_moments.npz') as z:
        ref = (z['mean'], z['cov'], z['sqrt'])
    features = {s['name']: np.load(out / f"{s['name']}.npy", mmap_mode='r') for s in audit['sets']}
    path = out / 'draws.jsonl'
    rows = [json.loads(line) for line in path.read_text().splitlines()] if path.exists() else []
    for rep in range(len(rows), reps):
        began = time.monotonic()
        rng = np.random.default_rng(np.random.SeedSequence([BOOT_SEED, rep]))
        weights = {kind: np.bincount(rng.integers(0, N, N), minlength=N) for kind in ('ode', 'em')}
        values = {s['name']: fid(features[s['name']], ref, weights[s['kind']]) for s in audit['sets']}
        row = dict(rep=rep, fid=values, contrasts=contrasts(values), seconds=time.monotonic() - began,
                   peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
        with path.open('a') as f:
            f.write(json.dumps(row) + '\n')
        rows.append(row)
        print(f"{len(rows)}/{reps} dpmpp FOX-VP={row['contrasts']['fox_minus_vp_ode_dpmpp_2m_tweedie_xt']:.4f} "
              f"{row['seconds']:.1f}s peak={row['peak_rss_kib']/1024:.0f} MiB", flush=True)
    point_c = contrasts(point)
    result = {}
    for name, p in point_c.items():
        draws = np.array([r['contrasts'][name] for r in rows])
        lo, mid, hi = np.quantile(draws, [.025, .5, .975])
        result[name] = dict(point=p, percentile_95=[lo, hi], basic_95=[2*p - hi, 2*p - lo], median=mid,
                            sd=float(draws.std(ddof=1)) if len(draws) > 1 else None,
                            opposite_sign_fraction=float(np.mean(np.sign(draws) != np.sign(p))))
    save(out / 'summary.json', dict(reps=len(rows), bootstrap_seed=BOOT_SEED, point_fid=point, contrasts=result,
        resampling='one row multiset shared by all ODE sets; independent multiset for the EM pair; fixed reference',
        scope='fixed checkpoints, sampling seed 0, fixed reference; finite generated-sample uncertainty only'))


def check():
    """Synthetic checks of the FID core; no experiment inputs."""
    rng = np.random.default_rng(0)
    x, real = rng.normal(size=(300, 8)), rng.normal(0.3, 1.2, size=(400, 8))
    m, c = moments(real)
    vals, vecs = linalg.eigh(c)
    ref = (m, c, (vecs * np.sqrt(vals)) @ vecs.T)
    mx, cx = np.mean(x, 0), np.cov(x, rowvar=False)
    direct = np.square(mx - m).sum() + np.trace(c + cx) - 2 * np.trace(linalg.sqrtm(cx @ c)).real
    assert abs(fid(x, ref) - direct) < 1e-8, 'FID core mismatch'
    w = np.bincount(rng.integers(0, 300, 300), minlength=300)
    assert abs(fid(x, ref, w) - fid(np.repeat(x, w, axis=0), ref)) < 1e-8, 'Weighted moments mismatch'
    assert round_seed(0, 0) == int.from_bytes(hashlib.sha256(b'0\x00sampling-round\x000').digest()[:8], 'big') % (2**31 - 1)
    names = {s['name']: 1.0 for s in feature_sets(True)}
    assert len(contrasts(names)) == 17
    print('check passed', flush=True)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--check', action='store_true')
    ap.add_argument('--ode', type=Path)
    ap.add_argument('--em', type=Path, help='repository with historical E4 extended evals; omit to skip EM')
    ap.add_argument('--reference', type=Path)
    ap.add_argument('--out', type=Path)
    ap.add_argument('--prepare', action='store_true')
    ap.add_argument('--reps', type=int, default=3)
    a = ap.parse_args()
    if a.check:
        check()
    else:
        a.out.mkdir(parents=True, exist_ok=True)
        if a.prepare:
            prepare(a.ode, a.em, a.reference, a.out)
        bootstrap(a.out, a.reps)
