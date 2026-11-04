"""Conditional independent-cell bootstrap for completed matched35 RTX features.

Run only in the CPU Docker launcher. Each cell is resampled independently; the
reference and checkpoints are fixed. Sampling seeds are reported separately.
"""
import argparse
import hashlib
import json
import time
import zipfile
from pathlib import Path

import numpy as np
from scipy import linalg
import celeba_final_ode_bootstrap as base

N, BATCH, DIM = base.N, base.BATCH, base.DIM
METHODS = ('ode_edm_heun', 'ode_dpmpp_2m')
SEED = 20261104


def specs():
    return [dict(name=f'{method}_{arm}_seed{seed}', method=method, arm=arm, seed=seed)
            for seed in (0, 1) for method in METHODS for arm in ('vp', 'fox')]


def directory(root, s):
    return root / 'workdirs' / base.WORK[s['arm']] / (
        f"eval_matched35rtx_{s['method']}_nfe34_{s['arm']}_50000_seed{s['seed']}")


def contrasts(values):
    out = {}
    for seed in (0, 1):
        get = lambda method, arm: values[f'{method}_{arm}_seed{seed}']
        for arm in ('vp', 'fox'):
            out[f'dpmpp_minus_heun_{arm}_seed{seed}'] = get(METHODS[1], arm) - get(METHODS[0], arm)
        for method in METHODS:
            out[f'fox_minus_vp_{method}_seed{seed}'] = get(method, 'fox') - get(method, 'vp')
        out[f'interaction_seed{seed}'] = (out[f'dpmpp_minus_heun_fox_seed{seed}'] -
                                            out[f'dpmpp_minus_heun_vp_seed{seed}'])
    return out


def prepare(root, reference, out):
    if base.digest(reference) != base.REFERENCE_SHA:
        raise ValueError('Reference SHA256 mismatch')
    if (root / 'logs/queue.state').read_text().strip() != 'completed':
        raise ValueError('Queue not completed')
    summary = json.loads((root / 'summary.json').read_text())
    pre = json.loads((root / 'logs/preflight.json').read_text())
    if (summary['status'] != 'completed' or summary['completed_cells'] != 8 or
            summary['total_cells'] != 8 or pre['status'] != 'passed' or
            pre['checkpoint_ids'] != base.CHECKPOINT_IDS):
        raise ValueError('Unexpected queue summary or checkpoints')
    cells = {(c['method'], c['arm'], c['sampling_seed']): c for c in summary['cells']}
    if set(cells) != {(s['method'], s['arm'], s['seed']) for s in specs()}:
        raise ValueError('Unexpected summary cells')
    audit = dict(reference_sha256=base.REFERENCE_SHA, source_sha256=pre['source_sha256'],
                 queue_summary_sha256=base.digest(root / 'summary.json'), sets=[])
    for s in specs():
        d = directory(root, s)
        manifest = json.loads((d / 'manifest.json').read_text())
        cfg, ev = manifest['config'], manifest['config']['eval']
        if (manifest['source_sha256'] != pre['source_sha256'] or
            (ev['sampling_seed'], ev['num_samples'], ev['batch_size'], ev['sampling_num_scales']) !=
                (s['seed'], N, BATCH, 34) or cfg['sampling']['method'] != s['method']):
            raise ValueError(f'Unexpected manifest {d}')
        with np.load(d / 'report_260.npz', allow_pickle=False) as z:
            expected = dict(checkpoint_id=base.CHECKPOINT_IDS[s['arm']], checkpoint_step=1300002,
                            dataset_stats_id=base.REFERENCE_SHA, num_samples=N, nfe=35)
            for k, v in expected.items():
                if base.scalar(z, k) != v:
                    raise ValueError(f'Incorrect report {k}: {d}')
            report_fid = float(base.scalar(z, 'fid_tweedie_xt'))
            protocol = base.scalar(z, 'protocol_sha256')
            training = base.scalar(z, 'training_protocol_sha256')
        if not np.isfinite(report_fid) or abs(report_fid - cells[(s['method'], s['arm'], s['seed'])]['fid_tweedie_xt']) > 1e-10:
            raise ValueError(f'Summary and report differ: {d}')
        files = [d / 'ckpt_260' / f'statistics_{i}.npz' for i in range(base.ROUNDS)]
        if set((d / 'ckpt_260').glob('statistics_*.npz')) != set(files):
            raise ValueError(f'Incorrect shards: {d}')
        x = np.lib.format.open_memmap(out / (s['name'] + '.npy'), mode='w+', dtype=np.float32, shape=(N, DIM))
        hashes = []
        for i, path in enumerate(files):
            hashes.append(base.digest(path))
            with np.load(path, allow_pickle=False) as z:
                expected = dict(round_id=i, sampling_seed=base.round_seed(s['seed'], i), num_samples=BATCH,
                                nfe=35, checkpoint_id=base.CHECKPOINT_IDS[s['arm']], checkpoint_step=1300002,
                                protocol_sha256=protocol, training_protocol_sha256=training)
                if any(base.scalar(z, k) != v for k, v in expected.items()):
                    raise ValueError(f'Shard identity mismatch: {path}')
                a = z['pool_3_tweedie_xt']
                if a.shape != (BATCH, DIM) or a.dtype != np.float32 or not np.isfinite(a).all():
                    raise ValueError(f'Invalid features: {path}')
                take = min(BATCH, N - i * BATCH)
                x[i * BATCH:i * BATCH + take] = a[:take]
        x.flush()
        del x
        audit['sets'].append(dict(**s, report_fid=report_fid, shard_sha256=hashes,
                                  report_sha256=base.digest(d / 'report_260.npz'),
                                  manifest_sha256=base.digest(d / 'manifest.json'),
                                  feature_sha256=base.digest(out / (s['name'] + '.npy'))))
        print('Audited', s['name'], flush=True)
    raw = out / 'reference_pool.npy'
    with zipfile.ZipFile(reference) as z, z.open('pool_3.npy') as src, raw.open('wb') as dst:
        for block in iter(lambda: src.read(1048576), b''):
            dst.write(block)
    mean, cov = base.moments(np.load(raw, mmap_mode='r'))
    raw.unlink()
    vals, vecs = linalg.eigh(cov, check_finite=False)
    sqrt = (vecs * np.sqrt(np.maximum(vals, 0))) @ vecs.T
    np.savez(out / 'reference_moments.npz', mean=mean, cov=cov, sqrt=sqrt)
    ref = (mean, cov, sqrt)
    checks = []
    for s in audit['sets']:
        f = base.fid(np.load(out / (s['name'] + '.npy'), mmap_mode='r'), ref)
        error = f - s['report_fid']
        checks.append(dict(name=s['name'], recomputed_fid=f, report_fid=s['report_fid'], error=error))
        print('FID check', checks[-1], flush=True)
        if not np.isfinite(f) or abs(error) > base.TOLERANCE:
            raise ValueError('Point FID differs by more than 0.001')
    audit['reference_moments_sha256'] = base.digest(out / 'reference_moments.npz')
    base.save(out / 'audit.json', audit)
    base.save(out / 'point_checks.json', checks)


def bootstrap(out, reps):
    audit = json.loads((out / 'audit.json').read_text())
    if base.digest(out / 'reference_moments.npz') != audit['reference_moments_sha256']:
        raise ValueError('Reference moments changed')
    features = {}
    for s in audit['sets']:
        path = out / (s['name'] + '.npy')
        if base.digest(path) != s['feature_sha256']:
            raise ValueError(f'Features changed: {path}')
        features[s['name']] = np.load(path, mmap_mode='r')
    checks = json.loads((out / 'point_checks.json').read_text())
    points = {x['name']: x['report_fid'] for x in checks}
    with np.load(out / 'reference_moments.npz') as z:
        ref = (z['mean'], z['cov'], z['sqrt'])
    path = out / 'draws.jsonl'
    rows = [json.loads(line) for line in path.read_text().splitlines()] if path.exists() else []
    if len(rows) > reps or any(r['rep'] != i or set(r['fid']) != set(features) for i, r in enumerate(rows)):
        raise ValueError('Existing draws inconsistent with requested run')
    for rep in range(len(rows), reps):
        began = time.monotonic()
        values = {}
        for index, s in enumerate(audit['sets']):
            rng = np.random.default_rng(np.random.SeedSequence([SEED, rep, index]))
            weights = np.bincount(rng.integers(0, N, N), minlength=N)
            values[s['name']] = base.fid(features[s['name']], ref, weights)
        if not all(np.isfinite(v) for v in values.values()):
            raise ValueError('Nonfinite bootstrap FID')
        row = dict(rep=rep, fid=values, contrasts=contrasts(values), seconds=time.monotonic() - began)
        with path.open('a') as f:
            f.write(json.dumps(row, allow_nan=False) + '\n')
            f.flush()
        rows.append(row)
        print(f"{rep+1}/{reps}, {row['seconds']:.1f}s", flush=True)
    result = {}
    for name, point in contrasts(points).items():
        values = np.asarray([row['contrasts'][name] for row in rows])
        lo, median, hi = np.quantile(values, [.025, .5, .975])
        result[name] = dict(point=point, percentile_95=[float(lo), float(hi)],
                            basic_95=[float(2*point-hi), float(2*point-lo)],
                            bootstrap_median=float(median))
    base.save(out / 'summary.json', dict(status='completed', repeats=len(rows), bootstrap_seed=SEED,
        uncertainty='generated-sample uncertainty conditional on fixed checkpoints and reference',
        resampling='independent per cell; separate reported sampling seeds',
        interval='95% percentile; basic also reported; no multiplicity correction',
        point_fid=points, contrasts=result))


def check():
    base.check()
    values = {s['name']: float(i) for i, s in enumerate(specs())}
    c = contrasts(values)
    assert len(c) == 10 and c['dpmpp_minus_heun_vp_seed0'] == 2
    assert c['fox_minus_vp_ode_edm_heun_seed1'] == 1
    assert c['interaction_seed1'] == 0
    print('matched35 bootstrap checks passed', flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--check', action='store_true')
    p.add_argument('--prepare', action='store_true')
    p.add_argument('--root', type=Path)
    p.add_argument('--reference', type=Path)
    p.add_argument('--out', type=Path)
    p.add_argument('--reps', type=int, default=3)
    a = p.parse_args()
    if a.check:
        check()
        return
    if not a.root or not a.reference or not a.out or a.reps < 1:
        p.error('root, reference, out and positive reps required')
    if a.prepare:
        prepare(a.root, a.reference, a.out)
    bootstrap(a.out, a.reps)


if __name__ == '__main__':
    main()
