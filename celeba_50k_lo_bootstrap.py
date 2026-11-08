"""Conditional independent-cell bootstrap for the 16 CelebA cells at 50k on the lower window.

A at 400k, VP at 200k, FOX at 800k and 1.3M, training seed 42; DPM++ and EDM Heun with 34
calls plus Tweedie, sampling seeds 0 and 1, window [-10; 9.115429865459795]. Run only in the
CPU Docker launcher run_celeba_50k_lo_bootstrap.sh. Each cell is resampled independently.

An interval shows the spread of generated samples at fixed weights and a fixed reference.
It does not show the spread between trainings: there is one training per model. Sampling
seeds are reported separately, and the contrasts ending in _mean average the two.

The cells come from two roots. celeba_50k_lo_eval.py has gpu1 with A and VP and gpu2 with
FOX at 1.3M; celeba_fox800_lo_eval.py has FOX at 800k in a root of its own, from another
commit, so sources are compared inside a root only. Identities are repeated here because
this image lacks the evaluation stack.
"""
import argparse
import json
import time
import zipfile
from pathlib import Path

import numpy as np
from scipy import linalg
import celeba_final_ode_bootstrap as base

N, BATCH, DIM = base.N, base.BATCH, base.DIM
METHODS = ('ode_edm_heun', 'ode_dpmpp_2m')
SEED = 20261107
LOGSNR_MAX = 9.115429865459795
MODELS = {
    'A': dict(lane='gpu1', work='celeba_factorial_A_vp_pvp_400k_rtx_seed42', ckpt=80, step=400001,
              checkpoint_id='7de461feeceb4a488583921d9f7f132d'),
    'vp': dict(lane='gpu1', work='celeba_vp_continuous_50k', ckpt=40, step=200001,
               checkpoint_id='legacy-sha256-922cc63cb505177fe0d86ca919d39bc0b68a221706c80b75b14bd7910f352073'),
    'fox': dict(lane='gpu2', work='celeba_fox_matern32_kappa1000_50k', ckpt=260, step=1300002,
                checkpoint_id='legacy-sha256-86ab70f2ce55dde46453eca08073ade220693e866f722e0e64fe5d1ce0e7f7e2'),
    'fox800': dict(lane='fox800', work='celeba_fox_matern32_kappa1000_50k', ckpt=160, step=800001, cell_model='fox',
                   checkpoint_id='legacy-sha256-7874545fa077e31812c4ca6cddf196d6102ec91e4ce88423a6ef8f43bc35b148'),
}
LANES = {'gpu1': ('A', 'vp'), 'gpu2': ('fox',), 'fox800': ('fox800',)}
GROUPS = (('gpu1', 'gpu2'), ('fox800',))  # Lanes of one root share their sources.
PAIRS = (('fox800', 'A'), ('fox800', 'vp'), ('fox800', 'fox'), ('fox', 'A'), ('fox', 'vp'), ('A', 'vp'))
INTERACTIONS = (('fox800', 'A'), ('fox', 'A'), ('fox', 'vp'))
MAIN = ('fox800', 'A')
SCOPE = ('The intervals show the spread of samples at fixed weights, not the spread between trainings; '
         'one training per model, and the models have unequal training lengths.')


def specs():
    return [dict(name=f'{method}_{model}_seed{seed}', method=method, model=model, seed=seed)
            for seed in (0, 1) for method in METHODS for model in MODELS]


def cell_model(model):
    return MODELS[model].get('cell_model', model)


def lane_root(roots, lane):
    return roots['fox800'] if lane == 'fox800' else roots['lo'] / lane


def directory(roots, s):
    m = MODELS[s['model']]
    folder = 'eval_fox800_lo' if m['lane'] == 'fox800' else 'eval_50k_lo'
    return lane_root(roots, m['lane']) / 'workdirs' / m['work'] / (
        f"{folder}_{cell_model(s['model'])}_c{m['ckpt']}_lo_{s['method']}_seed{s['seed']}")


def contrasts(values):
    """Left minus right, within one sampling seed; the models differ in training length."""
    out = {}
    for seed in (0, 1):
        get = lambda method, model: values[f'{method}_{model}_seed{seed}']
        for model in MODELS:
            out[f'dpmpp_minus_heun_{model}_seed{seed}'] = get(METHODS[1], model) - get(METHODS[0], model)
        for left, right in PAIRS:
            for method in METHODS:
                out[f'{left}_minus_{right}_{method}_seed{seed}'] = get(method, left) - get(method, right)
        for left, right in INTERACTIONS:
            out[f'interaction_{left}_{right}_seed{seed}'] = (out[f'dpmpp_minus_heun_{left}_seed{seed}'] -
                                                             out[f'dpmpp_minus_heun_{right}_seed{seed}'])
    left, right = MAIN
    for method in METHODS:  # The sign of the comparison rule is read from this mean.
        name = f'{left}_minus_{right}_{method}'
        out[name + '_mean'] = (out[name + '_seed0'] + out[name + '_seed1']) / 2
    return out


def read_lanes(roots):
    """One source inside each root; returns the source of each lane, the cells and the summary hashes."""
    cells, sources, hashes = {}, {}, {}
    for lane, models in LANES.items():
        logs = lane_root(roots, lane) / 'logs'
        summary = json.loads((logs / 'summary.json').read_text())
        pre = json.loads((logs / 'preflight.json').read_text())
        total = 4 * len(models)
        if (summary['status'] != 'completed' or summary['completed_cells'] != total or
                summary['total_cells'] != total or pre['status'] != 'passed'):
            raise ValueError('Unexpected queue summary: ' + lane)
        for model in models:
            expected = dict(checkpoint_id=MODELS[model]['checkpoint_id'], step=MODELS[model]['step'])
            if pre['checkpoints'][cell_model(model)] != expected:
                raise ValueError('Unexpected checkpoint: ' + model)
        keep = {cell_model(model): model for model in models}
        cells.update({(c['method'], keep[c['model']], c['sampling_seed']): c for c in summary['cells']})
        sources[lane] = pre['source_sha256']
        hashes[lane] = base.digest(logs / 'summary.json')
    if set(cells) != {(s['method'], s['model'], s['seed']) for s in specs()}:
        raise ValueError('Unexpected summary cells')
    for group in GROUPS:
        if len({sources[lane] for lane in group}) != 1:
            raise ValueError('The lanes of one root come from different sources')
    return sources, cells, hashes


def prepare(roots, reference, out):
    if base.digest(reference) != base.REFERENCE_SHA:
        raise ValueError('Reference SHA256 mismatch')
    sources, cells, hashes = read_lanes(roots)
    audit = dict(reference_sha256=base.REFERENCE_SHA, source_sha256=sources, lane_summary_sha256=hashes, sets=[])
    for s in specs():
        d = directory(roots, s)
        m = MODELS[s['model']]
        report, rounds = d / f"report_{m['ckpt']}.npz", d / f"ckpt_{m['ckpt']}"
        manifest = json.loads((d / 'manifest.json').read_text())
        cfg, ev = manifest['config'], manifest['config']['eval']
        if (manifest['source_sha256'] != sources[m['lane']] or
            (ev['sampling_seed'], ev['num_samples'], ev['batch_size'], ev['sampling_num_scales']) !=
                (s['seed'], N, BATCH, 34) or cfg['sampling']['method'] != s['method'] or
                cfg['sampling']['logsnr_max'] != LOGSNR_MAX):
            raise ValueError(f'Unexpected manifest {d}')
        with np.load(report, allow_pickle=False) as z:
            expected = dict(checkpoint_id=m['checkpoint_id'], checkpoint_step=m['step'],
                            dataset_stats_id=base.REFERENCE_SHA, num_samples=N, nfe=35)
            for k, v in expected.items():
                if base.scalar(z, k) != v:
                    raise ValueError(f'Incorrect report {k}: {d}')
            report_fid = float(base.scalar(z, 'fid_tweedie_xt'))
            protocol = base.scalar(z, 'protocol_sha256')
            training = base.scalar(z, 'training_protocol_sha256')
        if not np.isfinite(report_fid) or abs(report_fid - cells[(s['method'], s['model'], s['seed'])]['fid_tweedie_xt']) > 1e-10:
            raise ValueError(f'Summary and report differ: {d}')
        files = [rounds / f'statistics_{i}.npz' for i in range(base.ROUNDS)]
        if set(rounds.glob('statistics_*.npz')) != set(files):
            raise ValueError(f'Incorrect shards: {d}')
        x = np.lib.format.open_memmap(out / (s['name'] + '.npy'), mode='w+', dtype=np.float32, shape=(N, DIM))
        shard_hashes = []
        for i, path in enumerate(files):
            shard_hashes.append(base.digest(path))
            with np.load(path, allow_pickle=False) as z:
                expected = dict(round_id=i, sampling_seed=base.round_seed(s['seed'], i), num_samples=BATCH,
                                nfe=35, checkpoint_id=m['checkpoint_id'], checkpoint_step=m['step'],
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
        audit['sets'].append(dict(**s, report_fid=report_fid, shard_sha256=shard_hashes,
                                  report_sha256=base.digest(report),
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
        scope=SCOPE,
        point_fid=points, contrasts=result))


def check():
    base.check()
    values = {s['name']: float(i) for i, s in enumerate(specs())}
    c = contrasts(values)
    assert len(specs()) == 16 and len(c) == 40
    assert c['dpmpp_minus_heun_A_seed0'] == 4  # Sets run A, vp, fox, fox800 within a method.
    assert c['fox_minus_A_ode_edm_heun_seed1'] == 2
    assert c['fox800_minus_A_ode_edm_heun_seed1'] == 3
    assert c['fox800_minus_A_ode_dpmpp_2m_mean'] == 3
    assert c['A_minus_vp_ode_dpmpp_2m_seed1'] == -1
    assert c['interaction_fox_A_seed0'] == 0
    print('50k lo bootstrap checks passed', flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--check', action='store_true')
    p.add_argument('--prepare', action='store_true')
    p.add_argument('--root', type=Path)
    p.add_argument('--root-fox800', type=Path)
    p.add_argument('--reference', type=Path)
    p.add_argument('--out', type=Path)
    p.add_argument('--reps', type=int, default=3)
    a = p.parse_args()
    if a.check:
        check()
        return
    if not a.root or not a.root_fox800 or not a.reference or not a.out or a.reps < 1:
        p.error('both roots, reference, out and positive reps required')
    if a.prepare:
        prepare(dict(lo=a.root, fox800=a.root_fox800), a.reference, a.out)
    bootstrap(a.out, a.reps)


if __name__ == '__main__':
    main()
