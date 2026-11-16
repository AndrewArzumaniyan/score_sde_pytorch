"""Conditional bootstrap intervals for the AFHQv2 10k differences at 257 NFE of nfe_limit_eval.py (CPU only).

The method of nfe_limit_bootstrap.py and paper_diff_bootstrap.py: each cell is resampled independently, rows
drawn with replacement from its own 10k pool_3_tweedie_xt features; the checkpoints and the AFHQv2 reference
statistics are fixed. The interval shows the spread of generated samples, not the spread between trainings:
one training per model, one sampling seed. Point values are the report FIDs; recomputed FIDs must match
within 0.001.

Cells (lane a of nfe_limit_eval.py, H200): AFHQv2 DIAL@200k and VP-lambda@200k, DPM++ 2M and EDM Heun with
256 trajectory calls plus Tweedie, sampling seed 0, window [-10; 9.115429865459795], 10k images, batch 512,
20 shards. Differences: DIAL - VP-lambda per solver. nfe_limit_eval.py claims only the sign of this contrast;
the interval is reported beside it and adds no threshold.

Inputs under --inputs: a/ (lane folder of nfe_limit_eval.py) and reference_afhqv2.npz.
Writes --out: audit.json, draws.jsonl, summary.{json,md}; every 25 repeats and at the end. A rerun continues.
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

import celeba_final_ode_bootstrap as base
import paper_diff_bootstrap as pdb

N, BATCH, DIM = 10000, base.BATCH, base.DIM
ROUNDS = (N + BATCH - 1) // BATCH  # 20.
CALLS, NFE, SEED_SAMPLING, STEP, CKPT = 256, 257, 0, 200001, 4
LOGSNR_MIN, LOGSNR_MAX = pdb.LOGSNR_MIN, pdb.LOGSNR_MAX
FEATURES = pdb.FEATURES
SEED = 20261118
SUMMARY_EVERY = 25
CELLS = {  # name: (eval workdir, method, label)
    'afhq_dial_nfe257_dpm_10k': ('afhq_fox', 'ode_dpmpp_2m', 'AFHQv2 DIAL@200k, DPM++'),
    'afhq_vpl_nfe257_dpm_10k': ('afhq_a', 'ode_dpmpp_2m', 'AFHQv2 VP-lambda@200k, DPM++'),
    'afhq_dial_nfe257_heun_10k': ('afhq_fox', 'ode_edm_heun', 'AFHQv2 DIAL@200k, EDM Heun'),
    'afhq_vpl_nfe257_heun_10k': ('afhq_a', 'ode_edm_heun', 'AFHQv2 VP-lambda@200k, EDM Heun'),
}
INDEX = {name: index for index, name in enumerate(CELLS)}
DIFFS = (('afhq_dial_nfe257_dpm_10k', 'afhq_vpl_nfe257_dpm_10k'),
         ('afhq_dial_nfe257_heun_10k', 'afhq_vpl_nfe257_heun_10k'))


def paths(inputs, name):
    work = CELLS[name][0]
    d = inputs / 'a' / 'workdirs' / work / ('eval_nfelimit_' + name)
    return d, d / 'report_{}.npz'.format(CKPT), d / 'ckpt_{}'.format(CKPT), inputs / 'a' / 'logs' / (name + '_result.json')


def missing(inputs):
    out = {}
    for name in CELLS:
        d, report, rounds, result = paths(inputs, name)
        files = [d / 'manifest.json', report, result] + [rounds / 'statistics_{}.npz'.format(i) for i in range(ROUNDS)]
        absent = [str(f) for f in files if not f.is_file()]
        if absent:
            out[name] = absent
    if not (inputs / 'reference_afhqv2.npz').is_file():
        out['reference'] = [str(inputs / 'reference_afhqv2.npz')]
    return out


def audit_cell(inputs, name, reference_sha, out):
    work, method, label = CELLS[name]
    d, report, rounds, result_path = paths(inputs, name)
    manifest = json.loads((d / 'manifest.json').read_text())
    cfg, ev = manifest['config'], manifest['config']['eval']
    if ((ev['sampling_seed'], ev['num_samples'], ev['batch_size'], ev['sampling_num_scales']) !=
            (SEED_SAMPLING, N, BATCH, CALLS) or cfg['sampling']['method'] != method or
            cfg['sampling']['logsnr_max'] != LOGSNR_MAX or cfg['sampling'].get('logsnr_min', LOGSNR_MIN) != LOGSNR_MIN):
        raise ValueError('Unexpected manifest: {}'.format(d))
    result = json.loads(result_path.read_text())
    if (result['step'], result['method'], result['sampling_seed'], result['num_samples'], result['nfe']) != \
            (STEP, method, SEED_SAMPLING, N, NFE):
        raise ValueError('Unexpected queue result: {}'.format(d))
    checkpoint_id = result['checkpoint_id']
    with np.load(report, allow_pickle=False) as z:
        expected = dict(checkpoint_id=checkpoint_id, checkpoint_step=STEP, dataset_stats_id=reference_sha,
                        num_samples=N, nfe=NFE)
        for key, value in expected.items():
            if base.scalar(z, key) != value:
                raise ValueError('Incorrect report {}: {}'.format(key, d))
        report_fid = float(base.scalar(z, 'fid_tweedie_xt'))
        protocol = base.scalar(z, 'protocol_sha256')
        training = base.scalar(z, 'training_protocol_sha256')
    if not np.isfinite(report_fid) or abs(report_fid - result['fid_tweedie_xt']) > 1e-10:
        raise ValueError('Queue result and report differ: {}'.format(d))
    files = [rounds / 'statistics_{}.npz'.format(i) for i in range(ROUNDS)]
    if set(rounds.glob('statistics_*.npz')) != set(files):
        raise ValueError('Expected {} shards: {}'.format(ROUNDS, rounds))
    path = out / (name + '.npy')
    x = np.lib.format.open_memmap(path, mode='w+', dtype=np.float32, shape=(N, DIM))
    shard_hashes = []
    try:
        for i, shard in enumerate(files):
            shard_hashes.append(base.digest(shard))
            with np.load(shard, allow_pickle=False) as z:
                expected = dict(round_id=i, sampling_seed=base.round_seed(SEED_SAMPLING, i), num_samples=BATCH,
                                nfe=NFE, checkpoint_id=checkpoint_id, checkpoint_step=STEP,
                                protocol_sha256=protocol, training_protocol_sha256=training)
                if any(base.scalar(z, k) != v for k, v in expected.items()):
                    raise ValueError('Shard identity mismatch: {}'.format(shard))
                a = z[FEATURES]
                if a.shape != (BATCH, DIM) or a.dtype != np.float32 or not np.isfinite(a).all():
                    raise ValueError('Invalid features: {}'.format(shard))
                take = min(BATCH, N - i * BATCH)
                x[i * BATCH:i * BATCH + take] = a[:take]
        x.flush()
    finally:
        del x
    return dict(name=name, label=label, method=method, report_fid=report_fid, checkpoint_id=checkpoint_id,
                step=STEP, sampling_gpu=result.get('gpu_name'), eval_dir=str(d), report_sha256=base.digest(report),
                manifest_sha256=base.digest(d / 'manifest.json'), shard_sha256=shard_hashes,
                feature_sha256=base.digest(path))


def prepare(inputs, out):
    out.mkdir(parents=True, exist_ok=True)
    reference = inputs / 'reference_afhqv2.npz'
    reference_sha = base.digest(reference)
    ref = pdb.reference_moments(reference, 'AFHQV2', out)
    audit = dict(reference_sha256=reference_sha, reference_moments_sha256=base.digest(out / 'reference_moments.npz'),
                 sets=[])
    for name in CELLS:  # Every cell is needed by a difference: any failure stops the run.
        record = audit_cell(inputs, name, reference_sha, out)
        features = np.load(out / (name + '.npy'), mmap_mode='r')
        record['recomputed_fid'] = base.fid(features, ref)
        record['point_error'] = record['recomputed_fid'] - record['report_fid']
        if not np.isfinite(record['recomputed_fid']) or abs(record['point_error']) > base.TOLERANCE:
            raise ValueError('Point FID differs by more than {}: {}'.format(base.TOLERANCE, record['point_error']))
        audit['sets'].append(record)
        print('Audited', name, record['report_fid'], record['point_error'], flush=True)
    pdb.save(out / 'audit.json', audit)
    return audit


def summarize(out):
    audit = json.loads((out / 'audit.json').read_text())
    sets = {s['name']: s for s in audit['sets']}
    rows = pdb.read_draws(out)
    target = json.loads((out / 'target.json').read_text())['reps'] if (out / 'target.json').exists() else None
    diffs, lines = {}, ['# Bootstrap of the AFHQv2 10k differences at 257 NFE', '',
                        'FID after Tweedie, 257 NFE, sampling seed 0, 10k; 95% percentile interval '
                        '(basic in summary.json); conditional on fixed weights and reference.', '',
                        '| Difference | Point | 95% percentile | Repeats |', '|---|---|---|---|']
    for left, right in DIFFS:
        point = sets[left]['report_fid'] - sets[right]['report_fid']
        entry = dict(left=sets[left]['label'], right=sets[right]['label'], point=point, repeats=len(rows))
        interval = '—'
        if rows:
            values = np.asarray([r['fid'][left] - r['fid'][right] for r in rows])
            low, median, high = np.quantile(values, [.025, .5, .975])
            entry.update(percentile_95=[float(low), float(high)], basic_95=[float(2 * point - high), float(2 * point - low)],
                         bootstrap_median=float(median), bootstrap_sd=float(values.std(ddof=1)) if len(values) > 1 else None)
            interval = '[{:+.3f}; {:+.3f}]'.format(low, high)
        diffs['{}_minus_{}'.format(left, right)] = entry
        lines.append('| {} − {} | {:+.3f} | {} | {} |'.format(entry['left'], entry['right'], point, interval, len(rows)))
    status = 'completed' if target is not None and len(rows) >= target else 'running'
    pdb.save(out / 'summary.json', dict(status=status, repeats=len(rows), target=target, bootstrap_seed=SEED,
        updated_unix=time.time(), scope=pdb.SCOPE + ' One sampling seed, 10k images.',
        resampling='independent per cell', interval='95% percentile; basic also reported; no multiplicity correction',
        points={k: v['report_fid'] for k, v in sets.items()}, differences=diffs))
    pdb.write_text(out / 'summary.md', '\n'.join(lines) + '\n')
    return status


def bootstrap(out, reps):
    audit = json.loads((out / 'audit.json').read_text())
    if base.digest(out / 'reference_moments.npz') != audit['reference_moments_sha256']:
        raise ValueError('Reference moments changed')
    features = {}
    for s in audit['sets']:
        path = out / (s['name'] + '.npy')
        if base.digest(path) != s['feature_sha256']:
            raise ValueError('Features changed: {}'.format(path))
        features[s['name']] = np.load(path, mmap_mode='r')
    with np.load(out / 'reference_moments.npz') as z:
        ref = (z['mean'], z['cov'], z['sqrt'])
    rows = pdb.read_draws(out)
    if len(rows) > reps or any(r['rep'] != i or set(r['fid']) != set(features) for i, r in enumerate(rows)):
        raise ValueError('Existing draws inconsistent with requested run')
    pdb.save(out / 'target.json', dict(reps=reps))
    path = out / 'draws.jsonl'
    for rep in range(len(rows), reps):
        began = time.monotonic()
        values = {}
        for name, x in features.items():
            rng = np.random.default_rng(np.random.SeedSequence([SEED, INDEX[name], rep]))
            weights = np.bincount(rng.integers(0, N, N), minlength=N)
            values[name] = base.fid(x, ref, weights)
        if not all(np.isfinite(v) for v in values.values()):
            raise ValueError('Nonfinite bootstrap FID')
        with path.open('a') as f:
            f.write(json.dumps(dict(rep=rep, fid=values, seconds=time.monotonic() - began), allow_nan=False) + '\n')
        print('{}/{}, {:.1f}s'.format(rep + 1, reps, time.monotonic() - began), flush=True)
        if (rep + 1) % SUMMARY_EVERY == 0:
            summarize(out)
    return summarize(out)


def check():
    """Synthetic checks: cell names follow nfe_limit_eval.cell(), and the weighted FID is consistent."""
    assert ROUNDS == 20 and len(CELLS) == 4 and len(DIFFS) == 2
    for name, (work, method, _) in CELLS.items():
        model = 'afhq_dial' if work == 'afhq_fox' else 'afhq_vpl'
        short = {'ode_dpmpp_2m': 'dpm', 'ode_edm_heun': 'heun'}[method]
        assert name == '{}_nfe{}_{}_{}k'.format(model, CALLS + 1, short, N // 1000), name
    rng = np.random.default_rng(0)
    real = rng.standard_normal((4000, 8))
    mean, cov = base.moments(real)
    vals, vecs = np.linalg.eigh(cov)
    ref = (mean, cov, (vecs * np.sqrt(np.maximum(vals, 0))) @ vecs.T)
    assert abs(base.fid(real, ref)) < 1e-8
    ones = np.ones(len(real), dtype=np.int64)
    assert abs(base.fid(real, ref, ones) - base.fid(real, ref)) < 1e-10
    shifted = real + 0.5
    assert abs(base.fid(shifted, ref) - 8 * 0.25) < 1e-6
    print('nfe limit AFHQv2 bootstrap checks passed', flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--check', action='store_true')
    p.add_argument('--missing', action='store_true')
    p.add_argument('--summarize', action='store_true')
    p.add_argument('--inputs', type=Path, default=Path('/input'))
    p.add_argument('--out', type=Path)
    p.add_argument('--reps', type=int, default=1000)
    a = p.parse_args()
    if a.check:
        check()
        return
    if a.missing:
        absent = missing(a.inputs)
        for name, files in absent.items():
            print('Missing for {}: {} files, e.g. {}'.format(name, len(files), files[0]), flush=True)
        sys.exit(1 if absent else 0)
    if not a.out or a.reps < 1:
        p.error('--out and positive --reps required')
    if a.summarize:
        print('Summary:', summarize(a.out), flush=True)
        return
    if not (a.out / 'audit.json').exists():
        absent = missing(a.inputs)
        if absent:
            print('Inputs not ready: {}'.format({k: v[:2] for k, v in absent.items()}), flush=True)
            sys.exit(3)
        prepare(a.inputs, a.out)
    print('Summary:', bootstrap(a.out, a.reps), flush=True)


if __name__ == '__main__':
    main()
