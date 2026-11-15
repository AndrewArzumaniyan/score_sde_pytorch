"""Conditional bootstrap intervals for the new 50k differences of the paper (CPU only).

The method of celeba_50k_lo_bootstrap.py (c7a85e0) and celeba_matched35_bootstrap.py (151fc6d):
each cell is resampled independently, rows drawn with replacement from its own 50k
pool_3_tweedie_xt features; the checkpoints and the reference statistics are fixed. The
interval shows the spread of generated samples, not the spread between trainings: one
training per model. Point values are the report FIDs; recomputed FIDs must match within 0.001.

Every cell is matched35: 34 trajectory calls plus Tweedie, DPM++ 2M and EDM Heun, sampling
seeds 0 and 1, window [-10; 9.115429865459795], 50k images, batch 512, 98 shards.

For each difference Left - Right: per solver and seed, the mean over seeds per solver, and the
difference of contrasts between solvers, (L - R) under DPM++ minus (L - R) under EDM Heun,
per seed and averaged.

Cells are split into parts that run as separate containers:
  celeba    DIAL 800k, A 400k, B 800k, A 500k (CelebA statistics)
  afhq      DIAL 200k, A 250k, VP 250k (AFHQv2 statistics)
  afhq200k  A 200k, VP 200k, after afhq200k_eval.py
The draws of a cell depend only on (SEED, its index in CELLS, repeat), so parts combine.
A cell that fails its audit fails the differences that use it; the rest go on.
Each part writes out/PART/; summary.{json,csv,md} at the top cover every part present.
"""
import argparse
import csv
import json
import os
import sys
import tempfile
import time
import zipfile
from pathlib import Path

import numpy as np
from scipy import linalg
import celeba_final_ode_bootstrap as base
import celeba_50k_lo_bootstrap as lo50k

N, BATCH, DIM = base.N, base.BATCH, base.DIM
ROUNDS = base.ROUNDS  # 98.
METHODS = ('ode_dpmpp_2m', 'ode_edm_heun')
SHORT = {'ode_dpmpp_2m': 'dpmpp', 'ode_edm_heun': 'heun'}
SEEDS = (0, 1)
SEED = 20261115
LOGSNR_MIN, LOGSNR_MAX = -10.0, 9.115429865459795
CALLS, NFE = 34, 35
FEATURES = 'pool_3_tweedie_xt'
SUMMARY_EVERY = 25
AFHQ_IMAGES = 15803
SCOPE = ('Conditional on fixed checkpoints and fixed reference statistics: the spread of generated samples, '
         'not the spread between trainings; one training per model.')

# Roots, as mounted under --inputs: the 50k output folders on the server.
ROOTS = {
    'fox800': 'celeba_fox800_lo_20261107', 'lo_gpu1': 'celeba_50k_lo_20261107/gpu1',
    'gpu2q': 'gpu2_queue_20261114', 'gpu2a': 'gpu2_a50k_grids_20261114', 'gpu1vp': 'gpu1_vp_tail_20261114',
    'afhq200k_a': 'afhq200k_bootstrap_20261115/a', 'afhq200k_vp': 'afhq200k_bootstrap_20261115/vp',
}
# folder and result take {method} and {seed}. checkpoint_id None: taken from logs/<result>_result.json.
# training: where the weights were trained, from the launchers and session records, not from the cells.
# The 50k cells themselves were all sampled on RTX; sampling_gpu is read from each root's preflight.
MODELS = {
    'celeba_dial800': dict(dataset='CELEBA', label='DIAL 800k', part='celeba', root='fox800',
        work='celeba_fox_matern32_kappa1000_50k', folder='eval_fox800_lo_fox_c160_lo_{method}_seed{seed}',
        ckpt=160, step=800001, checkpoint_id=lo50k.MODELS['fox800']['checkpoint_id'], result=None,
        training='H200 (FOX run in the training checkout)', training_gpu=('H200',)),
    'celeba_a400': dict(dataset='CELEBA', label='A 400k', part='celeba', root='lo_gpu1',
        work='celeba_factorial_A_vp_pvp_400k_rtx_seed42', folder='eval_50k_lo_A_c80_lo_{method}_seed{seed}',
        ckpt=80, step=400001, checkpoint_id=lo50k.MODELS['A']['checkpoint_id'], result=None,
        training='H200 to 200k, RTX 200k-400k (celeba_a_400k_rtx_20261106)',
        training_gpu=('H200', 'RTX')),
    'celeba_b800': dict(dataset='CELEBA', label='B 800k', part='celeba', root='gpu2q', work='celeba_b',
        folder='eval_gpu2q_celeba_b_c160_m35_{method}_seed{seed}', ckpt=160, step=800001, checkpoint_id=None,
        result='celeba_b_c160_m35_{method}_seed{seed}', training='H200 (celeba_b_800k_h200_20261111)', training_gpu=('H200',)),
    'celeba_a500': dict(dataset='CELEBA', label='A 500k', part='celeba', root='gpu2a', work='celeba_a',
        folder='eval_gpu2a_celeba_a_cbest_m35_{method}_seed{seed}', ckpt=100, step=500001, checkpoint_id=None,
        result='celeba_a_cbest_m35_{method}_seed{seed}',
        training='H200 to 200k, RTX 200k-400k, H200 400k-500k (celeba_a_800k_h200_20261113)',
        training_gpu=('H200', 'RTX')),
    'afhq_dial200': dict(dataset='AFHQV2', label='DIAL 200k', part='afhq', root='gpu2q', work='afhq_fox',
        folder='eval_gpu2q_afhq_fox_c4_m35_{method}_seed{seed}', ckpt=4, step=200001, checkpoint_id=None,
        result='afhq_fox_c4_m35_{method}_seed{seed}', training='RTX (afhqv2_rtx_20261108)', training_gpu=('RTX',)),
    'afhq_a250': dict(dataset='AFHQV2', label='A 250k', part='afhq', root='gpu2q', work='afhq_a',
        folder='eval_gpu2q_afhq_a_cbest_m35_{method}_seed{seed}', ckpt=5, step=250001, checkpoint_id=None,
        result='afhq_a_cbest_m35_{method}_seed{seed}', training='RTX (afhqv2_rtx_20261108, afhqv2_a400k_20261113)',
        training_gpu=('RTX',)),
    'afhq_vp250': dict(dataset='AFHQV2', label='VP 250k', part='afhq', root='gpu1vp', work='afhq_vp',
        folder='eval_gpu1vp_afhq_vp_cbest_m35_{method}_seed{seed}', ckpt=5, step=250001, checkpoint_id=None,
        result='afhq_vp_cbest_m35_{method}_seed{seed}', training='RTX (afhqv2_rtx_20261108, afhqv2_vp400k_20261113)',
        training_gpu=('RTX',)),
    'afhq_a200': dict(dataset='AFHQV2', label='A 200k', part='afhq200k', root='afhq200k_a', work='afhq_a',
        folder='eval_afhq200k_afhq_a_c4_m35_{method}_seed{seed}', ckpt=4, step=200001, checkpoint_id=None,
        result='afhq_a_c4_m35_{method}_seed{seed}', training='RTX (afhqv2_rtx_20261108)', training_gpu=('RTX',)),
    'afhq_vp200': dict(dataset='AFHQV2', label='VP 200k', part='afhq200k', root='afhq200k_vp', work='afhq_vp',
        folder='eval_afhq200k_afhq_vp_c4_m35_{method}_seed{seed}', ckpt=4, step=200001, checkpoint_id=None,
        result='afhq_vp_c4_m35_{method}_seed{seed}', training='RTX (afhqv2_rtx_20261108)', training_gpu=('RTX',)),
}
PARTS = ('celeba', 'afhq', 'afhq200k')
REFERENCES = {'CELEBA': 'reference_celeba.npz', 'AFHQV2': 'reference_afhqv2.npz'}  # Under --inputs.
DIFFS = (('celeba_dial800', 'celeba_b800'), ('celeba_dial800', 'celeba_a500'), ('celeba_dial800', 'celeba_a400'),
         ('celeba_b800', 'celeba_a500'), ('celeba_b800', 'celeba_a400'),
         ('afhq_dial200', 'afhq_a250'), ('afhq_dial200', 'afhq_vp250'),
         ('afhq_dial200', 'afhq_a200'), ('afhq_dial200', 'afhq_vp200'))
# One feature set per (model, method, seed); its index here seeds its draws.
CELLS = [dict(name=f'{model}_{method}_seed{seed}', model=model, method=method, seed=seed)
         for model in MODELS for seed in SEEDS for method in METHODS]
INDEX = {cell['name']: index for index, cell in enumerate(CELLS)}


def cell_name(model, method, seed):
    return f'{model}_{method}_seed{seed}'


def diff_name(left, right):
    return f'{left}_minus_{right}'


def part_cells(part):
    return [cell for cell in CELLS if MODELS[cell['model']]['part'] == part]


def contrasts(left, right, values):
    """{cell: FID} -> the contrasts of Left - Right."""
    out = {}
    for seed in SEEDS:
        for method in METHODS:
            out[f'{SHORT[method]}_seed{seed}'] = (values[cell_name(left, method, seed)] -
                                                  values[cell_name(right, method, seed)])
        out[f'interaction_seed{seed}'] = out[f'dpmpp_seed{seed}'] - out[f'heun_seed{seed}']
    for kind in ('dpmpp', 'heun', 'interaction'):
        out[kind + '_mean'] = sum(out[f'{kind}_seed{seed}'] for seed in SEEDS) / len(SEEDS)
    return out


def save(path, obj):
    write_text(path, json.dumps(obj, indent=2, allow_nan=False) + '\n')


def write_text(path, text):
    """Atomic; the temporary name is unique across containers, which share pids under --init."""
    fd, tmp = tempfile.mkstemp(dir=str(path.parent), prefix='.' + path.name + '.', suffix='.tmp')
    try:
        with os.fdopen(fd, 'w', newline='') as stream:
            stream.write(text)
        os.chmod(tmp, 0o644)  # mkstemp gives 0600; others on the server read the summary.
        os.replace(tmp, str(path))
    except BaseException:
        if os.path.exists(tmp):
            os.unlink(tmp)
        raise


def expected_files(inputs, part):
    """The files every cell of the part needs before its audit, by cell."""
    out = {}
    for cell in part_cells(part):
        m = MODELS[cell['model']]
        fmt = dict(method=cell['method'], seed=cell['seed'])
        d = inputs / m['root'] / 'workdirs' / m['work'] / m['folder'].format(**fmt)
        files = [d / 'manifest.json', d / f"report_{m['ckpt']}.npz"]
        files += [d / f"ckpt_{m['ckpt']}" / f'statistics_{i}.npz' for i in range(ROUNDS)]
        if m['result']:
            files.append(inputs / m['root'] / 'logs' / (m['result'].format(**fmt) + '_result.json'))
        out[cell['name']] = files
    return out


def missing(inputs, part):
    """{cell: missing files}; a cell still being computed is missing, not failed."""
    out = {}
    for name, files in expected_files(inputs, part).items():
        absent = [str(f) for f in files if not f.is_file()]
        if absent:
            out[name] = absent
    reference = inputs / REFERENCES[MODELS[part_cells(part)[0]['model']]['dataset']]
    if not reference.is_file():
        out['reference'] = [str(reference)]
    return out


class MissingInputs(RuntimeError):
    pass


def environment(root):
    path = root / 'logs/preflight.json'
    if not path.exists():
        return None
    return json.loads(path.read_text()).get('environment', {}).get('visible_cuda_devices')


def audit_cell(inputs, cell, reference_sha, out):
    """Every check of the earlier bootstraps; writes the features. Returns the audit record."""
    m = MODELS[cell['model']]
    fmt = dict(method=cell['method'], seed=cell['seed'])
    root = inputs / m['root']
    d = root / 'workdirs' / m['work'] / m['folder'].format(**fmt)
    report, rounds = d / f"report_{m['ckpt']}.npz", d / f"ckpt_{m['ckpt']}"
    manifest = json.loads((d / 'manifest.json').read_text())
    cfg, ev = manifest['config'], manifest['config']['eval']
    if ((ev['sampling_seed'], ev['num_samples'], ev['batch_size'], ev['sampling_num_scales']) !=
            (cell['seed'], N, BATCH, CALLS) or cfg['sampling']['method'] != cell['method'] or
            cfg['sampling']['logsnr_max'] != LOGSNR_MAX or cfg['sampling'].get('logsnr_min', LOGSNR_MIN) != LOGSNR_MIN):
        raise ValueError(f'Unexpected manifest: {d}')
    checkpoint_id, queue_fid = m['checkpoint_id'], None
    if m['result']:
        result = json.loads((root / 'logs' / (m['result'].format(**fmt) + '_result.json')).read_text())
        if (result['step'], result['method'], result['sampling_seed'], result['num_samples']) != \
                (m['step'], cell['method'], cell['seed'], N):
            raise ValueError(f'Unexpected queue result: {d}')
        checkpoint_id, queue_fid = result['checkpoint_id'], result['fid_tweedie_xt']
    with np.load(report, allow_pickle=False) as z:
        expected = dict(checkpoint_id=checkpoint_id, checkpoint_step=m['step'], dataset_stats_id=reference_sha,
                        num_samples=N, nfe=NFE)
        for key, value in expected.items():
            if base.scalar(z, key) != value:
                raise ValueError(f'Incorrect report {key}: {d}')
        report_fid = float(base.scalar(z, 'fid_tweedie_xt'))
        protocol = base.scalar(z, 'protocol_sha256')
        training = base.scalar(z, 'training_protocol_sha256')
    if not np.isfinite(report_fid) or (queue_fid is not None and abs(report_fid - queue_fid) > 1e-10):
        raise ValueError(f'Queue result and report differ: {d}')
    files = [rounds / f'statistics_{i}.npz' for i in range(ROUNDS)]
    if set(rounds.glob('statistics_*.npz')) != set(files):
        raise ValueError(f'Expected {ROUNDS} shards: {rounds}')
    path = out / (cell['name'] + '.npy')
    x = np.lib.format.open_memmap(path, mode='w+', dtype=np.float32, shape=(N, DIM))
    shard_hashes = []
    try:
        for i, shard in enumerate(files):
            shard_hashes.append(base.digest(shard))
            with np.load(shard, allow_pickle=False) as z:
                expected = dict(round_id=i, sampling_seed=base.round_seed(cell['seed'], i), num_samples=BATCH,
                                nfe=NFE, checkpoint_id=checkpoint_id, checkpoint_step=m['step'],
                                protocol_sha256=protocol, training_protocol_sha256=training)
                if any(base.scalar(z, k) != v for k, v in expected.items()):
                    raise ValueError(f'Shard identity mismatch: {shard}')
                if FEATURES not in z.files:
                    raise ValueError(f'No {FEATURES}: {shard}')
                a = z[FEATURES]
                if a.shape != (BATCH, DIM) or a.dtype != np.float32 or not np.isfinite(a).all():
                    raise ValueError(f'Invalid features: {shard}')
                take = min(BATCH, N - i * BATCH)
                x[i * BATCH:i * BATCH + take] = a[:take]
        x.flush()
    finally:
        del x
    return dict(cell, report_fid=report_fid, checkpoint_id=checkpoint_id, step=m['step'],
                eval_dir=str(d), source_sha256=manifest.get('source_sha256'), report_sha256=base.digest(report),
                manifest_sha256=base.digest(d / 'manifest.json'), shard_sha256=shard_hashes,
                feature_sha256=base.digest(path), sampling_gpu=environment(root))


def reference_moments(path, dataset, out):
    if dataset == 'CELEBA' and base.digest(path) != base.REFERENCE_SHA:
        raise ValueError('CelebA reference SHA256 mismatch')
    raw = out / 'reference_pool.npy'
    with zipfile.ZipFile(path) as z, z.open('pool_3.npy') as src, raw.open('wb') as dst:
        for block in iter(lambda: src.read(1048576), b''):
            dst.write(block)
    pool = np.load(raw, mmap_mode='r')
    if dataset == 'AFHQV2' and pool.shape != (AFHQ_IMAGES, DIM):
        raise ValueError('Unexpected AFHQv2 reference size')
    mean, cov = base.moments(pool)
    del pool
    raw.unlink()
    vals, vecs = linalg.eigh(cov, check_finite=False)
    sqrt = (vecs * np.sqrt(np.maximum(vals, 0))) @ vecs.T
    np.savez(out / 'reference_moments.npz', mean=mean, cov=cov, sqrt=sqrt)
    return (mean, cov, sqrt)


def prepare(inputs, part, out):
    """Audit and point checks of every cell of the part; a failed cell is recorded, not fatal."""
    absent = missing(inputs, part)
    if absent:  # Nothing is written: a rerun after the cells finish audits them anew.
        raise MissingInputs('Inputs not ready for part {}: {}'.format(part, {k: v[:3] for k, v in absent.items()}))
    out.mkdir(parents=True, exist_ok=True)
    dataset = {MODELS[c['model']]['dataset'] for c in part_cells(part)}.pop()
    reference = inputs / REFERENCES[dataset]
    reference_sha = base.digest(reference)
    ref = reference_moments(reference, dataset, out)
    audit = dict(part=part, dataset=dataset, reference_sha256=reference_sha,
                 reference_moments_sha256=base.digest(out / 'reference_moments.npz'), sets=[], failed={})
    for cell in part_cells(part):
        try:
            record = audit_cell(inputs, cell, reference_sha, out)
            features = np.load(out / (cell['name'] + '.npy'), mmap_mode='r')
            record['recomputed_fid'] = base.fid(features, ref)
            record['point_error'] = record['recomputed_fid'] - record['report_fid']
            if not np.isfinite(record['recomputed_fid']) or abs(record['point_error']) > base.TOLERANCE:
                raise ValueError('Point FID differs by more than {}: {}'.format(base.TOLERANCE, record['point_error']))
            audit['sets'].append(record)
            print('Audited', cell['name'], record['report_fid'], record['point_error'], flush=True)
        except Exception as error:  # Missing files raise whatever the reader does.
            audit['failed'][cell['name']] = repr(error)
            (out / (cell['name'] + '.npy')).unlink(missing_ok=True)
            print('Audit failed:', cell['name'], repr(error), flush=True)
    save(out / 'audit.json', audit)
    return audit


def bootstrap(part, out, reps):
    audit = json.loads((out / 'audit.json').read_text())
    if base.digest(out / 'reference_moments.npz') != audit['reference_moments_sha256']:
        raise ValueError('Reference moments changed')
    features = {}
    for s in audit['sets']:
        path = out / (s['name'] + '.npy')
        if base.digest(path) != s['feature_sha256']:
            raise ValueError(f'Features changed: {path}')
        features[s['name']] = np.load(path, mmap_mode='r')
    with np.load(out / 'reference_moments.npz') as z:
        ref = (z['mean'], z['cov'], z['sqrt'])
    rows = read_draws(out)
    if len(rows) > reps or any(r['rep'] != i or set(r['fid']) != set(features) for i, r in enumerate(rows)):
        raise ValueError('Existing draws inconsistent with requested run')
    save(out / 'target.json', dict(reps=reps))
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
            f.flush()
        print(f'{part}: {rep + 1}/{reps}, {time.monotonic() - began:.1f}s', flush=True)
        if (rep + 1) % SUMMARY_EVERY == 0:
            try:  # An interim summary must not stop the repeats; the final one is strict.
                summarize(out.parent)
            except Exception as error:
                print('Interim summary failed: {!r}'.format(error), flush=True)


def read_draws(out):
    path = out / 'draws.jsonl'
    rows = []
    if path.exists():
        for line in path.read_text().splitlines():
            try:
                rows.append(json.loads(line))
            except ValueError:  # A line another container is still writing.
                break
    return rows


def summarize(top):
    """Every difference whose cells are present, at the repeats all its parts share."""
    parts = {}
    for part in PARTS:
        if (top / part / 'audit.json').exists():
            audit = json.loads((top / part / 'audit.json').read_text())
            target = top / part / 'target.json'
            parts[part] = dict(audit=audit, rows=read_draws(top / part),
                               target=json.loads(target.read_text())['reps'] if target.exists() else None,
                               sets={s['name']: s for s in audit['sets']})
    diffs, table = {}, []
    for left, right in DIFFS:
        name = diff_name(left, right)
        needed = [cell_name(model, method, seed) for model in (left, right) for seed in SEEDS for method in METHODS]
        involved = sorted({MODELS[left]['part'], MODELS[right]['part']})
        entry = dict(dataset=MODELS[left]['dataset'], left=MODELS[left]['label'], right=MODELS[right]['label'],
                     cells=needed, training=dict(left=MODELS[left]['training'], right=MODELS[right]['training']))
        diffs[name] = entry
        if any(part not in parts for part in involved):
            entry.update(status='pending', reason='part not prepared: ' + ', '.join(p for p in involved if p not in parts))
            continue
        records = {}
        failed = {}
        for cell in needed:
            part = parts[MODELS[cell.rsplit('_ode_', 1)[0]]['part']]
            if cell in part['sets']:
                records[cell] = part['sets'][cell]
            else:
                failed[cell] = part['audit']['failed'].get(cell, 'missing from the audit')
        if failed:
            entry.update(status='failed', failed_cells=failed)
            continue
        gpus = {side: sorted({json.dumps(records[c]['sampling_gpu']) for c in needed if c.startswith(model + '_')})
                for side, model in (('left', left), ('right', right))}
        entry['sampling_gpu'] = {side: [json.loads(g) for g in value] for side, value in gpus.items()}
        known = all('null' not in value for value in gpus.values())
        entry['sampling_hardware'] = ('unknown' if not known else
                                      'same' if gpus['left'] == gpus['right'] and len(gpus['left']) == 1 else 'differs')
        entry['training_hardware_differs'] = MODELS[left]['training_gpu'] != MODELS[right]['training_gpu']
        repeats = min(len(parts[p]['rows']) for p in involved)
        targets = [parts[p]['target'] for p in involved]
        entry['point_fid'] = {c: records[c]['report_fid'] for c in needed}
        points = contrasts(left, right, entry['point_fid'])
        entry['repeats'] = repeats
        if repeats == 0:
            entry.update(status='running', contrasts={k: dict(point=v) for k, v in points.items()})
            continue
        draws = {c: [parts[MODELS[c.rsplit('_ode_', 1)[0]]['part']]['rows'][i]['fid'][c] for i in range(repeats)]
                 for c in needed}
        samples = [contrasts(left, right, {c: draws[c][i] for c in needed}) for i in range(repeats)]
        result = {}
        for key, point in points.items():
            values = np.asarray([s[key] for s in samples])
            low, median, high = np.quantile(values, [.025, .5, .975])
            result[key] = dict(point=point, percentile_95=[float(low), float(high)],
                               basic_95=[float(2 * point - high), float(2 * point - low)],
                               bootstrap_median=float(median), repeats=repeats)
            table.append((name, entry['dataset'], key, point, float(low), float(high), float(2 * point - high),
                          float(2 * point - low), repeats))
        entry['contrasts'] = result
        entry['status'] = 'completed' if all(t is not None and repeats >= t for t in targets) else 'partial'
    statuses = [d['status'] for d in diffs.values()]
    status = 'completed' if all(s == 'completed' for s in statuses) else 'failed' if 'failed' in statuses and all(
        s in ('completed', 'failed') for s in statuses) else 'running'
    save(top / 'summary.json', dict(status=status, bootstrap_seed=SEED, updated_unix=time.time(),
        uncertainty='generated-sample uncertainty conditional on fixed checkpoints and reference',
        resampling='independent per cell; sampling seeds reported separately; _mean averages seeds 0 and 1',
        interaction='(Left - Right) under DPM++ 2M minus (Left - Right) under EDM Heun',
        interval='95% percentile; basic also reported; no multiplicity correction',
        hardware=('All 50k cells were sampled on RTX by their launchers; sampling_gpu is read from each '
                  'root logs/preflight.json. training is from the launchers and records, not from the cells.'),
        scope=SCOPE, window=[LOGSNR_MIN, LOGSNR_MAX], parts={p: dict(repeats=len(v['rows']), target=v['target'],
            failed_cells=v['audit']['failed']) for p, v in parts.items()}, differences=diffs))
    import io
    stream = io.StringIO()
    writer = csv.writer(stream)
    writer.writerow(('difference', 'dataset', 'contrast', 'point', 'percentile_low', 'percentile_high',
                     'basic_low', 'basic_high', 'repeats'))
    writer.writerows(table)
    write_text(top / 'summary.csv', stream.getvalue())
    lines = ['# Bootstrap of the new 50k differences', '',
             'FID after Tweedie, Left − Right, 95% percentile interval; conditional on fixed weights and reference. '
             'Window [−10; 9.115].', '',
             '| Dataset | Difference | DPM++ s0 | DPM++ s1 | DPM++ mean | Heun s0 | Heun s1 | Heun mean '
             '| DPM++ − Heun mean | Repeats | Status | 50k GPU | Training GPU differs |',
             '|---|---|---|---|---|---|---|---|---|---|---|---|---|']
    for name, entry in diffs.items():
        cells = []
        for key in ('dpmpp_seed0', 'dpmpp_seed1', 'dpmpp_mean', 'heun_seed0', 'heun_seed1', 'heun_mean',
                    'interaction_mean'):
            value = entry.get('contrasts', {}).get(key)
            cells.append('—' if value is None else '{:+.3f}'.format(value['point']) if 'percentile_95' not in value
                         else '{:+.3f} [{:+.3f}; {:+.3f}]'.format(value['point'], *value['percentile_95']))
        lines.append('| {} | {} − {} | {} | {} | {} | {} | {} |'.format(
            entry['dataset'], entry['left'], entry['right'], ' | '.join(cells), entry.get('repeats', 0),
            entry['status'], entry.get('sampling_hardware', '—'),
            'yes' if entry.get('training_hardware_differs') else 'no' if 'training_hardware_differs' in entry else '—'))
    write_text(top / 'summary.md', '\n'.join(lines) + '\n')
    return status


def check():
    base.check()
    assert len(CELLS) == 36 and len(set(INDEX)) == 36 and ROUNDS == 98
    assert [len(part_cells(p)) for p in PARTS] == [16, 12, 8]
    values = {c['name']: float(i) for i, c in enumerate(CELLS)}
    c = contrasts('celeba_dial800', 'celeba_b800', values)
    # Within a model, cells run seed 0 (DPM++, Heun), seed 1 (DPM++, Heun): models are 4 cells apart.
    assert c['dpmpp_seed0'] == -8 and c['heun_seed1'] == -8 and c['interaction_mean'] == 0
    assert all('training_gpu' in m for m in MODELS.values())
    assert len(c) == 9
    for left, right in DIFFS:
        assert MODELS[left]['dataset'] == MODELS[right]['dataset']
    print('paper diff bootstrap checks passed', flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--check', action='store_true')
    p.add_argument('--summarize', action='store_true')
    p.add_argument('--missing', action='store_true', help='list the inputs of --part that are not there yet')
    p.add_argument('--part', choices=PARTS)
    p.add_argument('--inputs', type=Path, default=Path('/input'))
    p.add_argument('--out', type=Path)
    p.add_argument('--reps', type=int, default=1000)
    a = p.parse_args()
    if a.check:
        check()
        return
    if not a.out:
        p.error('--out required')
    if a.summarize:
        print('Summary:', summarize(a.out), flush=True)
        return
    if not a.part or a.reps < 1:
        p.error('--part and positive --reps required')
    out = a.out / a.part
    if a.missing:
        absent = missing(a.inputs, a.part)
        for name, files in absent.items():
            print('Missing for {}: {} files, e.g. {}'.format(name, len(files), files[0]), flush=True)
        sys.exit(1 if absent else 0)
    if not (out / 'audit.json').exists():
        try:
            prepare(a.inputs, a.part, out)
        except MissingInputs as error:
            print(error, flush=True)
            sys.exit(3)
    bootstrap(a.part, out, a.reps)
    print('Summary:', summarize(a.out), flush=True)


if __name__ == '__main__':
    main()
