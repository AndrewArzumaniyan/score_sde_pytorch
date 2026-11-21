"""Conditional bootstrap intervals for the 257-NFE cells of nfe_limit_extra_eval.py (CPU only).

The rule is fixed on 21 November 2026, before any repeat of these cells.

Cells (nfe_limit_extra_20261116): VP-lambda@500k (lane a500), DIAL seed 43 @800k (fox43), VP-lambda seed 43 @400k
(a43). DPM++ 2M with 256 trajectory calls plus Tweedie (NFE 257), sampling seed 0, window [-10; 9.115429865459795],
50k images, batch 512, 98 shards.

Method: nfe_limit_bootstrap.py unchanged, loaded as a private copy with these cells. Each cell is resampled
independently: rows with replacement from its own 50k pool_3_tweedie_xt features. Checkpoints and reference statistics
are fixed. Recomputed point FIDs must match the reports within 0.001. The bootstrap seed is that of
nfe_limit_20261116/bootstrap (20261117). That run used stream indices 0, 1, 2 for DIAL@800k, B@800k and
VP-lambda@400k. These cells use 3, 4, 5, so all six cells are independent streams of one design.

Differences.
- Within this run: DIAL43 - VP-lambda43, the second training run (point -0.325).
- combine, with the 1000 finished draws of nfe_limit_20261116/bootstrap paired by repeat number: DIAL@800k -
  VP-lambda@500k (-0.270), B@800k - VP-lambda@500k (-0.245), VP-lambda@500k - VP-lambda@400k (+0.098). Pairing by
  number is a valid joint draw because the cells are independent. The earlier DIAL - B and DIAL - VP-lambda@400k are
  recomputed from its draws and must equal its summary.

Reading. Nothing here changes a statement of the main paper.
- The interval is the spread of generated samples at fixed weights, not the spread between trainings. One sampling
  seed, no multiplicity correction. 95% percentile interval; the basic interval is stored as well.
- DIAL43 - VP-lambda43: an interval below zero means the second run favors DIAL with an interval below zero.
  Otherwise the supplement says the point estimate of the second run favors DIAL and its interval includes zero.
- VP-lambda@500k - @400k: an interval above zero supports "400k is the stronger VP-lambda at 257 NFE". Otherwise the
  supplement says the point favors 400k and the interval includes zero.
- The contrasts with VP-lambda@500k fill the dash of main Table 5 in the supplement. Point values do not change.

Inputs under --inputs: a500/, fox43/, a43/ (lane folders of nfe_limit_extra_eval.py) and reference_celeba.npz.
--previous: the finished nfe_limit_20261116/bootstrap folder, read only.
Writes --out: audit.json, draws.jsonl, summary.{json,md} of this run, combined.{json,md} with --previous; every 25
repeats and at the end. A rerun continues the draws.
"""
import argparse
import importlib.util
import json
import sys
import time
from pathlib import Path

import numpy as np

import celeba_final_ode_bootstrap as base


def _private_copy():
    """nfe_limit_bootstrap.py as a separate module object: its cells change here, the imported original does not."""
    spec = importlib.util.spec_from_file_location('_nfe_limit_bootstrap_extra',
                                                  str(Path(__file__).with_name('nfe_limit_bootstrap.py')))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


nb = _private_copy()
PREVIOUS_CELLS = ('dial_nfe257_dpm_50k', 'b_nfe257_dpm_50k', 'vpl_nfe257_dpm_50k')  # Stream indices 0, 1, 2.
nb.CELLS = {  # name: (lane, eval workdir, checkpoint, step, label); workdir = source of nfe_limit_extra_eval.py.
    'vpl500_nfe257_dpm_50k': ('a500', 'celeba_a', 100, 500001, 'VP-lambda@500k'),
    'dial43_nfe257_dpm_50k': ('fox43', 'celeba_fox43', 160, 800001, 'DIAL seed 43 @800k'),
    'vpl43_nfe257_dpm_50k': ('a43', 'celeba_a43', 8, 400001, 'VP-lambda seed 43 @400k'),
}
nb.INDEX = {name: len(PREVIOUS_CELLS) + i for i, name in enumerate(nb.CELLS)}
nb.DIFFS = (('dial43_nfe257_dpm_50k', 'vpl43_nfe257_dpm_50k'),)
CROSS = (('dial_nfe257_dpm_50k', 'vpl500_nfe257_dpm_50k'), ('b_nfe257_dpm_50k', 'vpl500_nfe257_dpm_50k'),
         ('vpl500_nfe257_dpm_50k', 'vpl_nfe257_dpm_50k'))
PREVIOUS_DIFFS = (('dial_nfe257_dpm_50k', 'b_nfe257_dpm_50k'), ('dial_nfe257_dpm_50k', 'vpl_nfe257_dpm_50k'))
LANES = tuple(lane for lane, *_ in nb.CELLS.values())
MOMENTS_TOLERANCE = 1e-6  # Same reference file; only summation order may differ between the two runs.
PREVIOUS = []  # Set by main; the summary every 25 repeats then also refreshes combined.*.


def read_previous(previous):
    """The finished earlier run: its audit, summary and draws, checked for the design this run extends."""
    audit = json.loads((previous / 'audit.json').read_text())
    summary = json.loads((previous / 'summary.json').read_text())
    rows = nb.pdb.read_draws(previous)
    if summary.get('status') != 'completed' or summary.get('bootstrap_seed') != nb.SEED:
        raise ValueError('The previous bootstrap must be completed with seed {}'.format(nb.SEED))
    if tuple(s['name'] for s in audit['sets']) != PREVIOUS_CELLS:
        raise ValueError('Unexpected previous cells: {}'.format([s['name'] for s in audit['sets']]))
    if len(rows) != summary['repeats'] or any(r['rep'] != i or set(r['fid']) != set(PREVIOUS_CELLS)
                                              for i, r in enumerate(rows)):
        raise ValueError('Previous draws inconsistent with its summary')
    return audit, summary, rows


def _entry(sets, left, right, values, source):
    point = sets[left]['report_fid'] - sets[right]['report_fid']
    entry = dict(left=sets[left]['label'], right=sets[right]['label'], point=point, repeats=len(values), source=source,
                 sampling_gpu=dict(left=sets[left]['sampling_gpu'], right=sets[right]['sampling_gpu']))
    if len(values):
        low, median, high = np.quantile(values, [.025, .5, .975])
        entry.update(percentile_95=[float(low), float(high)], basic_95=[float(2 * point - high), float(2 * point - low)],
                     bootstrap_median=float(median), bootstrap_sd=float(values.std(ddof=1)) if len(values) > 1 else None)
    return entry


def combine(out, previous):
    """Differences across this run and the finished earlier one; reads both, writes combined.{json,md} in --out."""
    audit = json.loads((out / 'audit.json').read_text())
    prev_audit, prev_summary, prev_rows = read_previous(previous)
    if audit['reference_sha256'] != prev_audit['reference_sha256']:
        raise ValueError('The two runs use different reference files')
    with np.load(out / 'reference_moments.npz') as a, np.load(previous / 'reference_moments.npz') as b:
        moments_gap = max(float(np.abs(a[k] - b[k]).max()) for k in ('mean', 'cov', 'sqrt'))
    if moments_gap > MOMENTS_TOLERANCE:
        raise ValueError('Reference moments differ by {}'.format(moments_gap))
    sets = {s['name']: s for s in prev_audit['sets']}
    if set(sets) & {s['name'] for s in audit['sets']}:
        raise ValueError('The runs share cells')
    sets.update({s['name']: s for s in audit['sets']})
    rows = nb.pdb.read_draws(out)
    if len(rows) > len(prev_rows) or any(r['rep'] != i for i, r in enumerate(rows)):
        raise ValueError('This run has more or misnumbered draws than the previous one')
    joint = [dict(prev_rows[i]['fid'], **rows[i]['fid']) for i in range(len(rows))]

    def values(pairs, left, right):
        return np.asarray([r[left] - r[right] for r in pairs])

    diffs = {}
    for left, right in nb.DIFFS:
        diffs['{}_minus_{}'.format(left, right)] = _entry(sets, left, right, values(joint, left, right), 'this run')
    for left, right in CROSS:
        diffs['{}_minus_{}'.format(left, right)] = _entry(sets, left, right, values(joint, left, right),
                                                          'this run + previous, paired by repeat')
    for left, right in PREVIOUS_DIFFS:  # From all previous draws; must reproduce its summary exactly.
        key = '{}_minus_{}'.format(left, right)
        entry = _entry(sets, left, right, values([r['fid'] for r in prev_rows], left, right), 'previous')
        if np.abs(np.subtract(entry['percentile_95'], prev_summary['differences'][key]['percentile_95'])).max() > 1e-12:
            raise ValueError('Recomputed previous interval differs: {}'.format(key))
        diffs[key] = entry
    target = json.loads((out / 'target.json').read_text())['reps'] if (out / 'target.json').exists() else None
    status = 'completed' if target is not None and len(rows) >= target else 'running'
    nb.pdb.save(out / 'combined.json', dict(status=status, repeats=len(rows), previous_repeats=len(prev_rows),
        target=target, bootstrap_seed=nb.SEED, stream_index=dict(nb.INDEX, **{n: i for i, n in enumerate(PREVIOUS_CELLS)}),
        previous=str(previous), reference_moments_max_abs_gap=moments_gap, updated_unix=time.time(),
        scope=nb.pdb.SCOPE + ' One sampling seed.', resampling='independent per cell; cells of the two runs paired by '
        'repeat number', interval='95% percentile; basic also reported; no multiplicity correction',
        points={k: v['report_fid'] for k, v in sets.items()}, differences=diffs))
    lines = ['# Bootstrap of the 257-NFE differences: extra cells with the earlier run', '',
             'FID after Tweedie, DPM++ 2M, 257 NFE, sampling seed 0, 50k; 95% percentile interval (basic in combined.json); '
             'conditional on fixed weights and reference. Status: {}, {} repeats.'.format(status, len(rows)), '',
             '| Difference | Point | 95% percentile | Repeats | Source | 50k GPU |', '|---|---|---|---|---|---|']
    for entry in diffs.values():
        interval = '[{:+.3f}; {:+.3f}]'.format(*entry['percentile_95']) if 'percentile_95' in entry else '—'
        same = entry['sampling_gpu']['left'] == entry['sampling_gpu']['right']
        lines.append('| {} − {} | {:+.3f} | {} | {} | {} | {} |'.format(entry['left'], entry['right'], entry['point'],
                     interval, entry['repeats'], entry['source'], 'same' if same else 'differs'))
    nb.pdb.write_text(out / 'combined.md', '\n'.join(lines) + '\n')
    return status


_summarize = nb.summarize


def _summarize_and_combine(out):
    status = _summarize(out)
    if PREVIOUS:
        combine(out, PREVIOUS[0])
    return status


nb.summarize = _summarize_and_combine  # nb.bootstrap calls it every 25 repeats and at the end.


def check():
    import nfe_limit_bootstrap as original
    base.check()
    assert nb.ROUNDS == 98 and nb.N == 50000 and nb.SEED == original.SEED == 20261117
    assert tuple(original.CELLS) == PREVIOUS_CELLS and original.INDEX == {n: i for i, n in enumerate(PREVIOUS_CELLS)}
    assert sorted(nb.INDEX.values()) == [3, 4, 5] and not set(nb.CELLS) & set(PREVIOUS_CELLS)
    assert len(set(LANES)) == 3 and len(nb.DIFFS) == 1 and len(CROSS) == 3
    names = set(nb.CELLS) | set(PREVIOUS_CELLS)
    assert all(left in names and right in names for left, right in nb.DIFFS + CROSS + PREVIOUS_DIFFS)
    print('nfe limit extra bootstrap checks passed', flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--check', action='store_true')
    p.add_argument('--missing', action='store_true')
    p.add_argument('--check-previous', action='store_true')
    p.add_argument('--summarize', action='store_true')
    p.add_argument('--combine', action='store_true')
    p.add_argument('--inputs', type=Path, default=Path('/input'))
    p.add_argument('--out', type=Path)
    p.add_argument('--previous', type=Path)
    p.add_argument('--reps', type=int, default=1000)
    a = p.parse_args()
    if a.check:
        check()
        return
    if a.missing:
        absent = nb.missing(a.inputs)
        for name, files in absent.items():
            print('Missing for {}: {} files, e.g. {}'.format(name, len(files), files[0]), flush=True)
        sys.exit(1 if absent else 0)
    if a.previous:
        _, summary, rows = read_previous(a.previous)  # Fails before any work if the earlier run does not fit.
        print('Previous bootstrap: {} repeats, seed {}'.format(len(rows), summary['bootstrap_seed']), flush=True)
        PREVIOUS.append(a.previous)
    if a.check_previous:
        if not a.previous:
            p.error('--check-previous needs --previous')
        return
    if not a.out or a.reps < 1:
        p.error('--out and positive --reps required')
    if a.combine:
        if not a.previous:
            p.error('--combine needs --previous')
        print('Combined:', combine(a.out, a.previous), flush=True)
        return
    if a.summarize:
        print('Summary:', nb.summarize(a.out), flush=True)
        return
    if not (a.out / 'audit.json').exists():
        absent = nb.missing(a.inputs)
        if absent:
            print('Inputs not ready: {}'.format({k: v[:2] for k, v in absent.items()}), flush=True)
            sys.exit(3)
        nb.prepare(a.inputs, a.out)
    print('Summary:', nb.bootstrap(a.out, a.reps), flush=True)


if __name__ == '__main__':
    main()
