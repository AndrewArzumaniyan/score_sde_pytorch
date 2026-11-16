"""Reference ODE solve (257 NFE) for two more checks of the main contrast; the rule is fixed on 16 November
before any of these cells.

Questions.
1. VP-lambda@500k. The step rule picked 500k for VP-lambda; the main contrast uses 400k, its 10k minimum.
   At 35 NFE 500k is worse under DPM++ (1.908 vs 1.778) and better under EDM Heun (1.672 vs 1.702).
   Does the rule-selected checkpoint change the strongest VP-lambda in the reference solve?
2. Second training run (seed 43). DIAL seed 43 at 800k minus VP-lambda seed 43 at 400k gave -0.437
   (DPM++) and -0.019 (EDM Heun) at 35 NFE. Where is it in the reference solve?

Rule.
- Solver convergence: |FID(DPM++) - FID(EDM Heun)| at 257 NFE, 10k, per model; at most 0.05 calls the solve
  converged, as in nfe_limit_eval.py.
- VP-lambda@500k minus VP-lambda@400k (1.6193, nfe_limit_20261116/a), 50k, DPM++ 2M with 257 NFE, sampling
  seed 0. If it is below -0.05, the rule-selected 500k is the strongest VP-lambda in the reference solve and
  the contrasts DIAL@800k - VP-lambda@500k and B@800k - VP-lambda@500k are reported next to those with
  400k. Otherwise 400k stays the strongest VP-lambda and 500k is reported alongside. Both are always reported.
- Seed 43: DIAL43@800k minus VP-lambda43@400k under the same conditions. |delta| < 0.05 is indistinguishable.
  The gap replicates if it has the sign of seed 42 (-0.172) and |delta| >= 0.05. Absolute FID of the two
  training runs is not mixed. B was not trained for seed 43, so this pair tests the gap, not its decomposition.
- A negative result is reported. One training run per model, one sampling seed; intervals come later from
  the saved features.

Cells. Window [-10; 9.115429865459795], Tweedie, sampling seed 0, 256 trajectory calls (NFE 257). Each lane
first runs its 50k DPM++ cell, then 10k DPM++ and 10k EDM Heun. Lanes: a500 (VP-lambda@500k, checkpoint 100
of the 800k H200 run of A), fox43 (DIAL seed 43, 800k), a43 (VP-lambda seed 43, 400k). One lane per card;
any card type, the card goes to preflight.json and every result.

Configs are those of the existing 50k matched35 cells of each model at 34 calls (VP-lambda@500k:
gpu2_a50k_grids with checkpoint 100; seed 43: celeba_pair43_lo_eval), then only eval.sampling_num_scales and
eval.num_samples change. Everything else is nfe_limit_eval.py: preflight, queue, report, summary.

Run through run_nfe_limit_extra.sh from a separate worktree; NFE_LIMIT_LANE=a500|fox43|a43.
`nfe_limit_extra_eval.py combine DIR...` prints the contrasts of the rule from lane summaries of this queue
and of nfe_limit_20261116 (lanes dial, b, a).
"""
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import celeba_pair43_lo_eval as p
import gpu2_a50k_grids as grids
import gpu2_queue_eval as q
import nfe_limit_eval as n

DPM, HEUN, TOP = n.DPM, n.HEUN, n.TOP
TEN_K, FIFTY_K = n.TEN_K, n.FIFTY_K
LANES = ('a500', 'fox43', 'a43')
A43_WORK, FOX43_WORK = p.MODELS['A']['work'], p.MODELS['fox']['work']
THRESHOLD = 0.05
SEED42 = dict(DIAL=1.4476, B=1.4729, VPL=1.6193)  # nfe_limit_20261116, 50k DPM++-257, for the rule text.

q.SOURCES.setdefault('celeba_a43', dict(dataset='CELEBA', arm='A', manifest=A43_WORK, freq=50000))
q.SOURCES.setdefault('celeba_fox43', dict(dataset='CELEBA', arm='fox', manifest=FOX43_WORK, freq=5000))

MODELS = {
    'VPL500': dict(source='celeba_a', ckpt=100, dataset='CELEBA', lane='a500', label='VP-lambda@500k'),
    'DIAL43': dict(source='celeba_fox43', ckpt=160, dataset='CELEBA', lane='fox43', label='DIAL seed 43 @800k'),
    'VPL43': dict(source='celeba_a43', ckpt=8, dataset='CELEBA', lane='a43', label='VP-lambda seed 43 @400k'),
}
PAIR43 = {'DIAL43': 'fox', 'VPL43': 'A'}


def lane_of(model):
    return [n.cell(model, TOP, DPM, FIFTY_K), n.cell(model, TOP, DPM, TEN_K), n.cell(model, TOP, HEUN, TEN_K)]


def matched35_config(spec):
    """The config of the model's existing 50k matched35 cell with this solver: 34 calls, 50k, seed 0."""
    if spec['model'] in PAIR43:
        if spec['model'] == 'DIAL43':
            saved = q.manifest('celeba_fox43')
            if saved['config']['seed'] != p.SEED:
                raise ValueError('Expected training seed 43 behind DIAL seed 43')
        return p.get_config(p.cell_name(PAIR43[spec['model']], spec['method'], 0))
    return grids.config_of(dict(spec, kind='m35', calls=q.M35_CALLS, count=FIFTY_K))


def lane():
    value = os.environ.get('NFE_LIMIT_LANE', '')
    if value not in LANES:
        raise ValueError('NFE_LIMIT_LANE must be one of {}, not {!r}'.format(', '.join(LANES), value))
    return value


def run(name):
    spec = n.resolve(name)
    began, clock = time.time(), time.monotonic()
    result = subprocess.run([sys.executable, '-u', 'main.py', '--mode=eval',
                             '--config=nfe_limit_extra_eval.py:' + name, '--workdir=workdirs/' + spec['work'],
                             '--eval_folder=' + n.eval_folder(name)], cwd=str(n.ROOT))
    n.write_json(n.LOGS / (name + '_timing.json'), dict(started_unix=began, finished_unix=time.time(),
        elapsed_seconds=time.monotonic() - clock, returncode=result.returncode))
    result.check_returncode()


# The queue machinery of nfe_limit_eval.py, on the cells of this queue. The order is fixed in advance.
n.MODELS = MODELS
n.CELLS = dict(lane_of('VPL500') + lane_of('DIAL43') + lane_of('VPL43'))
n.LANES = LANES
n.LANE_CARD = {name: os.environ.get('NFE_LIMIT_CARD', '') for name in LANES}  # Set by the launcher.
n.matched35_config = matched35_config
n.lane = lane
n.run = run
CELLS = n.CELLS


def get_config(name):
    return n.get_config(name)


def read_cells(directories):
    cells, lanes = {}, {}
    for directory in directories:
        path = Path(directory) / 'summary.json'
        if not path.exists():
            continue
        summary = json.loads(path.read_text())
        lanes[summary['lane']] = '{} ({}/{}), {}'.format(summary['status'], summary['completed_cells'],
                                                        summary['total_cells'],
                                                        summary['cells'][0]['gpu_name'] if summary['cells'] else '-')
        for value in summary['cells']:
            cells[value['cell']] = value
    return cells, lanes


def value(cells, model, method=DPM, count=FIFTY_K):
    """FID after Tweedie at 257 NFE; models of both queues share the cell naming of nfe_limit_eval.cell."""
    name = '{}_nfe{}_{}_{}k'.format(model.lower(), TOP + 1, n.SHORT[method], count // 1000)
    cell = cells.get(name)
    return None if cell is None else cell['fid_tweedie_xt']


def difference(cells, first, second):
    a, b = value(cells, first), value(cells, second)
    out = dict(first=first, second=second, first_fid=a, second_fid=b)
    if a is not None and b is not None:
        out['difference'] = a - b
    return out


def combine(directories):
    """The contrasts of the rule; reads summaries only, prints JSON."""
    cells, lanes = read_cells(directories)
    gaps = {}
    for model in ('VPL500', 'DIAL43', 'VPL43'):
        dpm, heun = value(cells, model, DPM, TEN_K), value(cells, model, HEUN, TEN_K)
        if dpm is not None and heun is not None:
            gaps[model] = dict(dpm=dpm, heun=heun, dpm_minus_heun=dpm - heun,
                               converged=abs(dpm - heun) <= n.CONVERGED + 1e-9)
    step = difference(cells, 'VPL500', 'VPL')
    if 'difference' in step:
        step['status'] = ('500k is the strongest VP-lambda in the reference solve' if step['difference'] < -THRESHOLD
                          else '400k stays the strongest VP-lambda; 500k reported alongside')
    with_500k = [difference(cells, 'DIAL', 'VPL500'), difference(cells, 'B', 'VPL500')]
    pair = difference(cells, 'DIAL43', 'VPL43')
    if 'difference' in pair:
        delta = pair['difference']
        pair['status'] = 'indistinguishable' if abs(delta) < THRESHOLD else (
            'DIAL seed 43 better' if delta < 0 else 'VP-lambda seed 43 better')
        pair['replicates_seed42'] = delta <= -THRESHOLD
    out = dict(lanes=lanes, rule='nfe_limit_extra_eval.py docstring (16 November)', solver_gap_257=gaps,
               vpl500_minus_vpl400=step, contrasts_with_vpl500=with_500k, seed43=pair,
               seed42_reference=dict(SEED42, dial_minus_vpl=SEED42['DIAL'] - SEED42['VPL']))
    print(json.dumps(out, indent=2))
    return 0


if __name__ == '__main__':
    mode = sys.argv[1] if len(sys.argv) > 1 else ''
    if mode == 'preflight':
        sys.exit(n.preflight())
    elif mode == 'queue':
        sys.exit(n.queue())
    elif mode == 'plan':
        for cell_label, cell_spec in CELLS.items():
            print(cell_label, cell_spec)
    elif mode == 'combine':
        sys.exit(combine(sys.argv[2:]))
    else:
        raise SystemExit('Usage: NFE_LIMIT_LANE=a500|fox43|a43 nfe_limit_extra_eval.py preflight|queue|plan; '
                         'nfe_limit_extra_eval.py combine DIR...')
