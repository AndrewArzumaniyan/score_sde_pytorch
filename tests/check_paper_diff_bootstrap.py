"""CPU checks for paper_diff_bootstrap.py and its launcher on a small synthetic tree; NumPy and SciPy only."""
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import time
import unittest
from unittest.mock import patch

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

import celeba_final_ode_bootstrap as base
import paper_diff_bootstrap as pdb

N, BATCH, DIM = 40, 16, 6
ROUNDS = 3
REF_ROWS = 30


class Tree(unittest.TestCase):
    """Synthetic 50k outputs for every cell, shrunk to N rows of DIM features."""

    def setUp(self):
        self.root = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, str(self.root))
        self.inputs, self.out = self.root / 'input', self.root / 'out'
        self.inputs.mkdir()
        rng = np.random.default_rng(1)
        refs = {}
        for dataset, filename in pdb.REFERENCES.items():
            pool = rng.normal(0, 1, (REF_ROWS, DIM)).astype(np.float32)
            np.savez(str(self.inputs / filename), pool_3=pool)
            mean, cov = base.moments(pool)
            vals, vecs = np.linalg.eigh(cov)
            refs[dataset] = (mean, cov, (vecs * np.sqrt(np.maximum(vals, 0))) @ vecs.T)
        self.refs = refs
        self.sha = {d: base.digest(self.inputs / f) for d, f in pdb.REFERENCES.items()}
        for target, name, value in ((pdb, 'N', N), (pdb, 'BATCH', BATCH), (pdb, 'DIM', DIM), (pdb, 'ROUNDS', ROUNDS),
                                    (pdb, 'AFHQ_IMAGES', REF_ROWS), (pdb, 'SUMMARY_EVERY', 2),
                                    (base, 'REFERENCE_SHA', self.sha['CELEBA'])):
            patcher = patch.object(target, name, value)
            patcher.start()
            self.addCleanup(patcher.stop)
        self.points = {}
        for index, cell in enumerate(pdb.CELLS):
            self.make(cell, shift=0.05 * index)
        for key in pdb.ROOTS:
            (self.inputs / key / 'logs').mkdir(parents=True, exist_ok=True)
            self.preflight(key, ['NVIDIA RTX PRO 6000 Blackwell'])

    def preflight(self, key, gpus):
        (self.inputs / key / 'logs/preflight.json').write_text(json.dumps(dict(environment=dict(visible_cuda_devices=gpus))))

    def directory(self, cell):
        m = pdb.MODELS[cell['model']]
        return self.inputs / m['root'] / 'workdirs' / m['work'] / m['folder'].format(method=cell['method'], seed=cell['seed'])

    def make(self, cell, shift, features=True, checkpoint_id=None):
        m = pdb.MODELS[cell['model']]
        d = self.directory(cell)
        rounds = d / 'ckpt_{}'.format(m['ckpt'])
        rounds.mkdir(parents=True, exist_ok=True)
        ident = checkpoint_id or m['checkpoint_id'] or 'id-' + cell['model']
        rng = np.random.default_rng(pdb.INDEX[cell['name']] + 100)
        x = rng.normal(shift, 1, (ROUNDS * BATCH, DIM)).astype(np.float32)
        for i in range(ROUNDS):
            values = dict(round_id=i, sampling_seed=base.round_seed(cell['seed'], i), num_samples=BATCH, nfe=35,
                          checkpoint_id=ident, checkpoint_step=m['step'], protocol_sha256='p', training_protocol_sha256='t')
            if features:
                values[pdb.FEATURES] = x[i * BATCH:(i + 1) * BATCH]
            np.savez(str(rounds / 'statistics_{}.npz'.format(i)), **values)
        fid = base.fid(x[:N], self.refs[m['dataset']])
        self.points[cell['name']] = fid
        np.savez(str(d / 'report_{}.npz'.format(m['ckpt'])), checkpoint_id=m['checkpoint_id'] or 'id-' + cell['model'],
                 checkpoint_step=m['step'], dataset_stats_id=self.sha[m['dataset']], num_samples=N, nfe=35,
                 fid_tweedie_xt=fid, protocol_sha256='p', training_protocol_sha256='t')
        sampling = dict(method=cell['method'], logsnr_max=pdb.LOGSNR_MAX, logsnr_min=-10.0)
        evaluation = dict(sampling_seed=cell['seed'], num_samples=N, batch_size=BATCH, sampling_num_scales=34)
        (d / 'manifest.json').write_text(json.dumps(dict(config=dict(sampling=sampling, eval=evaluation),
                                                          source_sha256='src-' + m['root'])))
        if m['result']:
            logs = self.inputs / m['root'] / 'logs'
            logs.mkdir(parents=True, exist_ok=True)
            (logs / (m['result'].format(method=cell['method'], seed=cell['seed']) + '_result.json')).write_text(
                json.dumps(dict(step=m['step'], method=cell['method'], sampling_seed=cell['seed'], num_samples=N,
                                checkpoint_id='id-' + cell['model'], fid_tweedie_xt=fid)))

    def run_part(self, part, reps=3):
        out = self.out / part
        if not (out / 'audit.json').exists():
            pdb.prepare(self.inputs, part, out)
        pdb.bootstrap(part, out, reps)
        return pdb.summarize(self.out)

    def summary(self):
        return json.loads((self.out / 'summary.json').read_text())


class Plan(unittest.TestCase):
    def test_check(self):
        pdb.check()

    def test_differences_from_the_task(self):
        labels = [(pdb.MODELS[l]['dataset'], pdb.MODELS[l]['label'], pdb.MODELS[r]['label']) for l, r in pdb.DIFFS]
        self.assertEqual(labels, [('CELEBA', 'DIAL 800k', 'B 800k'), ('CELEBA', 'DIAL 800k', 'A 500k'),
                                  ('CELEBA', 'DIAL 800k', 'A 400k'), ('CELEBA', 'B 800k', 'A 500k'),
                                  ('CELEBA', 'B 800k', 'A 400k'), ('AFHQV2', 'DIAL 200k', 'A 250k'),
                                  ('AFHQV2', 'DIAL 200k', 'VP 250k'), ('AFHQV2', 'DIAL 200k', 'A 200k'),
                                  ('AFHQV2', 'DIAL 200k', 'VP 200k')])

    def test_cells_match_the_queues(self):
        try:  # The CPU bootstrap image may lack the evaluation stack; check_afhq200k.py runs in the RTX image.
            import afhq200k_eval as k
            import gpu1_vp_tail_eval as tail
            import gpu2_a50k_grids as grids
            import gpu2_queue_eval as q
        except Exception as error:
            self.skipTest('queue modules do not import here: {!r}'.format(error))
        folders = {m: pdb.MODELS[m]['folder'].format(method='ode_dpmpp_2m', seed=0) for m in pdb.MODELS}
        self.assertEqual(folders['celeba_b800'], q.eval_folder('celeba_b_c160_m35_ode_dpmpp_2m_seed0'))
        self.assertEqual(folders['afhq_dial200'], q.eval_folder('afhq_fox_c4_m35_ode_dpmpp_2m_seed0'))
        self.assertEqual(folders['afhq_a250'], q.eval_folder('afhq_a_cbest_m35_ode_dpmpp_2m_seed0'))
        self.assertEqual(folders['afhq_vp250'], tail.eval_folder('afhq_vp_cbest_m35_ode_dpmpp_2m_seed0'))
        self.assertEqual(folders['celeba_a500'], grids.eval_folder('celeba_a_cbest_m35_ode_dpmpp_2m_seed0'))
        self.assertEqual(folders['afhq_a200'], k.eval_folder('afhq_a_c4_m35_ode_dpmpp_2m_seed0'))
        self.assertEqual(folders['afhq_vp200'], k.eval_folder('afhq_vp_c4_m35_ode_dpmpp_2m_seed0'))
        for model, cells in (('celeba_b800', q.CELLS), ('afhq_dial200', q.CELLS), ('afhq_a250', q.CELLS),
                             ('afhq_vp250', tail.CELLS), ('celeba_a500', grids.CELLS), ('afhq_a200', k.CELLS),
                             ('afhq_vp200', k.CELLS)):
            self.assertIn(pdb.MODELS[model]['result'].format(method='ode_edm_heun', seed=1), cells)


class Bootstrap(Tree):
    def test_all_parts(self):
        for part in pdb.PARTS:
            status = self.run_part(part)
        self.assertEqual(status, 'completed')
        summary = self.summary()
        self.assertEqual({d['status'] for d in summary['differences'].values()}, {'completed'})
        entry = summary['differences']['celeba_dial800_minus_celeba_a400']
        p = self.points
        point = (p['celeba_dial800_ode_dpmpp_2m_seed0'] - p['celeba_a400_ode_dpmpp_2m_seed0'] -
                 p['celeba_dial800_ode_edm_heun_seed0'] + p['celeba_a400_ode_edm_heun_seed0'])
        self.assertAlmostEqual(entry['contrasts']['interaction_seed0']['point'], point, places=10)
        self.assertEqual(entry['contrasts']['heun_mean']['repeats'], 3)
        self.assertEqual(entry['sampling_hardware'], 'same')
        self.assertIs(entry['training_hardware_differs'], True)  # H200 against H200 and RTX.
        self.assertEqual(len((self.out / 'summary.csv').read_text().splitlines()), 1 + 9 * 9)
        self.assertEqual(len((self.out / 'summary.md').read_text().splitlines()), 6 + 9)

    def test_draws_depend_only_on_cell_and_repeat(self):
        self.run_part('afhq', reps=2)
        row = pdb.read_draws(self.out / 'afhq')[1]
        name = 'afhq_dial200_ode_edm_heun_seed1'
        rng = np.random.default_rng(np.random.SeedSequence([pdb.SEED, pdb.INDEX[name], 1]))
        weights = np.bincount(rng.integers(0, N, N), minlength=N)
        x = np.load(self.out / 'afhq' / (name + '.npy'))
        with np.load(self.out / 'afhq/reference_moments.npz') as z:
            ref = (z['mean'], z['cov'], z['sqrt'])
        self.assertAlmostEqual(row['fid'][name], base.fid(x, ref, weights), places=10)

    def test_pending_parts_and_partial_repeats(self):
        self.run_part('celeba', reps=3)
        self.run_part('afhq', reps=2)
        summary = self.summary()
        statuses = {k: v['status'] for k, v in summary['differences'].items()}
        self.assertEqual(statuses['afhq_dial200_minus_afhq_a200'], 'pending')
        self.assertEqual(statuses['afhq_dial200_minus_afhq_vp250'], 'completed')
        self.assertEqual(statuses['celeba_b800_minus_celeba_a500'], 'completed')
        self.assertEqual(summary['status'], 'running')
        with open(self.out / 'afhq/draws.jsonl', 'a') as stream:
            stream.write('{"rep": 2, "fid": {"af')  # Another container mid-write.
        self.assertEqual(len(pdb.read_draws(self.out / 'afhq')), 2)

    def test_failed_cell_fails_only_its_differences(self):
        self.make(dict(name='afhq_vp250_ode_edm_heun_seed0', model='afhq_vp250', method='ode_edm_heun', seed=0),
                  shift=0, features=False)
        rounds = self.directory(dict(model='afhq_a250', method='ode_dpmpp_2m', seed=1)) / 'ckpt_5'
        shutil.copy(str(rounds / 'statistics_2.npz'), str(rounds / 'statistics_3.npz'))  # A shard too many.
        self.run_part('afhq')
        audit = json.loads((self.out / 'afhq/audit.json').read_text())
        self.assertEqual(sorted(audit['failed']), ['afhq_a250_ode_dpmpp_2m_seed1', 'afhq_vp250_ode_edm_heun_seed0'])
        self.assertFalse((self.out / 'afhq/afhq_a250_ode_dpmpp_2m_seed1.npy').exists())
        self.run_part('afhq200k')
        summary = self.summary()
        statuses = {k: v['status'] for k, v in summary['differences'].items()}
        self.assertEqual((statuses['afhq_dial200_minus_afhq_a250'], statuses['afhq_dial200_minus_afhq_vp250'],
                          statuses['afhq_dial200_minus_afhq_a200']), ('failed', 'failed', 'completed'))
        self.assertIn('pool_3_tweedie_xt', summary['differences']['afhq_dial200_minus_afhq_vp250']['failed_cells'][
            'afhq_vp250_ode_edm_heun_seed0'])

    def test_unfinished_cell_writes_nothing(self):
        report = self.directory(dict(model='celeba_a500', method='ode_edm_heun', seed=1)) / 'report_100.npz'
        report.rename(report.with_suffix('.bak'))
        result = subprocess.run([sys.executable, str(REPO / 'paper_diff_bootstrap.py'), '--part', 'celeba', '--inputs',
                                 str(self.inputs), '--out', str(self.out), '--missing'], stdout=subprocess.PIPE)
        self.assertEqual(result.returncode, 1)
        self.assertIn(b'celeba_a500_ode_edm_heun_seed1', result.stdout)
        with self.assertRaisesRegex(pdb.MissingInputs, 'celeba_a500_ode_edm_heun_seed1'):
            pdb.prepare(self.inputs, 'celeba', self.out / 'celeba')
        self.assertFalse((self.out / 'celeba').exists())
        shard = self.directory(dict(model='afhq_vp250', method='ode_dpmpp_2m', seed=0)) / 'ckpt_5/statistics_1.npz'
        shard.unlink()
        self.assertEqual(list(pdb.missing(self.inputs, 'afhq')), ['afhq_vp250_ode_dpmpp_2m_seed0'])
        (self.inputs / 'gpu2q/logs/afhq_fox_c4_m35_ode_dpmpp_2m_seed1_result.json').unlink()
        self.assertEqual(sorted(pdb.missing(self.inputs, 'afhq')),
                         ['afhq_dial200_ode_dpmpp_2m_seed1', 'afhq_vp250_ode_dpmpp_2m_seed0'])
        report.with_suffix('.bak').rename(report)
        self.assertEqual(pdb.missing(self.inputs, 'celeba'), {})

    def test_summaries_from_two_containers_with_one_pid(self):
        self.run_part('afhq', reps=1)
        with patch('os.getpid', lambda: 7):
            for _ in range(3):
                pdb.summarize(self.out)
        self.assertEqual([p.name for p in self.out.iterdir() if p.name.endswith('.tmp')], [])
        with patch.object(pdb, 'summarize', side_effect=OSError('race')):
            pdb.bootstrap('afhq', self.out / 'afhq', 3)  # Interim summaries at 2; repeats go on.
        self.assertEqual(len(pdb.read_draws(self.out / 'afhq')), 3)

    def test_identity_refusals(self):
        cell = dict(name='celeba_a400_ode_dpmpp_2m_seed0', model='celeba_a400', method='ode_dpmpp_2m', seed=0)
        self.make(cell, shift=0, checkpoint_id='other')
        manifest = self.directory(dict(model='celeba_b800', method='ode_edm_heun', seed=1)) / 'manifest.json'
        saved = json.loads(manifest.read_text())
        saved['config']['sampling']['logsnr_max'] = 8.0
        manifest.write_text(json.dumps(saved))
        report = self.directory(dict(model='celeba_a500', method='ode_edm_heun', seed=0)) / 'report_100.npz'
        values = dict(np.load(report))
        values['fid_tweedie_xt'] = values['fid_tweedie_xt'] + 0.01
        np.savez(str(report), **values)
        audit = pdb.prepare(self.inputs, 'celeba', self.out / 'celeba')
        self.assertEqual(sorted(audit['failed']), ['celeba_a400_ode_dpmpp_2m_seed0', 'celeba_a500_ode_edm_heun_seed0',
                                                   'celeba_b800_ode_edm_heun_seed1'])
        self.assertIn('Shard identity', audit['failed']['celeba_a400_ode_dpmpp_2m_seed0'])
        self.assertIn('Queue result and report differ', audit['failed']['celeba_a500_ode_edm_heun_seed0'])

    def test_hardware(self):
        self.preflight('gpu2a', ['NVIDIA H200'])
        self.run_part('celeba', reps=1)
        differences = self.summary()['differences']
        self.assertEqual(differences['celeba_b800_minus_celeba_a500']['sampling_hardware'], 'differs')
        self.assertEqual(differences['celeba_dial800_minus_celeba_b800']['sampling_hardware'], 'same')
        self.assertFalse(differences['celeba_dial800_minus_celeba_b800']['training_hardware_differs'])
        self.assertTrue(differences['celeba_dial800_minus_celeba_a400']['training_hardware_differs'])
        (self.inputs / 'fox800/logs/preflight.json').unlink()
        pdb.summarize(self.out)  # Recorded at audit time, so the summary keeps the GPU.
        self.assertEqual(self.summary()['differences']['celeba_dial800_minus_celeba_b800']['sampling_hardware'], 'same')

    def test_resume(self):
        self.run_part('afhq200k', reps=2)
        self.assertEqual(self.summary()['differences']['afhq_dial200_minus_afhq_a200']['status'], 'pending')
        self.run_part('afhq', reps=3)
        self.assertEqual(self.summary()['differences']['afhq_dial200_minus_afhq_a200']['status'], 'partial')
        self.run_part('afhq200k', reps=3)
        self.assertEqual(self.summary()['differences']['afhq_dial200_minus_afhq_a200']['status'], 'completed')
        with self.assertRaisesRegex(ValueError, 'inconsistent'):
            pdb.bootstrap('afhq200k', self.out / 'afhq200k', 2)


DOCKER = r"""#!/usr/bin/env bash
last="${@: -1}"
case "$1" in
  image) echo sha256:image ;;
  inspect)
    [[ -e "$STATE/$last.polls" ]] || { echo; exit 1; }
    case "$*" in
      *State.Running*) n="$(cat "$STATE/$last.polls")"; if (( n > 0 )); then echo $((n - 1)) > "$STATE/$last.polls"; echo true; else echo false; fi ;;
      *ExitCode*) cat "$STATE/$last.code" ;;
      *) echo '[]' ;;
    esac ;;
  run)
    if [[ "$*" == *--missing* ]]; then
      [[ -e "$STATE/missing" ]] && { echo "Missing for cell"; exit 1; }
      echo "$*" > "$STATE/missing.args"; exit 0
    fi
    name=""; prev=""; for arg in "$@"; do [[ "$prev" == --name ]] && name="$arg"; prev="$arg"; done
    [[ -z "$name" ]] && name=check
    echo 0 > "$STATE/$name.polls"; echo 0 > "$STATE/$name.code"
    echo "run $name" >> "$STATE/events"; printf '%s\n' "$@" > "$STATE/$name.args" ;;
esac
exit 0
"""
STUBS = {
    'git': '#!/usr/bin/env bash\n[[ "$1" == rev-parse ]] && echo commit\nexit 0\n',
    'df': '#!/usr/bin/env bash\necho "Filesystem 1 2 3 4"\necho "d 0 0 999999999999999 0"\n',
}


class Launcher(unittest.TestCase):
    def setUp(self):
        self.root = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, str(self.root))
        self.bin, self.state, self.base = self.root / 'bin', self.root / 'state', self.root / 'base'
        self.worktree = self.root / 'worktree'
        for path in (self.bin, self.state, self.worktree):
            path.mkdir()
        for name, text in dict(STUBS, docker=DOCKER).items():
            (self.bin / name).write_text(text)
            (self.bin / name).chmod(0o755)
        for name in ('run_paper_diff_bootstrap.sh', 'paper_diff_bootstrap.py', 'celeba_final_ode_bootstrap.py',
                     'celeba_50k_lo_bootstrap.py'):
            shutil.copy(str(REPO / name), str(self.worktree))
        stats = self.base / 'score_sde_pytorch/assets/stats'
        stats.mkdir(parents=True)
        for name in ('celeba_64_0c2d89311192_stats.npz', 'afhqv2_64_cd017afab009_stats.npz'):
            (stats / name).write_text('x')
        for path in pdb.ROOTS.values():
            for sub in ('workdirs', 'logs'):
                (self.base / path / sub).mkdir(parents=True, exist_ok=True)
        self.container('afhqv2-200k-a', 2, 0)
        self.container('afhqv2-200k-vp', 3, 0)

    def container(self, name, polls, code):
        (self.state / (name + '.polls')).write_text(str(polls))
        (self.state / (name + '.code')).write_text(str(code))

    def launch(self, *arguments, **env):
        environment = dict(os.environ, PATH='{}:{}'.format(self.bin, os.environ['PATH']), STATE=str(self.state),
                           PDB_SERVER_BASE=str(self.base), PDB_WAIT_SECONDS='0', **env)
        return subprocess.run(['bash', str(self.worktree / 'run_paper_diff_bootstrap.sh')] + list(arguments),
                              stdout=subprocess.PIPE, stderr=subprocess.STDOUT, env=environment)

    def events(self):
        path = self.state / 'events'
        return path.read_text().splitlines() if path.exists() else []

    def test_start_celeba_mounts_its_roots_read_only(self):
        result = self.launch('start', 'celeba')
        self.assertEqual(result.returncode, 0, result.stdout)
        args = (self.state / 'paper-boot-celeba.args').read_text()
        for key in ('fox800', 'lo_gpu1', 'gpu2q', 'gpu2a'):
            self.assertIn('{},dst=/input/{},readonly'.format(self.base / pdb.ROOTS[key], key), args)
        self.assertNotIn('gpu1vp', args)
        self.assertIn('celeba_64_0c2d89311192_stats.npz,dst=/input/reference_celeba.npz,readonly', args)
        self.assertIn('--cpus\n8\n', args)
        self.assertIn('--reps\n1000', args)
        runtime = self.base / 'paper_diff_bootstrap_20261115/runtime'
        self.assertEqual(sorted(p.name for p in runtime.iterdir()),
                         ['celeba_50k_lo_bootstrap.py', 'celeba_final_ode_bootstrap.py', 'git_head.txt', 'image_id.txt',
                          'paper_diff_bootstrap.py'])
        result = self.launch('start', 'afhq')
        self.assertEqual(result.returncode, 0, result.stdout)
        args = (self.state / 'paper-boot-afhq.args').read_text()
        self.assertIn('afhqv2_64_cd017afab009_stats.npz,dst=/input/reference_afhqv2.npz,readonly', args)

    def test_unfinished_cells_start_nothing(self):
        (self.state / 'missing').write_text('x')
        result = self.launch('start', 'celeba')
        self.assertEqual(result.returncode, 1)
        self.assertIn(b'Inputs of part celeba are not ready; nothing started', result.stdout)
        self.assertEqual(self.events(), [])
        (self.state / 'missing').unlink()
        self.assertEqual(self.launch('start', 'afhq').returncode, 0)
        args = (self.state / 'missing.args').read_text()
        self.assertIn('--part afhq --inputs /input --out /tmp --missing', args)
        self.assertIn('dst=/input/gpu1vp,readonly', args)

    def test_after_part1_waits_for_both_lanes(self):
        result = self.launch('_wait')
        self.assertEqual(result.returncode, 0, result.stdout)
        self.assertEqual(self.events(), ['run paper-boot-afhq200k'])
        args = (self.state / 'paper-boot-afhq200k.args').read_text()
        self.assertIn('afhq200k_bootstrap_20261115/vp,dst=/input/afhq200k_vp,readonly', args)

    def test_failed_lane_starts_nothing(self):
        self.container('afhqv2-200k-vp', 1, 1)
        self.assertIn(b'afhqv2-200k-vp did not exit with 0; nothing started', self.launch('_wait').stdout)
        self.container('afhqv2-200k-a', 99, 0)
        self.assertIn(b'still runs; nothing started', self.launch('_wait', PDB_WAIT_MAX_SECONDS='0').stdout)
        self.assertEqual(self.events(), [])

    def test_after_part1_returns_at_once(self):
        self.assertIn(b'Waiting for afhqv2-200k-a afhqv2-200k-vp', self.launch('after-part1').stdout)
        for _ in range(100):
            if self.events():
                break
            time.sleep(0.1)
        self.assertEqual(self.events(), ['run paper-boot-afhq200k'])

    def test_refusals(self):
        self.assertIn(b'still runs; use after-part1', self.launch('start', 'afhq200k').stdout)
        self.assertEqual(self.launch('start', 'cifar').returncode, 2)
        self.assertEqual(self.launch('start').returncode, 2)
        self.assertEqual(self.launch('start', 'celeba').returncode, 0)
        self.assertIn(b'Container exists: paper-boot-celeba', self.launch('start', 'celeba').stdout)
        (self.worktree / 'paper_diff_bootstrap.py').write_text('changed')
        self.assertIn(b'Script differs from the first run', self.launch('start', 'afhq').stdout)
        shutil.rmtree(str(self.base / pdb.ROOTS['gpu1vp']))
        shutil.copy(str(REPO / 'paper_diff_bootstrap.py'), str(self.worktree))
        self.assertIn(b'Missing input', self.launch('start', 'afhq').stdout)
        (self.state / 'afhqv2-200k-b.polls').write_text('0')
        (self.state / 'afhqv2-200k-a.polls').unlink()
        self.assertIn(b'No container afhqv2-200k-a', self.launch('after-part1').stdout)
        self.assertEqual(self.events(), ['run paper-boot-celeba'])


if __name__ == '__main__':
    unittest.main()
