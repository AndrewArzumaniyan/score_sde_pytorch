"""CPU checks for celeba_a800k_bootstrap.py and its launcher on a small synthetic tree; NumPy and SciPy only."""
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / 'tests'))

import celeba_final_ode_bootstrap as base
import celeba_a800k_bootstrap as b
import check_paper_diff_bootstrap as shared
import paper_diff_bootstrap as pdb

PART = b.PART
RTX, H200 = 'NVIDIA RTX PRO 6000 Blackwell', 'NVIDIA H200'


def installed():
    """The tables of celeba_a800k_bootstrap in paper_diff_bootstrap, restored on exit."""
    return patch.multiple(pdb, ROOTS=b.ROOTS, MODELS=b.MODELS, DIFFS=b.DIFFS, PARTS=(PART,), CELLS=b.CELLS,
                          INDEX=b.INDEX, audit_cell=b.audit_cell)


class Plan(unittest.TestCase):
    def test_check_leaves_paper_diff_bootstrap_as_it_was(self):
        with patch.multiple(pdb, ROOTS=pdb.ROOTS, MODELS=pdb.MODELS, DIFFS=pdb.DIFFS, PARTS=pdb.PARTS,
                            CELLS=pdb.CELLS, INDEX=pdb.INDEX, audit_cell=pdb.audit_cell):
            b.check()
            self.assertEqual(pdb.PARTS, (PART,))
        self.assertEqual(pdb.PARTS, ('celeba', 'afhq', 'afhq200k'))
        self.assertEqual(len(pdb.CELLS), 36)

    def test_differences_and_indices(self):
        labels = [(b.MODELS[l]['label'], b.MODELS[r]['label']) for l, r in b.DIFFS]
        self.assertEqual(labels, [('DIAL 800k', 'A 800k'), ('B 800k', 'A 800k')])
        for model in ('celeba_dial800', 'celeba_b800'):
            mine = {k: v for k, v in b.MODELS[model].items() if k != 'part'}
            theirs = {k: v for k, v in pdb.MODELS[model].items() if k != 'part'}
            self.assertEqual(mine, theirs)
        self.assertEqual(sorted(b.INDEX.values()), [0, 1, 2, 3, 8, 9, 10, 11, 36, 37, 38, 39])
        self.assertEqual((b.A800['ckpt'], b.A800['step'], b.A800['work']), (160, 800001, 'celeba_a'))

    def test_cells_match_the_eval_queue(self):
        try:  # The CPU bootstrap image may lack the evaluation stack; check_celeba_a800k.py runs in the RTX image.
            import celeba_a800k_eval as e
        except Exception as error:
            self.skipTest('the eval module does not import here: {!r}'.format(error))
        for seed in (0, 1):
            for method in pdb.METHODS:
                cell = b.A800['result'].format(method=method, seed=seed)
                self.assertIn(cell, e.CELLS)
                self.assertEqual(b.A800['folder'].format(method=method, seed=seed), e.eval_folder(cell))
        self.assertEqual(b.ROOTS['a800k'], 'celeba_a800k_50k_20261115')


class Bootstrap(shared.Tree):
    def setUp(self):
        patcher = installed()
        patcher.start()
        self.addCleanup(patcher.stop)
        super().setUp()
        self.lanes(RTX, RTX)

    def lanes(self, s0, s1):
        for seed, gpu in ((0, s0), (1, s1)):
            (self.inputs / 'a800k/logs/preflight_s{}.json'.format(seed)).write_text(
                json.dumps(dict(environment=dict(visible_cuda_devices=[gpu]))))

    def test_part(self):
        self.assertEqual(self.run_part(PART), 'completed')
        summary = self.summary()
        self.assertEqual(list(summary['differences']), ['celeba_dial800_minus_celeba_a800',
                                                        'celeba_b800_minus_celeba_a800'])
        entry = summary['differences']['celeba_b800_minus_celeba_a800']
        p = self.points
        self.assertAlmostEqual(entry['contrasts']['heun_seed1']['point'],
                               p['celeba_b800_ode_edm_heun_seed1'] - p['celeba_a800_ode_edm_heun_seed1'], places=10)
        self.assertEqual((entry['status'], entry['sampling_hardware'], entry['training_hardware_differs']),
                         ('completed', 'same', True))
        self.assertEqual(len((self.out / 'summary.csv').read_text().splitlines()), 1 + 2 * 9)

    def test_draws_follow_the_indices(self):
        self.run_part(PART, reps=2)
        row = pdb.read_draws(self.out / PART)[1]
        with np.load(self.out / PART / 'reference_moments.npz') as z:
            ref = (z['mean'], z['cov'], z['sqrt'])
        for name, index in (('celeba_dial800_ode_edm_heun_seed1', 3), ('celeba_a800_ode_dpmpp_2m_seed0', 36)):
            rng = np.random.default_rng(np.random.SeedSequence([pdb.SEED, index, 1]))
            weights = np.bincount(rng.integers(0, shared.N, shared.N), minlength=shared.N)
            x = np.load(self.out / PART / (name + '.npy'))
            self.assertAlmostEqual(row['fid'][name], base.fid(x, ref, weights), places=10)

    def test_gpu_of_each_lane(self):
        self.lanes(RTX, H200)
        self.run_part(PART, reps=1)
        audit = json.loads((self.out / PART / 'audit.json').read_text())
        gpus = {s['name']: s['sampling_gpu'] for s in audit['sets']}
        self.assertEqual((gpus['celeba_a800_ode_edm_heun_seed0'], gpus['celeba_a800_ode_edm_heun_seed1']),
                         ([RTX], [H200]))
        self.assertEqual(self.summary()['differences']['celeba_dial800_minus_celeba_a800']['sampling_hardware'],
                         'differs')

    def test_unfinished_a800k_writes_nothing(self):
        result = self.inputs / 'a800k/logs/celeba_a_c160_m35_ode_edm_heun_seed1_result.json'
        result.unlink()
        with self.assertRaisesRegex(pdb.MissingInputs, 'celeba_a800_ode_edm_heun_seed1'):
            pdb.prepare(self.inputs, PART, self.out / PART)
        self.assertFalse((self.out / PART).exists())

    def test_wrong_checkpoint_fails_only_a800k(self):
        self.make(dict(name='celeba_a800_ode_dpmpp_2m_seed1', model='celeba_a800', method='ode_dpmpp_2m', seed=1),
                  shift=0, checkpoint_id='other')
        self.run_part(PART, reps=1)
        summary = self.summary()
        self.assertEqual({d['status'] for d in summary['differences'].values()}, {'failed'})
        self.assertIn('Shard identity', summary['differences']['celeba_b800_minus_celeba_a800']['failed_cells'][
            'celeba_a800_ode_dpmpp_2m_seed1'])


class Launcher(unittest.TestCase):
    def setUp(self):
        self.root = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, str(self.root))
        self.bin, self.state, self.base = self.root / 'bin', self.root / 'state', self.root / 'base'
        self.worktree = self.root / 'worktree'
        for path in (self.bin, self.state, self.worktree):
            path.mkdir()
        for name, text in dict(shared.STUBS, docker=shared.DOCKER).items():
            (self.bin / name).write_text(text)
            (self.bin / name).chmod(0o755)
        for name in ('run_celeba_a800k_bootstrap.sh', 'celeba_a800k_bootstrap.py', 'paper_diff_bootstrap.py',
                     'celeba_final_ode_bootstrap.py', 'celeba_50k_lo_bootstrap.py'):
            shutil.copy(str(REPO / name), str(self.worktree))
        stats = self.base / 'score_sde_pytorch/assets/stats'
        stats.mkdir(parents=True)
        (stats / 'celeba_64_0c2d89311192_stats.npz').write_text('x')
        for path in b.ROOTS.values():
            for sub in ('workdirs', 'logs'):
                (self.base / path / sub).mkdir(parents=True, exist_ok=True)

    def launch(self, *arguments):
        environment = dict(os.environ, PATH='{}:{}'.format(self.bin, os.environ['PATH']), STATE=str(self.state),
                           A800KB_SERVER_BASE=str(self.base))
        return subprocess.run(['bash', str(self.worktree / 'run_celeba_a800k_bootstrap.sh')] + list(arguments),
                              stdout=subprocess.PIPE, stderr=subprocess.STDOUT, env=environment)

    def events(self):
        path = self.state / 'events'
        return path.read_text().splitlines() if path.exists() else []

    def test_start_mounts_three_roots_read_only(self):
        result = self.launch('start')
        self.assertEqual(result.returncode, 0, result.stdout)
        args = (self.state / 'paper-boot-a800k.args').read_text()
        for key in ('fox800', 'gpu2q', 'a800k'):
            self.assertIn('{},dst=/input/{},readonly'.format(self.base / b.ROOTS[key], key), args)
        self.assertNotIn('gpu2a', args)
        self.assertIn('celeba_64_0c2d89311192_stats.npz,dst=/input/reference_celeba.npz,readonly', args)
        self.assertIn('/code/celeba_a800k_bootstrap.py\n--part\nceleba_a800k\n', args)
        self.assertIn('--missing', (self.state / 'missing.args').read_text())
        runtime = self.base / 'celeba_a800k_bootstrap_20261115/runtime'
        self.assertEqual(sorted(p.name for p in runtime.iterdir()),
                         ['celeba_50k_lo_bootstrap.py', 'celeba_a800k_bootstrap.py', 'celeba_final_ode_bootstrap.py',
                          'git_head.txt', 'image_id.txt', 'paper_diff_bootstrap.py'])
        self.assertFalse((self.base / 'paper_diff_bootstrap_20261115').exists())

    def test_refusals(self):
        (self.state / 'celeba-a800k-s1.polls').write_text('5')
        self.assertIn(b'celeba-a800k-s1 still runs', self.launch('start').stdout)
        (self.state / 'celeba-a800k-s1.polls').write_text('0')
        (self.state / 'missing').write_text('x')
        self.assertIn(b'not ready; nothing started', self.launch('start').stdout)
        (self.state / 'missing').unlink()
        self.assertEqual(self.launch('start', 'celeba').returncode, 2)
        self.assertEqual(self.launch('start').returncode, 0)
        self.assertIn(b'Container exists: paper-boot-a800k', self.launch('start').stdout)
        self.assertEqual(self.events(), ['run paper-boot-a800k'])


if __name__ == '__main__':
    unittest.main()
