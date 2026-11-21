"""CPU checks for nfe_limit_extra_bootstrap.py on synthetic shards; no server data.

A synthetic earlier run is made with nfe_limit_bootstrap.py itself, then the extra cells are bootstrapped and
combined with it.
"""
import io
import json
from pathlib import Path
import shutil
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

import celeba_final_ode_bootstrap as base
import nfe_limit_bootstrap as nb
import nfe_limit_extra_bootstrap as xb

N, BATCH, DIM = 1000, 100, 8
ROUNDS = N // BATCH
SHIFT = {'dial_nfe257_dpm_50k': 0.0, 'b_nfe257_dpm_50k': 0.05, 'vpl_nfe257_dpm_50k': 0.3,
         'vpl500_nfe257_dpm_50k': 0.35, 'dial43_nfe257_dpm_50k': 0.02, 'vpl43_nfe257_dpm_50k': 0.3}
GPU = {'a': 'H200', 'a500': 'H200'}


def make_cells(module, inputs, reference_bytes, sha, rng):
    inputs.mkdir()
    (inputs / 'reference_celeba.npz').write_bytes(reference_bytes)
    reference = np.load(io.BytesIO(reference_bytes))['pool_3']
    m, c = base.moments(reference)
    vals, vecs = np.linalg.eigh(c)
    ref = (m, c, (vecs * np.sqrt(np.maximum(vals, 0))) @ vecs.T)
    fids = {}
    for name, (lane, work, ckpt, step, _) in module.CELLS.items():
        d, report, rounds, result = module.paths(inputs, name)
        rounds.mkdir(parents=True)
        result.parent.mkdir(parents=True, exist_ok=True)
        features = (rng.normal(size=(N, DIM)) + SHIFT[name]).astype(np.float32)
        for i in range(ROUNDS):
            np.savez(str(rounds / 'statistics_{}.npz'.format(i)), round_id=i,
                     sampling_seed=base.round_seed(0, i), num_samples=BATCH, nfe=257, checkpoint_id='id-' + name,
                     checkpoint_step=step, protocol_sha256='p', training_protocol_sha256='t',
                     **{module.FEATURES: features[i * BATCH:(i + 1) * BATCH]})
        fids[name] = fid = base.fid(features, ref)
        np.savez(str(report), checkpoint_id='id-' + name, checkpoint_step=step, dataset_stats_id=sha,
                 num_samples=N, nfe=257, fid_tweedie_xt=fid, protocol_sha256='p', training_protocol_sha256='t')
        (d / 'manifest.json').write_text(json.dumps(dict(config=dict(
            eval=dict(sampling_seed=0, num_samples=N, batch_size=BATCH, sampling_num_scales=256),
            sampling=dict(method='ode_dpmpp_2m', logsnr_max=9.115429865459795, logsnr_min=-10.0)))))
        result.write_text(json.dumps(dict(step=step, method='ode_dpmpp_2m', sampling_seed=0, num_samples=N,
                                          nfe=257, checkpoint_id='id-' + name, fid_tweedie_xt=fid,
                                          gpu_name=GPU.get(lane, 'RTX'))))
    return fids


class Synthetic(unittest.TestCase):
    def setUp(self):
        self.root = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, str(self.root))
        rng = np.random.default_rng(0)
        buffer = io.BytesIO()
        np.savez(buffer, pool_3=rng.normal(size=(2000, DIM)).astype(np.float32))
        reference_bytes = buffer.getvalue()
        probe = self.root / 'probe.npz'
        probe.write_bytes(reference_bytes)
        sha = base.digest(probe)
        patches = [(base, 'REFERENCE_SHA', sha)]
        for module in (nb, xb.nb):
            patches += [(module, 'N', N), (module, 'BATCH', BATCH), (module, 'DIM', DIM), (module, 'ROUNDS', ROUNDS)]
        for module, attr, value in patches:
            patcher = patch.object(module, attr, value)
            patcher.start()
            self.addCleanup(patcher.stop)
        self.addCleanup(xb.PREVIOUS.clear)
        self.previous_inputs, self.previous = self.root / 'prev_input', self.root / 'prev_out'
        self.inputs, self.out = self.root / 'input', self.root / 'out'
        self.fids = make_cells(nb, self.previous_inputs, reference_bytes, sha, rng)
        self.fids.update(make_cells(xb.nb, self.inputs, reference_bytes, sha, rng))
        nb.prepare(self.previous_inputs, self.previous)
        self.assertEqual(nb.bootstrap(self.previous, 30), 'completed')

    def test_original_module_untouched(self):
        self.assertEqual(tuple(nb.CELLS), xb.PREVIOUS_CELLS)
        self.assertEqual(nb.INDEX, {name: i for i, name in enumerate(xb.PREVIOUS_CELLS)})
        self.assertEqual(sorted(xb.nb.INDEX.values()), [3, 4, 5])
        self.assertIsNot(xb.nb, nb)

    def test_end_to_end_combine(self):
        self.assertEqual(xb.nb.missing(self.inputs), {})
        xb.nb.prepare(self.inputs, self.out)
        xb.PREVIOUS.append(self.previous)
        self.assertEqual(xb.nb.bootstrap(self.out, 25), 'completed')  # Summary at 25 also writes combined.*.
        combined = json.loads((self.out / 'combined.json').read_text())
        self.assertEqual((combined['status'], combined['repeats'], combined['previous_repeats']), ('completed', 25, 30))
        prev_rows, rows = nb.pdb.read_draws(self.previous), xb.nb.pdb.read_draws(self.out)
        for left, right in xb.CROSS:
            entry = combined['differences']['{}_minus_{}'.format(left, right)]
            self.assertAlmostEqual(entry['point'], self.fids[left] - self.fids[right])
            joint = [dict(p['fid'], **r['fid']) for p, r in zip(prev_rows, rows)]
            values = np.asarray([j[left] - j[right] for j in joint])
            self.assertEqual(entry['percentile_95'], [float(v) for v in np.quantile(values, [.025, .975])])
        pair = combined['differences']['dial43_nfe257_dpm_50k_minus_vpl43_nfe257_dpm_50k']
        self.assertLess(pair['percentile_95'][1], 0)
        self.assertEqual(pair['sampling_gpu'], dict(left='RTX', right='RTX'))
        previous_summary = json.loads((self.previous / 'summary.json').read_text())
        for key in ('dial_nfe257_dpm_50k_minus_b_nfe257_dpm_50k', 'dial_nfe257_dpm_50k_minus_vpl_nfe257_dpm_50k'):
            self.assertEqual(combined['differences'][key]['percentile_95'],
                             previous_summary['differences'][key]['percentile_95'])
        self.assertIn('differs', (self.out / 'combined.md').read_text())  # DIAL@800k on RTX, VP-lambda@500k on H200.
        own = json.loads((self.out / 'summary.json').read_text())
        self.assertEqual(list(own['differences']), ['dial43_nfe257_dpm_50k_minus_vpl43_nfe257_dpm_50k'])

    def test_streams_differ_from_previous(self):
        xb.nb.prepare(self.inputs, self.out)
        xb.nb.bootstrap(self.out, 3)
        prev_rows, rows = nb.pdb.read_draws(self.previous), xb.nb.pdb.read_draws(self.out)
        # Same seed and repeat, different stream index: the resampling weights, hence the values, differ.
        for i in range(3):
            self.assertNotAlmostEqual(prev_rows[i]['fid']['vpl_nfe257_dpm_50k'], rows[i]['fid']['vpl43_nfe257_dpm_50k'])
        for index in (3, 4, 5):
            w_old = np.random.default_rng(np.random.SeedSequence([nb.SEED, index - 3, 0])).integers(0, N, N)
            w_new = np.random.default_rng(np.random.SeedSequence([nb.SEED, index, 0])).integers(0, N, N)
            self.assertFalse(np.array_equal(w_old, w_new))

    def test_combine_refuses(self):
        xb.nb.prepare(self.inputs, self.out)
        xb.nb.bootstrap(self.out, 31)
        with self.assertRaisesRegex(ValueError, 'more or misnumbered'):  # 31 draws against 30 previous.
            xb.combine(self.out, self.previous)
        summary_path = self.previous / 'summary.json'
        summary = json.loads(summary_path.read_text())
        summary_path.write_text(json.dumps(dict(summary, bootstrap_seed=1)))
        with self.assertRaisesRegex(ValueError, 'completed with seed'):
            xb.read_previous(self.previous)
        summary_path.write_text(json.dumps(dict(summary, status='running')))
        with self.assertRaisesRegex(ValueError, 'completed with seed'):
            xb.read_previous(self.previous)

    def test_audit_refuses_wrong_step(self):
        _, _, _, result = xb.nb.paths(self.inputs, 'vpl43_nfe257_dpm_50k')
        values = json.loads(result.read_text())
        result.write_text(json.dumps(dict(values, step=400000)))
        with self.assertRaisesRegex(ValueError, 'Unexpected queue result'):
            xb.nb.prepare(self.inputs, self.out)

    def test_check(self):
        with patch.object(xb.nb, 'N', 50000), patch.object(xb.nb, 'ROUNDS', 98):
            xb.check()


if __name__ == '__main__':
    unittest.main()
