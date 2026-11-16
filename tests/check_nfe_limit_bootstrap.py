"""CPU checks for nfe_limit_bootstrap.py on synthetic shards; no server data."""
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

N, BATCH, DIM = 1000, 100, 8
ROUNDS = N // BATCH
SHIFT = {'dial_nfe257_dpm_50k': 0.0, 'b_nfe257_dpm_50k': 0.05, 'vpl_nfe257_dpm_50k': 0.3}


class Synthetic(unittest.TestCase):
    def setUp(self):
        self.root = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, str(self.root))
        self.inputs, self.out = self.root / 'input', self.root / 'out'
        rng = np.random.default_rng(0)
        reference = rng.normal(size=(2000, DIM)).astype(np.float32)
        buffer = io.BytesIO()
        np.savez(buffer, pool_3=reference)
        self.inputs.mkdir()
        (self.inputs / 'reference_celeba.npz').write_bytes(buffer.getvalue())
        sha = base.digest(self.inputs / 'reference_celeba.npz')
        for module, attr, value in ((nb, 'N', N), (nb, 'BATCH', BATCH), (nb, 'DIM', DIM), (nb, 'ROUNDS', ROUNDS),
                                    (base, 'REFERENCE_SHA', sha)):
            patcher = patch.object(module, attr, value)
            patcher.start()
            self.addCleanup(patcher.stop)
        self.fids = {}
        for name, (lane, work, ckpt, step, _) in nb.CELLS.items():
            d, report, rounds, result = nb.paths(self.inputs, name)
            rounds.mkdir(parents=True)
            result.parent.mkdir(parents=True, exist_ok=True)
            features = (rng.normal(size=(N, DIM)) + SHIFT[name]).astype(np.float32)
            for i in range(ROUNDS):
                np.savez(str(rounds / 'statistics_{}.npz'.format(i)), round_id=i,
                         sampling_seed=base.round_seed(0, i), num_samples=BATCH, nfe=257, checkpoint_id='id-' + name,
                         checkpoint_step=step, protocol_sha256='p', training_protocol_sha256='t',
                         **{nb.FEATURES: features[i * BATCH:(i + 1) * BATCH]})
            m, c = base.moments(reference)
            vals, vecs = np.linalg.eigh(c)
            fid = base.fid(features, (m, c, (vecs * np.sqrt(np.maximum(vals, 0))) @ vecs.T))
            self.fids[name] = fid
            np.savez(str(report), checkpoint_id='id-' + name, checkpoint_step=step, dataset_stats_id=sha,
                     num_samples=N, nfe=257, fid_tweedie_xt=fid, protocol_sha256='p', training_protocol_sha256='t')
            (d / 'manifest.json').write_text(json.dumps(dict(config=dict(
                eval=dict(sampling_seed=0, num_samples=N, batch_size=BATCH, sampling_num_scales=256),
                sampling=dict(method='ode_dpmpp_2m', logsnr_max=9.115429865459795, logsnr_min=-10.0)))))
            result.write_text(json.dumps(dict(step=step, method='ode_dpmpp_2m', sampling_seed=0, num_samples=N,
                                              nfe=257, checkpoint_id='id-' + name, fid_tweedie_xt=fid,
                                              gpu_name='H200' if lane == 'a' else 'RTX')))

    def test_end_to_end_and_resume(self):
        self.assertEqual(nb.missing(self.inputs), {})
        nb.prepare(self.inputs, self.out)
        self.assertEqual(nb.bootstrap(self.out, 30), 'completed')
        summary = json.loads((self.out / 'summary.json').read_text())
        main = summary['differences']['dial_nfe257_dpm_50k_minus_b_nfe257_dpm_50k']
        self.assertAlmostEqual(main['point'], self.fids['dial_nfe257_dpm_50k'] - self.fids['b_nfe257_dpm_50k'])
        self.assertEqual((main['repeats'], len(main['percentile_95'])), (30, 2))
        self.assertEqual(main['sampling_gpu'], dict(left='RTX', right='RTX'))
        second = summary['differences']['dial_nfe257_dpm_50k_minus_vpl_nfe257_dpm_50k']
        self.assertLess(second['percentile_95'][1], 0)
        self.assertIn('differs', (self.out / 'summary.md').read_text())
        self.assertEqual(nb.bootstrap(self.out, 40), 'completed')  # Continues the draws.
        self.assertEqual(len(nb.pdb.read_draws(self.out)), 40)
        with self.assertRaisesRegex(ValueError, 'inconsistent'):
            nb.bootstrap(self.out, 20)

    def test_audit_refuses_wrong_nfe(self):
        d, report, _, _ = nb.paths(self.inputs, 'b_nfe257_dpm_50k')
        with np.load(str(report)) as z:
            values = dict(z)
        values['nfe'] = 35
        np.savez(str(report), **values)
        with self.assertRaisesRegex(ValueError, 'Incorrect report nfe'):
            nb.prepare(self.inputs, self.out)

    def test_missing(self):
        (nb.paths(self.inputs, 'vpl_nfe257_dpm_50k')[2] / 'statistics_3.npz').unlink()
        self.assertEqual(list(nb.missing(self.inputs)), ['vpl_nfe257_dpm_50k'])


if __name__ == '__main__':
    unittest.main()
