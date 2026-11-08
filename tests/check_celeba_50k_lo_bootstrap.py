"""Small end-to-end fixtures for the CPU bootstrap of the 50k cells on the lower window."""
import json
from pathlib import Path
import shutil
import sys
import tempfile
from unittest import TestCase, main

import numpy as np
from scipy import linalg

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import celeba_50k_lo_bootstrap as boot


class Synthetic(TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, str(self.tmp))
        saved = boot.N, boot.BATCH, boot.DIM, boot.base.ROUNDS, boot.base.REFERENCE_SHA

        def restore():
            boot.N, boot.BATCH, boot.DIM, boot.base.ROUNDS, boot.base.REFERENCE_SHA = saved

        self.addCleanup(restore)
        boot.N, boot.BATCH, boot.DIM = 20, 8, 4
        boot.base.ROUNDS = 3
        self.roots = dict(lo=self.tmp / 'data', fox800=self.tmp / 'fox800')
        self.out = self.tmp / 'out'
        self.out.mkdir()
        self.reference = self.tmp / 'reference.npz'
        rng = np.random.default_rng(7)
        real = rng.normal(size=(30, 4)).astype(np.float32)
        np.savez(self.reference, pool_3=real)
        boot.base.REFERENCE_SHA = boot.base.digest(self.reference)
        mean, cov = boot.base.moments(real)
        vals, vecs = linalg.eigh(cov)
        ref = (mean, cov, (vecs * np.sqrt(vals)) @ vecs.T)
        cells = {lane: [] for lane in boot.LANES}
        for s in boot.specs():
            m = boot.MODELS[s['model']]
            d = boot.directory(self.roots, s)
            rounds = d / 'ckpt_{}'.format(m['ckpt'])
            rounds.mkdir(parents=True)
            cfg = dict(eval=dict(sampling_seed=s['seed'], num_samples=20, batch_size=8, sampling_num_scales=34),
                       sampling=dict(method=s['method'], logsnr_max=boot.LOGSNR_MAX))
            (d / 'manifest.json').write_text(json.dumps(dict(config=cfg, source_sha256=self.source(m['lane']))))
            identity = dict(checkpoint_id=m['checkpoint_id'], checkpoint_step=m['step'],
                            protocol_sha256='protocol-' + s['name'], training_protocol_sha256='training')
            features = rng.normal(size=(24, 4)).astype(np.float32)
            fid = boot.base.fid(features[:20], ref)
            np.savez(d / 'report_{}.npz'.format(m['ckpt']), **identity, dataset_stats_id=boot.base.REFERENCE_SHA,
                     num_samples=20, nfe=35, fid_tweedie_xt=fid)
            for i in range(3):
                np.savez(rounds / 'statistics_{}.npz'.format(i), **identity, round_id=i,
                         sampling_seed=boot.base.round_seed(s['seed'], i), num_samples=8, nfe=35,
                         pool_3_tweedie_xt=features[8*i:8*(i+1)])
            cells[m['lane']].append(dict(method=s['method'], model=boot.cell_model(s['model']), sampling_seed=s['seed'],
                                         fid_tweedie_xt=float(np.float64(fid))))
        for lane, models in boot.LANES.items():
            logs = boot.lane_root(self.roots, lane) / 'logs'
            logs.mkdir(parents=True)
            checkpoints = {boot.cell_model(model): dict(checkpoint_id=boot.MODELS[model]['checkpoint_id'],
                                       step=boot.MODELS[model]['step']) for model in models}
            (logs / 'preflight.json').write_text(json.dumps(dict(status='passed', checkpoints=checkpoints,
                                                                 source_sha256=self.source(lane))))
            (logs / 'summary.json').write_text(json.dumps(dict(status='completed', completed_cells=len(cells[lane]),
                                                               total_cells=len(cells[lane]), cells=cells[lane])))

    @staticmethod
    def source(lane):
        return 'other-commit' if lane == 'fox800' else 'source'

    def edit(self, path, change):
        saved = json.loads(path.read_text())
        change(saved)
        path.write_text(json.dumps(saved))

    def refused(self, pattern):
        with self.assertRaisesRegex(ValueError, pattern):
            boot.prepare(self.roots, self.reference, self.out)

    def test_prepare_resume_and_reject_corrupt_feature(self):
        boot.prepare(self.roots, self.reference, self.out)
        boot.bootstrap(self.out, 1)
        boot.bootstrap(self.out, 2)
        self.assertEqual(len((self.out / 'draws.jsonl').read_text().splitlines()), 2)
        result = json.loads((self.out / 'summary.json').read_text())
        self.assertEqual(len(result['contrasts']), 40)
        self.assertEqual(len(result['point_fid']), 16)
        self.assertIn('fox800_minus_A_ode_dpmpp_2m_seed1', result['contrasts'])
        self.assertIn('fox800_minus_A_ode_edm_heun_mean', result['contrasts'])
        self.assertIn('not the spread between trainings', result['scope'])
        with (self.out / (boot.specs()[0]['name'] + '.npy')).open('ab') as target:
            target.write(b'corrupt')
        with self.assertRaisesRegex(ValueError, 'Features changed'):
            boot.bootstrap(self.out, 3)

    def test_refuses_an_unfinished_lane(self):
        self.edit(self.roots['lo'] / 'gpu1/logs/summary.json', lambda s: s.update(status='running', completed_cells=5))
        self.refused('queue summary: gpu1')

    def test_refuses_another_checkpoint(self):
        self.edit(self.roots['lo'] / 'gpu2/logs/preflight.json', lambda p: p['checkpoints']['fox'].update(step=800001))
        self.refused('Unexpected checkpoint: fox')

    def test_refuses_the_1300k_checkpoint_in_the_800k_root(self):
        self.edit(self.roots['fox800'] / 'logs/preflight.json', lambda p: p['checkpoints']['fox'].update(step=1300002))
        self.refused('Unexpected checkpoint: fox800')

    def test_refuses_an_unfinished_second_root(self):
        self.edit(self.roots['fox800'] / 'logs/summary.json', lambda s: s.update(status='running', completed_cells=2))
        self.refused('queue summary: fox800')

    def test_refuses_lanes_of_one_root_from_different_sources(self):
        self.edit(self.roots['lo'] / 'gpu2/logs/preflight.json', lambda p: p.update(source_sha256='other'))
        self.refused('different sources')

    def test_a_manifest_must_match_the_source_of_its_own_root(self):
        spec = next(s for s in boot.specs() if s['model'] == 'fox800')
        self.edit(boot.directory(self.roots, spec) / 'manifest.json', lambda m: m.update(source_sha256='source'))
        self.refused('Unexpected manifest')

    def test_refuses_the_upper_window(self):
        spec = boot.specs()[5]
        self.edit(boot.directory(self.roots, spec) / 'manifest.json',
                  lambda m: m['config']['sampling'].update(logsnr_max=10.575749165617202))
        self.refused('Unexpected manifest')

    def test_refuses_a_missing_shard(self):
        spec = boot.specs()[2]
        m = boot.MODELS[spec['model']]
        (boot.directory(self.roots, spec) / 'ckpt_{}'.format(m['ckpt']) / 'statistics_1.npz').unlink()
        self.refused('Incorrect shards')

    def test_identities_match_the_evaluation_queue(self):
        repo = Path(__file__).resolve().parents[1]
        text = (repo / 'celeba_50k_lo_eval.py').read_text()
        for model in ('A', 'vp', 'fox'):
            value = boot.MODELS[model]
            self.assertIn(value['checkpoint_id'], text, model)
            self.assertIn('ckpt={}, step={}'.format(value['ckpt'], value['step']), text, model)
        text = (repo / 'celeba_fox800_lo_eval.py').read_text()
        self.assertIn(boot.MODELS['fox800']['checkpoint_id'], text)
        self.assertIn('CKPT = 160', text)
        self.assertIn('STEP = 800001', text)

    def test_core_checks(self):
        boot.check()


if __name__ == '__main__':
    main(verbosity=1)
