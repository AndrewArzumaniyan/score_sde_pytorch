"""Small end-to-end fixtures for the matched35 CPU bootstrap; run in Docker."""
import json
import tempfile
from pathlib import Path
from unittest import TestCase, main

import numpy as np
from scipy import linalg
import celeba_matched35_bootstrap as boot


class Synthetic(TestCase):
    def test_prepare_resume_and_reject_corrupt_feature(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            data, out = root / 'data', root / 'out'
            out.mkdir()
            (data / 'logs').mkdir(parents=True)
            (data / 'logs/queue.state').write_text('completed')
            boot.N, boot.BATCH, boot.DIM = 20, 8, 4
            boot.base.ROUNDS = 3
            reference = root / 'reference.npz'
            rng = np.random.default_rng(7)
            real = rng.normal(size=(30, 4)).astype(np.float32)
            np.savez(reference, pool_3=real)
            boot.base.REFERENCE_SHA = boot.base.digest(reference)
            mean, cov = boot.base.moments(real)
            vals, vecs = linalg.eigh(cov)
            ref = (mean, cov, (vecs * np.sqrt(vals)) @ vecs.T)
            pre = dict(status='passed', checkpoint_ids=boot.base.CHECKPOINT_IDS, source_sha256='source')
            (data / 'logs/preflight.json').write_text(json.dumps(pre))
            cells = []
            for s in boot.specs():
                d = boot.directory(data, s)
                (d / 'ckpt_260').mkdir(parents=True)
                cfg = dict(eval=dict(sampling_seed=s['seed'], num_samples=20, batch_size=8,
                                     sampling_num_scales=34), sampling=dict(method=s['method']))
                (d / 'manifest.json').write_text(json.dumps(dict(config=cfg, source_sha256='source')))
                identity = dict(checkpoint_id=boot.base.CHECKPOINT_IDS[s['arm']], checkpoint_step=1300002,
                                protocol_sha256='protocol-' + s['name'], training_protocol_sha256='training')
                features = rng.normal(size=(24, 4)).astype(np.float32)
                fid = boot.base.fid(features[:20], ref)
                np.savez(d / 'report_260.npz', **identity, dataset_stats_id=boot.base.REFERENCE_SHA,
                         num_samples=20, nfe=35, fid_tweedie_xt=fid)
                for i in range(3):
                    np.savez(d / 'ckpt_260' / f'statistics_{i}.npz', **identity, round_id=i,
                             sampling_seed=boot.base.round_seed(s['seed'], i), num_samples=8,
                             nfe=35, pool_3_tweedie_xt=features[8*i:8*(i+1)])
                cells.append(dict(method=s['method'], arm=s['arm'], sampling_seed=s['seed'], fid_tweedie_xt=fid))
            (data / 'summary.json').write_text(json.dumps(dict(status='completed', completed_cells=8,
                                                               total_cells=8, cells=cells)))
            boot.prepare(data, reference, out)
            boot.bootstrap(out, 1)
            boot.bootstrap(out, 2)
            rows = (out / 'draws.jsonl').read_text().splitlines()
            self.assertEqual(len(rows), 2)
            result = json.loads((out / 'summary.json').read_text())
            self.assertEqual(len(result['contrasts']), 10)
            f = out / (boot.specs()[0]['name'] + '.npy')
            with f.open('ab') as target:
                target.write(b'corrupt')
            with self.assertRaisesRegex(ValueError, 'Features changed'):
                boot.bootstrap(out, 3)


if __name__ == '__main__':
    main(verbosity=2)
