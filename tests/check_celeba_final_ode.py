"""CPU-only protocol and native FOX numerical checks."""
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import celeba_final_ode_eval as final
from check_celeba_ode import NumericalChecks
from check_low_nfe_overnight import QueueChecks


class FinalProtocol(unittest.TestCase):
    def test_matrix_and_configs(self):
        self.assertEqual(len(final.CELLS), 8)
        self.assertEqual(len(set(final.CELLS)), 8)
        for n, count, arm, method in final.CELLS:
            cfg = final.get_config('{}_{}_{}'.format(method, count, arm))
            self.assertEqual((n, cfg.eval.num_samples, cfg.eval.batch_size), (256, count, 512))
            self.assertEqual(cfg.eval.sampling_seed, 0)
            self.assertEqual(cfg.eval.begin_ckpt, 260)
            self.assertEqual(cfg.sampling.logsnr_min, -10.)
            self.assertEqual(cfg.sampling.logsnr_max, 10.575749165617202)
            self.assertEqual(cfg.training.sde.lower(), 'vpsde' if arm == 'vp' else 'foxvpsde')
            self.assertEqual(getattr(cfg.model, 'noise_conditioning', 'time'), 'time')
            self.assertEqual(tuple(cfg.eval.endpoint_denoise_outputs), final.VARIANTS)

    @patch.dict(sys.modules, tensorflow=SimpleNamespace())
    def test_report_checks_identity_and_matrix(self):
        import numpy as np
        from artifact_utils import _canonicalize
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            output = root / 'output'
            (output / 'logs').mkdir(parents=True)
            (output / 'logs/preflight.json').write_text(json.dumps({'source_sha256': 'source', 'checkpoint_ids': {'vp': 'v', 'fox': 'f'}}))
            with patch.object(final, 'ROOT', root), patch.object(final, 'OUTPUT', output):
                for n, count, arm, method in final.CELLS:
                    directory = root / 'workdirs' / final.ARMS[arm] / final.folder(count, arm, method)
                    (directory / 'ckpt_260').mkdir(parents=True)
                    cfg = final.get_config('{}_{}_{}'.format(method, count, arm))
                    config = {k: _canonicalize(cfg[k]) for k in ('model', 'data', 'sampling', 'eval')}
                    config['eval'].pop('begin_ckpt', None)
                    config['eval'].pop('end_ckpt', None)
                    (directory / 'manifest.json').write_text(json.dumps({'config': config, 'source_sha256': 'source'}))
                    values = dict(num_samples=count, nfe=257, checkpoint_step=1300002,
                        checkpoint_id='v' if arm == 'vp' else 'f', dataset_stats_id=final.REFERENCE_SHA)
                    values.update({m + '_' + v: 1.0 for m in ('IS', 'fid', 'kid') for v in final.VARIANTS})
                    np.savez(directory / 'report_260.npz', **values)
                    for i in range((count + 511) // 512):
                        (directory / 'ckpt_260' / ('statistics_%d.npz' % i)).touch()
                    final.report(n, count, arm, method)
                self.assertEqual(json.loads((output / 'summary.json').read_text())['status'], 'completed')
                values['checkpoint_step'] = 123
                np.savez(directory / 'report_260.npz', **values)
                with self.assertRaises(RuntimeError):
                    final.report(n, count, arm, method)


class FinalQueue(QueueChecks):
    def test_final_queue_completes_and_stops_on_error(self):
        result, calls, output = self.run_queue(profile='final_ode', cells=final.CELLS)
        self.assertEqual(result.returncode, 0, result.stdout)
        self.assertEqual(sum(x.startswith('meta report ') for x in calls), 8)
        self.assertEqual((output / 'logs/queue.state').read_text(), 'completed\n')

    def test_final_queue_stops_on_failure(self):
        result, calls, output = self.run_queue(fail='256:10000:fox', profile='final_ode', cells=final.CELLS)
        self.assertNotEqual(result.returncode, 0)
        self.assertEqual(sum(x.startswith('eval ') for x in calls), 2)
        self.assertEqual((output / 'logs/queue.state').read_text(), 'failed\n')


if __name__ == '__main__':
    unittest.main(verbosity=2)
