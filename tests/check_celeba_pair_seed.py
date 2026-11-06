"""CPU checks for the seed 43 pair; no GPU, data or training."""
import copy
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

import celeba_pair_seed as pair
from celeba_stage0_eval import native_config

ORCHESTRATION = {'seed', 'allow_tf32', 'cudnn_benchmark', 'training.n_iters', 'training.log_freq',
                 'training.eval_freq', 'training.snapshot_freq', 'training.snapshot_freq_for_preemption',
                 'training.snapshot_sampling'}


def manifest(arm):
    config = pair.get_config(arm).to_dict()
    config.pop('eval')
    config.pop('device')
    return dict(config=config, source_sha256='source', training_protocol_sha256='protocol-' + arm,
                environment=dict(torch='2.4.1', visible_cuda_devices=['NVIDIA H200 NVL']))


class Config(unittest.TestCase):
    def test_arms_are_the_configs_stage_0_evaluates(self):
        for arm in pair.ARMS:
            changed = pair.differing_keys(native_config(arm).to_dict(), pair.get_config(arm).to_dict())
            self.assertLessEqual(changed, ORCHESTRATION)

    def test_fox_is_native_not_a0(self):
        model = pair.get_config('fox').model
        self.assertEqual((model.fox_u, model.fox_drift_schedule, model.fox_kernel), (-5.0, 'constant', 'matern_3_2'))
        self.assertEqual(model.fox_matern_length_scale, 346.4101615138)
        self.assertEqual(model.fox_target_terminal_variance, 1.0)
        self.assertEqual(model.fox_schedule_grid_size, 4096)

    def test_run_settings(self):
        for arm in pair.ARMS:
            config = pair.get_config(arm)
            self.assertEqual(config.seed, 43)
            self.assertTrue(config.allow_tf32 and config.cudnn_benchmark)
            self.assertEqual((config.training.batch_size, config.training.n_iters), (128, 800000))
            self.assertFalse(config.training.snapshot_sampling)
        self.assertEqual(pair.FINAL_CKPT, 160)

    def test_config_flag_passes_the_arm(self):
        from absl import flags
        from ml_collections.config_flags import config_flags
        config_flags.DEFINE_config_file('config', None, 'config')  # On global FLAGS, as main.py does.
        argv = ['test', '--config={}:fox'.format(REPO / 'celeba_pair_seed.py'), '--config.training.n_iters=7']
        with patch.object(sys, 'argv', argv):  # config_flags reads the overrides from sys.argv.
            flags.FLAGS(argv)
        config = flags.FLAGS.config
        self.assertEqual((config.training.sde, config.training.n_iters, config.seed), ('foxvpsde', 7, 43))


class Pair(unittest.TestCase):
    def test_accepts_arms_that_differ_only_in_the_sde(self):
        pair.check_pair(manifest('fox'), manifest('vp'))

    def test_refuses_an_unfair_pair(self):
        def changed(edit):
            value = manifest('vp')
            edit(value)
            with self.assertRaises(ValueError):
                pair.check_pair(manifest('fox'), value)
        changed(lambda m: m.update(source_sha256='other'))
        changed(lambda m: m['environment'].update(torch='2.8'))
        changed(lambda m: m['environment'].update(visible_cuda_devices=['RTX PRO 6000']))
        changed(lambda m: m['config'].update(seed=42))
        changed(lambda m: m['config'].update(allow_tf32=False))
        changed(lambda m: m['config']['optim'].update(lr=1e-4))
        changed(lambda m: m['config']['training'].update(batch_size=64))


class Run(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name)
        patcher = patch.object(pair, 'ROOT', self.root)
        patcher.start()
        self.addCleanup(patcher.stop)
        for arm in pair.ARMS:
            (self.root / pair.WORK[arm] / 'checkpoints').mkdir(parents=True)
            (self.root / pair.WORK[arm] / 'run_manifest.json').write_text(json.dumps(manifest(arm)))

    def finish(self, arm, step=pair.TARGET_ITERS + 1, protocol=None):
        torch.save(dict(step=step, training_protocol_sha256=protocol or 'protocol-' + arm), str(pair.final_path(arm)))

    def test_train_runs_a_missing_arm_then_verifies(self):
        with patch.object(pair, 'run_main', side_effect=lambda arm, work: self.finish(arm)) as run:
            pair.train('fox')
        run.assert_called_once_with('fox', pair.WORK['fox'])

    def test_train_skips_a_finished_arm(self):
        self.finish('fox')
        with patch.object(pair, 'run_main') as run:
            pair.train('fox')
        run.assert_not_called()

    def test_verify_refuses_foreign_checkpoints(self):
        self.finish('fox', step=pair.TARGET_ITERS)
        with self.assertRaises(ValueError):
            pair.verify('fox')
        self.finish('fox', protocol='other')
        with self.assertRaises(ValueError):
            pair.verify('fox')

    def test_verify_of_vp_checks_the_pair(self):
        self.finish('vp')
        pair.verify('vp')
        other = manifest('fox')
        other['source_sha256'] = 'other'
        (self.root / pair.WORK['fox'] / 'run_manifest.json').write_text(json.dumps(other))
        with self.assertRaises(ValueError):
            pair.verify('vp')

    def test_speed_uses_the_first_run_of_a_resumed_log(self):
        log = self.root / 'stdout.txt'
        log.write_text('\n'.join((
            'INFO - run_lib.py - 2026-11-06 10:00:00,000 - step: 100, training_loss: 1e-02',
            'INFO - run_lib.py - 2026-11-06 10:00:00,500 - step: 100, eval_loss: 1e-02',
            'INFO - run_lib.py - 2026-11-06 10:01:06,000 - step: 300, training_loss: 1e-02',
            'INFO - run_lib.py - 2026-11-06 10:09:00,000 - step: 300, training_loss: 1e-02')))
        self.assertAlmostEqual(pair.seconds_per_step(log), 0.33)

    def test_smoke_resumes_each_arm_and_removes_its_directory(self):
        calls = []

        def fake(arm, work, override):
            calls.append((arm, override))
            path = self.root / work
            (path / 'checkpoints').mkdir(parents=True, exist_ok=True)
            (path / 'run_manifest.json').write_text(json.dumps(manifest(arm)))
            (path / 'stdout.txt').write_text(
                'INFO - run_lib.py - 2026-11-06 10:00:00,000 - step: 100, training_loss: 1\n'
                'INFO - run_lib.py - 2026-11-06 10:01:06,000 - step: 300, training_loss: 1\n')
            (path / 'checkpoints/checkpoint_final_step_351.pth').write_text('x')

        with patch.object(pair, 'run_main', side_effect=fake):
            pair.smoke()
        self.assertEqual([arm for arm, _ in calls], ['fox', 'fox', 'vp', 'vp'])
        self.assertEqual(calls[1][1], '--config.training.n_iters=350')
        self.assertFalse((self.root / pair.SMOKE).exists())


if __name__ == '__main__':
    unittest.main()
