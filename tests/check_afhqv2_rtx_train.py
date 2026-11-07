"""CPU checks for AFHQv2 on RTX; no GPU, data or training."""
import importlib.util
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import MagicMock, patch

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
if importlib.util.find_spec('tensorflow') is None:  # Outside the Docker image.
    import torch._dynamo  # noqa: F401, its import scans module specs, which stubs lack.
    for module in ('tensorflow', 'tensorflow_gan', 'tensorflow_hub', 'tensorflow_datasets',
                   'tensorflow_probability', 'tensorboard', 'torch.utils.tensorboard',
                   'torchvision', 'torchvision.utils', 'torchvision.transforms'):
        sys.modules.setdefault(module, MagicMock())

import afhqv2_rtx_train as afhq
import artifact_utils
import celeba_pair_seed as pair
import datasets
import evaluation
import run_lib
import sampling
from utils import checkpoint_filename

ORCHESTRATION = {'seed', 'allow_tf32', 'cudnn_benchmark', 'training.n_iters', 'training.log_freq',
                 'training.eval_freq', 'training.snapshot_freq', 'training.snapshot_freq_for_preemption',
                 'training.snapshot_sampling', 'training.batch_size'}
DATA = {'data.dataset', 'data.celeba_dir', 'data.celeba_train_take', 'data.celeba_validation_take',
        'data.afhqv2_dir', 'data.afhqv2_train_take', 'data.afhqv2_eval_take'}
IDENTITY = 'cd017afab009' + '0' * 52


class ZeroEps(torch.nn.Module):
    def forward(self, x, labels):
        return torch.zeros_like(x)


def on_cpu(config):
    config.device = torch.device('cpu') if isinstance(config.device, torch.device) else 'cpu'
    return config


class Config(unittest.TestCase):
    def test_each_arm_is_its_celeba_config_with_other_data(self):
        for arm in afhq.ARMS:
            changed = pair.differing_keys(afhq.base_config(arm).to_dict(), afhq.get_config(arm).to_dict())
            self.assertLessEqual(changed, ORCHESTRATION | DATA, arm)

    def test_data_is_centered_like_celeba(self):
        for arm in afhq.ARMS:
            config = afhq.get_config(arm)
            self.assertEqual((config.data.dataset, config.data.centered), ('AFHQV2', True))
            self.assertEqual(afhq.base_config(arm).data.centered, True)
            self.assertEqual((config.data.image_size, config.data.random_flip), (64, True))
            self.assertEqual(config.data.afhqv2_dir, 'datasets/afhqv2-64x64')

    def test_fox_is_the_native_recipe(self):
        config = afhq.get_config('fox')
        model = config.model
        self.assertEqual(config.training.sde, 'foxvpsde')
        self.assertEqual((model.fox_u, model.fox_drift_schedule, model.fox_kernel), (-5.0, 'constant', 'matern_3_2'))
        self.assertEqual(model.fox_matern_length_scale, 346.4101615138)
        self.assertEqual(model.fox_target_terminal_variance, 1.0)
        self.assertEqual(model.fox_schedule_grid_size, 4096)

    def test_a_and_vp(self):
        a, vp = afhq.get_config('a'), afhq.get_config('vp')
        self.assertEqual((a.training.factorial_arm, a.training.noise_distribution, a.model.noise_conditioning),
                         ('A_vp_pvp', 'vp_time', 'logsnr'))
        self.assertEqual(a.model.logsnr_max, 9.115429865459795)
        self.assertEqual(vp.training.sde, 'vpsde')

    def test_network_and_run_settings(self):
        for arm in afhq.ARMS:
            config = afhq.get_config(arm)
            self.assertEqual((config.model.name, config.model.nf, config.model.num_res_blocks), ('ncsnpp', 128, 4))
            self.assertEqual((config.seed, config.training.batch_size), (42, 128))
            self.assertEqual((config.training.snapshot_freq, config.training.snapshot_freq_for_preemption),
                             (50000, 5000))
            self.assertTrue(config.allow_tf32 and config.cudnn_benchmark)
            self.assertFalse(config.training.snapshot_sampling)
        self.assertEqual(afhq.ITERS, dict(fox=800000, a=200000, vp=200000))

    def test_the_target_is_outside_the_training_protocol(self):
        with patch.object(artifact_utils, '_environment_manifest', lambda: {}), \
                patch.object(artifact_utils, '_source_fingerprint', lambda root: 'source'):
            full = artifact_utils.build_training_manifest(afhq.get_config('fox'))
            config = afhq.get_config('fox')
            config.training.n_iters = 600000
            short = artifact_utils.build_training_manifest(config)
        self.assertEqual(full['training_protocol_sha256'], short['training_protocol_sha256'])

    def test_arms_differ_in_protocol(self):
        with patch.object(artifact_utils, '_environment_manifest', lambda: {}), \
                patch.object(artifact_utils, '_source_fingerprint', lambda root: 'source'):
            hashes = {artifact_utils.build_training_manifest(afhq.get_config(arm))['training_protocol_sha256']
                      for arm in afhq.ARMS}
        self.assertEqual(len(hashes), 3)


class Pixels(unittest.TestCase):
    def setUp(self):
        self.config = afhq.get_config('fox')

    def test_scaler_pair(self):
        scaler = datasets.get_data_scaler(self.config)
        inverse = datasets.get_data_inverse_scaler(self.config)
        self.assertEqual((scaler(0.), scaler(1.)), (-1., 1.))
        self.assertEqual((inverse(-1.), inverse(1.)), (0., 1.))

    def test_check_batch_passes_on_8_bit_images_and_refuses_others(self):
        images = np.random.default_rng(0).integers(0, 256, (4, 64, 64, 3)).astype(np.float32) / 255.
        afhq.check_batch(images, self.config)
        with self.assertRaisesRegex(ValueError, r'\[0, 1\]'):
            afhq.check_batch(images * 255., self.config)
        with self.assertRaisesRegex(ValueError, r'\[-1, 1\]'):
            uncentered = afhq.get_config('fox')
            uncentered.data.centered = False
            afhq.check_batch(images, uncentered)

    def test_every_sampler_output_passes_the_inverse_scaler_before_inception(self):
        """run_lib multiplies by 255 what the sampler returns, so it must be in [0, 1] space."""
        for arm in afhq.ARMS:
            config = on_cpu(afhq.get_config(arm))
            config.sampling.method = 'ode_dpmpp_2m'
            config.sampling.probability_flow = True
            config.sampling.time_grid = 'uniform_logsnr'
            config.sampling.logsnr_min, config.sampling.logsnr_max = -10.0, 9.115429865459795
            config.sampling.predictor = config.sampling.corrector = 'none'
            config.eval.endpoint_denoise_variants = True
            config.eval.endpoint_denoise_outputs = ('x_t', 'tweedie_xt')
            sde, eps = run_lib.get_sde(config, n_override=4)
            seen = []
            real = datasets.get_data_inverse_scaler(config)

            def spy(x):
                seen.append(x)
                return real(x)

            outputs, _ = sampling.get_sampling_fn(config, sde, (2, 3, 8, 8), spy, eps)(ZeroEps())
            self.assertEqual(len(seen), 2, arm)
            for raw, value in zip(seen, (outputs['x_t'], outputs['tweedie_xt'])):
                self.assertTrue(torch.equal(value, (raw + 1.) / 2.), arm)

    def test_evaluation_scales_what_the_sampler_returns(self):
        text = (REPO / 'run_lib.py').read_text()
        self.assertIn('samples = np.clip(samples.permute(0, 2, 3, 1).cpu().numpy() * 255., 0, 255).astype(np.uint8)', text)
        self.assertIn('sampling_fn = sampling.get_sampling_fn(config, sampling_sde, sampling_shape, inverse_scaler, sampling_eps)', text)
        self.assertIn('inverse_scaler = datasets.get_data_inverse_scaler(config)', text)


class Statistics(unittest.TestCase):
    def setUp(self):
        self.root = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.root)
        (self.root / 'assets/stats').mkdir(parents=True)
        for target, item in ((afhq, 'ROOT'),):
            patcher = patch.object(target, item, self.root)
            patcher.start()
            self.addCleanup(patcher.stop)
        patcher = patch.object(datasets, 'afhqv2_dataset_identity', lambda config=None: IDENTITY)
        patcher.start()
        self.addCleanup(patcher.stop)
        self.config = afhq.get_config('fox')

    def write(self, identity=IDENTITY, count=afhq.IMAGES):
        np.savez(str(self.root / afhq.STATS), pool_3=np.zeros((count, 2048), np.float16),
                 num_examples=np.asarray(count), dataset_identity=np.asarray(identity), resolution=np.asarray(64))

    def test_file_name_does_not_depend_on_centering(self):
        names = set()
        for centered in (True, False):
            self.config.data.centered = centered
            names.add(evaluation._get_stats_filename(self.config))
        self.assertEqual(names, {afhq.STATS})

    def test_reference_is_computed_from_raw_pixels_without_the_config_scaler(self):
        text = (REPO / 'evaluation.py').read_text()
        start = text.index('def _load_afhqv2_image')
        body = text[start:text.index('def load_dataset_stats')]
        self.assertNotIn('centered', body)
        self.assertNotIn('get_data_scaler', body)
        self.assertIn('return image * 255.', body)

    def test_check_passes_on_the_right_file(self):
        self.write()
        afhq.check_stats(self.config)

    def test_refuses_another_path(self):
        self.write()
        with patch.object(datasets, 'afhqv2_dataset_identity', lambda config=None: 'ffffffffffff' + '0' * 52):
            with self.assertRaisesRegex(ValueError, 'instead of'):
                afhq.check_stats(self.config)

    def test_refuses_foreign_or_short_statistics(self):
        self.write(identity='cd017afab009' + '1' * 52)
        with self.assertRaisesRegex(ValueError, 'another dataset'):
            afhq.check_stats(self.config)
        self.write(count=100)
        with self.assertRaisesRegex(ValueError, 'unexpected size'):
            afhq.check_stats(self.config)


class Run(unittest.TestCase):
    def setUp(self):
        self.root = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.root)
        patcher = patch.object(afhq, 'ROOT', self.root)
        patcher.start()
        self.addCleanup(patcher.stop)

    def write_run(self, arm, step, seed=42, centered=True, ckpt=16):
        work = self.root / afhq.WORK[arm]
        (work / 'checkpoints').mkdir(parents=True, exist_ok=True)
        manifest = dict(config=dict(seed=seed, data=dict(dataset='AFHQV2', centered=centered)),
                        training_protocol_sha256='protocol')
        (work / 'run_manifest.json').write_text(json.dumps(manifest))
        torch.save(dict(step=step, training_protocol_sha256='protocol'), work / 'checkpoints/checkpoint_{}.pth'.format(ckpt))

    def test_final_checkpoint_numbers(self):
        self.assertEqual(afhq.final_path('fox', 800000).name, 'checkpoint_16.pth')
        self.assertEqual(afhq.final_path('fox', 600000).name, 'checkpoint_12.pth')
        self.assertEqual(afhq.final_path('a', 200000).name, 'checkpoint_4.pth')

    def test_verify(self):
        self.write_run('fox', 800001)
        afhq.verify('fox', 800000)
        self.write_run('fox', 800000)
        with self.assertRaisesRegex(ValueError, 'final checkpoint'):
            afhq.verify('fox', 800000)

    def test_verify_refuses_uncentered_or_other_seed(self):
        for changes in (dict(centered=False), dict(seed=43)):
            self.write_run('fox', 800001, **changes)
            with self.assertRaisesRegex(ValueError, 'run manifest'):
                afhq.verify('fox', 800000)

    def test_train_passes_the_target_and_skips_a_finished_run(self):
        with patch.object(afhq.subprocess, 'check_call', side_effect=lambda *a, **k: self.write_run('fox', 600001, ckpt=12)) as call:
            afhq.train('fox', 600000)
            afhq.train('fox', 600000)
        self.assertEqual(call.call_count, 1)
        self.assertIn('--config.training.n_iters=600000', call.call_args[0][0])
        self.assertIn('--config=afhqv2_rtx_train.py:fox', call.call_args[0][0])

    def test_a_run_to_800k_stopped_past_600k_counts_as_trained_to_600k(self):
        """The periodic snapshot of the long run has the name and step of a final one at 600k."""
        self.assertEqual(checkpoint_filename(600000, 600001, 50000), afhq.final_path('fox', 600000).name)
        self.write_run('fox', 600001, ckpt=12)
        with patch.object(afhq.subprocess, 'check_call') as call:
            afhq.train('fox', 600000)
        call.assert_not_called()

    def test_smoke_reads_the_files_in_a_process_without_the_gpu(self):
        class Stop(Exception):
            pass

        with patch.object(afhq.subprocess, 'check_call', side_effect=Stop) as call:
            with self.assertRaises(Stop):
                afhq.smoke()
        self.assertEqual(call.call_args[0][0][-1], 'files')
        self.assertEqual(call.call_args[1]['env']['CUDA_VISIBLE_DEVICES'], '')

    def test_refuses_a_target_off_the_snapshots(self):
        with self.assertRaisesRegex(ValueError, 'multiple'):
            afhq.train('fox', 620000)

    def test_slow_speed_names_the_fallback(self):
        with patch('builtins.print') as output:
            afhq.report_speed(dict(fox=0.7, a=0.55, vp=0.55))
        self.assertIn('queue fox GPU_ID 600000', ' '.join(str(c) for c in output.call_args_list))
        with patch('builtins.print') as output:
            afhq.report_speed(dict(fox=0.55, a=0.55, vp=0.55))
        self.assertNotIn('600000', ' '.join(str(c) for c in output.call_args_list))


if __name__ == '__main__':
    unittest.main()
