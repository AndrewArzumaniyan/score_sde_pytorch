"""CPU checks for the arm A continuation; a small model stands in for NCSN++.

A_TEST_MANIFEST may point to a copy of the real run_manifest.json of arm A.
"""
import importlib.util
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile
from types import SimpleNamespace as NS
import unittest
from unittest.mock import MagicMock, patch

import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
STUBBED = importlib.util.find_spec('tensorflow') is None  # Outside the Docker image.
if STUBBED:
    import torch._dynamo  # noqa: F401, its import scans module specs, which stubs lack.
    for module in ('tensorflow', 'tensorflow_gan', 'tensorflow_hub', 'tensorflow_datasets',
                   'tensorflow_probability', 'tensorboard', 'torch.utils.tensorboard',
                   'torchvision', 'torchvision.utils', 'torchvision.transforms'):
        sys.modules.setdefault(module, MagicMock())
    import glob
    sys.modules['tensorflow'].io.gfile = NS(
        exists=os.path.exists, listdir=os.listdir, glob=glob.glob, GFile=open, remove=os.remove,
        makedirs=lambda path: os.makedirs(path, exist_ok=True),
        rename=lambda source, target, overwrite=False: os.replace(source, target))

import artifact_utils
import celeba_a_continue as cont
from configs.factorial.celeba_a_vp_pvp import get_config as get_a
from models import utils as mutils
from models.ema import ExponentialMovingAverage
import losses
from utils import latest_immutable_checkpoint, restore_checkpoint, save_checkpoint

H200_TRAINING = 'h200-training-protocol'


def small_model(config):
    torch.manual_seed(0)
    return torch.nn.Linear(3, 2)


def manifest_payload():
    path = os.environ.get('A_TEST_MANIFEST')
    if path:
        return json.loads(Path(path).read_text())
    config = get_a()
    config.allow_tf32 = config.cudnn_benchmark = True
    config.training.snapshot_freq = 5000
    resolved = artifact_utils._canonicalize(config)
    resolved.pop('eval')
    resolved['device'] = 'cuda:0'
    for key in ('n_iters', 'log_freq', 'eval_freq', 'snapshot_freq_for_preemption', 'snapshot_sampling'):
        resolved['training'].pop(key)
    return dict(config=resolved, environment=dict(gpu='H200'), training_protocol_sha256=H200_TRAINING,
                model_manifest=artifact_utils.build_model_manifest(config))


def source_state(config, protocol, step=cont.SOURCE_STEP):
    """A trained parent: Adam counts every update, the data position is non-zero."""
    model = small_model(config)
    optimizer = losses.get_optimizer(config, model.parameters())
    model(torch.ones(1, 3)).sum().backward()
    optimizer.step()
    for group in optimizer.state.values():
        group['step'] = torch.tensor(float(step))
    return dict(model=model, optimizer=optimizer, step=step, data_batches_consumed=step - 1,
                ema=ExponentialMovingAverage(model.parameters(), config.model.ema_rate),
                eval_batches_consumed=7, model_protocol_sha256=protocol,
                training_protocol_sha256=H200_TRAINING)


class Continuation(unittest.TestCase):
    def setUp(self):
        self.root = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.root)
        (self.root / 'workdirs/_training_inputs').mkdir(parents=True)
        (self.root / 'workdirs' / cont.SOURCE_WORK / 'checkpoints').mkdir(parents=True)
        source = self.root / 'workdirs' / cont.SOURCE_WORK / 'checkpoints/checkpoint_40.pth'
        self.payload = manifest_payload()
        self.payload['training_protocol_sha256'] = H200_TRAINING
        manifest = self.root / 'workdirs/_training_inputs' / (cont.SOURCE_WORK + '.json')
        manifest.write_text(json.dumps(self.payload))
        for target, value in (('ROOT', self.root), ('SOURCE', source), ('SOURCE_MANIFEST', manifest)):
            patcher = patch.object(cont, target, value)
            patcher.start()
            self.addCleanup(patcher.stop)
        patches = [patch.object(mutils, 'create_model', small_model),
                   patch.object(artifact_utils, '_environment_manifest', lambda: dict(gpu='RTX')),
                   patch.object(artifact_utils, '_source_fingerprint', lambda root: 'rtx-source'),
                   patch('reproducibility.configure_reproducibility', lambda *a, **k: 42),
                   patch('reproducibility.seed_everything', lambda *a, **k: 42)]
        for item in patches:
            item.start()
            self.addCleanup(item.stop)
        self.config = cont.make_config('cpu')
        self.protocol = self.payload['model_manifest']['model_protocol_sha256']
        self.write_source()

    def write_source(self, **changes):
        state = source_state(self.config, self.protocol)
        state.update(changes)
        with patch('utils.uuid.uuid4', return_value=NS(hex=cont.SOURCE_ID)):
            save_checkpoint(str(cont.SOURCE), state)

    def imported_path(self):
        return self.root / cont.WORK / 'checkpoints/checkpoint_40.pth'

    def run_import(self):
        with patch.object(cont, 'get_config', lambda: cont.make_config('cpu')):
            cont.import_parent()

    def test_config_keeps_training_protocol_and_extends_budget(self):
        config = cont.get_config()
        self.assertEqual(config.training.n_iters, 400000)
        self.assertEqual(config.training.snapshot_freq, 5000)
        self.assertFalse(config.training.snapshot_sampling)
        self.assertTrue(config.allow_tf32 and config.cudnn_benchmark)
        for section in ('model', 'data', 'optim', 'sampling'):
            self.assertEqual(artifact_utils._canonicalize(config[section]), self.payload['config'][section])
        self.assertEqual(artifact_utils.build_model_manifest(config)['model_protocol_sha256'], self.protocol)

    def test_manifest_is_the_same_for_import_and_training(self):
        first = artifact_utils.build_training_manifest(cont.get_config())
        second = artifact_utils.build_training_manifest(cont.get_config())
        self.assertEqual(first, second)
        self.assertNotEqual(first['training_protocol_sha256'], H200_TRAINING)

    def test_import_replaces_fingerprint_and_resets_data_only(self):
        parent = torch.load(str(cont.SOURCE), weights_only=False)
        self.run_import()
        saved = torch.load(str(self.imported_path()), weights_only=False)
        manifest = json.loads((self.root / cont.WORK / 'run_manifest.json').read_text())
        self.assertEqual(saved['training_protocol_sha256'], manifest['training_protocol_sha256'])
        self.assertEqual((saved['data_batches_consumed'], saved['eval_batches_consumed']), (0, 0))
        self.assertEqual(saved['step'], cont.SOURCE_STEP)
        for key, value in parent['model'].items():
            self.assertTrue(torch.equal(saved['model'][key], value))
        self.assertEqual(saved['optimizer']['state'][0]['step'], parent['optimizer']['state'][0]['step'])
        self.assertTrue(torch.equal(saved['optimizer']['state'][0]['exp_avg'], parent['optimizer']['state'][0]['exp_avg']))
        self.assertEqual(saved['ema']['decay'], parent['ema']['decay'])
        for left, right in zip(saved['ema']['shadow_params'], parent['ema']['shadow_params']):
            self.assertTrue(torch.equal(left, right))

    def test_training_resume_accepts_import_and_lands_on_parent_step(self):
        self.run_import()
        manifest = json.loads((self.root / cont.WORK / 'run_manifest.json').read_text())
        path, step = latest_immutable_checkpoint(str(self.imported_path().parent), self.config.training.snapshot_freq)
        self.assertEqual((Path(path), step), (self.imported_path(), cont.SOURCE_STEP))
        state = cont.new_state(self.config, self.protocol)
        state['training_protocol_sha256'] = manifest['training_protocol_sha256']
        restore_checkpoint(path, state, 'cpu', defer_rng=True)
        self.assertEqual(int(state['step']), cont.SOURCE_STEP)
        self.assertEqual(state['data_batches_consumed'], 0)
        self.assertIsNotNone(state['_rng_state_to_restore'])

    def test_rtx_fingerprint_refuses_the_parent_directly(self):
        state = cont.new_state(self.config, self.protocol)
        state['training_protocol_sha256'] = 'rtx-training-protocol'
        with self.assertRaisesRegex(ValueError, 'training code/config/seed fingerprint'):
            restore_checkpoint(str(cont.SOURCE), state, 'cpu', restore_rng=False)

    def test_rerun_keeps_the_imported_checkpoint(self):
        self.run_import()
        before = cont.sha256(self.imported_path())
        self.run_import()
        self.assertEqual(cont.sha256(self.imported_path()), before)

    def test_interrupted_import_is_finished(self):
        self.run_import()
        self.imported_path().unlink()
        self.run_import()
        self.assertTrue(self.imported_path().is_file())

    def test_other_parent_cannot_reuse_the_workdir(self):
        self.run_import()
        self.write_source(data_batches_consumed=1)
        with self.assertRaisesRegex(ValueError, 'different parent'):
            self.run_import()

    def assert_refused(self, pattern, **changes):
        self.write_source(**changes)
        with self.assertRaisesRegex(ValueError, pattern):
            cont.load_source(self.config)

    def test_refuses_foreign_training_fingerprint(self):
        self.assert_refused('does not belong', training_protocol_sha256='other')

    def test_refuses_wrong_step(self):
        self.assert_refused('Expected state.step', step=cont.SOURCE_STEP + 5000)

    def test_refuses_other_checkpoint_id(self):
        state = source_state(self.config, self.protocol)
        save_checkpoint(str(cont.SOURCE), state)
        with self.assertRaisesRegex(ValueError, 'stage 0 evaluated'):
            cont.load_source(self.config)

    def test_refuses_adam_step_mismatch(self):
        state = source_state(self.config, self.protocol, step=cont.SOURCE_STEP - 1)
        state['step'] = cont.SOURCE_STEP
        with patch('utils.uuid.uuid4', return_value=NS(hex=cont.SOURCE_ID)):
            save_checkpoint(str(cont.SOURCE), state)
        with self.assertRaisesRegex(ValueError, 'Adam step'):
            cont.load_source(self.config)

    def test_refuses_non_finite_weights(self):
        state = source_state(self.config, self.protocol)
        with torch.no_grad():
            state['model'].weight[0, 0] = float('nan')
        with patch('utils.uuid.uuid4', return_value=NS(hex=cont.SOURCE_ID)):
            save_checkpoint(str(cont.SOURCE), state)
        with self.assertRaisesRegex(ValueError, 'Non-finite'):
            cont.load_source(self.config)

    def test_verify_requires_the_final_step(self):
        self.run_import()
        manifest = json.loads((self.root / cont.WORK / 'run_manifest.json').read_text())
        final = self.imported_path().with_name('checkpoint_80.pth')
        torch.save(dict(step=cont.TARGET_ITERS, training_protocol_sha256=manifest['training_protocol_sha256']), final)
        with self.assertRaisesRegex(ValueError, 'final checkpoint'):
            cont.verify()
        torch.save(dict(step=cont.TARGET_ITERS + 1, training_protocol_sha256=manifest['training_protocol_sha256']), final)
        cont.verify()


if __name__ == '__main__':
    unittest.main()
