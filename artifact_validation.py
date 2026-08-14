"""Regression tests for immutable evaluation protocol manifests."""

import os
import tempfile
import unittest

from artifact_utils import (build_training_manifest, ensure_eval_manifest,
                            ensure_training_manifest)
from configs.edm.cifar10_canonical import get_config


class ArtifactValidationTest(unittest.TestCase):

  def test_manifest_allows_exact_resume_and_rejects_changes(self):
    manifest = {
      'schema_version': 1,
      'source_sha256': 'source-a',
      'config': {'seed': 42},
      'protocol_sha256': 'protocol-a',
    }
    with tempfile.TemporaryDirectory() as directory:
      self.assertEqual(ensure_eval_manifest(directory, manifest), 'protocol-a')
      self.assertEqual(ensure_eval_manifest(directory, manifest), 'protocol-a')
      changed = dict(manifest)
      changed['protocol_sha256'] = 'protocol-b'
      with self.assertRaises(ValueError):
        ensure_eval_manifest(directory, changed)

  def test_legacy_artifacts_require_a_new_folder(self):
    manifest = {
      'schema_version': 1,
      'source_sha256': 'source-a',
      'config': {'seed': 42},
      'protocol_sha256': 'protocol-a',
    }
    with tempfile.TemporaryDirectory() as directory:
      with open(os.path.join(directory, 'statistics_0.npz'), 'wb') as output:
        output.write(b'legacy')
      with self.assertRaises(ValueError):
        ensure_eval_manifest(directory, manifest)

  def test_training_manifest_rejects_seed_or_recipe_change(self):
    manifest = {
      'schema_version': 1,
      'source_sha256': 'source-a',
      'config': {'seed': 42},
      'model_manifest': {'model_protocol_sha256': 'model-a'},
      'training_protocol_sha256': 'training-a',
    }
    with tempfile.TemporaryDirectory() as directory:
      self.assertEqual(
        ensure_training_manifest(directory, manifest), 'training-a')
      self.assertEqual(
        ensure_training_manifest(directory, manifest), 'training-a')
      changed = dict(manifest)
      changed['training_protocol_sha256'] = 'training-b'
      with self.assertRaises(ValueError):
        ensure_training_manifest(directory, changed)

  def test_legacy_training_checkpoint_requires_explicit_migration(self):
    manifest = {
      'schema_version': 1,
      'source_sha256': 'source-a',
      'config': {'seed': 42},
      'model_manifest': {'model_protocol_sha256': 'model-a'},
      'training_protocol_sha256': 'training-a',
    }
    with tempfile.TemporaryDirectory() as directory:
      checkpoint_dir = os.path.join(directory, 'checkpoints')
      os.makedirs(checkpoint_dir)
      with open(os.path.join(checkpoint_dir, 'checkpoint_1.pth'), 'wb') as output:
        output.write(b'legacy')
      with self.assertRaises(ValueError):
        ensure_training_manifest(directory, manifest)

  def test_legacy_training_samples_require_explicit_migration(self):
    manifest = {
      'schema_version': 1,
      'source_sha256': 'source-a',
      'config': {'seed': 42},
      'model_manifest': {'model_protocol_sha256': 'model-a'},
      'training_protocol_sha256': 'training-a',
    }
    with tempfile.TemporaryDirectory() as directory:
      sample_dir = os.path.join(directory, 'samples', 'iter_10')
      os.makedirs(sample_dir)
      with open(os.path.join(sample_dir, 'sample.np'), 'wb') as output:
        output.write(b'legacy')
      with self.assertRaises(ValueError):
        ensure_training_manifest(directory, manifest)

  def test_training_manifest_allows_extension_but_not_seed_change(self):
    config = get_config()
    initial = build_training_manifest(config)
    config.training.n_iters = 100000
    extended = build_training_manifest(config)
    self.assertEqual(
      initial['training_protocol_sha256'],
      extended['training_protocol_sha256'])
    config.seed = 43
    changed_seed = build_training_manifest(config)
    self.assertNotEqual(
      initial['training_protocol_sha256'],
      changed_seed['training_protocol_sha256'])


if __name__ == '__main__':
  unittest.main()
