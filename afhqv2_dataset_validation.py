"""In-container AFHQv2 loader and reference-statistics validation."""

import json
import os
import tempfile
import unittest
from unittest import mock

import numpy as np
import tensorflow as tf

import datasets
import evaluation
from configs.vp.afhqv2_ncsnpp_continuous import get_config


class AFHQv2DatasetValidationTest(unittest.TestCase):

  def _write_dataset(self, root, size=64):
    with open(os.path.join(root, 'dataset.json'), 'w', encoding='utf-8') as fout:
      json.dump({'labels': None}, fout)
    image_dir = os.path.join(root, '00000')
    os.makedirs(image_dir)
    for index, value in enumerate((0, 64, 128, 255)):
      image = tf.fill([size, size, 3], tf.cast(value, tf.uint8))
      tf.io.write_file(
        os.path.join(image_dir, f'img{index:08d}.png'),
        tf.io.encode_png(image))

  def test_identity_loader_and_inception_preprocessing(self):
    with tempfile.TemporaryDirectory() as root:
      self._write_dataset(root)
      config = get_config()
      config.data.afhqv2_dir = root
      config.data.random_flip = False
      config.training.batch_size = 2
      config.eval.batch_size = 2
      config.data.afhqv2_train_take = 4
      config.data.afhqv2_eval_take = 4

      paths = datasets.get_local_afhqv2_paths(config)
      self.assertEqual(len(paths), 4)
      self.assertEqual(len(datasets.afhqv2_dataset_identity(config)), 64)
      self.assertRegex(
        evaluation._get_stats_filename(config),
        r'^assets/stats/afhqv2_64_[0-9a-f]{12}_stats\.npz$')

      reference_image = evaluation._load_afhqv2_image(paths[0], 64).numpy()
      self.assertEqual(reference_image.shape, (64, 64, 3))
      self.assertGreaterEqual(reference_image.min(), 0.)
      self.assertLessEqual(reference_image.max(), 255.)

      train_ds, eval_ds, builder = datasets.get_dataset(config, evaluation=True)
      self.assertEqual(builder, os.path.realpath(root))
      train_batch = next(iter(train_ds))['image'].numpy()
      eval_batch = next(iter(eval_ds))['image'].numpy()
      self.assertEqual(train_batch.shape, (2, 64, 64, 3))
      self.assertEqual(eval_batch.shape, (2, 64, 64, 3))
      self.assertGreaterEqual(train_batch.min(), 0.)
      self.assertLessEqual(train_batch.max(), 1.)

  def test_loader_rejects_non_64_resolution(self):
    with tempfile.TemporaryDirectory() as root:
      self._write_dataset(root, size=32)
      config = get_config()
      config.data.afhqv2_dir = root
      config.training.batch_size = 2
      config.eval.batch_size = 2
      with self.assertRaises((tf.errors.InvalidArgumentError, ValueError)):
        train_ds, _, _ = datasets.get_dataset(config, evaluation=True)
        next(iter(train_ds))

  def test_reference_stats_use_every_prepared_image(self):
    with tempfile.TemporaryDirectory() as root:
      self._write_dataset(root)
      config = get_config()
      config.data.afhqv2_dir = root
      config.data.afhqv2_train_take = 1
      config.data.afhqv2_eval_take = 1
      filename = os.path.join(root, 'stats.npz')

      def fake_inception(batch, inception_model):
        del inception_model
        return {'pool_3': tf.reduce_mean(batch, axis=(1, 2))}

      with mock.patch.object(
          evaluation, 'run_inception_distributed', side_effect=fake_inception):
        evaluation._compute_afhqv2_stats(
          filename, inception_model=object(), batch_size=3, config=config)

      with np.load(filename) as stats:
        self.assertEqual(stats['pool_3'].shape, (4, 3))
        self.assertEqual(int(stats['num_examples']), 4)
        self.assertEqual(str(stats['dataset_identity']),
                         datasets.afhqv2_dataset_identity(config))


if __name__ == '__main__':
  unittest.main()
