"""Dependency-light regression tests for deterministic RNG plumbing."""

import os
import random
import tempfile
from types import SimpleNamespace
import unittest

import numpy as np
import tensorflow as tf
import torch

import datasets as datasets_lib
from reproducibility import derive_seed, isolated_torch_rng, seed_everything


class ReproducibilityValidationTest(unittest.TestCase):

  def test_seed_everything_repeats_python_numpy_and_torch(self):
    seed_everything(12345, tensorflow_module=tf)
    first = (
      random.random(),
      np.random.standard_normal(4),
      torch.randn(4),
      tf.random.normal([4]).numpy(),
    )
    seed_everything(12345, tensorflow_module=tf)
    second = (
      random.random(),
      np.random.standard_normal(4),
      torch.randn(4),
      tf.random.normal([4]).numpy(),
    )
    self.assertEqual(first[0], second[0])
    self.assertTrue(np.array_equal(first[1], second[1]))
    self.assertTrue(torch.equal(first[2], second[2]))
    self.assertTrue(np.array_equal(first[3], second[3]))

    seed_everything(12346)
    self.assertFalse(torch.equal(first[2], torch.randn(4)))

  def test_derived_seeds_are_stable_and_namespaced(self):
    first = derive_seed(42, 'sampling-round', 3)
    self.assertEqual(first, derive_seed(42, 'sampling-round', 3))
    self.assertNotEqual(first, derive_seed(42, 'sampling-round', 4))
    self.assertNotEqual(first, derive_seed(42, 'snapshot', 3))

  def test_isolated_diagnostics_do_not_consume_training_rng(self):
    seed_everything(91)
    expected_first = torch.randn(5)
    expected_second = torch.randn(5)

    seed_everything(91)
    actual_first = torch.randn(5)
    with isolated_torch_rng(derive_seed(91, 'diagnostic')):
      _ = torch.randn(100)
    actual_second = torch.randn(5)

    self.assertTrue(torch.equal(expected_first, actual_first))
    self.assertTrue(torch.equal(expected_second, actual_second))

  def test_invalid_seed_is_rejected(self):
    with self.assertRaises(ValueError):
      seed_everything(-1)
    with self.assertRaises(ValueError):
      seed_everything(True)
    with self.assertRaises(ValueError):
      seed_everything(1.5)
    with self.assertRaises(ValueError):
      seed_everything(2 ** 31)

  def test_deterministic_tf_data_replay_skip_matches_full_suffix(self):
    def make_dataset(seed):
      dataset = tf.data.Dataset.range(24)
      # Match the production graph: repeat first, then shuffle the repeated
      # stream, enumerate it, and apply stateless per-example randomness.
      dataset = dataset.repeat(2).shuffle(
        8, seed=seed, reshuffle_each_iteration=True).enumerate()

      def add_stateless_noise(index, value):
        noise = tf.random.stateless_uniform(
          [2], seed=tf.stack([tf.cast(seed, tf.int32),
                              tf.cast(index, tf.int32)]))
        return tf.cast(value, tf.float32), noise

      options = tf.data.Options()
      options.experimental_deterministic = True
      dataset = dataset.with_options(options)
      return dataset.map(
        add_stateless_noise,
        num_parallel_calls=tf.data.experimental.AUTOTUNE).batch(4)

    full = list(make_dataset(77).as_numpy_iterator())
    repeated = list(make_dataset(77).as_numpy_iterator())
    different = list(make_dataset(78).as_numpy_iterator())
    self.assertEqual(len(full), len(repeated))
    for expected, actual in zip(full, repeated):
      self.assertTrue(np.array_equal(expected[0], actual[0]))
      self.assertTrue(np.array_equal(expected[1], actual[1]))
    self.assertFalse(np.array_equal(full[0][1], different[0][1]))

    skipped = list(make_dataset(77).skip(3).as_numpy_iterator())
    self.assertEqual(len(skipped), len(full) - 3)
    for expected, actual in zip(full[3:], skipped):
      self.assertTrue(np.array_equal(expected[0], actual[0]))
      self.assertTrue(np.array_equal(expected[1], actual[1]))

  @unittest.skipUnless(
    hasattr(datasets_lib.tfds, 'ReadConfig'),
    'The active TFDS installation is incomplete/incompatible.')
  def test_production_tfrecord_pipeline_replays_saved_batch_cursor(self):
    with tempfile.TemporaryDirectory() as directory:
      record_path = os.path.join(directory, 'tiny.tfrecords')
      with tf.io.TFRecordWriter(record_path) as writer:
        for image_id in range(8):
          image = np.full((3, 4, 4), image_id * 17, dtype=np.uint8)
          example = tf.train.Example(features=tf.train.Features(feature={
            'shape': tf.train.Feature(
              int64_list=tf.train.Int64List(value=image.shape)),
            'data': tf.train.Feature(
              bytes_list=tf.train.BytesList(value=[image.tobytes()])),
          }))
          writer.write(example.SerializeToString())

      config = SimpleNamespace(
        seed=31415,
        training=SimpleNamespace(batch_size=2),
        eval=SimpleNamespace(batch_size=2),
        data=SimpleNamespace(
          dataset='FFHQ', image_size=4, random_flip=True,
          centered=False, num_channels=3, tfrecords_path=record_path))

      def first_batches(skip=0, count=5):
        train_ds, _, _ = datasets_lib.get_dataset(
          config, uniform_dequantization=True, evaluation=False)
        return list(train_ds.skip(skip).take(count).as_numpy_iterator())

      full = first_batches()
      repeated = first_batches()
      suffix = first_batches(skip=2, count=3)
      self.assertEqual(len(full), 5)
      for expected, actual in zip(full, repeated):
        self.assertTrue(np.array_equal(
          expected['image'], actual['image']))
      for expected, actual in zip(full[2:], suffix):
        self.assertTrue(np.array_equal(
          expected['image'], actual['image']))


if __name__ == '__main__':
  unittest.main()
