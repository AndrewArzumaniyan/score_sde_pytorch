# coding=utf-8
# Copyright 2020 The Google Research Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Utility functions for computing FID/Inception scores."""

import gc
import hashlib
import os

import numpy as np
import six
import tensorflow as tf
import tensorflow_datasets as tfds
import tensorflow_gan as tfgan
import tensorflow_hub as tfhub
from absl import logging
import datasets as datasets_lib
from artifact_utils import atomic_savez

try:
  import jax
except ImportError:
  jax = None

INCEPTION_TFHUB = 'https://tfhub.dev/tensorflow/tfgan/eval/inception/1'
INCEPTION_OUTPUT = 'logits'
INCEPTION_FINAL_POOL = 'pool_3'
_DEFAULT_DTYPES = {
  INCEPTION_OUTPUT: tf.float32,
  INCEPTION_FINAL_POOL: tf.float32
}
INCEPTION_DEFAULT_IMAGE_SIZE = 299


def get_inception_model(inceptionv3=False):
  if inceptionv3:
    return tfhub.load(
      'https://tfhub.dev/google/imagenet/inception_v3/feature_vector/4')
  else:
    return tfhub.load(INCEPTION_TFHUB)


def _get_stats_filename(config):
  """Return the dataset statistics path for the configured dataset."""
  if config.data.dataset == 'CIFAR10':
    cifar10_class = getattr(config.data, 'cifar10_class', -1)
    stats_split = getattr(config.data, 'cifar10_stats_split', 'train')
    if stats_split not in ('train', 'test'):
      raise ValueError(
        f'CIFAR-10 statistics split must be train or test, got {stats_split}.')
    per_class = getattr(config.data, f'cifar10_{stats_split}_per_class', -1)
    if cifar10_class >= 0 and per_class >= 0:
      raise ValueError('Set either cifar10_class or a balanced per-class subset, not both.')
    if cifar10_class >= 0:
      if cifar10_class > 9:
        raise ValueError(f'CIFAR-10 class must be in [0, 9], got {cifar10_class}.')
      return f'assets/stats/cifar10_class_{cifar10_class}_{stats_split}_stats.npz'
    if per_class >= 0:
      return f'assets/stats/cifar10_balanced_{per_class}_per_class_{stats_split}_stats.npz'
    if stats_split == 'train':
      return 'assets/stats/cifar10_stats.npz'
    return 'assets/stats/cifar10_test_stats.npz'
  elif config.data.dataset == 'CELEBA':
    resolution = config.data.image_size
    source = datasets_lib.get_local_celeba_root(config) or 'tfds'
    if source != 'tfds':
      source = os.path.realpath(source)
      partition_path = os.path.join(source, 'list_eval_partition.txt')
      source_digest = hashlib.sha256(source.encode('utf-8'))
      with open(partition_path, 'rb') as partition_file:
        for block in iter(lambda: partition_file.read(1024 * 1024), b''):
          source_digest.update(block)
      source_hash = source_digest.hexdigest()[:12]
    else:
      source_hash = hashlib.sha256(b'tfds').hexdigest()[:12]
    return f'assets/stats/celeba_{resolution}_{source_hash}_stats.npz'
  elif config.data.dataset == 'AFHQV2':
    resolution = config.data.image_size
    identity = datasets_lib.afhqv2_dataset_identity(config)[:12]
    return f'assets/stats/afhqv2_{resolution}_{identity}_stats.npz'
  elif config.data.dataset == 'LSUN':
    return f'assets/stats/lsun_{config.data.category}_{config.data.image_size}_stats.npz'
  else:
    raise ValueError(f'Dataset {config.data.dataset} stats not found.')


def _compute_cifar10_stats(filename, inception_model, batch_size, config,
                           split='train'):
  """Compute and persist CIFAR-10 pool_3 statistics from a TFDS split."""
  tfds_data_dir = os.environ.get('TFDS_DATA_DIR')
  ds = tfds.load(
    'cifar10',
    split=split,
    data_dir=tfds_data_dir,
    shuffle_files=False)
  ds = datasets_lib.select_cifar10_subset(ds, config, split)
  ds = ds.map(lambda example: example['image'],
              num_parallel_calls=tf.data.experimental.AUTOTUNE)
  ds = ds.batch(batch_size)
  ds = ds.prefetch(tf.data.experimental.AUTOTUNE)

  pools = []
  total = 0
  for batch in ds:
    gc.collect()
    latents = run_inception_distributed(batch, inception_model)
    gc.collect()
    pools.append(latents['pool_3'].numpy())
    total += int(batch.shape[0])
    logging.info('Computed CIFAR-10 dataset stats for %d examples', total)

  pool_3 = np.concatenate(pools, axis=0)
  try:
    atomic_savez(
      filename, overwrite=False, pool_3=pool_3,
      split=np.asarray(split), num_examples=np.asarray(pool_3.shape[0]))
  except FileExistsError:
    logging.info('Dataset stats were published concurrently: %s', filename)


def _preprocess_celeba_image(image, resolution):
  """Match the CelebA preprocessing used by the training/eval dataloader."""
  image = tf.image.convert_image_dtype(image, tf.float32)
  image = datasets_lib.central_crop(image, 140)
  image = datasets_lib.resize_small(image, resolution)
  # Inception expects images in [0, 255].
  return image * 255.


def _load_local_celeba_image(path, resolution):
  image = tf.io.read_file(path)
  image = tf.image.decode_jpeg(image, channels=3)
  image.set_shape([218, 178, 3])
  return _preprocess_celeba_image(image, resolution)


def _compute_celeba_stats(filename, inception_model, batch_size, config):
  """Compute and persist CelebA pool_3 statistics from local data or TFDS."""
  resolution = config.data.image_size
  local_paths = datasets_lib.get_local_celeba_split_paths('train', config)
  if local_paths is not None:
    logging.info('Computing CelebA stats from local directory: %s',
                 datasets_lib.get_local_celeba_root(config))
    ds = tf.data.Dataset.from_tensor_slices(local_paths)
    ds = ds.map(lambda path: _load_local_celeba_image(path, resolution),
                num_parallel_calls=tf.data.experimental.AUTOTUNE)
  else:
    tfds_data_dir = os.environ.get('TFDS_DATA_DIR')
    ds = tfds.load(
      'celeb_a',
      split='train',
      data_dir=tfds_data_dir,
      shuffle_files=False)
    ds = ds.map(lambda example: _preprocess_celeba_image(example['image'], resolution),
                num_parallel_calls=tf.data.experimental.AUTOTUNE)
  ds = ds.batch(batch_size)
  ds = ds.prefetch(tf.data.experimental.AUTOTUNE)

  pools = []
  total = 0
  for batch in ds:
    gc.collect()
    latents = run_inception_distributed(batch, inception_model)
    gc.collect()
    pools.append(latents['pool_3'].numpy())
    total += int(batch.shape[0])
    logging.info('Computed CelebA dataset stats for %d examples', total)

  pool_3 = np.concatenate(pools, axis=0)
  try:
    atomic_savez(
      filename, overwrite=False, pool_3=pool_3,
      resolution=np.asarray(resolution), num_examples=np.asarray(pool_3.shape[0]))
  except FileExistsError:
    logging.info('Dataset stats were published concurrently: %s', filename)


def _load_afhqv2_image(path, resolution):
  """Load an official-converter AFHQv2 PNG for Inception evaluation."""
  image = tf.io.read_file(path)
  image = tf.image.decode_png(image, channels=3)
  expected = tf.constant([resolution, resolution, 3], dtype=tf.int32)
  with tf.control_dependencies([
      tf.debugging.assert_equal(
        tf.shape(image), expected,
        message='AFHQv2 reference images must have the configured resolution')]):
    image = tf.image.convert_image_dtype(tf.identity(image), tf.float32)
  return image * 255.


def _compute_afhqv2_stats(filename, inception_model, batch_size, config):
  """Compute pool_3 stats over the complete prepared AFHQv2 dataset."""
  resolution = config.data.image_size
  image_paths = datasets_lib.get_local_afhqv2_paths(config)
  if image_paths is None:
    raise FileNotFoundError(
      'Prepared AFHQv2-64 was not found. Run tools/prepare_afhqv2_64.sh.')
  identity = datasets_lib.afhqv2_dataset_identity(config)
  logging.info('Computing AFHQv2 stats from %s (%d images)',
               datasets_lib.get_local_afhqv2_root(config), len(image_paths))
  ds = tf.data.Dataset.from_tensor_slices(image_paths)
  ds = ds.map(lambda path: _load_afhqv2_image(path, resolution),
              num_parallel_calls=tf.data.experimental.AUTOTUNE)
  ds = ds.batch(batch_size)
  ds = ds.prefetch(tf.data.experimental.AUTOTUNE)

  pools = []
  total = 0
  for batch in ds:
    gc.collect()
    latents = run_inception_distributed(batch, inception_model)
    gc.collect()
    pools.append(latents['pool_3'].numpy())
    total += int(batch.shape[0])
    logging.info('Computed AFHQv2 dataset stats for %d examples', total)

  pool_3 = np.concatenate(pools, axis=0)
  try:
    atomic_savez(
      filename, overwrite=False, pool_3=pool_3,
      resolution=np.asarray(resolution),
      num_examples=np.asarray(pool_3.shape[0]),
      dataset_identity=np.asarray(identity))
  except FileExistsError:
    logging.info('Dataset stats were published concurrently: %s', filename)


def load_dataset_stats(config, inception_model=None):
  """Load pre-computed dataset statistics, computing them for supported TFDS datasets if needed."""
  filename = _get_stats_filename(config)
  if not tf.io.gfile.exists(filename):
    logging.info('Dataset stats file %s not found. Computing it.', filename)
    if inception_model is None:
      inception_model = get_inception_model()
    batch_size = getattr(getattr(config, 'eval', None), 'batch_size', 512)
    if config.data.dataset == 'CIFAR10':
      _compute_cifar10_stats(
        filename,
        inception_model,
        batch_size,
        config=config,
        split=getattr(config.data, 'cifar10_stats_split', 'train'))
    elif config.data.dataset == 'CELEBA':
      _compute_celeba_stats(filename, inception_model, batch_size, config)
    elif config.data.dataset == 'AFHQV2':
      _compute_afhqv2_stats(filename, inception_model, batch_size, config)
    else:
      raise FileNotFoundError(
        f'Dataset stats file {filename} does not exist. '
        'Download or compute it before evaluation.')

  with tf.io.gfile.GFile(filename, 'rb') as fin:
    stats = np.load(fin)
    return {key: stats[key] for key in stats.files}


def dataset_stats_fingerprint(config):
  """Content fingerprint used to bind a metric report to reference stats."""
  filename = _get_stats_filename(config)
  digest = hashlib.sha256()
  with tf.io.gfile.GFile(filename, 'rb') as stats_file:
    while True:
      block = stats_file.read(1024 * 1024)
      if not block:
        break
      digest.update(block)
  return digest.hexdigest()


def classifier_fn_from_tfhub(output_fields, inception_model,
                             return_tensor=False):
  """Returns a function that can be as a classifier function.

  Copied from tfgan but avoid loading the model each time calling _classifier_fn

  Args:
    output_fields: A string, list, or `None`. If present, assume the module
      outputs a dictionary, and select this field.
    inception_model: A model loaded from TFHub.
    return_tensor: If `True`, return a single tensor instead of a dictionary.

  Returns:
    A one-argument function that takes an image Tensor and returns outputs.
  """
  if isinstance(output_fields, six.string_types):
    output_fields = [output_fields]

  def _classifier_fn(images):
    output = inception_model(images)
    if output_fields is not None:
      output = {x: output[x] for x in output_fields}
    if return_tensor:
      assert len(output) == 1
      output = list(output.values())[0]
    return tf.nest.map_structure(tf.compat.v1.layers.flatten, output)

  return _classifier_fn


@tf.function
def run_inception_jit(inputs,
                      inception_model,
                      num_batches=1,
                      inceptionv3=False):
  """Running the inception network. Assuming input is within [0, 255]."""
  if not inceptionv3:
    inputs = (tf.cast(inputs, tf.float32) - 127.5) / 127.5
  else:
    inputs = tf.cast(inputs, tf.float32) / 255.

  return tfgan.eval.run_classifier_fn(
    inputs,
    num_batches=num_batches,
    classifier_fn=classifier_fn_from_tfhub(None, inception_model),
    dtypes=_DEFAULT_DTYPES)


@tf.function
def run_inception_distributed(input_tensor,
                              inception_model,
                              num_batches=1,
                              inceptionv3=False):
  """Distribute the inception network computation to all available TPUs.

  Args:
    input_tensor: The input images. Assumed to be within [0, 255].
    inception_model: The inception network model obtained from `tfhub`.
    num_batches: The number of batches used for dividing the input.
    inceptionv3: If `True`, use InceptionV3, otherwise use InceptionV1.

  Returns:
    A dictionary with key `pool_3` and `logits`, representing the pool_3 and
      logits of the inception network respectively.
  """
  num_tpus = jax.local_device_count() if jax is not None else 1
  input_tensors = tf.split(input_tensor, num_tpus, axis=0)
  pool3 = []
  logits = [] if not inceptionv3 else None
  if jax is None:
    device_format = '/CPU:{}'
  else:
    device_format = '/TPU:{}' if 'TPU' in str(jax.devices()[0]) else '/GPU:{}'
  for i, tensor in enumerate(input_tensors):
    with tf.device(device_format.format(i)):
      tensor_on_device = tf.identity(tensor)
      res = run_inception_jit(
        tensor_on_device, inception_model, num_batches=num_batches,
        inceptionv3=inceptionv3)

      if not inceptionv3:
        pool3.append(res['pool_3'])
        logits.append(res['logits'])  # pytype: disable=attribute-error
      else:
        pool3.append(res)

  with tf.device('/CPU'):
    return {
      'pool_3': tf.concat(pool3, axis=0),
      'logits': tf.concat(logits, axis=0) if not inceptionv3 else None
    }
