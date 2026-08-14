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

# pylint: skip-file
"""Return training and evaluation/test datasets from config files."""
import os

import tensorflow as tf
import tensorflow_datasets as tfds

from reproducibility import derive_seed

try:
  import jax
except ImportError:
  jax = None


_CELEBA_PARTITIONS = {
  'train': '0',
  'validation': '1',
  'test': '2',
}


def _normalize_local_celeba_root(path):
  """Return the CelebA root containing metadata files and img_align_celeba."""
  if not path:
    return None

  candidates = [path, os.path.join(path, 'celeba')]
  required_files = (
    'list_eval_partition.txt',
    'list_attr_celeba.txt',
    'list_landmarks_align_celeba.txt',
  )
  for candidate in candidates:
    if not os.path.isdir(candidate):
      continue
    image_dir = os.path.join(candidate, 'img_align_celeba')
    if not os.path.isdir(image_dir):
      continue
    if all(os.path.isfile(os.path.join(candidate, name)) for name in required_files):
      return candidate
  return None


def get_local_celeba_root(config=None):
  """Locate a locally prepared CelebA directory if present."""
  candidates = []
  if config is not None:
    celeba_dir = getattr(getattr(config, 'data', None), 'celeba_dir', None)
    if celeba_dir:
      candidates.append(celeba_dir)

  env_dir = os.environ.get('CELEBA_DIR')
  if env_dir:
    candidates.append(env_dir)

  repo_root = os.path.dirname(os.path.abspath(__file__))
  cwd = os.getcwd()
  candidates.extend([
    os.path.join(cwd, 'celeba'),
    os.path.join(repo_root, 'celeba'),
  ])

  seen = set()
  for candidate in candidates:
    normalized = _normalize_local_celeba_root(candidate)
    if normalized and normalized not in seen:
      return normalized
    if normalized:
      seen.add(normalized)
  return None


def get_local_celeba_split_paths(split, config=None):
  """Return absolute image paths for a local CelebA split, or None if absent."""
  celeba_root = get_local_celeba_root(config)
  if celeba_root is None:
    return None
  if split not in _CELEBA_PARTITIONS:
    raise ValueError(f'Unknown CelebA split: {split}')

  image_dir = os.path.join(celeba_root, 'img_align_celeba')
  partition_file = os.path.join(celeba_root, 'list_eval_partition.txt')
  partition_id = _CELEBA_PARTITIONS[split]
  image_paths = []
  with open(partition_file, 'r') as fin:
    for line in fin:
      line = line.strip()
      if not line:
        continue
      filename, partition = line.split()
      if partition == partition_id:
        image_paths.append(os.path.join(image_dir, filename))
  return image_paths


def get_data_scaler(config):
  """Data normalizer. Assume data are always in [0, 1]."""
  if config.data.centered:
    # Rescale to [-1, 1]
    return lambda x: x * 2. - 1.
  else:
    return lambda x: x


def get_data_inverse_scaler(config):
  """Inverse data normalizer."""
  if config.data.centered:
    # Rescale [-1, 1] to [0, 1]
    return lambda x: (x + 1.) / 2.
  else:
    return lambda x: x


def crop_resize(image, resolution):
  """Crop and resize an image to the given resolution."""
  crop = tf.minimum(tf.shape(image)[0], tf.shape(image)[1])
  h, w = tf.shape(image)[0], tf.shape(image)[1]
  image = image[(h - crop) // 2:(h + crop) // 2,
          (w - crop) // 2:(w + crop) // 2]
  image = tf.image.resize(
    image,
    size=(resolution, resolution),
    antialias=True,
    method=tf.image.ResizeMethod.BICUBIC)
  return tf.cast(image, tf.uint8)


def resize_small(image, resolution):
  """Shrink an image to the given resolution."""
  h, w = image.shape[0], image.shape[1]
  ratio = resolution / min(h, w)
  h = tf.cast(tf.round(h * ratio), tf.int32)
  w = tf.cast(tf.round(w * ratio), tf.int32)
  return tf.image.resize(image, [h, w], antialias=True)


def central_crop(image, size):
  """Crop the center of an image to the given size."""
  top = (image.shape[0] - size) // 2
  left = (image.shape[1] - size) // 2
  return tf.image.crop_to_bounding_box(image, top, left, size, size)


def select_cifar10_subset(ds, config, split):
  """Select a configured CIFAR-10 class or deterministic balanced subset."""
  cifar10_class = getattr(config.data, 'cifar10_class', -1)
  per_class_name = f'cifar10_{split}_per_class'
  per_class = getattr(config.data, per_class_name, -1)
  if split not in ('train', 'test'):
    raise ValueError(f'Unknown CIFAR-10 split: {split}')
  if cifar10_class >= 0 and per_class >= 0:
    raise ValueError('Set either cifar10_class or a balanced per-class subset, not both.')
  if cifar10_class >= 0:
    if cifar10_class > 9:
      raise ValueError(f'CIFAR-10 class must be in [0, 9], got {cifar10_class}.')
    return ds.filter(lambda example: tf.equal(example['label'], cifar10_class))
  if per_class < 0:
    return ds

  class_datasets = []
  for class_id in range(10):
    class_ds = ds.filter(
      lambda example, class_id=class_id: tf.equal(example['label'], class_id))
    class_datasets.append(class_ds.take(per_class))
  return class_datasets[0].concatenate(class_datasets[1]).concatenate(
    class_datasets[2]).concatenate(class_datasets[3]).concatenate(
      class_datasets[4]).concatenate(class_datasets[5]).concatenate(
        class_datasets[6]).concatenate(class_datasets[7]).concatenate(
          class_datasets[8]).concatenate(class_datasets[9])


def get_dataset(config, uniform_dequantization=False, evaluation=False):
  """Create data loaders for training and evaluation.

  Args:
    config: A ml_collection.ConfigDict parsed from config files.
    uniform_dequantization: If `True`, add uniform dequantization to images.
    evaluation: If `True`, fix number of epochs to 1.

  Returns:
    train_ds, eval_ds, dataset_builder.
  """
  # Compute batch size for this worker.
  batch_size = config.training.batch_size if not evaluation else config.eval.batch_size
  device_count = jax.device_count() if jax is not None else 1
  if batch_size % device_count != 0:
    raise ValueError(f'Batch sizes ({batch_size} must be divided by'
                     f'the number of devices ({device_count})')

  # Reduce this when image resolution is too large and data pointer is stored
  shuffle_buffer_size = 10000
  prefetch_size = tf.data.experimental.AUTOTUNE
  num_epochs = None if not evaluation else 1
  tfds_data_dir = os.environ.get('TFDS_DATA_DIR')

  # Create dataset builders for each dataset.
  if config.data.dataset == 'CIFAR10':
    dataset_builder = tfds.builder('cifar10', data_dir=tfds_data_dir)
    train_split_name = 'train'
    eval_split_name = 'test'

    def resize_op(img):
      img = tf.image.convert_image_dtype(img, tf.float32)
      return tf.image.resize(img, [config.data.image_size, config.data.image_size], antialias=True)

  elif config.data.dataset == 'SVHN':
    dataset_builder = tfds.builder('svhn_cropped', data_dir=tfds_data_dir)
    train_split_name = 'train'
    eval_split_name = 'test'

    def resize_op(img):
      img = tf.image.convert_image_dtype(img, tf.float32)
      return tf.image.resize(img, [config.data.image_size, config.data.image_size], antialias=True)

  elif config.data.dataset == 'CELEBA':
    local_celeba_root = get_local_celeba_root(config)
    dataset_builder = local_celeba_root or tfds.builder('celeb_a', data_dir=tfds_data_dir)
    train_split_name = 'train'
    eval_split_name = 'validation'

    def resize_op(img):
      img = tf.image.convert_image_dtype(img, tf.float32)
      img = central_crop(img, 140)
      img = resize_small(img, config.data.image_size)
      return img

  elif config.data.dataset == 'LSUN':
    dataset_builder = tfds.builder(f'lsun/{config.data.category}', data_dir=tfds_data_dir)
    train_split_name = 'train'
    eval_split_name = 'validation'

    if config.data.image_size == 128:
      def resize_op(img):
        img = tf.image.convert_image_dtype(img, tf.float32)
        img = resize_small(img, config.data.image_size)
        img = central_crop(img, config.data.image_size)
        return img

    else:
      def resize_op(img):
        img = crop_resize(img, config.data.image_size)
        img = tf.image.convert_image_dtype(img, tf.float32)
        return img

  elif config.data.dataset in ['FFHQ', 'CelebAHQ']:
    dataset_builder = tf.data.TFRecordDataset(config.data.tfrecords_path)
    train_split_name = eval_split_name = 'train'

  else:
    raise NotImplementedError(
      f'Dataset {config.data.dataset} not yet supported.')

  # Customize preprocess functions for each dataset.
  if config.data.dataset in ['FFHQ', 'CelebAHQ']:
    def preprocess_fn(d, is_training, example_index, dataset_seed):
      sample = tf.io.parse_single_example(d, features={
        'shape': tf.io.FixedLenFeature([3], tf.int64),
        'data': tf.io.FixedLenFeature([], tf.string)})
      data = tf.io.decode_raw(sample['data'], tf.uint8)
      data = tf.reshape(data, sample['shape'])
      data = tf.transpose(data, (1, 2, 0))
      img = tf.image.convert_image_dtype(data, tf.float32)
      if config.data.random_flip and is_training:
        flip_seed = tf.stack([
          tf.cast(derive_seed(dataset_seed, 'flip'), tf.int32),
          tf.cast(tf.math.floormod(example_index, 2 ** 31 - 1), tf.int32),
        ])
        should_flip = tf.random.stateless_uniform(
          [], seed=flip_seed, dtype=tf.float32) < 0.5
        img = tf.cond(
          should_flip, lambda: tf.image.flip_left_right(img), lambda: img)
      if uniform_dequantization:
        dequant_seed = tf.stack([
          tf.cast(derive_seed(dataset_seed, 'dequantization'), tf.int32),
          tf.cast(tf.math.floormod(example_index, 2 ** 31 - 1), tf.int32),
        ])
        img = (tf.random.stateless_uniform(
          tf.shape(img), seed=dequant_seed, dtype=tf.float32) + img * 255.) / 256.
      return dict(image=img, label=None)

  else:
    def preprocess_fn(d, is_training, example_index, dataset_seed):
      """Basic preprocessing function scales data to [0, 1) and randomly flips."""
      img = resize_op(d['image'])
      if config.data.random_flip and is_training:
        flip_seed = tf.stack([
          tf.cast(derive_seed(dataset_seed, 'flip'), tf.int32),
          tf.cast(tf.math.floormod(example_index, 2 ** 31 - 1), tf.int32),
        ])
        should_flip = tf.random.stateless_uniform(
          [], seed=flip_seed, dtype=tf.float32) < 0.5
        img = tf.cond(
          should_flip, lambda: tf.image.flip_left_right(img), lambda: img)
      if uniform_dequantization:
        dequant_seed = tf.stack([
          tf.cast(derive_seed(dataset_seed, 'dequantization'), tf.int32),
          tf.cast(tf.math.floormod(example_index, 2 ** 31 - 1), tf.int32),
        ])
        img = (tf.random.stateless_uniform(
          tf.shape(img), seed=dequant_seed, dtype=tf.float32) + img * 255.) / 256.

      return dict(image=img, label=d.get('label', None))

  def create_dataset(dataset_builder, split, is_training):
    dataset_seed = derive_seed(config.seed, 'tf-data', split,
                               'train' if is_training else 'eval')
    dataset_options = tf.data.Options()
    dataset_options.experimental_optimization.map_parallelization = True
    dataset_options.experimental_threading.private_threadpool_size = 48
    dataset_options.experimental_threading.max_intra_op_parallelism = 1
    dataset_options.experimental_deterministic = True
    read_config = tfds.ReadConfig(
      options=dataset_options,
      shuffle_seed=dataset_seed,
      shuffle_reshuffle_each_iteration=is_training)
    if config.data.dataset == 'CELEBA' and isinstance(dataset_builder, str):
      image_paths = get_local_celeba_split_paths(split, config)
      if image_paths is None:
        raise FileNotFoundError('Local CelebA directory was expected but not found.')
      ds = tf.data.Dataset.from_tensor_slices(image_paths)

      def load_local_celeba_example(path):
        image = tf.io.read_file(path)
        image = tf.image.decode_jpeg(image, channels=3)
        image.set_shape([218, 178, 3])
        return dict(image=image, label=None)

      ds = ds.with_options(dataset_options)
      ds = ds.map(load_local_celeba_example, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    elif isinstance(dataset_builder, tfds.core.DatasetBuilder):
      dataset_builder.download_and_prepare()
      ds = dataset_builder.as_dataset(
        split=split, shuffle_files=is_training, read_config=read_config)
    else:
      ds = dataset_builder.with_options(dataset_options)
    if config.data.dataset == 'CIFAR10':
      ds = select_cifar10_subset(ds, config, split)
    ds = ds.repeat(count=num_epochs)
    if is_training:
      ds = ds.shuffle(
        shuffle_buffer_size, seed=dataset_seed,
        reshuffle_each_iteration=True)
    ds = ds.enumerate()
    ds = ds.map(
      lambda index, example: preprocess_fn(
        example, is_training=is_training, example_index=index,
        dataset_seed=dataset_seed),
      num_parallel_calls=tf.data.experimental.AUTOTUNE)
    ds = ds.batch(batch_size, drop_remainder=True)
    return ds.prefetch(prefetch_size)

  train_ds = create_dataset(
    dataset_builder, train_split_name, is_training=not evaluation)
  eval_ds = create_dataset(dataset_builder, eval_split_name, is_training=False)
  return train_ds, eval_ds, dataset_builder
