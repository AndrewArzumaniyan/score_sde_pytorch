"""Atomic artifact writes and provenance checks for evaluation caches."""

import hashlib
import io
from importlib import metadata as importlib_metadata
import json
import logging
import math
import os
import platform
import uuid

import numpy as np
import tensorflow as tf
import torch


EVAL_ARTIFACT_SCHEMA_VERSION = 1
TRAINING_MANIFEST_SCHEMA_VERSION = 1


def _canonicalize(value):
  if hasattr(value, 'to_dict'):
    value = value.to_dict()
  if isinstance(value, dict):
    return {str(key): _canonicalize(value[key]) for key in sorted(value)}
  if isinstance(value, (list, tuple)):
    return [_canonicalize(item) for item in value]
  if isinstance(value, np.generic):
    return _canonicalize(value.item())
  if isinstance(value, float) and not math.isfinite(value):
    return str(value)
  if value is None or isinstance(value, (bool, int, float, str)):
    return value
  return str(value)


def _source_fingerprint(repo_root):
  """Hash runtime sources, including the installed official EDM CUDA ops."""
  digest = hashlib.sha256()
  ignored_directories = {'.git', '__pycache__', '.cache', '.tfds',
                         '.torch_extensions', 'figures', 'notes', 'tests',
                         'workdirs'}
  source_extensions = ('.py', '.c', '.cc', '.cpp', '.cu', '.cuh', '.h')
  for current_root, directories, filenames in os.walk(repo_root):
    directories[:] = sorted(
      directory for directory in directories
      if directory not in ignored_directories)
    for filename in sorted(filenames):
      if (not filename.endswith(source_extensions) and
          filename not in ('Dockerfile', 'requirements.txt')):
        continue
      if filename.endswith('_validation.py'):
        continue
      path = os.path.join(current_root, filename)
      relative_path = os.path.relpath(path, repo_root).replace(os.sep, '/')
      digest.update(relative_path.encode('utf-8'))
      digest.update(b'\0')
      with open(path, 'rb') as source_file:
        while True:
          block = source_file.read(1024 * 1024)
          if not block:
            break
          digest.update(block)
  return digest.hexdigest()


def _manifest_with_hash(payload, hash_field):
  serialized = json.dumps(
    payload, sort_keys=True, separators=(',', ':'), ensure_ascii=True)
  payload[hash_field] = hashlib.sha256(
    serialized.encode('utf-8')).hexdigest()
  return payload


def _environment_manifest():
  cuda_names = []
  if torch.cuda.is_available():
    try:
      cuda_names = [
        torch.cuda.get_device_name(index)
        for index in range(torch.cuda.device_count())
      ]
    except RuntimeError as error:
      # Recording provenance must not be a second, unrelated way for
      # training/eval startup to fail. A device-name query can raise if CUDA
      # context creation is transiently contended (observed: TensorFlow and
      # Torch both touching the driver in-process), even though CUDA itself
      # is usable a moment later.
      logging.warning('Could not read CUDA device name(s) for the '
                      'environment manifest: %s', error)
      cuda_names = [f'unknown ({error})']
  def package_version(distribution):
    try:
      return importlib_metadata.version(distribution)
    except importlib_metadata.PackageNotFoundError:
      return None
  return {
    'python': platform.python_version(),
    'numpy': np.__version__,
    'tensorflow': tf.__version__,
    'torch': torch.__version__,
    'torch_cuda': torch.version.cuda,
    'cudnn': torch.backends.cudnn.version(),
    'visible_cuda_devices': cuda_names,
    'packages': {
      name: package_version(name)
      for name in ('ml-collections', 'tensorflow-datasets',
                   'tensorflow-gan', 'tensorflow-probability', 'torchvision')
    },
    'runtime_paths': {
      name: os.environ.get(name)
      for name in ('TFDS_DATA_DIR', 'CELEBA_DIR', 'EDM_OFFICIAL_ROOT')
    },
  }


def build_model_manifest(config):
  """Describe fields that must match to construct checkpoint parameters."""
  resolved = _canonicalize(config)
  training = resolved.get('training', {})
  payload = {
    'model': resolved.get('model', {}),
    'data': {
      key: resolved.get('data', {}).get(key)
      for key in ('dataset', 'image_size', 'num_channels', 'centered')
    },
    'training_sde': training.get('sde'),
    'continuous': training.get('continuous'),
    'edm_augmentation': training.get('edm_augmentation'),
    'edm_augment_probability': training.get('edm_augment_probability'),
  }
  return _manifest_with_hash(payload, 'model_protocol_sha256')


def build_training_manifest(config, repo_root=None):
  """Build the immutable training identity, excluding orchestration knobs."""
  if repo_root is None:
    repo_root = os.path.dirname(os.path.abspath(__file__))
  resolved = _canonicalize(config)
  resolved.pop('eval', None)
  training = resolved.get('training', {})
  for key in ('n_iters', 'log_freq', 'eval_freq',
              'snapshot_freq_for_preemption', 'snapshot_sampling'):
    training.pop(key, None)
  payload = {
    'schema_version': TRAINING_MANIFEST_SCHEMA_VERSION,
    'source_sha256': _source_fingerprint(repo_root),
    'environment': _environment_manifest(),
    'config': resolved,
    'model_manifest': build_model_manifest(config),
  }
  return _manifest_with_hash(payload, 'training_protocol_sha256')


def build_eval_manifest(config, repo_root=None):
  """Build a content-addressed description of an evaluation protocol."""
  if repo_root is None:
    repo_root = os.path.dirname(os.path.abspath(__file__))
  resolved_config = _canonicalize(config)
  # The checkpoint range controls orchestration, not the contents of each
  # checkpoint evaluation. Excluding it allows a stopped range to resume.
  eval_config = resolved_config.get('eval', {})
  eval_config.pop('begin_ckpt', None)
  eval_config.pop('end_ckpt', None)
  manifest = {
    'schema_version': EVAL_ARTIFACT_SCHEMA_VERSION,
    'source_sha256': _source_fingerprint(repo_root),
    'environment': _environment_manifest(),
    'config': resolved_config,
  }
  return _manifest_with_hash(manifest, 'protocol_sha256')


def atomic_write_bytes(path, data, overwrite=False):
  """Publish bytes with a same-directory temporary file and atomic rename."""
  directory = os.path.dirname(path)
  if directory:
    tf.io.gfile.makedirs(directory)
  if tf.io.gfile.exists(path) and not overwrite:
    raise FileExistsError(f'Refusing to overwrite existing artifact: {path}')
  temporary_path = f'{path}.tmp-{uuid.uuid4().hex}'
  try:
    with tf.io.gfile.GFile(temporary_path, 'wb') as output_file:
      output_file.write(data)
    try:
      tf.io.gfile.rename(temporary_path, path, overwrite=overwrite)
    except tf.errors.AlreadyExistsError as error:
      raise FileExistsError(
        f'Refusing to overwrite existing artifact: {path}') from error
  finally:
    if tf.io.gfile.exists(temporary_path):
      tf.io.gfile.remove(temporary_path)


def atomic_savez(path, overwrite=False, **arrays):
  buffer = io.BytesIO()
  np.savez_compressed(buffer, **arrays)
  atomic_write_bytes(path, buffer.getvalue(), overwrite=overwrite)


def ensure_eval_manifest(eval_dir, manifest):
  """Create or validate the immutable manifest for an eval directory."""
  tf.io.gfile.makedirs(eval_dir)
  manifest_path = os.path.join(eval_dir, 'manifest.json')
  expected = json.dumps(manifest, sort_keys=True, indent=2) + '\n'
  if tf.io.gfile.exists(manifest_path):
    with tf.io.gfile.GFile(manifest_path, 'r') as manifest_file:
      actual = manifest_file.read()
    if actual != expected:
      raise ValueError(
        'The eval folder belongs to a different code/config/seed protocol. '
        f'Use a new --eval_folder instead of reusing {eval_dir}.')
    return manifest['protocol_sha256']

  existing = tf.io.gfile.listdir(eval_dir)
  if existing:
    raise ValueError(
      'The eval folder contains legacy artifacts without a manifest. Use a '
      f'new --eval_folder instead of reusing {eval_dir}: {existing[:5]}')
  atomic_write_bytes(manifest_path, expected.encode('utf-8'), overwrite=False)
  return manifest['protocol_sha256']


def ensure_training_manifest(workdir, manifest):
  """Create or validate a workdir identity before training or resuming."""
  tf.io.gfile.makedirs(workdir)
  manifest_path = os.path.join(workdir, 'run_manifest.json')
  expected = json.dumps(manifest, sort_keys=True, indent=2) + '\n'
  if tf.io.gfile.exists(manifest_path):
    with tf.io.gfile.GFile(manifest_path, 'r') as manifest_file:
      actual = manifest_file.read()
    if actual != expected:
      raise ValueError(
        'The workdir belongs to a different training code/config/seed. '
        f'Use a new --workdir instead of reusing {workdir}.')
    return manifest['training_protocol_sha256']

  checkpoint_locations = (
    os.path.join(workdir, 'checkpoints'),
    os.path.join(workdir, 'checkpoints-meta'),
    os.path.join(workdir, 'samples'),
  )
  if any(tf.io.gfile.exists(path) and tf.io.gfile.listdir(path)
         for path in checkpoint_locations):
    raise ValueError(
      'The workdir contains legacy checkpoints/samples without '
      'run_manifest.json. '
      'Do not resume it implicitly; migrate it explicitly or use a new '
      f'--workdir: {workdir}')
  atomic_write_bytes(manifest_path, expected.encode('utf-8'), overwrite=False)
  return manifest['training_protocol_sha256']


def scalar_string(value):
  """Read a scalar string saved by NumPy without accepting arrays silently."""
  array = np.asarray(value)
  if array.shape != ():
    raise ValueError(f'Expected scalar metadata, got shape {array.shape}.')
  item = array.item()
  if isinstance(item, bytes):
    return item.decode('utf-8')
  return str(item)


def validate_npz_metadata(archive, path, protocol_sha256, checkpoint_id,
                          round_id=None, sampling_seed=None,
                          required_arrays=(), checkpoint_step=None,
                          training_protocol_sha256=None):
  """Reject stale or partially incompatible cached evaluation artifacts."""
  required = {'protocol_sha256', 'checkpoint_id'}
  missing = sorted(required.difference(archive.files))
  if missing:
    raise ValueError(
      f'Cached artifact {path} has no current provenance metadata ({missing}). '
      'Use a new --eval_folder.')
  if scalar_string(archive['protocol_sha256']) != protocol_sha256:
    raise ValueError(f'Cached artifact uses a different eval protocol: {path}')
  if scalar_string(archive['checkpoint_id']) != str(checkpoint_id):
    raise ValueError(f'Cached artifact belongs to a different checkpoint: {path}')
  if checkpoint_step is not None:
    if ('checkpoint_step' not in archive.files or
        int(np.asarray(archive['checkpoint_step']).item()) != checkpoint_step):
      raise ValueError(f'Cached artifact has a different checkpoint step: {path}')
  if training_protocol_sha256 is not None:
    if ('training_protocol_sha256' not in archive.files or
        scalar_string(archive['training_protocol_sha256']) !=
        str(training_protocol_sha256)):
      raise ValueError(
        f'Cached artifact has a different training protocol: {path}')
  if round_id is not None:
    if 'round_id' not in archive.files or int(np.asarray(archive['round_id']).item()) != round_id:
      raise ValueError(f'Cached artifact has a different sampling round: {path}')
  if sampling_seed is not None:
    if ('sampling_seed' not in archive.files or
        int(np.asarray(archive['sampling_seed']).item()) != sampling_seed):
      raise ValueError(f'Cached artifact has a different sampling seed: {path}')
  missing_arrays = sorted(set(required_arrays).difference(archive.files))
  if missing_arrays:
    raise ValueError(
      f'Cached artifact {path} is missing payload arrays: {missing_arrays}')
  for name in required_arrays:
    array = np.asarray(archive[name])
    if array.size == 0:
      raise ValueError(f'Cached artifact {path} has an empty {name} array.')
    if np.issubdtype(array.dtype, np.number) and not np.isfinite(array).all():
      raise ValueError(f'Cached artifact {path} has non-finite values in {name}.')
