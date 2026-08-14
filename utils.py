import torch
import tensorflow as tf
import os
import logging
import hashlib
import re
import uuid

from reproducibility import capture_rng_state, restore_rng_state


CHECKPOINT_FORMAT_VERSION = 2


def _legacy_checkpoint_id(path):
  """Content-address an old checkpoint that has no embedded identity."""
  digest = hashlib.sha256()
  with tf.io.gfile.GFile(path, 'rb') as checkpoint_file:
    while True:
      block = checkpoint_file.read(1024 * 1024)
      if not block:
        break
      digest.update(block)
  return f'legacy-sha256-{digest.hexdigest()}'


def restore_checkpoint(ckpt_dir, state, device, restore_rng=True,
                       defer_rng=False):
  if not tf.io.gfile.exists(ckpt_dir):
    tf.io.gfile.makedirs(os.path.dirname(ckpt_dir))
    logging.warning(f"No checkpoint found at {ckpt_dir}. "
                    f"Returned the same state as input")
    return state
  else:
    loaded_state = torch.load(
      ckpt_dir, map_location=device, weights_only=False)
    format_version = loaded_state.get('checkpoint_format_version', 1)
    if format_version > CHECKPOINT_FORMAT_VERSION:
      raise ValueError(
        f'Checkpoint format {format_version} is newer than supported format '
        f'{CHECKPOINT_FORMAT_VERSION}: {ckpt_dir}')
    expected_model_protocol = state.get('model_protocol_sha256')
    loaded_model_protocol = loaded_state.get('model_protocol_sha256')
    if (format_version >= 2 and expected_model_protocol is not None and
        loaded_model_protocol is None):
      raise ValueError(
        f'Checkpoint is missing its model/config fingerprint: {ckpt_dir}')
    if (expected_model_protocol is not None and loaded_model_protocol is not None
        and expected_model_protocol != loaded_model_protocol):
      raise ValueError(
        'Checkpoint model/config fingerprint does not match the current '
        f'configuration: {ckpt_dir}')
    expected_training_protocol = state.get('training_protocol_sha256')
    loaded_training_protocol = loaded_state.get('training_protocol_sha256')
    if (format_version >= 2 and expected_training_protocol is not None and
        loaded_training_protocol is None):
      raise ValueError(
        f'Checkpoint is missing its training fingerprint: {ckpt_dir}')
    if (expected_training_protocol is not None and
        loaded_training_protocol is not None and
        expected_training_protocol != loaded_training_protocol):
      raise ValueError(
        'Checkpoint training code/config/seed fingerprint does not match the '
        f'current workdir: {ckpt_dir}')
    if format_version >= 2 and loaded_model_protocol is None:
      logging.warning(
        'Checkpoint %s has no model/config fingerprint.', ckpt_dir)
    current_ema_parameter_names = [
      name for name, parameter in state['model'].named_parameters()
      if parameter.requires_grad
    ]
    loaded_ema_parameter_names = loaded_state.get('ema_parameter_names')
    if (loaded_ema_parameter_names is not None and
        loaded_ema_parameter_names != current_ema_parameter_names):
      raise ValueError(
        'Checkpoint EMA parameter names/order do not match the current model: '
        f'{ckpt_dir}')
    state['optimizer'].load_state_dict(loaded_state['optimizer'])
    try:
      state['model'].load_state_dict(loaded_state['model'], strict=True)
    except RuntimeError as error:
      raise ValueError(
        f'Checkpoint model weights do not match the current model definition '
        f'(strict load failed): {ckpt_dir}\n{error}') from error
    state['ema'].load_state_dict(loaded_state['ema'])
    state['step'] = loaded_state['step']
    state['data_batches_consumed'] = loaded_state.get(
      'data_batches_consumed', state.get('data_batches_consumed', 0))
    state['eval_batches_consumed'] = loaded_state.get(
      'eval_batches_consumed', state.get('eval_batches_consumed', 0))
    state['model_protocol_sha256'] = (
      loaded_model_protocol or expected_model_protocol)
    state['training_protocol_sha256'] = (
      loaded_training_protocol or expected_training_protocol)
    checkpoint_id = loaded_state.get('checkpoint_id')
    if checkpoint_id is None:
      checkpoint_id = _legacy_checkpoint_id(ckpt_dir)
    state['checkpoint_id'] = checkpoint_id
    if restore_rng and defer_rng:
      state['_rng_state_to_restore'] = loaded_state.get('rng_state')
    elif restore_rng:
      rng_state = loaded_state.get('rng_state')
      if rng_state is None:
        logging.warning(
          'Checkpoint %s predates RNG-state saving. Resume is not exactly '
          'reproducible.', ckpt_dir)
      elif not restore_rng_state(rng_state):
        logging.warning(
          'Checkpoint %s RNG state was only partially restored because the '
          'visible CUDA topology changed.', ckpt_dir)
    return state


def save_checkpoint(ckpt_dir, state, overwrite=True):
  """Atomically save a checkpoint, optionally enforcing immutable naming."""
  if tf.io.gfile.exists(ckpt_dir) and not overwrite:
    raise FileExistsError(
      f'Refusing to overwrite immutable checkpoint: {ckpt_dir}')
  checkpoint_id = uuid.uuid4().hex
  saved_state = {
    'checkpoint_format_version': CHECKPOINT_FORMAT_VERSION,
    'checkpoint_id': checkpoint_id,
    'optimizer': state['optimizer'].state_dict(),
    'model': state['model'].state_dict(),
    'ema': state['ema'].state_dict(),
    'step': state['step'],
    'data_batches_consumed': state.get('data_batches_consumed', 0),
    'eval_batches_consumed': state.get('eval_batches_consumed', 0),
    'rng_state': capture_rng_state(),
    'model_protocol_sha256': state.get('model_protocol_sha256'),
    'training_protocol_sha256': state.get('training_protocol_sha256'),
    'ema_parameter_names': [
      name for name, parameter in state['model'].named_parameters()
      if parameter.requires_grad
    ],
  }
  directory = os.path.dirname(ckpt_dir)
  if directory:
    tf.io.gfile.makedirs(directory)
  temporary_path = f'{ckpt_dir}.tmp-{uuid.uuid4().hex}'
  try:
    torch.save(saved_state, temporary_path)
    tf.io.gfile.rename(temporary_path, ckpt_dir, overwrite=overwrite)
  finally:
    if tf.io.gfile.exists(temporary_path):
      tf.io.gfile.remove(temporary_path)
  return checkpoint_id


def checkpoint_filename(step, state_step, snapshot_freq, final=False):
  """Keep periodic indices stable and isolate a non-boundary final state."""
  if snapshot_freq <= 0:
    raise ValueError('training.snapshot_freq must be positive.')
  if final and step % snapshot_freq:
    return f'checkpoint_final_step_{int(state_step)}.pth'
  return f'checkpoint_{step // snapshot_freq}.pth'


def latest_immutable_checkpoint(checkpoint_dir, snapshot_freq):
  """Return the newest immutable checkpoint path and semantic state step."""
  if snapshot_freq <= 0:
    raise ValueError('training.snapshot_freq must be positive.')
  candidates = []
  for path in tf.io.gfile.glob(os.path.join(checkpoint_dir, 'checkpoint_*.pth')):
    basename = os.path.basename(path)
    final_match = re.fullmatch(r'checkpoint_final_step_(\d+)\.pth', basename)
    periodic_match = re.fullmatch(r'checkpoint_(\d+)\.pth', basename)
    if final_match:
      state_step = int(final_match.group(1))
    elif periodic_match:
      state_step = int(periodic_match.group(1)) * snapshot_freq + 1
    else:
      continue
    candidates.append((state_step, path))
  if not candidates:
    return None, None
  state_step, path = max(candidates, key=lambda item: item[0])
  return path, state_step
