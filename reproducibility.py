"""Reproducibility helpers shared by training, evaluation, and checkpoints."""

import contextlib
import hashlib
import logging
import numbers
import os
import random

import numpy as np
import torch


_MAX_SEED = 2 ** 31 - 1


def normalize_seed(seed):
  """Validate and normalize a user-provided seed for all supported libraries."""
  if isinstance(seed, bool) or not isinstance(seed, numbers.Integral):
    raise ValueError(f'seed must be an integer, got {seed!r}.')
  seed = int(seed)
  if seed < 0 or seed > _MAX_SEED:
    raise ValueError(
      f'seed must be in [0, {_MAX_SEED}], got {seed}.')
  return seed


def derive_seed(base_seed, *components):
  """Derive a stable namespaced seed without relying on Python's hash()."""
  base_seed = normalize_seed(base_seed)
  payload = '\0'.join([str(base_seed)] + [str(component) for component in components])
  digest = hashlib.sha256(payload.encode('utf-8')).digest()
  return int.from_bytes(digest[:8], byteorder='big') % _MAX_SEED


def seed_torch(seed):
  """Seed the CPU generator and every visible CUDA generator."""
  seed = normalize_seed(seed)
  torch.manual_seed(seed)
  if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)
  return seed


def seed_everything(seed, tensorflow_module=None):
  """Seed Python, NumPy, PyTorch, and optionally TensorFlow."""
  seed = normalize_seed(seed)
  random.seed(seed)
  np.random.seed(seed)
  seed_torch(seed)
  if tensorflow_module is not None:
    tensorflow_module.random.set_seed(seed)
  return seed


def configure_torch_backends(deterministic=False, cudnn_benchmark=False,
                             allow_tf32=False,
                             allow_fp16_reduced_precision_reduction=False):
  """Apply explicit backend settings instead of relying on version defaults."""
  deterministic = bool(deterministic)
  torch.backends.cudnn.benchmark = bool(cudnn_benchmark) and not deterministic
  torch.backends.cudnn.deterministic = deterministic
  if hasattr(torch.backends.cudnn, 'allow_tf32'):
    torch.backends.cudnn.allow_tf32 = bool(allow_tf32)
  if hasattr(torch.backends, 'cuda') and hasattr(torch.backends.cuda, 'matmul'):
    torch.backends.cuda.matmul.allow_tf32 = bool(allow_tf32)
    if hasattr(torch.backends.cuda.matmul,
               'allow_fp16_reduced_precision_reduction'):
      torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = bool(
        allow_fp16_reduced_precision_reduction)
  if hasattr(torch, 'use_deterministic_algorithms'):
    torch.use_deterministic_algorithms(deterministic)
  if (deterministic and torch.cuda.is_available() and
      os.environ.get('CUBLAS_WORKSPACE_CONFIG') not in (':4096:8', ':16:8')):
    raise RuntimeError(
      'Deterministic CUDA mode is enabled, but CUBLAS_WORKSPACE_CONFIG was not '
      'set to :4096:8 or :16:8 before Python startup.')


def configure_reproducibility(config, tensorflow_module=None):
  """Activate config.seed and explicit numerical backend settings."""
  seed = seed_everything(config.seed, tensorflow_module=tensorflow_module)
  if (bool(getattr(config, 'deterministic', False)) and
      os.environ.get('PYTHONHASHSEED') != str(seed)):
    raise RuntimeError(
      'config.deterministic=True requires PYTHONHASHSEED to equal config.seed '
      f'before Python startup (expected {seed}).')
  configure_torch_backends(
    deterministic=getattr(config, 'deterministic', False),
    cudnn_benchmark=getattr(config, 'cudnn_benchmark', False),
    allow_tf32=getattr(config, 'allow_tf32', False),
    allow_fp16_reduced_precision_reduction=getattr(
      config, 'allow_fp16_reduced_precision_reduction', False))
  if bool(getattr(config, 'deterministic', False)) and tensorflow_module is not None:
    experimental = getattr(tensorflow_module.config, 'experimental', None)
    enable_determinism = getattr(
      experimental, 'enable_op_determinism', None) if experimental else None
    if enable_determinism is not None:
      enable_determinism()
    elif os.environ.get('TF_DETERMINISTIC_OPS') != '1':
      raise RuntimeError(
        'This TensorFlow version requires TF_DETERMINISTIC_OPS=1 to be set '
        'before Python startup when config.deterministic=True.')
  logging.info(
    'Reproducibility: seed=%d deterministic=%s cudnn_benchmark=%s '
    'allow_tf32=%s allow_fp16_reduced_precision_reduction=%s',
    seed,
    bool(getattr(config, 'deterministic', False)),
    bool(getattr(config, 'cudnn_benchmark', False)) and not bool(
      getattr(config, 'deterministic', False)),
    bool(getattr(config, 'allow_tf32', False)),
    bool(getattr(config, 'allow_fp16_reduced_precision_reduction', False)))
  return seed


def capture_rng_state():
  """Capture portable process RNG state for a training checkpoint."""
  state = {
    'python': random.getstate(),
    'numpy': np.random.get_state(),
    'torch_cpu': torch.get_rng_state().cpu(),
  }
  if torch.cuda.is_available():
    state['torch_cuda'] = [rng_state.cpu() for rng_state in torch.cuda.get_rng_state_all()]
  return state


def restore_rng_state(state):
  """Restore a state produced by capture_rng_state()."""
  if not state:
    return False
  random.setstate(state['python'])
  np.random.set_state(state['numpy'])
  torch.set_rng_state(state['torch_cpu'].cpu())
  cuda_states = state.get('torch_cuda')
  if cuda_states is None:
    return True
  visible_devices = torch.cuda.device_count() if torch.cuda.is_available() else 0
  if len(cuda_states) != visible_devices:
    logging.warning(
      'Checkpoint has RNG state for %d CUDA devices, but %d are visible; '
      'CUDA RNG state was not restored.', len(cuda_states), visible_devices)
    return False
  torch.cuda.set_rng_state_all([rng_state.cpu() for rng_state in cuda_states])
  return True


@contextlib.contextmanager
def isolated_torch_rng(seed):
  """Run diagnostics with a deterministic seed without consuming train RNG."""
  devices = list(range(torch.cuda.device_count())) if torch.cuda.is_available() else []
  with torch.random.fork_rng(devices=devices, enabled=True):
    seed_torch(seed)
    yield
