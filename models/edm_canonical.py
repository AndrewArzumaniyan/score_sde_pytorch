"""Adapter for the unmodified official NVLabs EDM implementation.

The external source is CC BY-NC-SA 4.0 and is intentionally not vendored into
this Apache-licensed repository. Install the pinned checkout with
``tools/install_official_edm.sh`` before constructing this model.
"""

import importlib
import os
import subprocess
import sys

import torch
import torch.nn as nn

from . import utils


OFFICIAL_EDM_COMMIT = '008a4e5316c8e3bfe61a62f874bddba254295afb'


def _load_official_modules(config):
  configured_root = getattr(config.model, 'edm_official_root', 'third_party/edm')
  root = os.environ.get('EDM_OFFICIAL_ROOT', configured_root)
  root = os.path.abspath(os.path.expanduser(root))
  required = (
    os.path.join(root, 'training', 'networks.py'),
    os.path.join(root, 'training', 'augment.py'),
    os.path.join(root, 'torch_utils', 'persistence.py'),
  )
  missing = [path for path in required if not os.path.isfile(path)]
  if missing:
    raise FileNotFoundError(
      'Official NVLabs EDM checkout is missing. Run '
      '`tools/install_official_edm.sh` or set EDM_OFFICIAL_ROOT. Missing: '
      + ', '.join(missing))
  expected_commit = getattr(
    config.model, 'edm_official_commit', OFFICIAL_EDM_COMMIT)
  try:
    actual_commit = subprocess.check_output(
      ['git', '-c', f'safe.directory={root}', '-C', root,
       'rev-parse', 'HEAD'], text=True,
      stderr=subprocess.DEVNULL).strip()
  except (OSError, subprocess.CalledProcessError) as error:
    raise RuntimeError(
      'The official EDM source must be a Git checkout so its revision can be '
      f'verified: {root}') from error
  if actual_commit != expected_commit:
    raise RuntimeError(
      f'Official EDM checkout is at {actual_commit}; expected '
      f'{expected_commit}. Re-run `tools/install_official_edm.sh`.')
  dirty_files = subprocess.check_output(
    ['git', '-c', f'safe.directory={root}', '-C', root, 'status',
     '--porcelain', '--untracked-files=no'], text=True).strip()
  if dirty_files:
    raise RuntimeError(
      'Official EDM tracked sources have local modifications; pinned parity '
      f'cannot be guaranteed: {root}\n{dirty_files}')
  if root not in sys.path:
    sys.path.insert(0, root)
  networks = importlib.import_module('training.networks')
  augment = importlib.import_module('training.augment')
  for module in (networks, augment):
    module_path = os.path.realpath(module.__file__)
    if not module_path.startswith(os.path.realpath(root) + os.sep):
      raise ImportError(
        'A different `training` package was imported before official EDM: '
        + module_path)
  return networks, augment


@utils.register_model(name='edm_canonical_songunet')
class CanonicalEDMSongUNet(nn.Module):
  """Official EDMPrecond + DDPM++-style SongUNet + paper augmentation."""

  def __init__(self, config):
    super().__init__()
    networks, augment = _load_official_modules(config)
    augment_dim = 9 if config.training.edm_augmentation else 0
    self.network = networks.EDMPrecond(
      img_resolution=config.data.image_size,
      img_channels=config.data.num_channels,
      label_dim=0,
      use_fp16=getattr(config.model, 'edm_use_fp16', False),
      sigma_min=0.0,
      sigma_max=float('inf'),
      sigma_data=config.model.edm_sigma_data,
      model_type='SongUNet',
      augment_dim=augment_dim,
      model_channels=config.model.edm_model_channels,
      channel_mult=list(config.model.edm_channel_mult),
      channel_mult_emb=4,
      num_blocks=config.model.edm_num_blocks,
      attn_resolutions=list(config.model.edm_attn_resolutions),
      dropout=config.model.dropout,
      embedding_type='positional',
      channel_mult_noise=1,
      encoder_type='standard',
      decoder_type='standard',
      resample_filter=[1, 1],
    )
    self.augment_pipe = None
    if config.training.edm_augmentation:
      self.augment_pipe = augment.AugmentPipe(
        p=config.training.edm_augment_probability,
        xflip=1e8,
        yflip=1,
        scale=1,
        rotate_frac=1,
        aniso=1,
        translate_frac=1,
      )

  def augment(self, images):
    if self.augment_pipe is None:
      return images, None
    return self.augment_pipe(images)

  def forward(self, x, sigma, augment_labels=None):
    if not torch.is_tensor(sigma):
      sigma = torch.as_tensor(sigma, device=x.device, dtype=x.dtype)
    sigma = sigma.to(device=x.device, dtype=x.dtype)
    if sigma.ndim == 0:
      sigma = sigma.expand(x.shape[0])
    return self.network(x, sigma, class_labels=None,
                        augment_labels=augment_labels)
