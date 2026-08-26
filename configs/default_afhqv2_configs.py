"""Shared defaults for unconditional AFHQv2 at 64x64."""

from configs.default_celeba_configs import get_default_configs as get_celeba_defaults


def apply_afhqv2_data(config):
  """Replace the CelebA data contract while preserving common 64x64 defaults."""
  config.data.dataset = 'AFHQV2'
  config.data.afhqv2_dir = 'datasets/afhqv2-64x64'
  config.data.afhqv2_train_take = -1
  config.data.afhqv2_eval_take = -1
  config.data.image_size = 64
  config.data.num_channels = 3
  config.data.random_flip = True
  config.data.uniform_dequantization = False
  config.data.centered = False
  for field in ('celeba_dir', 'celeba_train_take', 'celeba_validation_take'):
    if field in config.data:
      del config.data[field]
  return config


def get_default_configs():
  return apply_afhqv2_data(get_celeba_defaults())
