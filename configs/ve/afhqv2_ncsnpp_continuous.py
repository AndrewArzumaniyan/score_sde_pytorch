"""Continuous VE/NCSN++ transfer baseline on AFHQv2-64.

This deliberately reuses the published repository's 64x64 CelebA VE recipe;
it is a controlled transfer baseline, not a separately tuned AFHQv2 recipe.
"""

from configs.default_afhqv2_configs import apply_afhqv2_data
from configs.ve.celeba_ncsnpp_continuous import get_config as get_celeba_config


def get_config():
  return apply_afhqv2_data(get_celeba_config())
