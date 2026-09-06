"""Arm C: normalized-FOX forward process with uniform-VP-time p(lambda)."""

from configs.factorial.celeba_common import apply_factorial_controls
from configs.fox.celeba_ncsnpp_continuous_normalized import get_config as get_base_config


def get_config():
  return apply_factorial_controls(
    get_base_config(), 'C_normfox_pvp', 'vp_time')
