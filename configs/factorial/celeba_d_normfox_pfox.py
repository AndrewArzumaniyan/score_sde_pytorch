"""Arm D: normalized-FOX forward process with uniform-FOX-time p(lambda)."""

from configs.factorial.celeba_common import apply_factorial_controls
from configs.fox.celeba_ncsnpp_continuous_normalized import get_config as get_base_config


def get_config():
  return apply_factorial_controls(
    get_base_config(), 'D_normfox_pfox', 'normalized_fox_time')
