"""Arm B: VP forward process with p(lambda) induced by uniform FOX time."""

from configs.factorial.celeba_common import apply_factorial_controls
from configs.vp.celeba_ncsnpp_continuous import get_config as get_base_config


def get_config():
  return apply_factorial_controls(
    get_base_config(), 'B_vp_pfox', 'normalized_fox_time')
