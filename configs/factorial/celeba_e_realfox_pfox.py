"""Geometry control: real FOX forward process with uniform-FOX-time p(lambda)."""

from configs.factorial.celeba_common import (
  KAPPA1000_ELL, apply_factorial_controls)
from configs.fox.celeba_ncsnpp_continuous_matern_3_2 import get_config as get_base_config


def get_config():
  config = get_base_config()
  config.model.fox_u = -5.0
  config.model.fox_drift_schedule = 'constant'
  config.model.fox_matern_length_scale = KAPPA1000_ELL
  config.model.fox_target_terminal_variance = 1.0
  config.model.fox_normalize_scale = False
  config.model.fox_schedule_grid_size = 8192
  return apply_factorial_controls(
    config, 'E_realfox_pfox', 'normalized_fox_time')
