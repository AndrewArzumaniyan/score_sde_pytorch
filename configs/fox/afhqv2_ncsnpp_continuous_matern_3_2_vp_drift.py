"""AFHQv2 Matern-3/2 FOX candidate with the same drift as linear VP."""

from configs.fox.afhqv2_ncsnpp_continuous_matern_3_2 import get_config as get_base_config


def get_config():
  config = get_base_config()
  config.model.fox_u = -5.0
  config.model.fox_drift_schedule = 'vp_linear'
  config.model.fox_beta_min = 0.1
  config.model.fox_beta_max = 20.0
  # kappa=1000 under ell = kappa * sqrt(3) / abs(u).
  config.model.fox_matern_length_scale = 346.4101615138
  config.model.fox_target_terminal_variance = 1.0
  return config
