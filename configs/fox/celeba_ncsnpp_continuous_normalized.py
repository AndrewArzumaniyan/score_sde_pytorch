# coding=utf-8
"""E4 'normalized-FOX': the CelebA kappa=1000 winner's log-SNR path, rescaled to
alpha^2 + q = 1 (VP-type overall scale r(t) == 1).

Isolates whether the CelebA FOX gain comes from the log-SNR *shape* lambda(t)
(shared with real-FOX) or from the FOX overall-scale path r_F(t) (which sags to
~0.71 mid-path).  Compare against: VP (lambda_VP, r == 1) and real-FOX
kappa=1000 (lambda_FOX, r_F(t)).
"""

from configs.fox.celeba_ncsnpp_continuous_matern_3_2 import get_config as get_base_config


def get_config():
  config = get_base_config()
  model = config.model
  model.fox_drift_schedule = 'constant'
  model.fox_u = -5.0
  model.fox_kernel = 'matern_3_2'
  # kappa = |u| * ell / sqrt(3) = 1000  ->  ell = 1000 * sqrt(3) / 5
  model.fox_matern_length_scale = 346.4101615138
  model.fox_target_terminal_variance = 1.0
  model.fox_normalize_scale = True
  model.fox_schedule_grid_size = 8192
  return config
