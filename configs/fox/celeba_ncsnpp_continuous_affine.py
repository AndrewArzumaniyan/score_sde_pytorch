# coding=utf-8
"""Affine-drift shape screen: k_a(t) = k_bar + a (t - 1/2), kernel matern_3_2
kappa=1000, CelebA.

int_0^1 k_a = k_bar for any a, so terminal alpha(1) / lambda(1) are pinned and
`a` only redistributes contraction along the path:
  a = 0    -> constant drift (the current CelebA winner)
  a < 0    -> back-loaded contraction (a = -9.95 == vp_linear, the known failure)
  a > 0    -> front-loaded / 'reversed' contraction  (untested region)

Set the screen point on the CLI, e.g.:
  --config.model.fox_drift_a=2.5
  --config.model.fox_drift_a=5.0
"""

from configs.fox.celeba_ncsnpp_continuous_matern_3_2 import get_config as get_base_config


def get_config():
  config = get_base_config()
  model = config.model
  model.fox_drift_schedule = 'affine'
  model.fox_drift_k_bar = -5.025
  model.fox_drift_a = 2.5           # override per screen point on the CLI
  model.fox_kernel = 'matern_3_2'
  # kappa = |u| * ell / sqrt(3) = 1000  ->  ell = 1000 * sqrt(3) / 5
  # (|u| is nominal here; with affine drift there is no single u -- the label
  #  is kept only so the kernel length scale matches the a = 0 winner.)
  model.fox_matern_length_scale = 346.4101615138
  model.fox_target_terminal_variance = 1.0
  model.fox_schedule_grid_size = 8192
  return config
