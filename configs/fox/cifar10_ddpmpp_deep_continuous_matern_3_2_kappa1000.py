"""Real FOX counterpart of the matched, eight-block DDPM++ continuous VP.

The network, optimizer, batch, loss, EMA, budget and evaluation preset are
inherited from the matched VP config. Only the forward SDE and its parameters
change: Matérn-3/2, constant drift -5, ell=346.4101615138 (tau=200, kappa=1000),
q(T)=1, no normalization of the marginal scale, schedule cache size 4096.
These are the historical real-FOX parameters, not the base FOX ell=0.1.

Use a new workdir; old four-block NCSN++ checkpoints cannot be continued here.
As for VP, :eval selects common log-SNR endpoints and explicit final Tweedie.
"""

from configs.vp.cifar10_ddpmpp_deep_continuous_matched import get_config as get_vp_config


def get_config(name=None):
  config = get_vp_config(name)
  config.training.sde = 'foxvpsde'

  model = config.model
  model.fox_u = -5.0
  model.fox_drift_schedule = 'constant'
  model.fox_beta_min = model.beta_min
  model.fox_beta_max = model.beta_max
  # The SDE calibrates the effective noise amplitude to q(T)=1.
  model.fox_diffusion_scale = 1.0
  model.fox_target_terminal_variance = 1.0
  model.fox_normalize_scale = False
  model.fox_kernel = 'matern_3_2'
  model.fox_matern_length_scale = 346.4101615138
  model.fox_schedule_grid_size = 4096
  # Valid unused-kernel defaults required by get_sde/FoxVPSDE validation.
  model.fox_gaussian_sigma = 0.2
  model.fox_power_law_alpha = 1.5
  model.fox_power_law_tau0 = 0.1

  return config
