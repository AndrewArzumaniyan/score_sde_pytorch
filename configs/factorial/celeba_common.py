"""Shared constants and controls for the CelebA VP/FOX factorial study."""


# Intersection of the VP and normalized-FOX log-SNR ranges.  The upper end is
# lambda_VP(1e-3) for beta(t)=0.1+19.9t; the lower end deliberately avoids the
# terminal rounding difference between the two processes.
LOGSNR_MIN = -10.0
LOGSNR_MAX = 9.115429865459795
KAPPA1000_ELL = 346.4101615138


def apply_factorial_controls(config, arm, noise_distribution):
  """Apply controls shared by all factorial arms without changing architecture."""
  config.training.factorial_arm = arm
  config.training.noise_distribution = noise_distribution

  # Parameters of the two clocks that induce p(lambda).  They are explicit in
  # every config/manifest, including when the clock matches the target SDE.
  config.training.noise_vp_beta_min = 0.1
  config.training.noise_vp_beta_max = 20.0
  config.training.noise_fox_u = -5.0
  config.training.noise_fox_drift_schedule = 'constant'
  config.training.noise_fox_beta_min = 0.1
  config.training.noise_fox_beta_max = 20.0
  config.training.noise_fox_diffusion_scale = 1.0
  config.training.noise_fox_kernel = 'matern_3_2'
  config.training.noise_fox_gaussian_sigma = 0.2
  config.training.noise_fox_power_law_alpha = 1.5
  config.training.noise_fox_power_law_tau0 = 0.1
  config.training.noise_fox_matern_length_scale = KAPPA1000_ELL
  config.training.noise_fox_drift_k_bar = -5.025
  config.training.noise_fox_drift_a = 0.0
  config.training.noise_fox_schedule_grid_size = 8192
  config.training.noise_fox_target_terminal_variance = 1.0

  # Common model input. lambda_max maps to label 0 and lambda_min to 999,
  # preserving the low-noise -> high-noise direction of the original t label.
  config.model.noise_conditioning = 'logsnr'
  config.model.logsnr_min = LOGSNR_MIN
  config.model.logsnr_max = LOGSNR_MAX

  # Publication evaluation defaults. Scripts may vary only NFE/grid for the
  # sampler ablation while keeping these common endpoints fixed.
  config.sampling.time_grid = 'uniform_logsnr'
  config.sampling.logsnr_min = LOGSNR_MIN
  config.sampling.logsnr_max = LOGSNR_MAX
  return config
