# coding=utf-8
"""Training NCSN++ on CelebA with exact Fox VP-style SDE."""

from configs.default_celeba_configs import get_default_configs


def get_config():
  config = get_default_configs()

  training = config.training
  training.sde = 'foxvpsde'
  training.continuous = True
  training.reduce_mean = True

  sampling = config.sampling
  sampling.method = 'pc'
  sampling.predictor = 'euler_maruyama'
  sampling.corrector = 'none'

  data = config.data
  data.centered = True

  model = config.model
  model.name = 'ncsnpp'
  model.scale_by_sigma = False
  model.ema_rate = 0.9999
  model.normalization = 'GroupNorm'
  model.nonlinearity = 'swish'
  model.nf = 128
  model.ch_mult = (1, 2, 2, 2)
  model.num_res_blocks = 4
  model.attn_resolutions = (16,)
  model.resamp_with_conv = True
  model.conditional = True
  model.fir = True
  model.fir_kernel = [1, 3, 3, 1]
  model.skip_rescale = True
  model.resblock_type = 'biggan'
  model.progressive = 'none'
  model.progressive_input = 'residual'
  model.progressive_combine = 'sum'
  model.attention_type = 'ddpm'
  model.embedding_type = 'positional'
  model.init_scale = 0.
  model.fourier_scale = 16
  model.conv_size = 3

  model.fox_u = -5.0
  model.fox_diffusion_scale = 1.0
  model.fox_kernel = 'gaussian'
  model.fox_gaussian_sigma = 0.2
  model.fox_power_law_alpha = 1.5
  model.fox_power_law_tau0 = 0.1
  model.fox_matern_length_scale = 0.1
  model.fox_schedule_grid_size = 4096
  model.fox_target_terminal_variance = 1.0

  return config
