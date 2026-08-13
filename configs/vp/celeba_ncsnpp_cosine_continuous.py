"""NCSN++ on CelebA with the Improved-DDPM cosine VP schedule."""

from configs.vp.celeba_ncsnpp_continuous import get_config as get_base_config


def get_config():
  config = get_base_config()
  config.training.sde = 'cosinevpsde'
  config.model.cosine_s = 0.008
  config.model.cosine_t_max = 0.999
  config.model.cosine_max_beta = 0.999
  config.eval.enable_sampling = True
  return config
