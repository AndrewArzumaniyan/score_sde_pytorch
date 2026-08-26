"""Training NCSN++ on AFHQv2-64 with the Gaussian FOX SDE."""

from configs.default_afhqv2_configs import apply_afhqv2_data
from configs.fox.celeba_ncsnpp_continuous import get_config as get_celeba_config


def get_config():
  return apply_afhqv2_data(get_celeba_config())
