# coding=utf-8
"""Training NCSN++ on CIFAR-10 with exact Fox Matérn-1/2 kernel."""
from configs.fox.cifar10_ncsnpp_continuous import get_config as get_base_config


def get_config():
  config = get_base_config()
  config.model.fox_kernel = 'matern_1_2'
  config.model.fox_matern_length_scale = 0.1
  return config
