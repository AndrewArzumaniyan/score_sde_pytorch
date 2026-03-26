# coding=utf-8
"""Training NCSN++ on CIFAR-10 with exact Fox power-law kernel."""
from configs.fox.cifar10_ncsnpp_continuous import get_config as get_base_config


def get_config():
  config = get_base_config()
  config.model.fox_kernel = 'power_law'
  config.model.fox_power_law_alpha = 1.5
  config.model.fox_power_law_tau0 = 0.1
  return config

