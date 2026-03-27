from configs.fox.cifar10_ncsnpp_continuous import get_config as get_base_config


def get_config():
  config = get_base_config()
  config.model.fox_kernel = 'matern_3_2'
  config.model.fox_matern_length_scale = 0.1
  return config
