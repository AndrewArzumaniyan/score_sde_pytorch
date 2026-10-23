"""Native CelebA VP/FOX endpoint screen on the final 1.3M checkpoints."""
from configs.vp.celeba_ncsnpp_continuous import get_config as get_vp
from configs.fox.celeba_ncsnpp_continuous_matern_3_2 import get_config as get_fox

# Six new points; common/extended anchors already have saved 50k features.
ENDPOINTS = (5, 6, 7, 8, 10, 12)


def get_config(name):
    model, endpoint = name.split('_')
    endpoint = int(endpoint)
    if endpoint not in ENDPOINTS:
        raise ValueError(f'Unexpected screen endpoint: {endpoint}')
    if model == 'vp':
        config = get_vp()
    elif model == 'fox':
        config = get_fox()
        config.model.fox_u = -5.0
        config.model.fox_drift_schedule = 'constant'
        config.model.fox_normalize_scale = False
        config.model.fox_matern_length_scale = 346.4101615138
        config.model.fox_target_terminal_variance = 1.0
    else:
        raise ValueError(model)
    config.sampling.time_grid = 'uniform_logsnr'
    config.sampling.logsnr_min = -10.0
    config.sampling.logsnr_max = float(endpoint)
    config.sampling.noise_removal = True
    config.sampling.probability_flow = False
    config.eval.endpoint_denoise_variants = True
    return config
