import importlib
import os
import sys
import types
import unittest

import torch

os.environ.setdefault('TORCH_EXTENSIONS_DIR', '/tmp/torch_extensions')
os.environ.setdefault('XDG_CACHE_HOME', '/tmp')


def _install_ml_collections_stub():
  if 'ml_collections' in sys.modules:
    return

  ml_collections = types.ModuleType('ml_collections')

  class ConfigDict(dict):
    def __getattr__(self, name):
      try:
        return self[name]
      except KeyError as exc:
        raise AttributeError(name) from exc

    def __setattr__(self, name, value):
      self[name] = value

  ml_collections.ConfigDict = ConfigDict
  sys.modules['ml_collections'] = ml_collections


def _get_sde(config):
  import sde_lib

  sde_name = config.training.sde.lower()
  if sde_name == 'vpsde':
    return sde_lib.VPSDE(
      beta_min=config.model.beta_min,
      beta_max=config.model.beta_max,
      N=config.model.num_scales,
    ), 1e-3
  if sde_name == 'vesde':
    return sde_lib.VESDE(
      sigma_min=config.model.sigma_min,
      sigma_max=config.model.sigma_max,
      N=config.model.num_scales,
    ), 1e-5
  if sde_name == 'foxvpsde':
    return sde_lib.FoxVPSDE(
      u=config.model.fox_u,
      diffusion_scale=config.model.fox_diffusion_scale,
      kernel=config.model.fox_kernel,
      gaussian_sigma=config.model.fox_gaussian_sigma,
      power_law_alpha=config.model.fox_power_law_alpha,
      power_law_tau0=config.model.fox_power_law_tau0,
      matern_length_scale=config.model.fox_matern_length_scale,
      schedule_grid_size=config.model.fox_schedule_grid_size,
      target_terminal_variance=getattr(config.model, 'fox_target_terminal_variance', None),
      N=config.model.num_scales,
    ), 1e-3
  raise ValueError(f'Unsupported SDE: {config.training.sde}')


class CelebaConfigSmokeTest(unittest.TestCase):

  @classmethod
  def setUpClass(cls):
    _install_ml_collections_stub()
    from models import ddpm, ncsnv2, ncsnpp  # noqa: F401

  def test_new_celeba_configs_build_and_step(self):
    import sampling
    import sde_lib
    from models import utils as mutils

    modules = [
      'configs.ve.celeba_ncsnpp',
      'configs.ve.celeba_ncsnpp_continuous',
      'configs.vp.celeba_ncsnpp_continuous',
      'configs.fox.celeba_ncsnpp_continuous',
      'configs.fox.celeba_ncsnpp_continuous_matern_3_2',
    ]

    for module_name in modules:
      with self.subTest(module=module_name):
        module = importlib.import_module(module_name)
        config = module.get_config()
        config.device = torch.device('cpu')

        self.assertEqual(config.sampling.time_grid, 'uniform_time')
        self.assertEqual(config.eval.sampling_num_scales, 0)

        sde, eps = _get_sde(config)
        model = mutils.create_model(config)
        predictor = sampling.get_predictor(config.sampling.predictor.lower())
        corrector = sampling.get_corrector(config.sampling.corrector.lower())

        shape = (1, config.data.num_channels, config.data.image_size, config.data.image_size)
        x = sde.prior_sampling(shape).to(config.device)
        t = torch.full((shape[0],), 0.5, device=config.device)
        if isinstance(sde, sde_lib.FoxVPSDE):
          time_grid = sde.sampling_time_grid(
            eps, grid=config.sampling.time_grid, device=config.device, dtype=torch.float32, N=2)
        else:
          time_grid = torch.linspace(sde.T, eps, 2, device=config.device)
        dt = time_grid[1] - time_grid[0]

        x, _ = sampling.shared_corrector_update_fn(
          x=x,
          t=t,
          sde=sde,
          model=model,
          corrector=corrector,
          continuous=config.training.continuous,
          snr=config.sampling.snr,
          n_steps=config.sampling.n_steps_each,
        )
        x, _ = sampling.shared_predictor_update_fn(
          x=x,
          t=t,
          sde=sde,
          model=model,
          predictor=predictor,
          probability_flow=config.sampling.probability_flow,
          continuous=config.training.continuous,
          dt=dt,
        )

        self.assertEqual(x.shape, shape)
        self.assertTrue(torch.isfinite(x).all().item())


if __name__ == '__main__':
  unittest.main()
