"""Validation tests for the controlled VP/normalized-FOX factorial experiment."""

import unittest

import torch

import losses
from configs.factorial import celeba_a_vp_pvp
from configs.factorial import celeba_b_vp_pfox
from configs.factorial import celeba_c_normfox_pvp
from configs.factorial import celeba_d_normfox_pfox
from configs.factorial import celeba_e_realfox_pfox
from configs.factorial.celeba_common import LOGSNR_MAX, LOGSNR_MIN
from models import utils as mutils
import sampling
import sde_lib


def make_vp():
  return sde_lib.VPSDE(beta_min=0.1, beta_max=20.0, N=1000)


def make_normalized_fox(normalize_scale=True):
  return sde_lib.FoxVPSDE(
    u=-5.0,
    drift_schedule='constant',
    beta_min=0.1,
    beta_max=20.0,
    diffusion_scale=1.0,
    kernel='matern_3_2',
    gaussian_sigma=0.2,
    power_law_alpha=1.5,
    power_law_tau0=0.1,
    matern_length_scale=346.4101615138,
    drift_k_bar=-5.025,
    drift_a=0.0,
    normalize_scale=normalize_scale,
    schedule_grid_size=8192,
    target_terminal_variance=1.0,
    N=1000)


class LabelRecorder(torch.nn.Module):

  def __init__(self):
    super().__init__()
    self.labels = None

  def forward(self, x, labels):
    self.labels = labels.detach().clone()
    return torch.zeros_like(x)


class FactorialValidationTest(unittest.TestCase):

  @classmethod
  def setUpClass(cls):
    cls.vp = make_vp()
    cls.fox = make_normalized_fox()

  def test_arm_matrix_and_geometry_control(self):
    configs = [
      celeba_a_vp_pvp.get_config(), celeba_b_vp_pfox.get_config(),
      celeba_c_normfox_pvp.get_config(),
      celeba_d_normfox_pfox.get_config(),
      celeba_e_realfox_pfox.get_config()]
    self.assertEqual(
      [config.training.noise_distribution for config in configs],
      ['vp_time', 'normalized_fox_time', 'vp_time',
       'normalized_fox_time', 'normalized_fox_time'])
    self.assertEqual(
      [config.training.sde for config in configs],
      ['vpsde', 'vpsde', 'foxvpsde', 'foxvpsde', 'foxvpsde'])
    for config in configs:
      self.assertEqual(config.model.noise_conditioning, 'logsnr')
      self.assertEqual(config.model.logsnr_min, LOGSNR_MIN)
      self.assertEqual(config.model.logsnr_max, LOGSNR_MAX)
      self.assertEqual(config.sampling.time_grid, 'uniform_logsnr')
      self.assertEqual(config.sampling.logsnr_min, LOGSNR_MIN)
      self.assertEqual(config.sampling.logsnr_max, LOGSNR_MAX)
    self.assertTrue(configs[2].model.fox_normalize_scale)
    self.assertTrue(configs[3].model.fox_normalize_scale)
    self.assertFalse(configs[4].model.fox_normalize_scale)

  def test_network_architecture_is_identical_in_all_arms(self):
    configs = [
      celeba_a_vp_pvp.get_config(), celeba_b_vp_pfox.get_config(),
      celeba_c_normfox_pvp.get_config(),
      celeba_d_normfox_pfox.get_config(),
      celeba_e_realfox_pfox.get_config()]
    architecture_fields = (
      'name', 'nf', 'ch_mult', 'num_res_blocks', 'attn_resolutions',
      'dropout', 'embedding_type', 'normalization', 'nonlinearity',
      'resblock_type', 'progressive', 'progressive_input',
      'progressive_combine', 'attention_type', 'fir', 'fir_kernel',
      'skip_rescale', 'scale_by_sigma', 'conv_size')
    reference = tuple(getattr(configs[0].model, key)
                      for key in architecture_fields)
    for config in configs[1:]:
      self.assertEqual(
        tuple(getattr(config.model, key) for key in architecture_fields),
        reference)

  def test_logsnr_inverse_round_trip(self):
    targets = torch.linspace(LOGSNR_MIN, LOGSNR_MAX, 101, dtype=torch.float64)
    for sde in (self.vp, self.fox):
      times = sde.time_from_log_snr(targets)
      recovered = sde.log_snr(times)
      torch.testing.assert_close(recovered, targets, rtol=0.0, atol=2e-8)
      self.assertTrue(bool(torch.all(times[1:] < times[:-1])))

  def test_common_logsnr_grid_has_common_physical_levels(self):
    grids = []
    for sde in (self.vp, self.fox):
      sde.sampling_logsnr_min = LOGSNR_MIN
      sde.sampling_logsnr_max = LOGSNR_MAX
      times = sde.sampling_time_grid(
        1e-3, grid='uniform_logsnr', dtype=torch.float64, N=101)
      grids.append(sde.log_snr(times))
    torch.testing.assert_close(grids[0], grids[1], rtol=0.0, atol=2e-8)

  def test_normalized_marginals_match_at_equal_logsnr(self):
    targets = torch.linspace(LOGSNR_MIN, LOGSNR_MAX, 31, dtype=torch.float64)
    zeros = torch.zeros(31, 1, 1, 1, dtype=torch.float64)
    vp_t = self.vp.time_from_log_snr(targets)
    fox_t = self.fox.time_from_log_snr(targets)
    vp_mean, vp_std = self.vp.marginal_prob(zeros, vp_t)
    fox_mean, fox_std = self.fox.marginal_prob(zeros, fox_t)
    torch.testing.assert_close(vp_mean, fox_mean, rtol=0.0, atol=0.0)
    torch.testing.assert_close(vp_std, fox_std, rtol=2e-6, atol=2e-7)
    vp_alpha = self.vp.marginal_prob(torch.ones_like(zeros), vp_t)[0]
    fox_alpha = self.fox.marginal_prob(torch.ones_like(zeros), fox_t)[0]
    torch.testing.assert_close(vp_alpha, fox_alpha, rtol=2e-6, atol=2e-7)

  def test_cross_sde_sampling_preserves_drawn_logsnr(self):
    for source in (self.vp, self.fox):
      torch.manual_seed(123)
      vp_t = losses.sample_sde_time(
        self.vp, 256, torch.device('cpu'), 1e-3,
        noise_distribution_sde=source,
        logsnr_min=LOGSNR_MIN, logsnr_max=LOGSNR_MAX,
        dtype=torch.float64)
      torch.manual_seed(123)
      fox_t = losses.sample_sde_time(
        self.fox, 256, torch.device('cpu'), 1e-3,
        noise_distribution_sde=source,
        logsnr_min=LOGSNR_MIN, logsnr_max=LOGSNR_MAX,
        dtype=torch.float64)
      torch.testing.assert_close(
        self.vp.log_snr(vp_t), self.fox.log_snr(fox_t),
        rtol=0.0, atol=2e-8)

  def test_logsnr_conditioning_is_process_independent(self):
    target = torch.tensor([LOGSNR_MAX, 2.0, -3.0, LOGSNR_MIN])
    labels = []
    x = torch.zeros(4, 1, 1, 1)
    for sde in (self.vp, self.fox):
      sde.noise_conditioning = 'logsnr'
      sde.conditioning_logsnr_min = LOGSNR_MIN
      sde.conditioning_logsnr_max = LOGSNR_MAX
      recorder = LabelRecorder()
      score_fn = mutils.get_score_fn(sde, recorder, continuous=True)
      score_fn(x, sde.time_from_log_snr(target))
      labels.append(recorder.labels)
    torch.testing.assert_close(labels[0], labels[1], rtol=0.0, atol=2e-4)
    torch.testing.assert_close(
      labels[0][[0, -1]], torch.tensor([0.0, 999.0]),
      rtol=0.0, atol=2e-4)

  def test_legacy_time_conditioning_is_unchanged(self):
    recorder = LabelRecorder()
    t = torch.tensor([0.001, 0.25, 1.0])
    score_fn = mutils.get_score_fn(self.vp, recorder, continuous=True)
    score_fn(torch.zeros(3, 1, 1, 1), t)
    torch.testing.assert_close(recorder.labels, t * 999)

  def test_legacy_uniform_time_grid_is_unchanged(self):
    expected = torch.linspace(1.0, 1e-3, 17)
    actual = make_vp().sampling_time_grid(1e-3, 'uniform_time', N=17)
    torch.testing.assert_close(actual, expected, rtol=0.0, atol=0.0)

  def test_common_endpoint_pc_sampler_smoke(self):
    for sde in (make_vp(), make_normalized_fox()):
      sde.N = 4
      sde.noise_conditioning = 'logsnr'
      sde.conditioning_logsnr_min = LOGSNR_MIN
      sde.conditioning_logsnr_max = LOGSNR_MAX
      sde.sampling_logsnr_min = LOGSNR_MIN
      sde.sampling_logsnr_max = LOGSNR_MAX
      sample_fn = sampling.get_pc_sampler(
        sde, (2, 3, 4, 4), sampling.EulerMaruyamaPredictor,
        sampling.NoneCorrector, lambda value: value, 0.17,
        continuous=True, denoise=False, time_grid='uniform_logsnr',
        eps=1e-3, device='cpu')
      samples, nfe = sample_fn(LabelRecorder())
      self.assertEqual(samples.shape, (2, 3, 4, 4))
      self.assertEqual(nfe, 4)
      self.assertTrue(bool(torch.isfinite(samples).all()))


if __name__ == '__main__':
  unittest.main()
