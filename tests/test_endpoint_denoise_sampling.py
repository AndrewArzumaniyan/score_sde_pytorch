"""Regression test for the paired endpoint-denoise sampler."""
import pytest
import torch

import sampling
import sde_lib


class _ToySDE:
  N = 2
  T = 1.0

  def prior_sampling(self, shape):
    return torch.randn(shape)

  def sampling_time_grid(self, eps, grid, device, dtype, N):
    assert N == 3
    return torch.tensor([1.0, 0.5, 0.25], device=device, dtype=dtype)

  def marginal_prob(self, x, t):
    return (2.0 - t)[:, None, None, None] * x, 0.1 + t

  def reverse(self, score_fn, probability_flow=False):
    class Reverse:
      N = 2

      def sde(self, x, t):
        return 0.1 * score_fn(x, t), torch.full_like(t, 0.2)

    return Reverse()


def test_endpoint_denoise_uses_noisy_state_and_final_time(monkeypatch):
  times = []

  def get_score_fn(sde, model, train=False, continuous=False):
    assert continuous

    def score(x, t):
      times.append(t.detach().clone())
      return -0.5 * x

    return score

  monkeypatch.setattr(sampling, 'get_score_fn', get_score_fn)
  monkeypatch.setattr(sampling.mutils, 'get_score_fn', get_score_fn)
  kwargs = dict(
    sde=_ToySDE(), shape=(2, 1, 2, 2),
    predictor=sampling.EulerMaruyamaPredictor,
    corrector=sampling.NoneCorrector,
    inverse_scaler=lambda x: x, snr=0.17,
    continuous=True, denoise=True, device='cpu')
  original = sampling.get_pc_sampler(**kwargs)
  paired = sampling.get_pc_sampler(**kwargs, endpoint_variants=True)
  torch.manual_seed(7)
  old_mean, old_nfe = original(None)
  times.clear()
  torch.manual_seed(7)
  outputs, nfe = paired(None)

  assert old_nfe == 2
  assert nfe == 3
  assert set(outputs) == {'x_mean', 'x_t', 'tweedie_xt'}
  assert torch.equal(outputs['x_mean'], old_mean)
  assert len(times) == 3
  assert torch.all(times[-1] == 0.25)
  assert not torch.equal(outputs['x_t'], outputs['x_mean'])
  alpha = 2.0 - 0.25
  sigma = 0.1 + 0.25
  expected = (outputs['x_t'] - 0.5 * sigma ** 2 * outputs['x_t']) / alpha
  torch.testing.assert_close(outputs['tweedie_xt'], expected)


@pytest.mark.parametrize('model', ('vp', 'fox'))
@pytest.mark.parametrize('logsnr_max', (9.115429865459795, 10.575749165617202))
def test_real_sde_endpoint_coefficients(monkeypatch, model, logsnr_max):
  def get_score_fn(sde, network, train=False, continuous=False):
    assert continuous
    return lambda x, t: torch.full_like(x, -0.25)

  monkeypatch.setattr(sampling, 'get_score_fn', get_score_fn)
  monkeypatch.setattr(sampling.mutils, 'get_score_fn', get_score_fn)
  if model == 'vp':
    sde = sde_lib.VPSDE(N=2)
  else:
    sde = sde_lib.FoxVPSDE(
      u=-5.0, kernel='matern_3_2', matern_length_scale=346.4101615138,
      target_terminal_variance=1.0, N=2)
  sde.sampling_logsnr_min = -10.0
  sde.sampling_logsnr_max = logsnr_max
  grid = sde.sampling_time_grid(
    1e-3, grid='uniform_logsnr', device='cpu', dtype=torch.float32, N=3)
  torch.testing.assert_close(
    sde.log_snr(grid[-1:]), torch.tensor([logsnr_max]), atol=1e-3, rtol=0)
  sampler = sampling.get_pc_sampler(
    sde=sde, shape=(2, 1, 2, 2),
    predictor=sampling.EulerMaruyamaPredictor,
    corrector=sampling.NoneCorrector,
    inverse_scaler=lambda x: x, snr=0.17,
    continuous=True, denoise=True, endpoint_variants=True,
    time_grid='uniform_logsnr', device='cpu')
  outputs, nfe = sampler(None)
  assert nfe == 3
  end_t = torch.ones(2) * grid[-1]
  alpha, sigma = sde.marginal_prob(torch.ones((2, 1, 1, 1)), end_t)
  expected = (outputs['x_t'] - 0.25 * sigma[:, None, None, None] ** 2) / alpha
  torch.testing.assert_close(outputs['tweedie_xt'], expected)
