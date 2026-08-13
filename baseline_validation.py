"""Dependency-light regression tests for Cosine-VP and EDM baselines."""

import math
import unittest

import numpy as np
import torch

import edm_lib
import losses
from models.edm import EDMPreconditionedNCSNpp
from models.ema import ExponentialMovingAverage
import sampling
import sde_lib


class ZeroDenoiser(torch.nn.Module):
  def forward(self, x, sigma):
    del sigma
    return torch.zeros_like(x)


class CaptureConditioning(torch.nn.Module):
  def __init__(self):
    super().__init__()
    self.time_cond = None

  def forward(self, x, time_cond):
    self.time_cond = time_cond.detach().clone()
    return torch.zeros_like(x)


class BaselineValidationTest(unittest.TestCase):

  def test_cosine_alpha_bar_matches_improved_ddpm_formula(self):
    sde = sde_lib.CosineVPSDE(N=1000)
    t = torch.tensor([0.0, 0.25, 0.5, 0.9], dtype=torch.float64)
    actual = sde.alpha_bar(t)
    f = torch.cos((t + 0.008) / 1.008 * math.pi / 2.0) ** 2
    f0 = math.cos(0.008 / 1.008 * math.pi / 2.0) ** 2
    expected = f / f0
    self.assertTrue(torch.allclose(actual, expected, atol=1e-12, rtol=1e-12))
    edges = np.linspace(0.0, sde.T, sde.N + 1)
    alpha_edges = np.cos((edges + 0.008) / 1.008 * np.pi / 2.0) ** 2 / f0
    reference_betas = np.minimum(1.0 - alpha_edges[1:] / alpha_edges[:-1], 0.999)
    self.assertTrue(np.allclose(sde.discrete_betas.numpy(), reference_betas, atol=1e-7))

  def test_cosine_marginal_satisfies_vp_variance_ode(self):
    sde = sde_lib.CosineVPSDE(N=1000)
    t = torch.linspace(0.02, 0.98, 1001, dtype=torch.float64)
    variance = 1.0 - sde.alpha_bar(t)
    dt = t[1] - t[0]
    derivative = (variance[2:] - variance[:-2]) / (2.0 * dt)
    expected = sde.beta(t[1:-1]) * (1.0 - variance[1:-1])
    self.assertLess(float(torch.max(torch.abs(derivative - expected))), 2e-3)

  def test_edm_preconditioning_matches_paper(self):
    sigma = torch.tensor([0.01, 0.5, 10.0], dtype=torch.float64)
    c_skip, c_out, c_in, c_noise = edm_lib.EDM.preconditioning(sigma, 0.5)
    self.assertTrue(torch.allclose(c_skip, 0.25 / (sigma ** 2 + 0.25)))
    self.assertTrue(torch.allclose(c_out, sigma * 0.5 / torch.sqrt(sigma ** 2 + 0.25)))
    self.assertTrue(torch.allclose(c_in, 1.0 / torch.sqrt(sigma ** 2 + 0.25)))
    self.assertTrue(torch.allclose(c_noise, torch.log(sigma) / 4.0))

  def test_edm_fourier_conditioning_round_trip(self):
    wrapper = EDMPreconditionedNCSNpp.__new__(EDMPreconditionedNCSNpp)
    torch.nn.Module.__init__(wrapper)
    wrapper.sigma_data = 0.5
    wrapper.model = CaptureConditioning()
    x = torch.randn(3, 2, 4, 4)
    sigma = torch.tensor([0.002, 0.5, 80.0])
    wrapper(x, sigma)

    # NCSN++ applies log() to this value before its Fourier projection.
    recovered_c_noise = torch.log(wrapper.model.time_cond)
    expected_c_noise = torch.log(sigma) / 4.0
    self.assertTrue(torch.allclose(
      recovered_c_noise, expected_c_noise, atol=1e-7, rtol=1e-7))

  def test_edm_loss_matches_official_formula(self):
    edm = edm_lib.EDM()
    batch = torch.randn(3, 2, 4, 4)
    torch.manual_seed(91)
    actual = losses.get_edm_loss_fn(
      edm, train=True, reduce_mean=False)(ZeroDenoiser(), batch)

    torch.manual_seed(91)
    rnd_normal = torch.randn(batch.shape[0])
    sigma = torch.exp(rnd_normal * edm.p_std + edm.p_mean)
    noise = torch.randn_like(batch) * sigma[:, None, None, None]
    del noise  # ZeroDenoiser makes the noisy input irrelevant to the output.
    weight = ((sigma ** 2 + edm.sigma_data ** 2)
              / (sigma * edm.sigma_data) ** 2)
    weighted_error = weight[:, None, None, None] * batch.square()
    expected = torch.sum(weighted_error) / batch.shape[0]
    self.assertTrue(torch.equal(actual, expected))

  def test_edm_karras_grid_and_nfe(self):
    edm = edm_lib.EDM(N=5)
    levels = edm.noise_levels('cpu')
    self.assertAlmostEqual(float(levels[0]), 80.0, places=10)
    self.assertAlmostEqual(float(levels[-2]), 0.002, places=10)
    self.assertEqual(float(levels[-1]), 0.0)
    self.assertTrue(torch.all(levels[:-1][1:] < levels[:-1][:-1]))

    sampler = sampling.get_edm_sampler(
      edm, (2, 1, 4, 4), inverse_scaler=lambda x: x, device='cpu')
    samples, nfe = sampler(ZeroDenoiser())
    self.assertEqual(samples.shape, (2, 1, 4, 4))
    self.assertTrue(torch.isfinite(samples).all())
    self.assertEqual(nfe, 2 * edm.N - 1)
    self.assertLess(float(torch.max(torch.abs(samples))), 1e-5)

  def test_existing_ema_update_is_unchanged_without_schedule(self):
    parameter = torch.nn.Parameter(torch.tensor([2.0]))
    ema = ExponentialMovingAverage([parameter], decay=0.9)
    initial_shadow = ema.shadow_params[0].clone()
    ema.update([parameter])
    legacy_decay = min(0.9, 2.0 / 11.0)
    expected = initial_shadow - (1.0 - legacy_decay) * (initial_shadow - parameter)
    self.assertTrue(torch.equal(ema.shadow_params[0], expected))
    self.assertEqual(ema.num_updates, 1)

  def test_scheduled_ema_matches_edm_half_life(self):
    parameter = torch.nn.Parameter(torch.tensor([2.0]))
    ema = ExponentialMovingAverage([parameter], decay=0.9999)
    parameter.data.fill_(4.0)
    decay = 0.5 ** (128.0 / 500000.0)
    ema.update([parameter], decay=decay)
    expected = torch.tensor([2.0]) - (1.0 - decay) * (torch.tensor([2.0]) - parameter)
    self.assertTrue(torch.allclose(ema.shadow_params[0], expected))
    self.assertEqual(ema.num_updates, 0)


if __name__ == '__main__':
  unittest.main()
