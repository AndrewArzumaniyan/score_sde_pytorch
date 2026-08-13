"""EDM design-space formulation from Karras et al. (NeurIPS 2022)."""

import math

import torch


class EDM:
  """Configuration and prior for EDM training and Algorithm 2 sampling."""

  def __init__(self,
               sigma_data=0.5,
               p_mean=-1.2,
               p_std=1.2,
               sigma_min=0.002,
               sigma_max=80.0,
               rho=7.0,
               s_churn=0.0,
               s_min=0.0,
               s_max=float('inf'),
               s_noise=1.0,
               augmentation=False,
               N=18):
    if sigma_data <= 0.0:
      raise ValueError('EDM sigma_data must be positive.')
    if p_std <= 0.0:
      raise ValueError('EDM p_std must be positive.')
    if sigma_min <= 0.0 or sigma_max <= sigma_min:
      raise ValueError('EDM requires 0 < sigma_min < sigma_max.')
    if rho <= 0.0:
      raise ValueError('EDM rho must be positive.')
    if s_churn < 0.0 or s_noise <= 0.0:
      raise ValueError('EDM churn must be non-negative and s_noise positive.')
    if N < 2:
      raise ValueError('EDM sampling requires at least two steps.')
    self.sigma_data = float(sigma_data)
    self.p_mean = float(p_mean)
    self.p_std = float(p_std)
    self.sigma_min = float(sigma_min)
    self.sigma_max = float(sigma_max)
    self.rho = float(rho)
    self.s_churn = float(s_churn)
    self.s_min = float(s_min)
    self.s_max = float(s_max)
    self.s_noise = float(s_noise)
    self.augmentation = bool(augmentation)
    self.N = int(N)

  def prior_sampling(self, shape):
    return torch.randn(*shape)

  def noise_levels(self, device, dtype=torch.float64):
    indices = torch.arange(self.N, device=device, dtype=dtype)
    ramp = indices / (self.N - 1)
    levels = (self.sigma_max ** (1.0 / self.rho)
              + ramp * (self.sigma_min ** (1.0 / self.rho)
                        - self.sigma_max ** (1.0 / self.rho))) ** self.rho
    return torch.cat([levels, torch.zeros_like(levels[:1])])

  @staticmethod
  def preconditioning(sigma, sigma_data):
    sigma2 = sigma ** 2
    data2 = sigma_data ** 2
    c_skip = data2 / (sigma2 + data2)
    c_out = sigma * sigma_data / torch.sqrt(sigma2 + data2)
    c_in = 1.0 / torch.sqrt(sigma2 + data2)
    c_noise = torch.log(sigma) / 4.0
    return c_skip, c_out, c_in, c_noise

  def churn_gamma(self, sigma):
    if self.s_min <= float(sigma) <= self.s_max:
      return min(self.s_churn / self.N, math.sqrt(2.0) - 1.0)
    return 0.0
