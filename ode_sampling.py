"""Fixed-budget VP probability-flow solvers in half-log-SNR coordinates.

DPM-Solver++ 2M uses the data-prediction midpoint multistep formula of
Lu et al., https://arxiv.org/abs/2211.01095 (algorithm_type=dpmsolver++,
solver_type=dpmsolver). First step is first order; no thresholding,
clipping, stochasticity, or implicit final integration to zero noise.

EDM Heun is Algorithm 2 of Karras et al., https://arxiv.org/abs/2206.00364,
without churn: rho=7 nodes in sigma/alpha = exp(-lambda/2) and Heun steps in
that variable on the state x/alpha. Its final Euler step to zero noise is the
explicit Tweedie call below, so 18 nodes give the canonical 35 calls.
"""
import torch

import sde_lib
from models.utils import get_score_fn

METHODS = ('ode_heun', 'ode_dpmpp_2m', 'ode_edm_heun')
VARIANTS = ('x_t', 'tweedie_xt')
KARRAS_RHO = 7.0


def time_grid_for(method):
  return 'karras' if method == 'ode_edm_heun' else 'uniform_logsnr'


def karras_time_grid(sde, eps, num_points, device, rho=KARRAS_RHO):
  """Karras sigma nodes between the configured log-SNR ends.

  Endpoint times are exactly those of the uniform-logSNR grid; interior nodes
  are inverted from log-SNR with the SDE's own schedule.
  """
  ends = sde.sampling_time_grid(eps, grid='uniform_logsnr', device=device,
                                dtype=torch.float64, N=2)
  lam_min = torch.tensor(float(sde.sampling_logsnr_min), dtype=torch.float64)
  lam_max = torch.tensor(float(sde.sampling_logsnr_max), dtype=torch.float64)
  ramp = torch.linspace(0, 1, num_points, dtype=torch.float64)
  inv_max, inv_min = torch.exp(-lam_min / (2 * rho)), torch.exp(-lam_max / (2 * rho))
  ve_sigma = (inv_max + ramp * (inv_min - inv_max)) ** rho
  grid = sde.time_from_log_snr(-2 * torch.log(ve_sigma)).to(device=device, dtype=torch.float64)
  grid[0], grid[-1] = ends[0], ends[-1]
  return grid


def dpmpp_2m_update(x, current, previous, alpha_next, sigma, sigma_next, h, h_prev=None):
  """One data-prediction exponential step; h is HALF-log-SNR increment."""
  estimate = current
  if previous is not None:
    estimate = current + (h / (2 * h_prev)) * (current - previous)
  return (sigma_next / sigma) * x - alpha_next * torch.expm1(-h) * estimate


def get_vp_ode_sampler(sde, shape, inverse_scaler, method, eps=1e-3, device='cuda'):
  """N trajectory network calls plus one explicit Tweedie call, for all methods.

  Heun and EDM Heun take N/2 two-stage steps, DPM++ takes N multistep steps.
  Heun and DPM++ use uniform full-log-SNR nodes, EDM Heun uses Karras nodes;
  all share the finite endpoints and Gaussian prior.
  """
  # Normalize native Gaussian marginals by their deterministic overall scale.
  # The normalized process has VP marginals, while the score is evaluated
  # at the original native state and native time.
  if type(sde) not in (sde_lib.VPSDE, sde_lib.FoxVPSDE) or method not in METHODS:
    raise ValueError('Fixed-budget ODE sampling requires VP or native FOX and a supported method.')
  budget = int(sde.N)
  two_stage = method in ('ode_heun', 'ode_edm_heun')
  if budget < 2 or (two_stage and budget % 2):
    raise ValueError('Heun requires a positive even trajectory NFE; minimum budget is 2.')
  if getattr(sde, 'sampling_logsnr_min', None) is None or getattr(sde, 'sampling_logsnr_max', None) is None:
    raise ValueError('Both finite sampling log-SNR endpoints must be explicit.')
  steps = budget // 2 if two_stage else budget

  @torch.no_grad()
  def sampler(model, z=None):
    from sampling import get_initial_sample
    if method == 'ode_edm_heun':
      grid = karras_time_grid(sde, eps, steps + 1, device)
    else:
      grid = sde.sampling_time_grid(eps, grid='uniform_logsnr', device=device,
                                    dtype=torch.float64, N=steps + 1)
    x = get_initial_sample(sde, shape, grid, device) if z is None else z.clone().to(device)
    if tuple(x.shape) != tuple(shape):
      raise ValueError('Initial state shape differs from configured sampling batch.')
    # Evaluate the network using the project's existing time/log-SNR wrapper.
    times = grid.to(dtype=x.dtype)
    ones = torch.ones((steps + 1, 1, 1, 1), device=device, dtype=x.dtype)
    native_alpha, native_sigma = sde.marginal_prob(ones, times)
    native_alpha = native_alpha.flatten()
    scale = torch.sqrt(native_alpha.square() + native_sigma.square())
    alpha, sigma = native_alpha / scale, native_sigma / scale
    x = x / scale[0]
    # Compute half-log-SNR from the actual coefficients; no factor-of-two ambiguity.
    half_lambda = torch.log(alpha.double()) - torch.log(sigma.double())
    if not bool(torch.all(half_lambda[1:] > half_lambda[:-1])):
      raise ValueError('ODE grid must increase in log-SNR.')
    score_fn = get_score_fn(sde, model, train=False, continuous=True)
    calls = 0

    def score(state, index):
      nonlocal calls
      calls += 1
      vec_t = torch.full((shape[0],), times[index].item(), device=device, dtype=state.dtype)
      return scale[index] * score_fn(state * scale[index], vec_t)

    previous = None
    for i in range(steps):
      h = (half_lambda[i + 1] - half_lambda[i]).to(x.dtype)
      current_score = score(x, i)
      if method == 'ode_heun':
        # VP PF-ODE: dx/dr = sigma(r)^2 * (x + score(x,r)), r=lambda/2.
        drift = sigma[i].square() * (x + current_score)
        proposal = x + h * drift
        next_drift = sigma[i + 1].square() * (proposal + score(proposal, i + 1))
        x = x + (h / 2) * (drift + next_drift)
      elif method == 'ode_edm_heun':
        # Algorithm 2 on y = x / alpha with D = (x + sigma^2 score) / alpha.
        ve, ve_next = sigma[i] / alpha[i], sigma[i + 1] / alpha[i + 1]
        y = x / alpha[i]
        d_cur = (y - (x + sigma[i].square() * current_score) / alpha[i]) / ve
        proposal = (y + (ve_next - ve) * d_cur) * alpha[i + 1]
        denoised_next = (proposal + sigma[i + 1].square() * score(proposal, i + 1)) / alpha[i + 1]
        d_next = (proposal / alpha[i + 1] - denoised_next) / ve_next
        x = (y + (ve_next - ve) * (0.5 * d_cur + 0.5 * d_next)) * alpha[i + 1]
      else:
        clean = (x + sigma[i].square() * current_score) / alpha[i]
        h_prev = None if i == 0 else (half_lambda[i] - half_lambda[i - 1]).to(x.dtype)
        x = dpmpp_2m_update(x, clean, previous, alpha[i + 1], sigma[i], sigma[i + 1], h, h_prev)
        previous = clean
      if not bool(torch.isfinite(x).all()):
        raise FloatingPointError('Non-finite ODE trajectory at step {}'.format(i))
    # Evaluate at the corrected final state, not Heun's Euler proposal.
    clean = (x + sigma[-1].square() * score(x, steps)) / alpha[-1]
    if calls != budget + 1 or not bool(torch.isfinite(clean).all()):
      raise RuntimeError('Incorrect ODE NFE or non-finite endpoint denoise.')
    return dict(x_t=inverse_scaler(x * scale[-1]), tweedie_xt=inverse_scaler(clean)), calls

  return sampler
