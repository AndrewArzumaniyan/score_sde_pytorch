"""Fixed-budget VP probability-flow solvers in half-log-SNR coordinates.

DPM-Solver++ 2M uses the data-prediction midpoint multistep formula of
Lu et al., https://arxiv.org/abs/2211.01095 (algorithm_type=dpmsolver++,
solver_type=dpmsolver). First step is first order; no thresholding,
clipping, stochasticity, or implicit final integration to zero noise.
"""
import torch

import sde_lib
from models.utils import get_score_fn

METHODS = ('ode_heun', 'ode_dpmpp_2m')
VARIANTS = ('x_t', 'tweedie_xt')


def dpmpp_2m_update(x, current, previous, alpha_next, sigma, sigma_next, h, h_prev=None):
  """One data-prediction exponential step; h is HALF-log-SNR increment."""
  estimate = current
  if previous is not None:
    estimate = current + (h / (2 * h_prev)) * (current - previous)
  return (sigma_next / sigma) * x - alpha_next * torch.expm1(-h) * estimate


def get_vp_ode_sampler(sde, shape, inverse_scaler, method, eps=1e-3, device='cuda'):
  """N trajectory network calls plus one explicit Tweedie call, for both methods.

  Heun takes N/2 two-stage steps, DPM++ takes N multistep steps. Both use
  uniform full-log-SNR nodes, identical finite endpoints and Gaussian prior.
  """
  # This implementation relies on alpha^2+sigma^2=1 and excludes native FOX.
  if type(sde) is not sde_lib.VPSDE or method not in METHODS:
    raise ValueError('Fixed-budget ODE sampling requires a VPSDE and a supported method.')
  budget = int(sde.N)
  if budget < 2 or (method == 'ode_heun' and budget % 2):
    raise ValueError('Heun requires a positive even trajectory NFE; minimum budget is 2.')
  if getattr(sde, 'sampling_logsnr_min', None) is None or getattr(sde, 'sampling_logsnr_max', None) is None:
    raise ValueError('Both finite sampling log-SNR endpoints must be explicit.')
  steps = budget // 2 if method == 'ode_heun' else budget

  @torch.no_grad()
  def sampler(model, z=None):
    from sampling import get_initial_sample
    grid = sde.sampling_time_grid(eps, grid='uniform_logsnr', device=device,
                                  dtype=torch.float64, N=steps + 1)
    x = get_initial_sample(sde, shape, grid, device) if z is None else z.clone().to(device)
    if tuple(x.shape) != tuple(shape):
      raise ValueError('Initial state shape differs from configured sampling batch.')
    # Evaluate the network using the project's existing time/log-SNR wrapper.
    times = grid.to(dtype=x.dtype)
    ones = torch.ones((steps + 1, 1, 1, 1), device=device, dtype=x.dtype)
    alpha, sigma = sde.marginal_prob(ones, times)
    alpha = alpha.flatten()
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
      return score_fn(state, vec_t)

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
    return dict(x_t=inverse_scaler(x), tweedie_xt=inverse_scaler(clean)), calls

  return sampler
