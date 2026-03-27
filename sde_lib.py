"""Abstract SDE classes, Reverse SDE, and VE/VP SDEs."""
import abc
import torch
import numpy as np


class SDE(abc.ABC):
  """SDE abstract class. Functions are designed for a mini-batch of inputs."""

  def __init__(self, N):
    """Construct an SDE.

    Args:
      N: number of discretization time steps.
    """
    super().__init__()
    self.N = N

  @property
  @abc.abstractmethod
  def T(self):
    """End time of the SDE."""
    pass

  @abc.abstractmethod
  def sde(self, x, t):
    pass

  @abc.abstractmethod
  def marginal_prob(self, x, t):
    """Parameters to determine the marginal distribution of the SDE, $p_t(x)$."""
    pass

  @abc.abstractmethod
  def prior_sampling(self, shape):
    """Generate one sample from the prior distribution, $p_T(x)$."""
    pass

  @abc.abstractmethod
  def prior_logp(self, z):
    """Compute log-density of the prior distribution.

    Useful for computing the log-likelihood via probability flow ODE.

    Args:
      z: latent code
    Returns:
      log probability density
    """
    pass

  def discretize(self, x, t):
    """Discretize the SDE in the form: x_{i+1} = x_i + f_i(x_i) + G_i z_i.

    Useful for reverse diffusion sampling and probabiliy flow sampling.
    Defaults to Euler-Maruyama discretization.

    Args:
      x: a torch tensor
      t: a torch float representing the time step (from 0 to `self.T`)

    Returns:
      f, G
    """
    dt = 1 / self.N
    drift, diffusion = self.sde(x, t)
    f = drift * dt
    G = diffusion * torch.sqrt(torch.tensor(dt, device=t.device))
    return f, G

  def reverse(self, score_fn, probability_flow=False):
    """Create the reverse-time SDE/ODE.

    Args:
      score_fn: A time-dependent score-based model that takes x and t and returns the score.
      probability_flow: If `True`, create the reverse-time ODE used for probability flow sampling.
    """
    N = self.N
    T = self.T
    sde_fn = self.sde
    discretize_fn = self.discretize

    # Build the class for reverse-time SDE.
    class RSDE(self.__class__):
      def __init__(self):
        self.N = N
        self.probability_flow = probability_flow

      @property
      def T(self):
        return T

      def sde(self, x, t):
        """Create the drift and diffusion functions for the reverse SDE/ODE."""
        drift, diffusion = sde_fn(x, t)
        score = score_fn(x, t)
        drift = drift - diffusion[:, None, None, None] ** 2 * score * (0.5 if self.probability_flow else 1.)
        # Set the diffusion function to zero for ODEs.
        diffusion = 0. if self.probability_flow else diffusion
        return drift, diffusion

      def discretize(self, x, t):
        """Create discretized iteration rules for the reverse diffusion sampler."""
        f, G = discretize_fn(x, t)
        rev_f = f - G[:, None, None, None] ** 2 * score_fn(x, t) * (0.5 if self.probability_flow else 1.)
        rev_G = torch.zeros_like(G) if self.probability_flow else G
        return rev_f, rev_G

    return RSDE()


class VPSDE(SDE):
  def __init__(self, beta_min=0.1, beta_max=20, N=1000):
    """Construct a Variance Preserving SDE.

    Args:
      beta_min: value of beta(0)
      beta_max: value of beta(1)
      N: number of discretization steps
    """
    super().__init__(N)
    self.beta_0 = beta_min
    self.beta_1 = beta_max
    self.N = N
    self.discrete_betas = torch.linspace(beta_min / N, beta_max / N, N)
    self.alphas = 1. - self.discrete_betas
    self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
    self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
    self.sqrt_1m_alphas_cumprod = torch.sqrt(1. - self.alphas_cumprod)

  @property
  def T(self):
    return 1

  def sde(self, x, t):
    beta_t = self.beta_0 + t * (self.beta_1 - self.beta_0)
    drift = -0.5 * beta_t[:, None, None, None] * x
    diffusion = torch.sqrt(beta_t)
    return drift, diffusion

  def marginal_prob(self, x, t):
    log_mean_coeff = -0.25 * t ** 2 * (self.beta_1 - self.beta_0) - 0.5 * t * self.beta_0
    mean = torch.exp(log_mean_coeff[:, None, None, None]) * x
    std = torch.sqrt(1. - torch.exp(2. * log_mean_coeff))
    return mean, std

  def prior_sampling(self, shape):
    return torch.randn(*shape)

  def prior_logp(self, z):
    shape = z.shape
    N = np.prod(shape[1:])
    logps = -N / 2. * np.log(2 * np.pi) - torch.sum(z ** 2, dim=(1, 2, 3)) / 2.
    return logps

  def discretize(self, x, t):
    """DDPM discretization."""
    timestep = (t * (self.N - 1) / self.T).long()
    beta = self.discrete_betas.to(x.device)[timestep]
    alpha = self.alphas.to(x.device)[timestep]
    sqrt_beta = torch.sqrt(beta)
    f = torch.sqrt(alpha)[:, None, None, None] * x - x
    G = sqrt_beta
    return f, G


class subVPSDE(SDE):
  def __init__(self, beta_min=0.1, beta_max=20, N=1000):
    """Construct the sub-VP SDE that excels at likelihoods.

    Args:
      beta_min: value of beta(0)
      beta_max: value of beta(1)
      N: number of discretization steps
    """
    super().__init__(N)
    self.beta_0 = beta_min
    self.beta_1 = beta_max
    self.N = N

  @property
  def T(self):
    return 1

  def sde(self, x, t):
    beta_t = self.beta_0 + t * (self.beta_1 - self.beta_0)
    drift = -0.5 * beta_t[:, None, None, None] * x
    discount = 1. - torch.exp(-2 * self.beta_0 * t - (self.beta_1 - self.beta_0) * t ** 2)
    diffusion = torch.sqrt(beta_t * discount)
    return drift, diffusion

  def marginal_prob(self, x, t):
    log_mean_coeff = -0.25 * t ** 2 * (self.beta_1 - self.beta_0) - 0.5 * t * self.beta_0
    mean = torch.exp(log_mean_coeff)[:, None, None, None] * x
    std = 1 - torch.exp(2. * log_mean_coeff)
    return mean, std

  def prior_sampling(self, shape):
    return torch.randn(*shape)

  def prior_logp(self, z):
    shape = z.shape
    N = np.prod(shape[1:])
    return -N / 2. * np.log(2 * np.pi) - torch.sum(z ** 2, dim=(1, 2, 3)) / 2.


class FoxVPSDE(SDE):
  """Variance-preserving-style SDE induced by the exact Fox reduction.

  The forward process uses a constant linear drift and a time-dependent
  effective diffusion coefficient obtained from the chosen colored-noise
  kernel:

    dx = u x dt + sqrt(2 D_eff(t)) dW_t

  Marginals remain Gaussian, so the rest of the score-matching pipeline can
  reuse the standard continuous-time VP path.
  """

  def __init__(self,
               u=-0.5,
               diffusion_scale=1.0,
               kernel='gaussian',
               gaussian_sigma=0.35,
               power_law_alpha=1.5,
               power_law_tau0=0.2,
               matern_length_scale=0.3,
               schedule_grid_size=4096,
               target_terminal_variance=None,
               N=1000):
    super().__init__(N)
    if schedule_grid_size < 2:
      raise ValueError('schedule_grid_size must be at least 2.')

    self.u = float(u)
    self.diffusion_scale = float(diffusion_scale)
    self.kernel = self._normalize_kernel_name(kernel)
    self.gaussian_sigma = float(gaussian_sigma)
    self.power_law_alpha = float(power_law_alpha)
    self.power_law_tau0 = float(power_law_tau0)
    self.matern_length_scale = float(matern_length_scale)
    self.schedule_grid_size = int(schedule_grid_size)
    self.target_terminal_variance = target_terminal_variance
    self.N = N

    self._validate_params()
    self._build_schedule_cache()

  @property
  def T(self):
    return 1

  def _normalize_kernel_name(self, kernel):
    normalized = kernel.lower()
    if normalized == 'ou':
      return 'matern_1_2'
    if normalized in ('matern32', 'matern3/2'):
      return 'matern_3_2'
    return normalized

  def _validate_params(self):
    if self.diffusion_scale <= 0.0:
      raise ValueError('diffusion_scale must be positive.')
    if self.kernel not in ('gaussian', 'power_law', 'matern_1_2', 'matern_3_2'):
      raise ValueError('Unsupported kernel: %s' % self.kernel)
    if self.gaussian_sigma <= 0.0:
      raise ValueError('gaussian_sigma must be positive.')
    if self.power_law_alpha <= 0.0:
      raise ValueError('power_law_alpha must be positive.')
    if self.power_law_tau0 <= 0.0:
      raise ValueError('power_law_tau0 must be positive.')
    if self.matern_length_scale <= 0.0:
      raise ValueError('matern_length_scale must be positive.')
    if self.target_terminal_variance is not None and self.target_terminal_variance <= 0.0:
      raise ValueError('target_terminal_variance must be positive.')

  def _kernel_correlation_np(self, tau):
    if self.kernel == 'gaussian':
      return self.diffusion_scale * np.exp(-(tau ** 2) / (2.0 * self.gaussian_sigma ** 2))
    if self.kernel == 'power_law':
      return self.diffusion_scale / ((1.0 + tau / self.power_law_tau0) ** self.power_law_alpha)
    if self.kernel == 'matern_1_2':
      return self.diffusion_scale * np.exp(-tau / self.matern_length_scale)
    if self.kernel == 'matern_3_2':
      scaled_tau = np.sqrt(3.0) * tau / self.matern_length_scale
      return self.diffusion_scale * (1.0 + scaled_tau) * np.exp(-scaled_tau)
    raise ValueError('Unsupported kernel: %s' % self.kernel)

  def _cumulative_trapezoid_np(self, x, y):
    integral = np.zeros_like(y)
    if y.size <= 1:
      return integral
    dx = np.diff(x)
    trapezoids = 0.5 * (y[1:] + y[:-1]) * dx
    integral[1:] = np.cumsum(trapezoids)
    return integral

  def _build_schedule_cache(self):
    times = np.linspace(0.0, self.T, self.schedule_grid_size, dtype=np.float64)
    kernel_correlation = self._kernel_correlation_np(times)
    effective_diffusion_integrand = kernel_correlation * np.exp(self.u * times)
    effective_diffusion = self._cumulative_trapezoid_np(times, effective_diffusion_integrand)

    variance_integrand = effective_diffusion * np.exp(-2.0 * self.u * times)
    variance = 2.0 * np.exp(2.0 * self.u * times) * self._cumulative_trapezoid_np(times, variance_integrand)

    if self.target_terminal_variance is not None:
      terminal_variance = variance[-1]
      if terminal_variance <= 0.0:
        raise ValueError('Terminal variance must be positive for normalization.')
      scale = self.target_terminal_variance / terminal_variance
      self.diffusion_scale *= scale
      effective_diffusion *= scale
      variance *= scale

    variance = np.maximum(variance, 0.0)
    effective_diffusion = np.maximum(effective_diffusion, 0.0)

    self._schedule_times_cpu = torch.from_numpy(times)
    self._effective_diffusion_cpu = torch.from_numpy(effective_diffusion)
    self._variance_cpu = torch.from_numpy(variance)
    self._schedule_cache = {}
    self._prior_variance = float(variance[-1])
    self._prior_std = float(np.sqrt(self._prior_variance))

  def _cached_schedule(self, t):
    key = (t.device.type, t.device.index, t.dtype)
    cached = self._schedule_cache.get(key)
    if cached is None:
      cached = (
        self._schedule_times_cpu.to(device=t.device, dtype=t.dtype),
        self._effective_diffusion_cpu.to(device=t.device, dtype=t.dtype),
        self._variance_cpu.to(device=t.device, dtype=t.dtype),
      )
      self._schedule_cache[key] = cached
    return cached

  def _interpolate(self, t, values):
    schedule_times, _, _ = self._cached_schedule(t)
    t = torch.clamp(t, 0.0, self.T)
    indices = torch.searchsorted(schedule_times, t)
    indices = torch.clamp(indices, 1, schedule_times.shape[0] - 1)
    left = indices - 1
    right = indices

    t0 = schedule_times[left]
    t1 = schedule_times[right]
    v0 = values[left]
    v1 = values[right]
    weight = (t - t0) / (t1 - t0)
    return v0 + weight * (v1 - v0)

  def mean_coeff(self, t):
    return torch.exp(self.u * t)

  def effective_diffusion(self, t):
    _, effective_diffusion, _ = self._cached_schedule(t)
    return self._interpolate(t, effective_diffusion)

  def marginal_variance(self, t):
    _, _, variance = self._cached_schedule(t)
    return torch.clamp(self._interpolate(t, variance), min=0.0)

  def sde(self, x, t):
    drift = self.u * x
    diffusion = torch.sqrt(torch.clamp(2.0 * self.effective_diffusion(t), min=0.0))
    return drift, diffusion

  def marginal_prob(self, x, t):
    mean = self.mean_coeff(t)[:, None, None, None] * x
    std = torch.sqrt(self.marginal_variance(t))
    return mean, std

  def prior_sampling(self, shape):
    return torch.randn(*shape) * self._prior_std

  def prior_logp(self, z):
    if self._prior_variance <= 0.0:
      raise ValueError('Prior variance must be positive.')
    shape = z.shape
    N = np.prod(shape[1:])
    quadratic = torch.sum(z ** 2, dim=tuple(range(1, len(shape)))) / (2.0 * self._prior_variance)
    normalizer = -N / 2.0 * np.log(2 * np.pi * self._prior_variance)
    return normalizer - quadratic


class VESDE(SDE):
  def __init__(self, sigma_min=0.01, sigma_max=50, N=1000):
    """Construct a Variance Exploding SDE.

    Args:
      sigma_min: smallest sigma.
      sigma_max: largest sigma.
      N: number of discretization steps
    """
    super().__init__(N)
    self.sigma_min = sigma_min
    self.sigma_max = sigma_max
    self.discrete_sigmas = torch.exp(torch.linspace(np.log(self.sigma_min), np.log(self.sigma_max), N))
    self.N = N

  @property
  def T(self):
    return 1

  def sde(self, x, t):
    sigma = self.sigma_min * (self.sigma_max / self.sigma_min) ** t
    drift = torch.zeros_like(x)
    diffusion = sigma * torch.sqrt(torch.tensor(2 * (np.log(self.sigma_max) - np.log(self.sigma_min)),
                                                device=t.device))
    return drift, diffusion

  def marginal_prob(self, x, t):
    std = self.sigma_min * (self.sigma_max / self.sigma_min) ** t
    mean = x
    return mean, std

  def prior_sampling(self, shape):
    return torch.randn(*shape) * self.sigma_max

  def prior_logp(self, z):
    shape = z.shape
    N = np.prod(shape[1:])
    return -N / 2. * np.log(2 * np.pi * self.sigma_max ** 2) - torch.sum(z ** 2, dim=(1, 2, 3)) / (2 * self.sigma_max ** 2)

  def discretize(self, x, t):
    """SMLD(NCSN) discretization."""
    timestep = (t * (self.N - 1) / self.T).long()
    sigma = self.discrete_sigmas.to(t.device)[timestep]
    adjacent_sigma = torch.where(timestep == 0, torch.zeros_like(t),
                                 self.discrete_sigmas[timestep - 1].to(t.device))
    f = torch.zeros_like(x)
    G = torch.sqrt(sigma ** 2 - adjacent_sigma ** 2)
    return f, G
