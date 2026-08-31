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


class CosineVPSDE(VPSDE):
  """VP SDE with the cosine cumulative-noise schedule of Nichol & Dhariwal.

  The paper defines

    alpha_bar(t) = f(t) / f(0),
    f(t) = cos^2(((t + s) / (1 + s)) * pi / 2),  s = 0.008.

  Its continuous-time beta diverges at t=1.  We therefore integrate up to
  ``t_max`` (0.999 by default), where the residual signal is already
  negligible, while retaining the paper's max_beta=0.999 safeguard for the
  cached discrete schedule.
  """

  def __init__(self, s=0.008, t_max=0.999, max_beta=0.999, N=1000):
    SDE.__init__(self, N)
    if s < 0.0:
      raise ValueError('cosine schedule offset s must be non-negative.')
    if not 0.0 < t_max < 1.0:
      raise ValueError('cosine t_max must lie strictly between 0 and 1.')
    if not 0.0 < max_beta < 1.0:
      raise ValueError('cosine max_beta must lie strictly between 0 and 1.')
    self.s = float(s)
    self.t_max = float(t_max)
    self.max_beta = float(max_beta)
    self.N = int(N)
    self._theta_0 = self.s / (1.0 + self.s) * np.pi / 2.0
    self._f0 = float(np.cos(self._theta_0) ** 2)

    # Same alpha_bar discretization and beta clipping as improved-diffusion,
    # evaluated over the numerically safe continuous horizon [0, t_max].
    edges = np.linspace(0.0, self.t_max, self.N + 1, dtype=np.float64)
    alpha_bar = self._alpha_bar_np(edges)
    betas = np.minimum(1.0 - alpha_bar[1:] / alpha_bar[:-1], self.max_beta)
    self.discrete_betas = torch.from_numpy(betas.astype(np.float32))
    self.alphas = 1.0 - self.discrete_betas
    self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
    self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
    self.sqrt_1m_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod)

  @property
  def T(self):
    return self.t_max

  def _alpha_bar_np(self, t):
    theta = (t + self.s) / (1.0 + self.s) * np.pi / 2.0
    return np.cos(theta) ** 2 / self._f0

  def alpha_bar(self, t):
    t = torch.clamp(t, 0.0, self.T)
    theta = (t + self.s) / (1.0 + self.s) * np.pi / 2.0
    return torch.clamp(torch.cos(theta) ** 2 / self._f0, min=0.0, max=1.0)

  def beta(self, t):
    t = torch.clamp(t, 0.0, self.T)
    theta = (t + self.s) / (1.0 + self.s) * np.pi / 2.0
    return np.pi / (1.0 + self.s) * torch.tan(theta)

  def sde(self, x, t):
    beta_t = self.beta(t)
    drift = -0.5 * beta_t[:, None, None, None] * x
    diffusion = torch.sqrt(beta_t)
    return drift, diffusion

  def marginal_prob(self, x, t):
    alpha_bar = self.alpha_bar(t)
    mean = torch.sqrt(alpha_bar)[:, None, None, None] * x
    std = torch.sqrt(torch.clamp(1.0 - alpha_bar, min=0.0))
    return mean, std

  def prior_sampling(self, shape):
    return torch.randn(*shape)

  def prior_logp(self, z):
    shape = z.shape
    num_dims = np.prod(shape[1:])
    reduce_dims = tuple(range(1, len(shape)))
    return -num_dims / 2.0 * np.log(2 * np.pi) - torch.sum(z ** 2, dim=reduce_dims) / 2.0

  def discretize(self, x, t):
    # Euler discretization is consistent with the continuous cosine SDE and
    # avoids silently using VPSDE's linear-beta DDPM coefficients.
    return SDE.discretize(self, x, t)


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

  The forward process uses a deterministic linear drift ``k(t) x`` and a
  time-dependent effective diffusion coefficient obtained from the chosen
  colored-noise kernel:

    dx = k(t) x dt + sqrt(2 D_eff(t)) dW_t

  For ``drift_schedule='vp_linear'``, ``k(t)=-beta(t)/2`` with the same
  linear beta schedule as ``VPSDE``.  This is an exact one-time-marginal
  reduction: D_eff(t) uses the response exp(integral_s^t k(r) dr), rather
  than the constant-drift response exp(u (t-s)).

  Marginals remain Gaussian, so the rest of the score-matching pipeline can
  reuse the standard continuous-time VP path.
  """

  def __init__(self,
               u=-0.5,
               drift_schedule='constant',
               beta_min=0.1,
               beta_max=20.0,
               diffusion_scale=1.0,
               kernel='gaussian',
               gaussian_sigma=0.35,
               power_law_alpha=1.5,
               power_law_tau0=0.2,
               matern_length_scale=0.3,
               drift_k_bar=-5.025,
               drift_a=0.0,
               normalize_scale=False,
               schedule_grid_size=4096,
               target_terminal_variance=None,
               N=1000):
    super().__init__(N)
    if schedule_grid_size < 2:
      raise ValueError('schedule_grid_size must be at least 2.')

    self.u = float(u)
    self.drift_schedule = self._normalize_drift_schedule_name(drift_schedule)
    self.beta_min = float(beta_min)
    self.beta_max = float(beta_max)
    self.diffusion_scale = float(diffusion_scale)
    self.kernel = self._normalize_kernel_name(kernel)
    self.gaussian_sigma = float(gaussian_sigma)
    self.power_law_alpha = float(power_law_alpha)
    self.power_law_tau0 = float(power_law_tau0)
    self.matern_length_scale = float(matern_length_scale)
    # Affine drift k_a(t) = drift_k_bar + drift_a (t - 1/2).  int_0^1 k_a =
    # drift_k_bar for any drift_a, so the terminal alpha(1) / lambda(1) are
    # pinned while drift_a only redistributes contraction along the path.
    # Used iff drift_schedule == 'affine'.
    self.drift_k_bar = float(drift_k_bar)
    self.drift_a = float(drift_a)
    # normalize_scale: keep the induced log-SNR path lambda(t) but rescale the
    # marginal to alpha^2 + q = 1 (VP-type overall scale r(t) == 1).
    # Schedule-only post-processing; see _build_schedule_cache.
    self.normalize_scale = bool(normalize_scale)
    self.schedule_grid_size = int(schedule_grid_size)
    if self.normalize_scale and self.schedule_grid_size < 8192:
      # normalize_scale recovers drift / D_eff by differentiating the schedule
      # numerically, so keep the quadrature grid fine regardless of the request.
      self.schedule_grid_size = 8192
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

  def _normalize_drift_schedule_name(self, schedule):
    normalized = schedule.lower()
    if normalized in ('vp', 'vp-linear', 'vp_linear_beta'):
      return 'vp_linear'
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
    if self.drift_schedule not in ('constant', 'vp_linear', 'affine'):
      raise ValueError('Unsupported Fox drift schedule: %s' % self.drift_schedule)
    if self.drift_schedule == 'affine':
      k_at_0 = self.drift_k_bar - 0.5 * self.drift_a
      k_at_1 = self.drift_k_bar + 0.5 * self.drift_a
      if k_at_0 >= 0.0 or k_at_1 >= 0.0:
        raise ValueError(
          'affine drift k_a(t) = k_bar + a (t - 1/2) must stay negative on '
          '[0, 1]; got k_a(0) = %.4g, k_a(1) = %.4g (need |a| < 2|k_bar|).'
          % (k_at_0, k_at_1))
    if self.beta_min <= 0.0 or self.beta_max <= 0.0:
      raise ValueError('Fox VP beta_min and beta_max must be positive.')

  def _drift_coefficient_np(self, t):
    if self.drift_schedule == 'constant':
      return np.full_like(t, self.u, dtype=np.float64)
    if self.drift_schedule == 'affine':
      return self.drift_k_bar + self.drift_a * (t - 0.5)
    beta_t = self.beta_min + t * (self.beta_max - self.beta_min)
    return -0.5 * beta_t

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
    drift_coefficient = self._drift_coefficient_np(times)
    log_mean_coeff = self._cumulative_trapezoid_np(times, drift_coefficient)

    if self.drift_schedule == 'constant':
      # Preserve the previous, one-dimensional quadrature path exactly.
      kernel_correlation = self._kernel_correlation_np(times)
      effective_diffusion_integrand = kernel_correlation * np.exp(self.u * times)
      effective_diffusion = self._cumulative_trapezoid_np(times, effective_diffusion_integrand)
    else:
      # D_eff(t_i) = int_0^t_i C(t_i-s) exp(K(t_i)-K(s)) ds, K'=k.
      # The kernels implemented here are stationary, so this is an exact
      # deterministic response factor evaluated by trapezoidal quadrature.
      effective_diffusion = np.zeros_like(times)
      for index in range(1, times.size):
        past_times = times[:index + 1]
        lag = times[index] - past_times
        response = np.exp(log_mean_coeff[index] - log_mean_coeff[:index + 1])
        integrand = self._kernel_correlation_np(lag) * response
        effective_diffusion[index] = np.sum(
          0.5 * (integrand[1:] + integrand[:-1]) * np.diff(past_times))

    variance_integrand = effective_diffusion * np.exp(-2.0 * log_mean_coeff)
    variance = 2.0 * np.exp(2.0 * log_mean_coeff) * self._cumulative_trapezoid_np(times, variance_integrand)

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

    if self.normalize_scale:
      # 'normalized-FOX' (E4): keep the induced log-SNR path lambda(t) but force
      # alpha^2 + q = 1, i.e. a VP-type overall scale r(t) == 1.  With
      #   alpha_tilde^2 = sigmoid(lambda),   q_tilde = sigmoid(-lambda),
      # the log-SNR lambda_tilde = log(alpha_tilde^2 / q_tilde) == lambda is
      # unchanged by construction.  Drift and D_eff are then recovered from the
      # moment ODEs by differentiating the analytic (alpha_tilde, q_tilde) on the
      # schedule grid -- the same numeric-inversion route used for VP / cosine.
      safe_variance = np.maximum(variance, 1e-300)
      log_snr_grid = 2.0 * log_mean_coeff - np.log(safe_variance)
      # index 0: q(0) = 0 => lambda(0) = +inf; use a finite placeholder, the
      # true t = 0 limits are written back explicitly below and t = 0 is never
      # used as the right interpolation node.
      log_snr_grid[0] = 2.0 * log_snr_grid[1] - log_snr_grid[2]
      log_alpha_tilde = -0.5 * np.logaddexp(0.0, -log_snr_grid)   # 0.5 log sigmoid(lambda)
      q_tilde = np.exp(-np.logaddexp(0.0, log_snr_grid))          # sigmoid(-lambda)
      log_alpha_tilde[0] = 0.0                                    # alpha_tilde(0) = 1
      q_tilde[0] = 0.0                                            # q_tilde(0) = 0
      drift_coefficient = np.gradient(log_alpha_tilde, times)     # d/dt log alpha_tilde
      q_tilde_dot = np.gradient(q_tilde, times)
      effective_diffusion = 0.5 * (q_tilde_dot - 2.0 * drift_coefficient * q_tilde)
      log_mean_coeff = log_alpha_tilde
      variance = np.maximum(q_tilde, 0.0)
      effective_diffusion = np.maximum(effective_diffusion, 0.0)

    self._schedule_times_cpu = torch.from_numpy(times)
    self._drift_coefficient_cpu = torch.from_numpy(drift_coefficient)
    self._log_mean_coeff_cpu = torch.from_numpy(log_mean_coeff)
    self._effective_diffusion_cpu = torch.from_numpy(effective_diffusion)
    self._variance_cpu = torch.from_numpy(variance)
    # Expose the cached variance under an explicit name for sanity checks/debugging.
    self.sigma2_grid = self._variance_cpu
    self._schedule_cache = {}
    self._prior_variance = float(variance[-1])
    self._prior_std = float(np.sqrt(self._prior_variance))

    if self.kernel == 'gaussian':
      assert torch.all(self.sigma2_grid >= 0), "sigma2 < 0 detected!"
      assert torch.isfinite(self.sigma2_grid).all(), "NaN/Inf in sigma2!"

  def _cached_schedule(self, t):
    key = (t.device.type, t.device.index, t.dtype)
    cached = self._schedule_cache.get(key)
    if cached is None:
      cached = (
        self._schedule_times_cpu.to(device=t.device, dtype=t.dtype),
        self._drift_coefficient_cpu.to(device=t.device, dtype=t.dtype),
        self._log_mean_coeff_cpu.to(device=t.device, dtype=t.dtype),
        self._effective_diffusion_cpu.to(device=t.device, dtype=t.dtype),
        self._variance_cpu.to(device=t.device, dtype=t.dtype),
      )
      self._schedule_cache[key] = cached
    return cached

  def _interpolate(self, t, values):
    schedule_times = self._cached_schedule(t)[0]
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
    _, _, log_mean_coeff, _, _ = self._cached_schedule(t)
    return torch.exp(self._interpolate(t, log_mean_coeff))

  def drift_coefficient(self, t):
    _, drift_coefficient, _, _, _ = self._cached_schedule(t)
    return self._interpolate(t, drift_coefficient)

  def effective_diffusion(self, t):
    _, _, _, effective_diffusion, _ = self._cached_schedule(t)
    return self._interpolate(t, effective_diffusion)

  def marginal_variance(self, t):
    _, _, _, _, variance = self._cached_schedule(t)
    return torch.clamp(self._interpolate(t, variance), min=0.0)

  def log_snr(self, t):
    """log-SNR lambda(t) = 2 log alpha(t) - log q(t).

    lambda'(t) = -2 D_eff(t) / q(t), so lambda is strictly decreasing on (0, T]
    whenever D_eff(t) > 0 -- even when the marginal variance q(t) itself is
    non-monotone (e.g. drift_schedule='vp_linear').  This makes log-SNR the
    robust monotone coordinate for building sampling grids.
    """
    _, _, log_mean_coeff, _, variance = self._cached_schedule(t)
    log_var = torch.log(torch.clamp(self._interpolate(t, variance), min=1e-30))
    return 2.0 * self._interpolate(t, log_mean_coeff) - log_var

  def _uniform_logsnr_grid(self, eps, num_steps, device, dtype):
    """Reverse-time grid (T -> eps) with uniformly spaced log-SNR.

    Well defined whenever log-SNR is monotone, i.e. D_eff(t) > 0 on (0, T].
    This is the case 'uniform_variance' cannot handle: for a non-monotone q(t)
    a searchsorted on the variance table is mathematically invalid.

    Inversion is by vectorized bisection against ``log_snr`` (monotone), so the
    grid is exact w.r.t. the model's own piecewise-linear schedule and does not
    depend on ``schedule_grid_size`` resolution.
    """
    log_mean_coeff = self._log_mean_coeff_cpu.to(device=device, dtype=dtype)
    variance = self._variance_cpu.to(device=device, dtype=dtype)
    interior = variance > 0
    lam_grid = 2.0 * log_mean_coeff[interior] - torch.log(variance[interior])
    if not bool(torch.all(lam_grid[1:] < lam_grid[:-1])):
      raise ValueError(
        "uniform_logsnr needs a strictly decreasing log-SNR on (0, T] "
        "(D_eff(t) > 0 everywhere); this schedule violates that. Use uniform_time.")

    eps_t = torch.tensor(float(eps), device=device, dtype=dtype)
    T_t = torch.tensor(float(self.T), device=device, dtype=dtype)
    lam_eps = self.log_snr(eps_t.reshape(1))[0]
    lam_T = self.log_snr(T_t.reshape(1))[0]
    targets = torch.linspace(lam_T.item(), lam_eps.item(), num_steps,
                             device=device, dtype=dtype)

    lo = torch.full_like(targets, float(eps))
    hi = torch.full_like(targets, float(self.T))
    for _ in range(64):
      mid = 0.5 * (lo + hi)
      too_small = self.log_snr(mid) > targets     # lam decreasing -> t below target
      lo = torch.where(too_small, mid, lo)
      hi = torch.where(too_small, hi, mid)
    timesteps = 0.5 * (lo + hi)
    timesteps[0] = T_t
    timesteps[-1] = eps_t
    return timesteps

  def sampling_time_grid(self, eps, grid='uniform_time', device=None, dtype=None, N=None):
    """Build reverse-time sampling grids for Fox-aware discretization."""
    num_steps = self.N if N is None else N
    if device is None:
      device = self._schedule_times_cpu.device
    if dtype is None:
      dtype = self._schedule_times_cpu.dtype

    if grid == 'uniform_time':
      return torch.linspace(self.T, eps, num_steps, device=device, dtype=dtype)
    if grid == 'uniform_logsnr':
      return self._uniform_logsnr_grid(eps, num_steps, device, dtype)
    if grid != 'uniform_variance':
      raise ValueError(
        "Unsupported Fox sampling grid: %r (expected 'uniform_time', "
        "'uniform_logsnr' or 'uniform_variance')" % (grid,))

    schedule_times = self._schedule_times_cpu.to(device=device, dtype=dtype)
    variance = self._variance_cpu.to(device=device, dtype=dtype)
    eps_tensor = torch.tensor([eps], device=device, dtype=dtype)
    eps_variance = self.marginal_variance(eps_tensor)[0]
    terminal_variance = variance[-1]
    target_variances = torch.linspace(
      terminal_variance.item(), eps_variance.item(), num_steps, device=device, dtype=dtype)

    indices = torch.searchsorted(variance, target_variances)
    indices = torch.clamp(indices, 1, variance.shape[0] - 1)
    left = indices - 1
    right = indices

    v0 = variance[left]
    v1 = variance[right]
    t0 = schedule_times[left]
    t1 = schedule_times[right]
    denom = torch.clamp(v1 - v0, min=torch.finfo(dtype).eps)
    weights = (target_variances - v0) / denom
    timesteps = t0 + weights * (t1 - t0)
    timesteps[0] = torch.tensor(self.T, device=device, dtype=dtype)
    timesteps[-1] = torch.tensor(eps, device=device, dtype=dtype)
    return timesteps

  def sde(self, x, t):
    drift = self.drift_coefficient(t)[:, None, None, None] * x
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
    self.discrete_sigmas = torch.exp(
      torch.linspace(np.log(self.sigma_min), np.log(self.sigma_max), N, dtype=torch.float32))
    self.N = N

  @property
  def T(self):
    return 1

  def sde(self, x, t):
    sigma = self.sigma_min * (self.sigma_max / self.sigma_min) ** t
    drift = torch.zeros_like(x)
    diffusion = sigma * torch.sqrt(torch.tensor(
      2 * (np.log(self.sigma_max) - np.log(self.sigma_min)),
      device=t.device,
      dtype=t.dtype))
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
    discrete_sigmas = self.discrete_sigmas.to(device=t.device, dtype=t.dtype)
    sigma = discrete_sigmas[timestep]
    adjacent_sigma = torch.where(timestep == 0, torch.zeros_like(t),
                                 discrete_sigmas[timestep - 1])
    f = torch.zeros_like(x)
    G = torch.sqrt(torch.clamp(sigma ** 2 - adjacent_sigma ** 2, min=0.0))
    return f, G
