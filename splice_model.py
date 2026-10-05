"""Route VP sampler score calls between VP and native FOX epsilon models."""

import torch
from torch import nn


class NativeEpsAdapter(nn.Module):
  """Evaluate a native model at the VP state's actual log-SNR and scale."""

  def __init__(self, model, base_sde, target_sde):
    super().__init__()
    self.model = model
    self.base = base_sde
    self.target = target_sde

  def forward(self, x, labels):
    tb = labels / 999
    xb = torch.ones_like(x[:, :1, :1, :1])
    ab = self.base.marginal_prob(xb, tb)[0]
    sb = self.base.marginal_prob(torch.zeros_like(xb), tb)[1][:, None, None, None]
    rb = torch.sqrt(ab.square() + sb.square())
    actual_lambda = 2 * torch.log(ab / sb)
    tm = self.target.time_from_log_snr(actual_lambda.flatten())
    am_native = self.target.marginal_prob(xb, tm)[0]
    sm_native = self.target.marginal_prob(torch.zeros_like(xb), tm)[1][:, None, None, None]
    rm = torch.sqrt(am_native.square() + sm_native.square())
    # Preserve native time and overall scale while matching the actual VP SNR.
    sm = rm / torch.sqrt(1 + torch.exp(actual_lambda))
    xm = x * (rm / rb)
    eps_m = self.model(xm, tm * 999)
    return eps_m * (sb / rb) * (rm / sm)


class SplicedEpsModel(nn.Module):
  """Select the VP or adapted FOX epsilon model once per sampler call."""

  ROUTE_KEYS = (
    'splice_vp_calls', 'splice_fox_calls',
    'splice_switch_step', 'splice_switch_lambda')

  def __init__(self, vp_model, fox_model, vp_sde, fox_sde, mode,
               threshold=None, sampling_eps=1e-3,
               time_grid='uniform_logsnr', device='cpu',
               shared_logsnr_conditioning=False):
    super().__init__()
    if mode not in ('full', 'none', 'tail', 'head'):
      raise ValueError(f'Unknown splice mode: {mode}')
    if mode in ('tail', 'head') and threshold is None:
      raise ValueError(f'{mode} requires a log-SNR threshold.')
    self.vp_model = vp_model
    self.fox_adapter = (fox_model if shared_logsnr_conditioning else
                        NativeEpsAdapter(fox_model, vp_sde, fox_sde))
    self.vp_sde = vp_sde
    self.mode = mode
    self.threshold = None if threshold is None else float(threshold)
    self.expected_routes, self.expected_route_metadata = self._expected_route(
      sampling_eps, time_grid, device)
    self.reset_route()

  def _lambda(self, t):
    one = torch.ones((1, 1, 1, 1), device=t.device, dtype=t.dtype)
    alpha = self.vp_sde.marginal_prob(one, t)[0]
    sigma = self.vp_sde.marginal_prob(torch.zeros_like(one), t)[1]
    return 2 * torch.log(alpha.flatten() / sigma)

  def _use_fox(self, actual_lambda):
    if self.mode == 'full':
      return True
    if self.mode == 'none':
      return False
    above = bool((actual_lambda >= self.threshold).item())
    return above if self.mode == 'tail' else not above

  def _expected_route(self, sampling_eps, time_grid, device):
    grid = self.vp_sde.sampling_time_grid(
      sampling_eps, grid=time_grid, device=device, dtype=torch.float32,
      N=self.vp_sde.N + 1)
    # The predictor uses every node except the last; Tweedie uses the last.
    lambdas = []
    routes = []
    for t in grid:
      actual_lambda = self._lambda(t.reshape(1))
      lambdas.append(float(actual_lambda.item()))
      routes.append(self._use_fox(actual_lambda))
    routes = tuple(routes)
    switch = next((i for i in range(1, len(routes))
                   if routes[i] != routes[0]), -1)
    metadata = {
      'splice_vp_calls': routes.count(False),
      'splice_fox_calls': routes.count(True),
      'splice_switch_step': switch,
      'splice_switch_lambda': lambdas[switch] if switch >= 0 else 0.0,
    }
    return routes, metadata

  def reset_route(self):
    self.route_history = []

  def checked_route_metadata(self):
    if tuple(self.route_history) != self.expected_routes:
      raise ValueError(
        'Splice routes differ from the VP sampling grid: '
        f'actual VP/FOX={self.route_history.count(False)}/'
        f'{self.route_history.count(True)}, expected VP/FOX='
        f'{self.expected_route_metadata["splice_vp_calls"]}/'
        f'{self.expected_route_metadata["splice_fox_calls"]}.')
    return self.expected_route_metadata

  def forward(self, x, labels):
    if self.training:
      raise RuntimeError('SplicedEpsModel is only defined for evaluation.')
    if labels.ndim != 1 or labels.shape[0] != x.shape[0] or \
        not bool(torch.all(labels == labels[0]).item()):
      raise ValueError('Every sampler call must use one time for the batch.')
    actual_lambda = self._lambda((labels[:1] / 999).to(x.dtype))
    fox = self._use_fox(actual_lambda)
    self.route_history.append(fox)
    if fox:
      return self.fox_adapter(x, labels)
    return self.vp_model(x, labels)
