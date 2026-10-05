"""E3b routing, native epsilon conversion and cache provenance tests."""
import numpy as np
import pytest
import torch

from artifact_utils import validate_npz_metadata
from splice_model import NativeEpsAdapter, SplicedEpsModel
import sde_lib
import sampling


class ExactGaussianEps(torch.nn.Module):
  """Exact epsilon network for x_0 distributed as standard normal."""

  def __init__(self, sde):
    super().__init__()
    self.sde = sde

  def forward(self, x, labels):
    t = labels / 999
    one = torch.ones_like(x[:, :1, :1, :1])
    alpha = self.sde.marginal_prob(one, t)[0]
    sigma = self.sde.marginal_prob(torch.zeros_like(one), t)[1][:, None, None, None]
    radius_sq = alpha.square() + sigma.square()
    return sigma * x / radius_sq


def sdes(n=8):
  vp = sde_lib.VPSDE(N=n)
  vp.sampling_logsnr_min = -10.0
  vp.sampling_logsnr_max = 10.575749165617202
  fox = sde_lib.FoxVPSDE(
    u=-5.0, kernel='matern_3_2', matern_length_scale=346.4101615138,
    target_terminal_variance=1.0, N=n)
  return vp, fox


@pytest.mark.parametrize('mode,threshold', [
  ('full', None), ('none', None), ('tail', 8.0), ('head', -3.3)])
def test_exact_gaussian_score_after_splice(mode, threshold):
  vp, fox = sdes()
  model = SplicedEpsModel(
    ExactGaussianEps(vp), ExactGaussianEps(fox), vp, fox,
    mode, threshold, device='cpu').eval()
  x = torch.linspace(-2, 2, 16).reshape(2, 1, 2, 4)
  for lam in (-8.0, 0.0, 8.0, 10.0):
    t = vp.time_from_log_snr(torch.tensor([lam])).expand(2)
    labels = t * 999
    eps = model(x, labels)
    alpha = vp.marginal_prob(torch.ones_like(x[:, :1, :1, :1]), t)[0]
    sigma = vp.marginal_prob(torch.zeros_like(x[:, :1, :1, :1]), t)[1]
    radius_sq = alpha.square() + sigma[:, None, None, None].square()
    expected = sigma[:, None, None, None] * x / radius_sq
    torch.testing.assert_close(eps, expected, atol=1e-4, rtol=1e-4)


@pytest.mark.parametrize('mode,threshold', [
  ('full', None), ('none', None), ('tail', 8.0), ('head', -3.3)])
def test_route_matches_real_vp_grid(mode, threshold):
  vp, fox = sdes()
  model = SplicedEpsModel(
    ExactGaussianEps(vp), ExactGaussianEps(fox), vp, fox,
    mode, threshold, device='cpu').eval()
  grid = vp.sampling_time_grid(
    1e-3, grid='uniform_logsnr', device='cpu', dtype=torch.float32,
    N=vp.N + 1)
  x = torch.ones((2, 1, 2, 2))
  for t in grid:
    model(x, torch.full((2,), float(t)) * 999)
  meta = model.checked_route_metadata()
  assert meta['splice_vp_calls'] + meta['splice_fox_calls'] == vp.N + 1
  if mode == 'full':
    assert meta['splice_fox_calls'] == vp.N + 1
  if mode == 'none':
    assert meta['splice_fox_calls'] == 0
  if mode in ('tail', 'head'):
    expected = sum((float(vp.log_snr(t.reshape(1))) >= threshold)
                   == (mode == 'tail') for t in grid)
    assert meta['splice_fox_calls'] == expected


def test_none_is_exact_identity():
  vp, fox = sdes()
  network = ExactGaussianEps(vp)
  model = SplicedEpsModel(
    network, ExactGaussianEps(fox), vp, fox, 'none', device='cpu').eval()
  x = torch.randn((2, 1, 2, 2))
  labels = torch.full((2,), 125.0)
  assert torch.equal(model(x, labels), network(x, labels))


@pytest.mark.parametrize('mode,expected_network', [('none', 'A'), ('full', 'B')])
def test_shared_logsnr_all_same_network_routes_are_exact(mode, expected_network):
  vp, _ = sdes()
  model_a = ExactGaussianEps(vp)
  model_b = ExactGaussianEps(vp)
  wrapper = SplicedEpsModel(
    model_a, model_b, vp, vp, mode, device='cpu',
    shared_logsnr_conditioning=True).eval()
  x = torch.randn((2, 1, 2, 2))
  labels = torch.full((2,), 125.0)
  expected = model_a if expected_network == 'A' else model_b
  assert torch.equal(wrapper(x, labels), expected(x, labels))



def test_none_sampler_outputs_match_unwrapped_vp():
  vp, fox = sdes()
  network = ExactGaussianEps(vp)
  wrapper = SplicedEpsModel(
    network, ExactGaussianEps(fox), vp, fox, 'none', device='cpu').eval()
  sampler = sampling.get_pc_sampler(
    sde=vp, shape=(2, 1, 2, 2),
    predictor=sampling.EulerMaruyamaPredictor,
    corrector=sampling.NoneCorrector,
    inverse_scaler=lambda x: x, snr=0.17,
    continuous=True, denoise=True, endpoint_variants=True,
    time_grid='uniform_logsnr', device='cpu')
  torch.manual_seed(20261001)
  baseline, baseline_nfe = sampler(network)
  torch.manual_seed(20261001)
  switched, switched_nfe = sampler(wrapper)
  assert baseline_nfe == switched_nfe == vp.N + 1
  wrapper.checked_route_metadata()
  for variant in baseline:
    assert torch.equal(baseline[variant], switched[variant])

def test_fox_logsnr_roundtrip():
  _, fox = sdes()
  lambdas = torch.tensor([-10.0, -3.3, 0.0, 8.0, 10.575749], dtype=torch.float64)
  times = fox.time_from_log_snr(lambdas)
  torch.testing.assert_close(fox.log_snr(times), lambdas, atol=1e-4, rtol=0)


def test_cache_rejects_missing_or_different_alt_checkpoint(tmp_path):
  filename = tmp_path / 'samples.npz'
  common = dict(protocol_sha256=np.asarray('protocol'),
                checkpoint_id=np.asarray('vp-checkpoint'))
  np.savez(filename, **common)
  with np.load(filename) as archive:
    with pytest.raises(ValueError, match='alt_checkpoint_id'):
      validate_npz_metadata(
        archive, str(filename), 'protocol', 'vp-checkpoint',
        extra_identity={'alt_checkpoint_id': np.asarray('fox-checkpoint')})
  np.savez(filename, **common, alt_checkpoint_id=np.asarray('wrong-fox'))
  with np.load(filename) as archive:
    with pytest.raises(ValueError, match='alt_checkpoint_id'):
      validate_npz_metadata(
        archive, str(filename), 'protocol', 'vp-checkpoint',
        extra_identity={'alt_checkpoint_id': np.asarray('fox-checkpoint')})
