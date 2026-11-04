"""Checks for the k_a(t) shape experiment configs (configs/fox/celeba_kaffine.py).

Run:  python -m pytest -q tests/test_kaffine_config.py
"""

import json
import os
import sys

import numpy as np
import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import run_lib
from artifact_utils import _canonicalize
from configs.fox.celeba_kaffine import ARMS, COMMON_LOGSNR_MAX, COMMON_LOGSNR_MIN, get_config


def _scientific(config):
  result = json.loads(json.dumps(_canonicalize(config)))
  result['model'].pop('fox_drift_a')
  result['model'].pop('fox_normalize_scale')
  result['training'].pop('kaffine_arm')
  return result


def _path(arm, n=20001):
  sde, _ = run_lib.get_sde(get_config(arm))
  t = torch.linspace(1e-5, 1.0, n)
  mean, std = sde.marginal_prob(torch.ones(n, 1, 1, 1), t)
  alpha = mean.flatten().double().numpy()
  sigma = std.flatten().double().numpy()
  return sde, alpha, sigma


def test_arms_differ_only_in_drift_shape():
  base = _scientific(get_config('a0'))
  for arm in ARMS:
    assert _scientific(get_config(arm)) == base, arm


def test_unknown_arm_rejected():
  with pytest.raises(ValueError):
    get_config('a1')


@pytest.mark.parametrize('arm', sorted(ARMS))
def test_common_window_inside_training_support(arm):
  sde, _ = run_lib.get_sde(get_config(arm))
  assert sde.log_snr(torch.tensor([1e-5])).item() > COMMON_LOGSNR_MAX
  assert sde.log_snr(torch.tensor([1.0])).item() < COMMON_LOGSNR_MIN


def test_terminal_logsnr_shared():
  ends = [run_lib.get_sde(get_config(arm))[0].log_snr(torch.tensor([1.0])).item() for arm in ARMS]
  assert max(ends) - min(ends) < 1e-3


def test_vplin_norm_keeps_vplin_path_with_unit_scale():
  _, alpha_raw, sigma_raw = _path('vplin_raw')
  _, alpha, sigma = _path('vplin_norm')
  lam_raw = 2 * np.log(alpha_raw / sigma_raw)
  lam = 2 * np.log(alpha / sigma)
  assert np.max(np.abs(lam - lam_raw)) < 1e-3
  assert np.max(np.abs(alpha ** 2 + sigma ** 2 - 1)) < 1e-4
  assert np.max(np.sqrt(alpha_raw ** 2 + sigma_raw ** 2)) > 2.0


def test_scale_excursions_match_design():
  # a0 dips to ~0.71 without overshoot; ap5 dips deeper (~0.47), still no overshoot.
  for arm, low, high in (('a0', 0.70, 0.72), ('ap5', 0.45, 0.49)):
    _, alpha, sigma = _path(arm)
    r = np.sqrt(alpha ** 2 + sigma ** 2)
    assert low < r.min() < high, (arm, r.min())
    assert r.max() <= 1 + 1e-4, (arm, r.max())


def test_uniform_logsnr_grid_covers_common_window():
  for arm in ARMS:
    config = get_config(arm)
    sde, eps = run_lib.get_sde(config)
    grid = sde.sampling_time_grid(eps, grid='uniform_logsnr', dtype=torch.float64, N=1001)
    lam = sde.log_snr(grid.float()).double().numpy()
    assert np.all(np.diff(lam) > 0) or np.all(np.diff(lam) < 0), arm
    assert abs(min(lam) - COMMON_LOGSNR_MIN) < 1e-3 and abs(max(lam) - COMMON_LOGSNR_MAX) < 1e-3, arm
