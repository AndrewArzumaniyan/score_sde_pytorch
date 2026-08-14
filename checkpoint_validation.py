"""Regression tests for atomic immutable checkpoints and RNG restoration."""

import os
import random
import tempfile
import unittest

import numpy as np
import torch

from models.ema import ExponentialMovingAverage
from reproducibility import seed_everything
from utils import (checkpoint_filename, latest_immutable_checkpoint,
                   restore_checkpoint, save_checkpoint)


def _new_state():
  model = torch.nn.Linear(3, 2)
  optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
  ema = ExponentialMovingAverage(model.parameters(), decay=0.999)
  optimizer.zero_grad()
  model(torch.ones(2, 3)).sum().backward()
  optimizer.step()
  ema.update(model.parameters())
  return dict(optimizer=optimizer, model=model, ema=ema, step=7)


class CheckpointValidationTest(unittest.TestCase):

  def test_round_trip_restores_rng_and_training_state(self):
    seed_everything(2026)
    state = _new_state()
    expected_parameters = [parameter.detach().clone()
                           for parameter in state['model'].parameters()]
    with tempfile.TemporaryDirectory() as directory:
      path = os.path.join(directory, 'checkpoint.pth')
      checkpoint_id = save_checkpoint(path, state, overwrite=False)
      expected_random = random.random()
      expected_numpy = np.random.standard_normal(3)
      expected_torch = torch.randn(3)

      _ = random.random(), np.random.standard_normal(20), torch.randn(20)
      restored = _new_state()
      restored = restore_checkpoint(path, restored, torch.device('cpu'))

      self.assertEqual(restored['step'], 7)
      self.assertEqual(restored['checkpoint_id'], checkpoint_id)
      for expected, actual in zip(
          expected_parameters, restored['model'].parameters()):
        self.assertTrue(torch.equal(expected, actual))
      self.assertEqual(expected_random, random.random())
      self.assertTrue(np.array_equal(expected_numpy, np.random.standard_normal(3)))
      self.assertTrue(torch.equal(expected_torch, torch.randn(3)))

  def test_immutable_checkpoint_refuses_overwrite(self):
    seed_everything(17)
    state = _new_state()
    with tempfile.TemporaryDirectory() as directory:
      path = os.path.join(directory, 'checkpoint_1.pth')
      save_checkpoint(path, state, overwrite=False)
      with open(path, 'rb') as checkpoint_file:
        original = checkpoint_file.read()
      with self.assertRaises(FileExistsError):
        save_checkpoint(path, state, overwrite=False)
      with open(path, 'rb') as checkpoint_file:
        self.assertEqual(original, checkpoint_file.read())

  def test_non_boundary_final_has_extension_safe_namespace(self):
    self.assertEqual(
      checkpoint_filename(30, 31, 30, final=False), 'checkpoint_1.pth')
    self.assertEqual(
      checkpoint_filename(31, 32, 30, final=True),
      'checkpoint_final_step_32.pth')
    self.assertEqual(
      checkpoint_filename(60, 61, 30, final=False), 'checkpoint_2.pth')
    self.assertEqual(
      checkpoint_filename(390624, 390625, 25000, final=True),
      'checkpoint_final_step_390625.pth')

  def test_model_protocol_mismatch_fails_before_partial_load(self):
    seed_everything(19)
    state = _new_state()
    state['model_protocol_sha256'] = 'model-a'
    state['training_protocol_sha256'] = 'training-a'
    with tempfile.TemporaryDirectory() as directory:
      path = os.path.join(directory, 'checkpoint.pth')
      save_checkpoint(path, state, overwrite=False)
      target = _new_state()
      target['model_protocol_sha256'] = 'model-b'
      with self.assertRaisesRegex(ValueError, 'model/config fingerprint'):
        restore_checkpoint(path, target, torch.device('cpu'))

  def test_latest_immutable_recovers_checkpoint_newer_than_meta(self):
    with tempfile.TemporaryDirectory() as directory:
      paths = [
        os.path.join(directory, 'checkpoint_1.pth'),
        os.path.join(directory, 'checkpoint_2.pth'),
        os.path.join(directory, 'checkpoint_final_step_47.pth'),
      ]
      for path in paths:
        with open(path, 'wb') as checkpoint_file:
          checkpoint_file.write(b'test')
      path, state_step = latest_immutable_checkpoint(directory, 30)
      self.assertEqual(path, paths[1])
      self.assertEqual(state_step, 61)


if __name__ == '__main__':
  unittest.main()
