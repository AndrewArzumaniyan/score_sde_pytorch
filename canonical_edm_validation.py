"""Parity and integration checks for the pinned official NVLabs EDM checkout."""

import sys
import unittest

import torch

import edm_lib
import losses
import sampling
from configs.edm.cifar10_canonical import get_config
from models.edm_canonical import CanonicalEDMSongUNet
from models.ema import ExponentialMovingAverage


class CanonicalEDMValidationTest(unittest.TestCase):

  def _small_config(self):
    config = get_config()
    config.device = torch.device('cpu')
    config.model.edm_model_channels = 16
    config.model.edm_channel_mult = (1, 1)
    config.model.edm_num_blocks = 1
    config.model.edm_attn_resolutions = ()
    return config

  def test_config_is_official_ddpmpp_recipe(self):
    config = get_config()
    self.assertEqual(config.model.name, 'edm_canonical_songunet')
    self.assertEqual(tuple(config.model.edm_channel_mult), (2, 2, 2))
    self.assertEqual(config.model.edm_model_channels, 128)
    self.assertEqual(config.model.edm_num_blocks, 4)
    self.assertEqual(config.training.batch_size, 128)
    self.assertEqual(config.training.effective_batch_size, 512)
    self.assertEqual(config.training.gradient_accumulation_steps, 4)
    self.assertEqual(config.training.n_iters + 1, 390625)
    self.assertEqual((config.training.n_iters + 1) * config.training.effective_batch_size,
                     200000000)
    self.assertTrue(config.training.edm_augmentation)
    self.assertEqual(config.training.edm_augment_probability, 0.12)
    self.assertFalse(config.data.random_flip)
    self.assertTrue(config.eval.enable_sampling)
    self.assertEqual(config.eval.num_samples, 50000)
    self.assertEqual(config.eval.end_ckpt, 15)
    self.assertEqual(config.eval.sampling_seed, 0)
    self.assertEqual(config.eval.loss_seed, 0)
    self.assertEqual(config.eval.bpd_seed, 0)
    self.assertFalse(config.eval.include_final_checkpoint)
    self.assertFalse(config.allow_tf32)
    self.assertTrue(config.cudnn_benchmark)

  def test_adapter_rejects_unpinned_checkout(self):
    config = self._small_config()
    config.model.edm_official_commit = '0' * 40
    with self.assertRaisesRegex(RuntimeError, 'expected'):
      CanonicalEDMSongUNet(config)

  def test_adapter_is_bitwise_identical_to_official_modules(self):
    config = self._small_config()
    torch.manual_seed(314)
    wrapped = CanonicalEDMSongUNet(config)

    root = str(config.model.edm_official_root)
    if root not in sys.path:
      sys.path.insert(0, root)
    from training import augment, loss as official_loss, networks

    kwargs = dict(
      img_resolution=32, img_channels=3, label_dim=0, use_fp16=False,
      sigma_min=0.0, sigma_max=float('inf'), sigma_data=0.5,
      model_type='SongUNet', augment_dim=9, model_channels=16,
      channel_mult=[1, 1], channel_mult_emb=4, num_blocks=1,
      attn_resolutions=[], dropout=0.13, embedding_type='positional',
      channel_mult_noise=1, encoder_type='standard',
      decoder_type='standard', resample_filter=[1, 1])
    torch.manual_seed(314)
    direct = networks.EDMPrecond(**kwargs)
    wrapped_state = wrapped.network.state_dict()
    direct_state = direct.state_dict()
    self.assertEqual(wrapped_state.keys(), direct_state.keys())
    for key in wrapped_state:
      self.assertTrue(torch.equal(wrapped_state[key], direct_state[key]), key)

    x = torch.randn(2, 3, 32, 32)
    sigma = torch.tensor([0.02, 2.0])
    labels = torch.randn(2, 9)
    wrapped.eval()
    direct.eval()
    self.assertTrue(torch.equal(
      wrapped(x, sigma, labels),
      direct(x, sigma, None, augment_labels=labels)))

    direct_augment = augment.AugmentPipe(
      p=0.12, xflip=1e8, yflip=1, scale=1, rotate_frac=1,
      aniso=1, translate_frac=1)
    torch.manual_seed(2718)
    wrapped_images, wrapped_labels = wrapped.augment(x)
    torch.manual_seed(2718)
    direct_images, direct_labels = direct_augment(x)
    self.assertTrue(torch.equal(wrapped_images, direct_images))
    self.assertTrue(torch.equal(wrapped_labels, direct_labels))

    wrapped.train()
    direct.train()
    torch.manual_seed(1618)
    wrapped_loss = losses.get_edm_loss_fn(
      edm_lib.EDM(augmentation=True), train=True,
      reduce_mean=False)(wrapped, x)
    torch.manual_seed(1618)
    direct_loss = official_loss.EDMLoss()(direct, x, None, direct_augment)
    direct_loss = direct_loss.sum() / x.shape[0]
    self.assertTrue(torch.equal(wrapped_loss, direct_loss))

  def test_loss_backward_and_sampler(self):
    config = self._small_config()
    # This is a capacity/plumbing diagnostic, not the paper schedule: step 0
    # of the official 10M-image warmup deliberately has LR=0.
    config.optim.warmup_kimg = None
    model = CanonicalEDMSongUNet(config)
    design = edm_lib.EDM(augmentation=True, N=3)
    optimizer = losses.get_optimizer(config, model.parameters())
    ema = ExponentialMovingAverage(model.parameters(), decay=config.model.ema_rate)
    state = dict(optimizer=optimizer, model=model, ema=ema, step=0)
    step_fn = losses.get_step_fn(
      design, train=True, optimize_fn=losses.optimization_manager(config),
      reduce_mean=False, ema_decay_fn=losses.get_ema_decay_fn(config))
    microbatches = [torch.randn(1, 3, 32, 32) for _ in range(4)]
    before = [parameter.detach().clone() for parameter in model.parameters()]
    loss = step_fn(state, microbatches)
    self.assertTrue(torch.isfinite(loss))
    self.assertEqual(state['step'], 1)
    self.assertTrue(any(
      not torch.equal(old, new)
      for old, new in zip(before, model.parameters())))

    model.train()
    sampler = sampling.get_edm_sampler(
      design, (1, 3, 32, 32), inverse_scaler=lambda x: x, device='cpu')
    samples, nfe = sampler(model)
    self.assertTrue(torch.isfinite(samples).all())
    self.assertEqual(samples.shape, (1, 3, 32, 32))
    self.assertEqual(nfe, 5)
    self.assertTrue(model.training)


if __name__ == '__main__':
  unittest.main()
