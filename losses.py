# coding=utf-8
# Copyright 2020 The Google Research Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""All functions related to loss computation and optimization.
"""

import torch
import torch.optim as optim
import numpy as np
from models import utils as mutils
from sde_lib import VESDE, VPSDE
import edm_lib


def get_optimizer(config, params):
  """Returns a flax optimizer object based on `config`."""
  if config.optim.optimizer == 'Adam':
    optimizer = optim.Adam(params, lr=config.optim.lr, betas=(config.optim.beta1, 0.999), eps=config.optim.eps,
                           weight_decay=config.optim.weight_decay)
  else:
    raise NotImplementedError(
      f'Optimizer {config.optim.optimizer} not supported yet!')

  return optimizer


def optimization_manager(config):
  """Returns an optimize_fn based on `config`."""

  def optimize_fn(optimizer, params, step, lr=config.optim.lr,
                  warmup=config.optim.warmup,
                  grad_clip=config.optim.grad_clip):
    """Optimizes with warmup and gradient clipping (disabled if negative)."""
    params = list(params)
    warmup_kimg = getattr(config.optim, 'warmup_kimg', None)
    if warmup_kimg is not None:
      warmup_nimg = warmup_kimg * 1000.0
      effective_batch_size = getattr(
        config.training, 'effective_batch_size', config.training.batch_size)
      lr_scale = np.minimum(step * effective_batch_size / max(warmup_nimg, 1e-8), 1.0)
      for g in optimizer.param_groups:
        g['lr'] = lr * lr_scale
    elif warmup > 0:
      for g in optimizer.param_groups:
        g['lr'] = lr * np.minimum(step / warmup, 1.0)
    if getattr(config.optim, 'sanitize_gradients', False):
      for param in params:
        if param.grad is not None:
          torch.nan_to_num(param.grad, nan=0.0, posinf=1e5, neginf=-1e5,
                           out=param.grad)
    if grad_clip >= 0:
      torch.nn.utils.clip_grad_norm_(params, max_norm=grad_clip)
    optimizer.step()

  return optimize_fn


def sample_sde_time(sde, batch_size, device, eps, noise_distribution_sde=None,
                    logsnr_min=None, logsnr_max=None, dtype=torch.float32):
  """Sample target time, optionally through another SDE's uniform clock."""
  random_values = torch.rand(batch_size, device=device, dtype=dtype)
  if noise_distribution_sde is None:
    return random_values * (sde.T - eps) + eps
  if logsnr_min is None or logsnr_max is None:
    raise ValueError(
      'A cross-SDE noise distribution requires common log-SNR endpoints.')
  if not float(logsnr_max) > float(logsnr_min):
    raise ValueError('logsnr_max must be greater than logsnr_min.')
  lambda_min = torch.tensor(
    [float(logsnr_min)], device=device, dtype=dtype)
  lambda_max = torch.tensor(
    [float(logsnr_max)], device=device, dtype=dtype)
  source_start = noise_distribution_sde.time_from_log_snr(lambda_max)[0]
  source_end = noise_distribution_sde.time_from_log_snr(lambda_min)[0]
  source_t = source_start + random_values * (source_end - source_start)
  sampled_logsnr = noise_distribution_sde.log_snr(source_t)
  return sde.time_from_log_snr(sampled_logsnr)


def get_sde_loss_fn(sde, train, reduce_mean=True, continuous=True,
                    likelihood_weighting=True, eps=1e-5,
                    noise_distribution_sde=None, logsnr_min=None,
                    logsnr_max=None):
  """Create a loss function for training with arbirary SDEs.

  Args:
    sde: An `sde_lib.SDE` object that represents the forward SDE.
    train: `True` for training loss and `False` for evaluation loss.
    reduce_mean: If `True`, average the loss across data dimensions. Otherwise sum the loss across data dimensions.
    continuous: `True` indicates that the model is defined to take continuous time steps. Otherwise it requires
      ad-hoc interpolation to take continuous time steps.
    likelihood_weighting: If `True`, weight the mixture of score matching losses
      according to https://arxiv.org/abs/2101.09258; otherwise use the weighting recommended in our paper.
    eps: A `float` number. The smallest time step to sample from.

  Returns:
    A loss function.
  """
  reduce_op = torch.mean if reduce_mean else lambda *args, **kwargs: 0.5 * torch.sum(*args, **kwargs)

  def loss_fn(model, batch):
    """Compute the loss function.

    Args:
      model: A score model.
      batch: A mini-batch of training data.

    Returns:
      loss: A scalar that represents the average loss value across the mini-batch.
    """
    score_fn = mutils.get_score_fn(sde, model, train=train, continuous=continuous)
    t = sample_sde_time(
      sde, batch.shape[0], batch.device, eps,
      noise_distribution_sde=noise_distribution_sde,
      logsnr_min=logsnr_min, logsnr_max=logsnr_max,
      dtype=batch.dtype)
    z = torch.randn_like(batch)
    mean, std = sde.marginal_prob(batch, t)
    perturbed_data = mean + std[:, None, None, None] * z
    score = score_fn(perturbed_data, t)

    if not likelihood_weighting:
      losses = torch.square(score * std[:, None, None, None] + z)
      losses = reduce_op(losses.reshape(losses.shape[0], -1), dim=-1)
    else:
      g2 = sde.sde(torch.zeros_like(batch), t)[1] ** 2
      losses = torch.square(score + z / std[:, None, None, None])
      losses = reduce_op(losses.reshape(losses.shape[0], -1), dim=-1) * g2

    loss = torch.mean(losses)
    return loss

  return loss_fn


def get_edm_loss_fn(edm, train, reduce_mean=True):
  """EDM denoising loss from Karras et al., Eq. (5) and Table 1."""

  def loss_fn(model, batch):
    # Match the official EDMLoss RNG order: sigma, augmentation, additive noise.
    rnd_normal = torch.randn(batch.shape[0], device=batch.device, dtype=batch.dtype)
    sigma = torch.exp(rnd_normal * edm.p_std + edm.p_mean)
    target = batch
    augment_labels = None
    if edm.augmentation and train:
      model_without_parallel = model.module if hasattr(model, 'module') else model
      if not hasattr(model_without_parallel, 'augment'):
        raise ValueError('EDM augmentation was enabled but the model has no augment() method.')
      target, augment_labels = model_without_parallel.augment(batch)
    noise = torch.randn_like(target) * sigma[:, None, None, None]
    if augment_labels is None:
      denoised = model(target + noise, sigma)
    else:
      denoised = model(target + noise, sigma, augment_labels)
    weight = ((sigma ** 2 + edm.sigma_data ** 2)
              / (sigma * edm.sigma_data) ** 2)
    per_element = weight[:, None, None, None] * torch.square(denoised - target)
    if reduce_mean:
      return torch.mean(per_element)
    # Official EDM calls loss.sum() / batch_gpu_total. Keeping the same global
    # reduction also preserves its floating-point reduction order.
    return torch.sum(per_element) / per_element.shape[0]

  return loss_fn


def get_smld_loss_fn(vesde, train, reduce_mean=False):
  """Legacy code to reproduce previous results on SMLD(NCSN). Not recommended for new work."""
  assert isinstance(vesde, VESDE), "SMLD training only works for VESDEs."

  # Previous SMLD models assume descending sigmas
  smld_sigma_array = torch.flip(vesde.discrete_sigmas, dims=(0,))
  reduce_op = torch.mean if reduce_mean else lambda *args, **kwargs: 0.5 * torch.sum(*args, **kwargs)

  def loss_fn(model, batch):
    model_fn = mutils.get_model_fn(model, train=train)
    labels = torch.randint(0, vesde.N, (batch.shape[0],), device=batch.device)
    sigmas = smld_sigma_array.to(batch.device)[labels]
    noise = torch.randn_like(batch) * sigmas[:, None, None, None]
    perturbed_data = noise + batch
    score = model_fn(perturbed_data, labels)
    target = -noise / (sigmas ** 2)[:, None, None, None]
    losses = torch.square(score - target)
    losses = reduce_op(losses.reshape(losses.shape[0], -1), dim=-1) * sigmas ** 2
    loss = torch.mean(losses)
    return loss

  return loss_fn


def get_ddpm_loss_fn(vpsde, train, reduce_mean=True):
  """Legacy code to reproduce previous results on DDPM. Not recommended for new work."""
  assert isinstance(vpsde, VPSDE), "DDPM training only works for VPSDEs."

  reduce_op = torch.mean if reduce_mean else lambda *args, **kwargs: 0.5 * torch.sum(*args, **kwargs)

  def loss_fn(model, batch):
    model_fn = mutils.get_model_fn(model, train=train)
    labels = torch.randint(0, vpsde.N, (batch.shape[0],), device=batch.device)
    sqrt_alphas_cumprod = vpsde.sqrt_alphas_cumprod.to(batch.device)
    sqrt_1m_alphas_cumprod = vpsde.sqrt_1m_alphas_cumprod.to(batch.device)
    noise = torch.randn_like(batch)
    perturbed_data = sqrt_alphas_cumprod[labels, None, None, None] * batch + \
                     sqrt_1m_alphas_cumprod[labels, None, None, None] * noise
    score = model_fn(perturbed_data, labels)
    losses = torch.square(score - noise)
    losses = reduce_op(losses.reshape(losses.shape[0], -1), dim=-1)
    loss = torch.mean(losses)
    return loss

  return loss_fn


def get_step_fn(sde, train, optimize_fn=None, reduce_mean=False, continuous=True,
                likelihood_weighting=False, ema_decay_fn=None,
                noise_distribution_sde=None, logsnr_min=None,
                logsnr_max=None):
  """Create a one-step training/evaluation function.

  Args:
    sde: An `sde_lib.SDE` object that represents the forward SDE.
    optimize_fn: An optimization function.
    reduce_mean: If `True`, average the loss across data dimensions. Otherwise sum the loss across data dimensions.
    continuous: `True` indicates that the model is defined to take continuous time steps.
    likelihood_weighting: If `True`, weight the mixture of score matching losses according to
      https://arxiv.org/abs/2101.09258; otherwise use the weighting recommended by our paper.

  Returns:
    A one-step function for training or evaluation.
  """
  if isinstance(sde, edm_lib.EDM):
    loss_fn = get_edm_loss_fn(sde, train, reduce_mean=reduce_mean)
  elif continuous:
    loss_fn = get_sde_loss_fn(sde, train, reduce_mean=reduce_mean,
                              continuous=True,
                              likelihood_weighting=likelihood_weighting,
                              noise_distribution_sde=noise_distribution_sde,
                              logsnr_min=logsnr_min,
                              logsnr_max=logsnr_max)
  else:
    assert not likelihood_weighting, "Likelihood weighting is not supported for original SMLD/DDPM training."
    if isinstance(sde, VESDE):
      loss_fn = get_smld_loss_fn(sde, train, reduce_mean=reduce_mean)
    elif isinstance(sde, VPSDE):
      loss_fn = get_ddpm_loss_fn(sde, train, reduce_mean=reduce_mean)
    else:
      raise ValueError(f"Discrete training for {sde.__class__.__name__} is not recommended.")

  def step_fn(state, batch):
    """Running one step of training or evaluation.

    This function will undergo `jax.lax.scan` so that multiple steps can be pmapped and jit-compiled together
    for faster execution.

    Args:
      state: A dictionary of training information, containing the score model, optimizer,
       EMA status, and number of optimization steps.
      batch: A mini-batch of training/evaluation data.

    Returns:
      loss: The average loss value of this state.
    """
    model = state['model']
    if train:
      model.train()
      optimizer = state['optimizer']
      optimizer.zero_grad()
      if isinstance(batch, (list, tuple)):
        if not batch:
          raise ValueError('Gradient accumulation received no microbatches.')
        accumulated_loss = 0.0
        for microbatch in batch:
          microbatch_loss = loss_fn(model, microbatch)
          (microbatch_loss / len(batch)).backward()
          accumulated_loss = accumulated_loss + microbatch_loss.detach()
        loss = accumulated_loss / len(batch)
      else:
        loss = loss_fn(model, batch)
        loss.backward()
      current_step = state['step']
      optimize_fn(optimizer, model.parameters(), step=current_step)
      state['step'] += 1
      ema_decay = None if ema_decay_fn is None else ema_decay_fn(current_step)
      state['ema'].update(model.parameters(), decay=ema_decay)
    else:
      was_training = model.training
      model.eval()
      ema = state['ema']
      ema.store(model.parameters())
      try:
        with torch.no_grad():
          ema.copy_to(model.parameters())
          loss = loss_fn(model, batch)
      finally:
        ema.restore(model.parameters())
        model.train(was_training)

    return loss

  return step_fn


def get_ema_decay_fn(config):
  """Return the image-count EMA schedule used by the official EDM recipe."""
  if config.training.sde.lower() != 'edm':
    return None
  batch_size = getattr(
    config.training, 'effective_batch_size', config.training.batch_size)
  halflife_nimg = config.model.edm_ema_halflife_kimg * 1000.0
  rampup_ratio = config.model.edm_ema_rampup_ratio

  def decay_fn(step):
    current_nimg = step * batch_size
    current_halflife = halflife_nimg
    if rampup_ratio is not None:
      current_halflife = min(current_halflife, current_nimg * rampup_ratio)
    return 0.5 ** (batch_size / max(current_halflife, 1e-8))

  return decay_fn
