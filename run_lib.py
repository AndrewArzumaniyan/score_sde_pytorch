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

# pylint: skip-file
"""Training and evaluation for score-based generative models. """

import gc
import io
import math
import os
import time

import torch
from torch.utils import tensorboard
from torchvision.utils import make_grid, save_image
import numpy as np

# In mixed TensorFlow/PyTorch environments (notably WSL with older TF stacks),
# importing TensorFlow before PyTorch lazily initializes CUDA can trigger
# `RuntimeError: random_device could not be read: File exists` on the first
# `.to("cuda")`. Initialize CUDA eagerly while Torch still owns startup.
if torch.cuda.is_available():
  torch.cuda.init()

import tensorflow as tf
import tensorflow_gan as tfgan
import logging
# Keep the import below for registering all model definitions
from models import ddpm, edm, edm_canonical, ncsnv2, ncsnpp
import losses
import sampling
from models import utils as mutils
from models.ema import ExponentialMovingAverage
import datasets
import evaluation
import likelihood
import sde_lib
import edm_lib
from artifact_utils import (atomic_savez, atomic_write_bytes,
                            build_eval_manifest, build_model_manifest,
                            build_training_manifest, ensure_eval_manifest,
                            ensure_training_manifest,
                            validate_npz_metadata)
from reproducibility import (configure_reproducibility, derive_seed,
                             isolated_torch_rng, restore_rng_state)
from absl import flags
from utils import (checkpoint_filename, latest_immutable_checkpoint,
                   save_checkpoint, restore_checkpoint)

FLAGS = flags.FLAGS


def get_sde(config, n_override=None):
  num_scales = config.model.num_scales if n_override is None else n_override
  sde_name = config.training.sde.lower()
  if sde_name == 'vpsde':
    return sde_lib.VPSDE(
      beta_min=config.model.beta_min,
      beta_max=config.model.beta_max,
      N=num_scales,
    ), 1e-3
  if sde_name == 'cosinevpsde':
    return sde_lib.CosineVPSDE(
      s=config.model.cosine_s,
      t_max=config.model.cosine_t_max,
      max_beta=config.model.cosine_max_beta,
      N=num_scales,
    ), 1e-3
  if sde_name == 'subvpsde':
    return sde_lib.subVPSDE(
      beta_min=config.model.beta_min,
      beta_max=config.model.beta_max,
      N=num_scales,
    ), 1e-3
  if sde_name == 'vesde':
    return sde_lib.VESDE(
      sigma_min=config.model.sigma_min,
      sigma_max=config.model.sigma_max,
      N=num_scales,
    ), 1e-5
  if sde_name == 'foxvpsde':
    return sde_lib.FoxVPSDE(
      u=config.model.fox_u,
      drift_schedule=getattr(config.model, 'fox_drift_schedule', 'constant'),
      beta_min=getattr(config.model, 'fox_beta_min', config.model.beta_min),
      beta_max=getattr(config.model, 'fox_beta_max', config.model.beta_max),
      diffusion_scale=config.model.fox_diffusion_scale,
      kernel=config.model.fox_kernel,
      gaussian_sigma=config.model.fox_gaussian_sigma,
      power_law_alpha=config.model.fox_power_law_alpha,
      power_law_tau0=config.model.fox_power_law_tau0,
      matern_length_scale=config.model.fox_matern_length_scale,
      schedule_grid_size=config.model.fox_schedule_grid_size,
      target_terminal_variance=getattr(config.model, 'fox_target_terminal_variance', None),
      N=num_scales,
    ), 1e-3
  if sde_name == 'edm':
    num_steps = (getattr(config.sampling, 'edm_num_steps', 18)
                 if n_override is None else n_override)
    return edm_lib.EDM(
      sigma_data=config.model.edm_sigma_data,
      p_mean=config.training.edm_p_mean,
      p_std=config.training.edm_p_std,
      sigma_min=config.sampling.edm_sigma_min,
      sigma_max=config.sampling.edm_sigma_max,
      rho=config.sampling.edm_rho,
      s_churn=config.sampling.edm_s_churn,
      s_min=config.sampling.edm_s_min,
      s_max=config.sampling.edm_s_max,
      s_noise=config.sampling.edm_s_noise,
      augmentation=getattr(config.training, 'edm_augmentation', False),
      N=num_steps,
    ), 0.0
  raise NotImplementedError(f"SDE {config.training.sde} unknown.")


def train(config, workdir):
  """Runs the training pipeline.

  Args:
    config: Configuration to use.
    workdir: Working directory for checkpoints and TF summaries. If this
      contains checkpoint training will be resumed from the latest checkpoint.
  """

  base_seed = configure_reproducibility(config, tensorflow_module=tf)
  training_manifest = build_training_manifest(config)
  training_protocol_sha256 = ensure_training_manifest(
    workdir, training_manifest)
  model_protocol_sha256 = training_manifest[
    'model_manifest']['model_protocol_sha256']

  # Create directories for experimental logs
  sample_dir = os.path.join(workdir, "samples")
  tf.io.gfile.makedirs(sample_dir)

  tb_dir = os.path.join(workdir, "tensorboard")
  tf.io.gfile.makedirs(tb_dir)

  # Initialize model.
  score_model = mutils.create_model(config)
  ema = ExponentialMovingAverage(score_model.parameters(), decay=config.model.ema_rate)
  optimizer = losses.get_optimizer(config, score_model.parameters())
  state = dict(
    optimizer=optimizer, model=score_model, ema=ema, step=0,
    data_batches_consumed=0,
    eval_batches_consumed=0,
    model_protocol_sha256=model_protocol_sha256,
    training_protocol_sha256=training_protocol_sha256)

  # Create checkpoints directory
  checkpoint_dir = os.path.join(workdir, "checkpoints")
  # Intermediate checkpoints to resume training after pre-emption in cloud environments
  checkpoint_meta_dir = os.path.join(workdir, "checkpoints-meta", "checkpoint.pth")
  tf.io.gfile.makedirs(checkpoint_dir)
  tf.io.gfile.makedirs(os.path.dirname(checkpoint_meta_dir))
  # Resume training when intermediate checkpoints are detected
  state = restore_checkpoint(
    checkpoint_meta_dir, state, config.device, defer_rng=True)
  immutable_path, immutable_step = latest_immutable_checkpoint(
    checkpoint_dir, config.training.snapshot_freq)
  if immutable_step is not None and immutable_step > int(state['step']):
    logging.warning(
      'Resume meta lags immutable checkpoint state %d; recovering from %s.',
      immutable_step, immutable_path)
    state = restore_checkpoint(
      immutable_path, state, config.device, defer_rng=True)
    if int(state['step']) != immutable_step:
      raise ValueError(
        'Immutable checkpoint filename/state mismatch: '
        f'{immutable_path} implies step {immutable_step}, but contains '
        f"step {state['step']}.")
  initial_step = int(state['step'])
  if initial_step > int(config.training.n_iters) + 1:
    raise ValueError(
      f'Workdir already reached state.step={initial_step}, which is beyond '
      f'the requested n_iters+1={int(config.training.n_iters) + 1}. Refusing '
      'to label a longer run as a shorter experiment.')
  writer = tensorboard.SummaryWriter(
    tb_dir, purge_step=initial_step if initial_step > 0 else None)

  # Build data iterators
  train_ds, eval_ds, _ = datasets.get_dataset(config,
                                              uniform_dequantization=config.data.uniform_dequantization)
  if state['data_batches_consumed']:
    logging.info(
      'Replaying and skipping %d deterministic training batches to restore '
      'the logical data position.', state['data_batches_consumed'])
    train_ds = train_ds.skip(state['data_batches_consumed'])
  if state['eval_batches_consumed']:
    eval_ds = eval_ds.skip(state['eval_batches_consumed'])
  train_iter = iter(train_ds)  # pytype: disable=wrong-arg-types
  eval_iter = iter(eval_ds)  # pytype: disable=wrong-arg-types
  # Create data normalizer and its inverse
  scaler = datasets.get_data_scaler(config)
  inverse_scaler = datasets.get_data_inverse_scaler(config)

  # Setup SDEs
  sde, sampling_eps = get_sde(config)

  # Build one-step training and evaluation functions
  optimize_fn = losses.optimization_manager(config)
  continuous = config.training.continuous
  reduce_mean = config.training.reduce_mean
  likelihood_weighting = config.training.likelihood_weighting
  ema_decay_fn = losses.get_ema_decay_fn(config)
  train_step_fn = losses.get_step_fn(sde, train=True, optimize_fn=optimize_fn,
                                     reduce_mean=reduce_mean, continuous=continuous,
                                     likelihood_weighting=likelihood_weighting,
                                     ema_decay_fn=ema_decay_fn)
  eval_step_fn = losses.get_step_fn(sde, train=False, optimize_fn=optimize_fn,
                                    reduce_mean=reduce_mean, continuous=continuous,
                                    likelihood_weighting=likelihood_weighting)

  # Building sampling functions
  if config.training.snapshot_sampling:
    sampling_shape = (config.training.batch_size, config.data.num_channels,
                      config.data.image_size, config.data.image_size)
    sampling_fn = sampling.get_sampling_fn(config, sde, sampling_shape, inverse_scaler, sampling_eps)

  # Dataset/model/runtime construction must not advance a restored process RNG.
  # The tf.data iterator position itself is not checkpointed; see the explicit
  # warning below for resumed runs.
  pending_rng_state = state.pop('_rng_state_to_restore', None)
  if pending_rng_state is not None:
    if not restore_rng_state(pending_rng_state):
      logging.warning(
        'Checkpoint RNG state was only partially restored because the visible '
        'CUDA topology changed.')
  elif initial_step > 0:
    logging.warning(
      'The resume checkpoint has no RNG state; stochastic continuation is not '
      'reproducible.')
  if initial_step > 0:
    logging.warning(
      'tf.data iterator state is restored by deterministic replay/skip rather '
      'than an iterator snapshot. Resume can be slow and requires unchanged '
      'dataset contents, TensorFlow version, and device topology.')

  num_train_steps = config.training.n_iters

  # In case there are multiple hosts (e.g., TPU pods), only log to host 0
  logging.info("Starting training loop at step %d." % (initial_step,))

  for step in range(initial_step, num_train_steps + 1):
    accumulation_steps = getattr(config.training, 'gradient_accumulation_steps', 1)
    if accumulation_steps < 1:
      raise ValueError('training.gradient_accumulation_steps must be positive.')
    microbatches = []
    for _ in range(accumulation_steps):
      # Convert data to Torch tensors and normalize them. Use ._numpy() to avoid copy.
      microbatch = torch.from_numpy(next(train_iter)['image']._numpy()).to(config.device).float()
      state['data_batches_consumed'] += 1
      microbatch = microbatch.permute(0, 3, 1, 2)
      microbatches.append(scaler(microbatch))
    batch = microbatches[0] if accumulation_steps == 1 else microbatches
    # Execute one training step
    loss = train_step_fn(state, batch)
    if step % config.training.log_freq == 0:
      logging.info("step: %d, training_loss: %.5e" % (step, loss.item()))
      writer.add_scalar("training_loss", loss, step)

    # Report the loss on an evaluation dataset periodically
    if step % config.training.eval_freq == 0:
      eval_batch = torch.from_numpy(next(eval_iter)['image']._numpy()).to(config.device).float()
      state['eval_batches_consumed'] += 1
      eval_batch = eval_batch.permute(0, 3, 1, 2)
      eval_batch = scaler(eval_batch)
      eval_seed = derive_seed(base_seed, 'periodic-eval', state['step'])
      with isolated_torch_rng(eval_seed):
        eval_loss = eval_step_fn(state, eval_batch)
      logging.info("step: %d, eval_loss: %.5e" % (step, eval_loss.item()))
      writer.add_scalar("eval_loss", eval_loss.item(), step)

    # Save a checkpoint periodically and generate samples if needed
    is_periodic_snapshot = (
      step != 0 and step % config.training.snapshot_freq == 0)
    is_final_step = step == num_train_steps
    if is_periodic_snapshot or is_final_step:
      # Save the checkpoint.
      filename = checkpoint_filename(
        step, state['step'], config.training.snapshot_freq,
        final=is_final_step)
      save_checkpoint(
        os.path.join(checkpoint_dir, filename),
        state, overwrite=False)

    # The mutable resume checkpoint is published after all training/eval RNG
    # consumption for this update, and always includes the final state.
    should_save_meta = (
      (step != 0 and
       step % config.training.snapshot_freq_for_preemption == 0) or
      is_final_step)
    if should_save_meta:
      save_checkpoint(checkpoint_meta_dir, state, overwrite=True)
      writer.flush()

      # Generate and save samples
    if (is_periodic_snapshot or is_final_step) and config.training.snapshot_sampling:
      snapshot_seed = derive_seed(base_seed, 'snapshot', state['step'])
      with isolated_torch_rng(snapshot_seed):
        ema.store(score_model.parameters())
        try:
          ema.copy_to(score_model.parameters())
          sample, n = sampling_fn(score_model)
        finally:
          ema.restore(score_model.parameters())
        this_sample_dir = os.path.join(sample_dir, "iter_{}".format(step))
        tf.io.gfile.makedirs(this_sample_dir)
        nrow = int(np.sqrt(sample.shape[0]))
        image_grid = make_grid(sample, nrow, padding=2)
        sample = np.clip(sample.permute(0, 2, 3, 1).cpu().numpy() * 255, 0, 255).astype(np.uint8)
        sample_buffer = io.BytesIO()
        np.save(sample_buffer, sample)
        atomic_write_bytes(
          os.path.join(this_sample_dir, "sample.np"),
          sample_buffer.getvalue(), overwrite=False)

        image_buffer = io.BytesIO()
        save_image(image_grid, image_buffer, format='png')
        atomic_write_bytes(
          os.path.join(this_sample_dir, "sample.png"),
          image_buffer.getvalue(), overwrite=False)

  writer.flush()
  writer.close()


def evaluate(config,
             workdir,
             eval_folder="eval"):
  """Evaluate trained models.

  Args:
    config: Configuration to use.
    workdir: Working directory for checkpoints.
    eval_folder: The subfolder for storing evaluation results. Default to
      "eval".
  """
  configure_reproducibility(config, tensorflow_module=tf)

  # Create and lock the evaluation protocol before accepting cached artifacts.
  eval_dir = os.path.join(workdir, eval_folder)
  eval_manifest = build_eval_manifest(config)
  protocol_sha256 = ensure_eval_manifest(eval_dir, eval_manifest)

  # Build data pipeline
  train_ds, eval_ds, _ = datasets.get_dataset(config,
                                              uniform_dequantization=config.data.uniform_dequantization,
                                              evaluation=True)

  # Create data normalizer and its inverse
  scaler = datasets.get_data_scaler(config)
  inverse_scaler = datasets.get_data_inverse_scaler(config)

  # Initialize model
  score_model = mutils.create_model(config)
  optimizer = losses.get_optimizer(config, score_model.parameters())
  ema = ExponentialMovingAverage(score_model.parameters(), decay=config.model.ema_rate)
  model_manifest = build_model_manifest(config)
  state = dict(
    optimizer=optimizer, model=score_model, ema=ema, step=0,
    model_protocol_sha256=model_manifest['model_protocol_sha256'])

  checkpoint_dir = os.path.join(workdir, "checkpoints")

  # Setup SDEs. Keep the training SDE for loss/BPD, but allow eval-only
  # sampling to use a different number of discretization steps.
  sde, sampling_eps = get_sde(config)
  sampling_num_scales = getattr(config.eval, 'sampling_num_scales', 0)
  sampling_sde = sde
  if sampling_num_scales and sampling_num_scales > 0:
    sampling_sde, sampling_eps = get_sde(config, n_override=sampling_num_scales)

  # Create the one-step evaluation function when loss computation is enabled
  if config.eval.enable_loss:
    optimize_fn = losses.optimization_manager(config)
    continuous = config.training.continuous
    likelihood_weighting = config.training.likelihood_weighting

    reduce_mean = config.training.reduce_mean
    eval_step = losses.get_step_fn(sde, train=False, optimize_fn=optimize_fn,
                                   reduce_mean=reduce_mean,
                                   continuous=continuous,
                                   likelihood_weighting=likelihood_weighting)


  # Build raw [0, 1] data for likelihood evaluation. Uniform dequantization is
  # applied below with a separate per-batch Torch seed so partial resume and
  # repeated test passes remain deterministic but distinct.
  train_ds_bpd, eval_ds_bpd, _ = datasets.get_dataset(config,
                                                      uniform_dequantization=False, evaluation=True)
  if config.eval.bpd_dataset.lower() == 'train':
    ds_bpd = train_ds_bpd
    bpd_num_repeats = 1
  elif config.eval.bpd_dataset.lower() == 'test':
    # Go over the dataset 5 times when computing likelihood on the test dataset
    ds_bpd = eval_ds_bpd
    bpd_num_repeats = 5
  else:
    raise ValueError(f"No bpd dataset {config.eval.bpd_dataset} recognized.")

  # Build the likelihood computation function when likelihood is enabled
  if config.eval.enable_bpd:
    if isinstance(sde, edm_lib.EDM):
      raise ValueError('Probability-flow BPD is not defined for the EDM baseline.')
    likelihood_fn = likelihood.get_likelihood_fn(sde, inverse_scaler)

  # Build the sampling function when sampling is enabled
  if config.eval.enable_sampling:
    sampling_shape = (config.eval.batch_size,
                      config.data.num_channels,
                      config.data.image_size, config.data.image_size)
    sampling_fn = sampling.get_sampling_fn(config, sampling_sde, sampling_shape, inverse_scaler, sampling_eps)

  # Use inceptionV3 for images with resolution higher than 256.
  inceptionv3 = config.data.image_size >= 256
  inception_model = evaluation.get_inception_model(inceptionv3=inceptionv3)

  begin_ckpt = config.eval.begin_ckpt
  checkpoint_targets = [
    (ckpt, os.path.join(checkpoint_dir, f'checkpoint_{ckpt}.pth'))
    for ckpt in range(begin_ckpt, config.eval.end_ckpt + 1)
  ]
  data_stats = None
  data_stats_id = None
  if (getattr(config.eval, 'include_final_checkpoint', False) and
      config.training.n_iters % config.training.snapshot_freq):
    final_state_step = int(config.training.n_iters) + 1
    final_label = f'final_step_{final_state_step}'
    checkpoint_targets.append((
      final_label,
      os.path.join(
        checkpoint_dir, f'checkpoint_final_step_{final_state_step}.pth')))
  logging.info("begin checkpoint: %s", begin_ckpt)
  for ckpt, ckpt_path in checkpoint_targets:
    # Wait if the target checkpoint doesn't exist yet
    waiting_message_printed = False
    while not tf.io.gfile.exists(ckpt_path):
      if not waiting_message_printed:
        logging.warning("Waiting for the arrival of checkpoint_%s", ckpt)
        waiting_message_printed = True
      time.sleep(60)

    # New checkpoints are atomically published, so an existing path is ready
    # for reading. Configuration/corruption errors must fail immediately.
    state = restore_checkpoint(
      ckpt_path, state, device=config.device, restore_rng=False)
    checkpoint_id = state['checkpoint_id']
    checkpoint_step = int(state['step'])
    training_protocol = state.get(
      'training_protocol_sha256') or 'legacy-unknown'
    artifact_identity = dict(
      protocol_sha256=np.asarray(protocol_sha256),
      checkpoint_id=np.asarray(checkpoint_id),
      checkpoint_step=np.asarray(checkpoint_step),
      training_protocol_sha256=np.asarray(training_protocol))
    ema.copy_to(score_model.parameters())
    # Compute the loss function on the full evaluation dataset if loss computation is enabled
    if config.eval.enable_loss:
      loss_path = os.path.join(eval_dir, f"ckpt_{ckpt}_loss.npz")
      if tf.io.gfile.exists(loss_path):
        with tf.io.gfile.GFile(loss_path, 'rb') as loss_file:
          loss_archive = np.load(loss_file)
          validate_npz_metadata(
            loss_archive, loss_path, protocol_sha256, checkpoint_id,
            required_arrays=('all_losses', 'mean_loss'),
            checkpoint_step=checkpoint_step,
            training_protocol_sha256=training_protocol)
        logging.info('Reusing validated loss artifact: %s', loss_path)
      else:
        all_losses = []
        eval_iter = iter(eval_ds)  # pytype: disable=wrong-arg-types
        with isolated_torch_rng(
            derive_seed(config.eval.loss_seed, 'eval-loss')):
          for i, batch in enumerate(eval_iter):
            eval_batch = torch.from_numpy(batch['image']._numpy()).to(config.device).float()
            eval_batch = eval_batch.permute(0, 3, 1, 2)
            eval_batch = scaler(eval_batch)
            eval_loss = eval_step(state, eval_batch)
            all_losses.append(eval_loss.item())
            if (i + 1) % 1000 == 0:
              logging.info("Finished %dth step loss evaluation" % (i + 1))

        all_losses = np.asarray(all_losses)
        atomic_savez(
          loss_path, overwrite=False, all_losses=all_losses,
          mean_loss=all_losses.mean(),
          **artifact_identity)

    # Compute log-likelihoods (bits/dim) if enabled
    if config.eval.enable_bpd:
      bpds = []
      for repeat in range(bpd_num_repeats):
        bpd_iter = iter(ds_bpd)  # pytype: disable=wrong-arg-types
        for batch_id in range(len(ds_bpd)):
          batch = next(bpd_iter)
          bpd_round_id = batch_id + len(ds_bpd) * repeat
          bpd_seed = derive_seed(
            config.eval.bpd_seed, 'eval-bpd', bpd_round_id)
          bpd_dequant_seed = derive_seed(
            config.eval.bpd_seed, 'bpd-dequantization', bpd_round_id)
          bpd_path = os.path.join(
            eval_dir,
            f"{config.eval.bpd_dataset}_ckpt_{ckpt}_bpd_{bpd_round_id}.npz")
          if tf.io.gfile.exists(bpd_path):
            with tf.io.gfile.GFile(bpd_path, 'rb') as bpd_file:
              bpd_archive = np.load(bpd_file)
              validate_npz_metadata(
                bpd_archive, bpd_path, protocol_sha256, checkpoint_id,
                round_id=bpd_round_id, sampling_seed=bpd_seed,
                required_arrays=('bpd', 'dequantization_seed'),
                checkpoint_step=checkpoint_step,
                training_protocol_sha256=training_protocol)
              if (int(np.asarray(
                  bpd_archive['dequantization_seed']).item()) !=
                  bpd_dequant_seed):
                raise ValueError(
                  f'Cached BPD artifact has a different dequantization seed: '
                  f'{bpd_path}')
              cached_bpd = np.asarray(bpd_archive['bpd']).reshape(-1).copy()
            bpds.extend(cached_bpd)
            logging.info('Reusing validated BPD artifact: %s', bpd_path)
            continue
          eval_batch = torch.from_numpy(batch['image']._numpy()).to(config.device).float()
          eval_batch = eval_batch.permute(0, 3, 1, 2)
          with isolated_torch_rng(bpd_dequant_seed):
            eval_batch = (
              torch.rand_like(eval_batch) + eval_batch * 255.) / 256.
          eval_batch = scaler(eval_batch)
          with isolated_torch_rng(bpd_seed):
            bpd = likelihood_fn(score_model, eval_batch)[0]
          bpd = bpd.detach().cpu().numpy().reshape(-1)
          bpds.extend(bpd)
          logging.info(
            "ckpt: %s, repeat: %d, batch: %d, mean bpd: %6f" % (
              ckpt, repeat, batch_id, np.mean(np.asarray(bpds))))
          atomic_savez(
            bpd_path, overwrite=False, bpd=bpd,
            round_id=np.asarray(bpd_round_id),
            sampling_seed=np.asarray(bpd_seed),
            dequantization_seed=np.asarray(bpd_dequant_seed),
            **artifact_identity)

    # Generate samples and compute IS/FID/KID when enabled
    if config.eval.enable_sampling:
      requested_samples = int(config.eval.num_samples)
      sampling_batch_size = int(config.eval.batch_size)
      if requested_samples <= 0 or sampling_batch_size <= 0:
        raise ValueError('eval.num_samples and eval.batch_size must be positive.')
      num_sampling_rounds = int(math.ceil(
        requested_samples / float(sampling_batch_size)))
      sampling_seed_base = getattr(config.eval, 'sampling_seed', 0)
      required_stat_arrays = (
        ('pool_3', 'num_samples') if inceptionv3 else
        ('pool_3', 'logits', 'num_samples'))
      this_sample_dir = os.path.join(eval_dir, f"ckpt_{ckpt}")
      tf.io.gfile.makedirs(this_sample_dir)
      for r in range(num_sampling_rounds):
        sample_file = os.path.join(this_sample_dir, f"samples_{r}.npz")
        stat_file = os.path.join(this_sample_dir, f"statistics_{r}.npz")
        sampling_seed = derive_seed(
          sampling_seed_base, 'sampling-round', r)
        if tf.io.gfile.exists(stat_file):
          with tf.io.gfile.GFile(stat_file, 'rb') as stat_input:
            stat_archive = np.load(stat_input)
            validate_npz_metadata(
              stat_archive, stat_file, protocol_sha256, checkpoint_id,
              round_id=r, sampling_seed=sampling_seed,
              required_arrays=required_stat_arrays,
              checkpoint_step=checkpoint_step,
              training_protocol_sha256=training_protocol)
            if (int(np.asarray(stat_archive['num_samples']).item()) !=
                np.asarray(stat_archive['pool_3']).shape[0]):
              raise ValueError(
                f'Cached statistics have an inconsistent sample count: {stat_file}')
          logging.info(
            "sampling -- ckpt: %s, round: %d (reuse validated statistics)" %
            (ckpt, r))
          continue

        logging.info("sampling -- ckpt: %s, round: %d" % (ckpt, r))
        if tf.io.gfile.exists(sample_file):
          with tf.io.gfile.GFile(sample_file, "rb") as fin:
            sample_archive = np.load(fin)
            validate_npz_metadata(
              sample_archive, sample_file, protocol_sha256, checkpoint_id,
              round_id=r, sampling_seed=sampling_seed,
              required_arrays=('samples', 'nfe'),
              checkpoint_step=checkpoint_step,
              training_protocol_sha256=training_protocol)
            samples = np.asarray(sample_archive["samples"])
            if samples.shape[0] != sampling_batch_size:
              raise ValueError(
                f'Cached sample batch has {samples.shape[0]} images; expected '
                f'{sampling_batch_size}: {sample_file}')
        else:
          with isolated_torch_rng(sampling_seed):
            samples, n = sampling_fn(score_model)
          samples = np.clip(samples.permute(0, 2, 3, 1).cpu().numpy() * 255., 0, 255).astype(np.uint8)
          samples = samples.reshape(
            (-1, config.data.image_size, config.data.image_size, config.data.num_channels))
          atomic_savez(
            sample_file, overwrite=False, samples=samples,
            nfe=np.asarray(n), round_id=np.asarray(r),
            sampling_seed=np.asarray(sampling_seed), **artifact_identity)

        # Force garbage collection before calling TensorFlow code for Inception network
        gc.collect()
        latents = evaluation.run_inception_distributed(samples, inception_model,
                                                       inceptionv3=inceptionv3)
        # Force garbage collection again before returning to JAX code
        gc.collect()
        stat_arrays = dict(
          pool_3=latents["pool_3"],
          round_id=np.asarray(r),
          sampling_seed=np.asarray(sampling_seed),
          num_samples=np.asarray(samples.shape[0]), **artifact_identity)
        if not inceptionv3:
          stat_arrays['logits'] = latents['logits']
        atomic_savez(stat_file, overwrite=False, **stat_arrays)

      # Compute inception scores, FIDs and KIDs.
      # Load exactly the expected validated rounds; do not glob stale extras.
      all_logits = []
      all_pools = []
      for r in range(num_sampling_rounds):
        stat_file = os.path.join(this_sample_dir, f"statistics_{r}.npz")
        sampling_seed = derive_seed(
          sampling_seed_base, 'sampling-round', r)
        with tf.io.gfile.GFile(stat_file, "rb") as fin:
          stat = np.load(fin)
          validate_npz_metadata(
            stat, stat_file, protocol_sha256, checkpoint_id,
            round_id=r, sampling_seed=sampling_seed,
            required_arrays=required_stat_arrays,
            checkpoint_step=checkpoint_step,
            training_protocol_sha256=training_protocol)
          if not inceptionv3:
            all_logits.append(np.asarray(stat["logits"]))
          all_pools.append(np.asarray(stat["pool_3"]))

      if not inceptionv3:
        all_logits = np.concatenate(all_logits, axis=0)[:requested_samples]
      all_pools = np.concatenate(all_pools, axis=0)
      if all_pools.shape[0] < requested_samples:
        raise ValueError(
          f'Collected only {all_pools.shape[0]} validated samples; '
          f'{requested_samples} were requested.')
      all_pools = all_pools[:requested_samples]

      # Load pre-computed dataset statistics.
      if data_stats is None:
        data_stats = evaluation.load_dataset_stats(
          config, inception_model=inception_model)
        data_stats_id = evaluation.dataset_stats_fingerprint(config)
      data_pools = data_stats["pool_3"]

      # Compute FID/KID/IS on all samples together.
      if not inceptionv3:
        inception_score = tfgan.eval.classifier_score_from_logits(all_logits)
      else:
        inception_score = -1

      fid = tfgan.eval.frechet_classifier_distance_from_activations(
        data_pools, all_pools)
      # Hack to get tfgan KID work for eager execution.
      tf_data_pools = tf.convert_to_tensor(data_pools)
      tf_all_pools = tf.convert_to_tensor(all_pools)
      kid = tfgan.eval.kernel_classifier_distance_from_activations(
        tf_data_pools, tf_all_pools).numpy()
      del tf_data_pools, tf_all_pools

      logging.info(
        "ckpt-%s --- inception_score: %.6e, FID: %.6e, KID: %.6e" % (
          ckpt, inception_score, fid, kid))

      report_path = os.path.join(eval_dir, f"report_{ckpt}.npz")
      if tf.io.gfile.exists(report_path):
        with tf.io.gfile.GFile(report_path, 'rb') as report_file:
          report_archive = np.load(report_file)
          validate_npz_metadata(
            report_archive, report_path, protocol_sha256, checkpoint_id,
            required_arrays=('IS', 'fid', 'kid', 'num_samples',
                             'dataset_stats_id'),
            checkpoint_step=checkpoint_step,
            training_protocol_sha256=training_protocol)
          if (str(np.asarray(report_archive['dataset_stats_id']).item()) !=
              data_stats_id):
            raise ValueError(
              f'Cached report uses different reference dataset stats: {report_path}')
          current_metrics = {
            'IS': float(np.asarray(inception_score)),
            'fid': float(np.asarray(fid)),
            'kid': float(np.asarray(kid)),
          }
          for name, current_value in current_metrics.items():
            saved_value = float(np.asarray(report_archive[name]))
            if not np.isclose(saved_value, current_value, rtol=1e-6, atol=1e-8):
              raise ValueError(
                f'Cached report {name}={saved_value} disagrees with the '
                f'recomputed value {current_value}: {report_path}')
        logging.info('Keeping existing validated report: %s', report_path)
      else:
        atomic_savez(
          report_path, overwrite=False, IS=inception_score, fid=fid, kid=kid,
          num_samples=np.asarray(requested_samples),
          dataset_stats_id=np.asarray(data_stats_id), **artifact_identity)
