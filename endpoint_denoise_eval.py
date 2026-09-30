"""Paired endpoint-denoise evaluation for the VP/FOX E4 experiment."""
import gc
import logging
import math
import os

import numpy as np
import tensorflow as tf
import tensorflow_gan as tfgan

from artifact_utils import atomic_savez, validate_npz_metadata
from reproducibility import derive_seed, isolated_torch_rng
import evaluation

VARIANTS = ('x_mean', 'x_t', 'tweedie_xt')


def _read_npz(filename, protocol_sha256, checkpoint_id, checkpoint_step,
              training_protocol, round_id=None, sampling_seed=None,
              required_arrays=(), extra_identity=None):
  with tf.io.gfile.GFile(filename, 'rb') as input_file:
    with np.load(input_file) as archive:
      validate_npz_metadata(
        archive, filename, protocol_sha256, checkpoint_id,
        round_id=round_id, sampling_seed=sampling_seed,
        required_arrays=required_arrays,
        checkpoint_step=checkpoint_step,
        training_protocol_sha256=training_protocol,
        extra_identity=extra_identity)
      return {key: np.asarray(archive[key]).copy() for key in archive.files}


def evaluate_checkpoint(config, eval_dir, ckpt, score_model, sampling_fn,
                        inception_model, protocol_sha256, checkpoint_id,
                        checkpoint_step, training_protocol, artifact_identity,
                        extra_identity=None):
  """Generate three image outputs per trajectory and evaluate each identically."""
  if config.data.dataset != 'CELEBA' or config.data.image_size != 64:
    raise ValueError('E4 endpoint denoise evaluation is restricted to CelebA 64.')
  requested = int(config.eval.num_samples)
  batch_size = int(config.eval.batch_size)
  if requested <= 0 or batch_size <= 0:
    raise ValueError('eval.num_samples and eval.batch_size must be positive.')
  rounds = int(math.ceil(requested / float(batch_size)))
  sample_dir = os.path.join(eval_dir, f'ckpt_{ckpt}')
  tf.io.gfile.makedirs(sample_dir)
  pools = {variant: [] for variant in VARIANTS}
  logits = {variant: [] for variant in VARIANTS}
  route_metadata = getattr(score_model, 'expected_route_metadata', None)
  route_keys = tuple(route_metadata) if route_metadata is not None else ()
  sample_keys = tuple(f'samples_{variant}' for variant in VARIANTS) + ('nfe',) + route_keys
  stat_keys = tuple(
    key for variant in VARIANTS
    for key in (f'pool_3_{variant}', f'logits_{variant}')) + ('num_samples', 'nfe')

  for round_id in range(rounds):
    sample_file = os.path.join(sample_dir, f'samples_{round_id}.npz')
    stat_file = os.path.join(sample_dir, f'statistics_{round_id}.npz')
    seed = derive_seed(config.eval.sampling_seed, 'sampling-round', round_id)
    if tf.io.gfile.exists(sample_file):
      samples = _read_npz(
        sample_file, protocol_sha256, checkpoint_id, checkpoint_step,
        training_protocol, round_id=round_id, sampling_seed=seed,
        required_arrays=sample_keys, extra_identity=extra_identity)
      logging.info('E4: reuse sample batch %d', round_id)
    else:
      if route_metadata is not None:
        score_model.reset_route()
      with isolated_torch_rng(seed):
        outputs, nfe = sampling_fn(score_model)
      if route_metadata is not None:
        score_model.checked_route_metadata()
      if set(outputs) != set(VARIANTS):
        raise ValueError(f'Unexpected E4 sampler outputs: {tuple(outputs)}')
      samples = {
        f'samples_{variant}': np.clip(
          outputs[variant].permute(0, 2, 3, 1).cpu().numpy() * 255.,
          0, 255).astype(np.uint8)
        for variant in VARIANTS
      }
      samples.update(nfe=np.asarray(nfe), round_id=np.asarray(round_id),
                     sampling_seed=np.asarray(seed), **artifact_identity)
      if route_metadata is not None:
        samples.update({key: np.asarray(value)
                        for key, value in route_metadata.items()})
      atomic_savez(sample_file, overwrite=False, **samples)
    if route_metadata is not None:
      for key, expected in route_metadata.items():
        if not np.isclose(np.asarray(samples[key]), expected, rtol=0, atol=1e-5):
          raise ValueError(f'Incorrect splice route {key}: {sample_file}')
    if int(np.asarray(samples['nfe'])) != int(config.eval.sampling_num_scales) + 1:
      raise ValueError(f'Incorrect E4 network evaluation count: {sample_file}')
    for variant in VARIANTS:
      image_batch = samples[f'samples_{variant}']
      if image_batch.shape != (batch_size, 64, 64, config.data.num_channels):
        raise ValueError(f'Incorrect E4 image batch shape: {sample_file}')
      if image_batch.dtype != np.uint8:
        raise ValueError(f'Incorrect E4 image dtype: {sample_file}')

    if tf.io.gfile.exists(stat_file):
      stats = _read_npz(
        stat_file, protocol_sha256, checkpoint_id, checkpoint_step,
        training_protocol, round_id=round_id, sampling_seed=seed,
        required_arrays=stat_keys, extra_identity=extra_identity)
      logging.info('E4: reuse Inception batch %d', round_id)
    else:
      stats = dict(round_id=np.asarray(round_id), sampling_seed=np.asarray(seed),
                   num_samples=np.asarray(batch_size),
                   nfe=np.asarray(samples['nfe']), **artifact_identity)
      for variant in VARIANTS:
        gc.collect()
        result = evaluation.run_inception_distributed(
          samples[f'samples_{variant}'], inception_model)
        stats[f'pool_3_{variant}'] = result['pool_3'].numpy()
        stats[f'logits_{variant}'] = result['logits'].numpy()
      atomic_savez(stat_file, overwrite=False, **stats)
    if int(np.asarray(stats['num_samples'])) != batch_size or \
        int(np.asarray(stats['nfe'])) != int(np.asarray(samples['nfe'])):
      raise ValueError(f'Inconsistent E4 Inception batch: {stat_file}')
    for variant in VARIANTS:
      pool = np.asarray(stats[f'pool_3_{variant}'])
      logit = np.asarray(stats[f'logits_{variant}'])
      if pool.shape[0] != batch_size or logit.shape[0] != batch_size:
        raise ValueError(f'Incorrect E4 Inception batch length: {stat_file}')
      pools[variant].append(pool)
      logits[variant].append(logit)

  data_stats = evaluation.load_dataset_stats(config, inception_model=inception_model)
  data_stats_id = evaluation.dataset_stats_fingerprint(config)
  report = dict(num_samples=np.asarray(requested),
                dataset_stats_id=np.asarray(data_stats_id),
                nfe=np.asarray(int(config.eval.sampling_num_scales) + 1),
                **artifact_identity)
  if route_metadata is not None:
    report.update({key: np.asarray(value)
                   for key, value in route_metadata.items()})
    report['splice_vp_calls_total'] = np.asarray(
      rounds * route_metadata['splice_vp_calls'])
    report['splice_fox_calls_total'] = np.asarray(
      rounds * route_metadata['splice_fox_calls'])
  data_pools = data_stats['pool_3']
  for variant in VARIANTS:
    all_pools = np.concatenate(pools[variant], axis=0)[:requested]
    all_logits = np.concatenate(logits[variant], axis=0)[:requested]
    report[f'IS_{variant}'] = np.asarray(
      tfgan.eval.classifier_score_from_logits(all_logits))
    report[f'fid_{variant}'] = np.asarray(
      tfgan.eval.frechet_classifier_distance_from_activations(
        data_pools, all_pools))
    report[f'kid_{variant}'] = np.asarray(
      tfgan.eval.kernel_classifier_distance_from_activations(
        tf.convert_to_tensor(data_pools),
        tf.convert_to_tensor(all_pools)).numpy())
    logging.info(
      'ckpt-%s %s --- inception_score: %.6e, FID: %.6e, KID: %.6e',
      ckpt, variant, report[f'IS_{variant}'],
      report[f'fid_{variant}'], report[f'kid_{variant}'])

  report_path = os.path.join(eval_dir, f'report_{ckpt}.npz')
  if tf.io.gfile.exists(report_path):
    saved = _read_npz(
      report_path, protocol_sha256, checkpoint_id, checkpoint_step,
      training_protocol,
      required_arrays=tuple(report.keys()), extra_identity=extra_identity)
    if str(np.asarray(saved['dataset_stats_id']).item()) != data_stats_id or \
        int(np.asarray(saved['num_samples'])) != requested or \
        int(np.asarray(saved['nfe'])) != int(np.asarray(report['nfe'])):
      raise ValueError(f'Cached E4 report uses different reference data: {report_path}')
    for variant in VARIANTS:
      for metric in ('IS', 'fid', 'kid'):
        key = f'{metric}_{variant}'
        if not np.isclose(saved[key], report[key], rtol=1e-6, atol=1e-8):
          raise ValueError(f'Cached E4 {key} disagrees with recomputed value: {report_path}')
    logging.info('Keeping existing validated E4 report: %s', report_path)
  else:
    atomic_savez(report_path, overwrite=False, **report)
