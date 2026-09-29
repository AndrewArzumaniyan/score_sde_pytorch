#!/usr/bin/env python3
"""E2: paired CelebA test-set epsilon MSE versus log-SNR for old VP/FOX.

Run this from a separate checkout while E1 is active: adding this Python file
changes the source fingerprint checked by E1 in its current checkout.
"""

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

import tensorflow as tf
import tensorflow_datasets as tfds
import torch

import datasets
import losses
from artifact_utils import (atomic_savez, atomic_write_bytes,
                            build_model_manifest, _source_fingerprint)
from configs.fox.celeba_ncsnpp_continuous_matern_3_2 import get_config as fox_config
from configs.vp.celeba_ncsnpp_continuous import get_config as vp_config
from models import ddpm, ncsnpp  # Register model classes.
from models.ema import ExponentialMovingAverage
from models import utils as mutils
from run_lib import get_sde
from utils import restore_checkpoint


FOX_WORKDIR = 'workdirs/celeba_fox_matern32_kappa1000_50k'
VP_WORKDIR = 'workdirs/celeba_vp_continuous_50k'
E1_ENDPOINT = 10.575749165617202


def sha256_file(path):
  digest = hashlib.sha256()
  with open(path, 'rb') as source:
    for block in iter(lambda: source.read(1024 * 1024), b''):
      digest.update(block)
  return digest.hexdigest()


def make_grid(pilot):
  if pilot:
    return np.asarray([-8.0, 0.0, E1_ENDPOINT], dtype=np.float64)
  # Include the E1 clean endpoint exactly; keep the farther clean tail separate.
  return np.concatenate((
    np.linspace(-10.0, E1_ENDPOINT, 35),
    np.linspace(E1_ENDPOINT, 13.8, 6)[1:],
  ))


def prepared_config(name, device):
  config = vp_config() if name == 'vp' else fox_config()
  if name == 'fox':
    # Match run_1300k_endpoint_controls_eval.sh and its completed manifests.
    config.model.fox_u = -5.0
    config.model.fox_matern_length_scale = 346.4101615138
    config.model.fox_target_terminal_variance = 1.0
  config.device = device
  config.allow_tf32 = True
  config.cudnn_benchmark = True
  return config


def load_test_images(config, limit):
  """Use the existing CelebA transform, selecting test explicitly."""
  local_paths = datasets.get_local_celeba_split_paths('test', config)
  if local_paths is not None:
    source = 'local-celeba-test'
    paths = local_paths[:limit]
    raw = tf.data.Dataset.from_tensor_slices(paths)

    def decode(path):
      image = tf.image.decode_jpeg(tf.io.read_file(path), channels=3)
      image.set_shape([218, 178, 3])
      return image

    images = raw.map(decode, num_parallel_calls=tf.data.experimental.AUTOTUNE)
  else:
    source = 'tfds-celeb_a-test'
    builder = tfds.builder('celeb_a', data_dir=os.environ.get('TFDS_DATA_DIR'))
    images = builder.as_dataset(split='test', shuffle_files=False).take(limit)
    images = images.map(lambda example: example['image'],
                        num_parallel_calls=tf.data.experimental.AUTOTUNE)

  def preprocess(image):
    image = tf.image.convert_image_dtype(image, tf.float32)
    image = datasets.central_crop(image, 140)
    return datasets.resize_small(image, config.data.image_size)

  images = images.map(preprocess, num_parallel_calls=tf.data.experimental.AUTOTUNE)
  result = np.empty((limit, 3, 64, 64), dtype=np.float32)
  count = 0
  for image in tfds.as_numpy(images):
    if count >= limit:
      break
    result[count] = np.transpose(image, (2, 0, 1))
    count += 1
  if count != limit:
    raise ValueError('CelebA test has %d images; requested %d' % (count, limit))
  return result, source


def level_parameters(sde, levels, device):
  result = []
  for target in levels:
    desired = torch.tensor([float(target)], device=device, dtype=torch.float32)
    with torch.no_grad():
      t = sde.time_from_log_snr(desired)
      mean, std = sde.marginal_prob(
        torch.ones((1, 1, 1, 1), device=device), t)
      alpha = float(mean.item())
      sigma = float(std.item())
      actual = float(sde.log_snr(t).item())
    if not (np.isfinite(alpha) and np.isfinite(sigma) and
            np.isfinite(actual) and 1e-5 <= float(t.item()) <= 1.0 and
            abs(actual - target) < 0.003):
      raise ValueError('Invalid SDE inversion at lambda=%g: t=%g, actual=%g' %
                       (target, t.item(), actual))
    result.append((float(t.item()), alpha, sigma, actual))
  return np.asarray(result, dtype=np.float64)


def paired_noise(shape, seed, level_index, batch_start):
  rng = np.random.default_rng(np.random.SeedSequence(
    [seed, level_index, batch_start]))
  return rng.standard_normal(shape).astype(np.float32)


def evaluate_checkpoint(config, sde, parameters, levels, images, args, ckpt):
  path = args.vp_workdir if config.training.sde == 'vpsde' else args.fox_workdir
  path = path / 'checkpoints' / ('checkpoint_%d.pth' % ckpt)
  if not path.is_file():
    raise FileNotFoundError(str(path))
  model = mutils.create_model(config)
  optimizer = losses.get_optimizer(config, model.parameters())
  ema = ExponentialMovingAverage(model.parameters(), decay=config.model.ema_rate)
  state = dict(model=model, optimizer=optimizer, ema=ema, step=0,
               model_protocol_sha256=build_model_manifest(config)[
                 'model_protocol_sha256'])
  state = restore_checkpoint(str(path), state, config.device, restore_rng=False)
  ema.copy_to(model.parameters())
  model.eval()
  score_fn = mutils.get_score_fn(sde, model, train=False, continuous=True)
  errors = np.empty((len(levels), len(images)), dtype=np.float32)
  scaler = datasets.get_data_scaler(config)
  with torch.inference_mode():
    for i, target in enumerate(levels):
      t_value, alpha, sigma, _ = parameters[i]
      for start in range(0, len(images), args.batch_size):
        stop = min(start + args.batch_size, len(images))
        clean = torch.from_numpy(images[start:stop]).to(config.device)
        clean = scaler(clean)
        noise = torch.from_numpy(paired_noise(
          tuple(clean.shape), args.seed, i, start)).to(config.device)
        noisy = alpha * clean + sigma * noise
        t = torch.full((stop - start,), float(t_value),
                       device=config.device, dtype=torch.float32)
        estimated = -sigma * score_fn(noisy, t)
        per_image = (estimated - noise).square().flatten(1).mean(dim=1)
        errors[i, start:stop] = per_image.cpu().numpy()
      print('%s ckpt=%d lambda=%+.5f mse=%.7f' %
            (config.training.sde, ckpt, target, errors[i].mean()), flush=True)
  return errors, str(state['checkpoint_id']), int(state['step'])


def main():
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument('--pilot', action='store_true')
  parser.add_argument('--checkpoints', type=int, nargs='+', default=[20, 80, 160, 260])
  parser.add_argument('--num-images', type=int, default=None)
  parser.add_argument('--batch-size', type=int, default=128)
  parser.add_argument('--seed', type=int, default=20260929)
  parser.add_argument('--output', type=Path)
  parser.add_argument('--vp-workdir', type=Path, default=ROOT / VP_WORKDIR)
  parser.add_argument('--fox-workdir', type=Path, default=ROOT / FOX_WORKDIR)
  args = parser.parse_args()
  if args.num_images is None:
    args.num_images = 128 if args.pilot else 19962
  if args.output is None:
    args.output = Path('workdirs/e2_pilot' if args.pilot else 'workdirs/e2_results')
  if args.num_images < 1 or args.num_images > 19962 or args.batch_size < 1:
    parser.error('num-images must be in [1,19962] and batch-size must be positive')
  if args.seed < 0:
    parser.error('seed must be nonnegative')
  if len(set(args.checkpoints)) != len(args.checkpoints):
    parser.error('checkpoint list contains duplicates')
  tf.config.set_visible_devices([], 'GPU')
  if not torch.cuda.is_available():
    raise RuntimeError('E2 requires the selected CUDA GPU')
  device = torch.device('cuda:0')
  torch.backends.cuda.matmul.allow_tf32 = True
  torch.backends.cudnn.allow_tf32 = True
  torch.backends.cudnn.benchmark = True
  levels = make_grid(args.pilot)
  vp = prepared_config('vp', device)
  fox = prepared_config('fox', device)
  # Fail before large allocations if any requested checkpoint is absent.
  for ckpt in args.checkpoints:
    for workdir in (args.vp_workdir, args.fox_workdir):
      path = workdir / 'checkpoints' / ('checkpoint_%d.pth' % ckpt)
      if not path.is_file():
        raise FileNotFoundError(str(path))
  images, source = load_test_images(vp, args.num_images)
  image_hash = hashlib.sha256(memoryview(images)).hexdigest()
  vp_sde, _ = get_sde(vp)
  fox_sde, _ = get_sde(fox)
  parameters = {
    'vp': level_parameters(vp_sde, levels, device),
    'fox': level_parameters(fox_sde, levels, device),
  }
  protocol = {
    'schema': 1,
    'script_sha256': sha256_file(__file__),
    'runtime_source_sha256': _source_fingerprint(str(ROOT)),
    'model_manifest_sha256': {
      'vp': build_model_manifest(vp)['model_protocol_sha256'],
      'fox': build_model_manifest(fox)['model_protocol_sha256'],
    },
    'device_name': torch.cuda.get_device_name(device),
    'dataset': source,
    'dataset_preprocessed_sha256': image_hash,
    'num_images': args.num_images,
    'batch_size': args.batch_size,
    'seed': args.seed,
    'checkpoints': args.checkpoints,
    'vp_workdir': str(args.vp_workdir.resolve()),
    'fox_workdir': str(args.fox_workdir.resolve()),
    'lambda': levels.tolist(),
    'parameters': {name: value.tolist() for name, value in parameters.items()},
    'metric': 'per-image mean squared epsilon prediction error in centered image coordinates',
  }
  args.output.mkdir(parents=True, exist_ok=True)
  manifest_path = args.output / 'manifest.json'
  manifest = (json.dumps(protocol, sort_keys=True, indent=2) + '\n').encode()
  if manifest_path.exists():
    if manifest_path.read_bytes() != manifest:
      raise ValueError('Existing E2 output has a different protocol: %s' % args.output)
  else:
    if any(args.output.iterdir()):
      raise ValueError('Output directory is nonempty without a manifest: %s' % args.output)
    atomic_write_bytes(str(manifest_path), manifest, overwrite=False)

  for ckpt in args.checkpoints:
    results = {}
    for name, config, sde in (('vp', vp, vp_sde), ('fox', fox, fox_sde)):
      result_path = args.output / ('%s_ckpt_%d.npz' % (name, ckpt))
      if result_path.exists():
        with np.load(str(result_path)) as saved:
          if not np.array_equal(saved['lambda'], levels):
            raise ValueError('Cached lambda grid mismatch: %s' % result_path)
          if saved['per_image_epsilon_mse'].shape != (len(levels), len(images)):
            raise ValueError('Cached result shape mismatch: %s' % result_path)
          results[name] = saved['per_image_epsilon_mse'].copy()
        print('Reused %s' % result_path, flush=True)
        continue
      errors, checkpoint_id, checkpoint_step = evaluate_checkpoint(
        config, sde, parameters[name], levels, images, args, ckpt)
      atomic_savez(str(result_path), overwrite=False,
                   per_image_epsilon_mse=errors, **{'lambda': levels},
                   checkpoint_id=np.asarray(checkpoint_id),
                   checkpoint_step=np.asarray(checkpoint_step))
      results[name] = errors
    difference = results['fox'].astype(np.float64) - results['vp']
    mean = difference.mean(axis=1)
    stderr = difference.std(axis=1, ddof=1) / np.sqrt(len(images)) if len(images) > 1 else np.zeros(len(levels))
    summary_path = args.output / ('paired_ckpt_%d.csv' % ckpt)
    if not summary_path.exists():
      rows = np.column_stack((levels, results['vp'].mean(axis=1),
                              results['fox'].mean(axis=1), mean, stderr))
      header = 'lambda,vp_epsilon_mse,fox_epsilon_mse,fox_minus_vp,paired_se'
      import io
      buffer = io.StringIO()
      np.savetxt(buffer, rows, delimiter=',', header=header, comments='')
      atomic_write_bytes(str(summary_path), buffer.getvalue().encode(), overwrite=False)
    print('Completed paired checkpoint %d: %s' % (ckpt, summary_path), flush=True)


if __name__ == '__main__':
  main()
