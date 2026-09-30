#!/usr/bin/env python3
"""Validate and summarize the four E4 endpoint-denoise reports."""
import argparse
import json
from pathlib import Path

import numpy as np

VARIANTS = ('x_mean', 'x_t', 'tweedie_xt')
ENDPOINTS = {'common': 9.115429865459795,
             'extended': 10.575749165617202}


def read_case(workdir, model, endpoint, args):
  folder = (f'eval_e4_endpoint_denoise_{args.tag}_{model}_{endpoint}'
            f'_ckpt{args.ckpt}_nfe1000_{args.samples}samples_'
            f'batch{args.batch_size}_seed{args.seed}')
  directory = workdir / folder
  with (directory / 'manifest.json').open() as input_file:
    manifest = json.load(input_file)
  config = manifest['config']
  expected = {
    ('eval', 'num_samples'): args.samples,
    ('eval', 'batch_size'): args.batch_size,
    ('eval', 'sampling_seed'): args.seed,
    ('eval', 'sampling_num_scales'): 1000,
    ('eval', 'endpoint_denoise_variants'): True,
    ('sampling', 'time_grid'): 'uniform_logsnr',
    ('sampling', 'logsnr_min'): -10.0,
    ('sampling', 'logsnr_max'): ENDPOINTS[endpoint],
    ('sampling', 'method'): 'pc',
    ('sampling', 'predictor'): 'euler_maruyama',
    ('sampling', 'corrector'): 'none',
  }
  for (section, key), value in expected.items():
    if config[section].get(key) != value:
      raise ValueError(f'{directory}: unexpected {section}.{key}')
  with np.load(directory / f'report_{args.ckpt}.npz') as report:
    if str(report['protocol_sha256'].item()) != manifest['protocol_sha256']:
      raise ValueError(f'{directory}: report/manifest protocol mismatch')
    if int(report['num_samples']) != args.samples or int(report['nfe']) != 1001:
      raise ValueError(f'{directory}: wrong sample count or NFE')
    result = {
      'reference': str(report['dataset_stats_id'].item()),
      'checkpoint_id': str(report['checkpoint_id'].item()),
      'step': int(report['checkpoint_step']),
      'fid': {v: float(report[f'fid_{v}']) for v in VARIANTS},
      'kid': {v: float(report[f'kid_{v}']) for v in VARIANTS},
    }
  return result


def main():
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument('--vp-workdir', type=Path,
                      default=Path('workdirs/celeba_vp_continuous_50k'))
  parser.add_argument('--fox-workdir', type=Path,
                      default=Path('workdirs/celeba_fox_matern32_kappa1000_50k'))
  parser.add_argument('--tag', default='v1')
  parser.add_argument('--ckpt', type=int, default=260)
  parser.add_argument('--samples', type=int, default=10000)
  parser.add_argument('--batch-size', type=int, default=512)
  parser.add_argument('--seed', type=int, default=0)
  args = parser.parse_args()
  results = {
    (model, endpoint): read_case(workdir, model, endpoint, args)
    for model, workdir in (('vp', args.vp_workdir), ('fox', args.fox_workdir))
    for endpoint in ENDPOINTS
  }
  if len({r['reference'] for r in results.values()}) != 1:
    raise ValueError('E4 cases use different reference statistics')
  for model in ('vp', 'fox'):
    if results[model, 'common']['checkpoint_id'] != \
        results[model, 'extended']['checkpoint_id']:
      raise ValueError(f'{model}: endpoints use different checkpoints')
  print('variant       model endpoint      FID        KID')
  for variant in VARIANTS:
    for model in ('vp', 'fox'):
      for endpoint in ENDPOINTS:
        case = results[model, endpoint]
        print(f'{variant:13} {model:5} {endpoint:9} '
              f'{case["fid"][variant]:9.6f} {case["kid"][variant]:10.8f}')
    for model in ('vp', 'fox'):
      difference = (results[model, 'common']['fid'][variant] -
                    results[model, 'extended']['fid'][variant])
      print(f'{variant}: {model} common−extended FID = {difference:+.6f}')
    for endpoint in ENDPOINTS:
      difference = (results['vp', endpoint]['fid'][variant] -
                    results['fox', endpoint]['fid'][variant])
      print(f'{variant}: {endpoint} VP−FOX FID = {difference:+.6f}')


if __name__ == '__main__':
  main()
