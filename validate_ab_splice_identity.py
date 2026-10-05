#!/usr/bin/env python3
"""Check that all-A/all-B shared-log-SNR routes reproduce direct A/B samples."""
import argparse
from pathlib import Path
import numpy as np

VARIANTS = ('x_mean', 'x_t', 'tweedie_xt')


def load_samples(folder, checkpoint):
  sample_dir = Path(folder) / f'ckpt_{checkpoint}'
  batches = {key: [] for key in VARIANTS}
  for path in sorted(sample_dir.glob('samples_*.npz'),
                     key=lambda item: int(item.stem.split('_')[-1])):
    with np.load(path) as archive:
      for key in VARIANTS:
        batches[key].append(np.asarray(archive[f'samples_{key}']))
  if not batches['x_t']:
    raise FileNotFoundError(f'No sample shards in {sample_dir}')
  return {key: np.concatenate(parts, axis=0) for key, parts in batches.items()}


def main():
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument('--direct-a', required=True)
  parser.add_argument('--direct-b', required=True)
  parser.add_argument('--route-a', required=True, help='splice mode none folder')
  parser.add_argument('--route-b', required=True, help='splice mode full folder')
  parser.add_argument('--checkpoint', type=int, default=40)
  args = parser.parse_args()
  comparisons = ((args.direct_a, args.route_a, 'A'),
                 (args.direct_b, args.route_b, 'B'))
  for direct, routed, label in comparisons:
    a = load_samples(direct, args.checkpoint)
    b = load_samples(routed, args.checkpoint)
    for key in VARIANTS:
      if a[key].shape != b[key].shape or not np.array_equal(a[key], b[key]):
        delta = (float(np.abs(a[key].astype(np.int16) -
                              b[key].astype(np.int16)).max())
                 if a[key].shape == b[key].shape else None)
        raise SystemExit(f'{label}/{key}: all-same-network route mismatch; max uint8 delta={delta}')
    print(f'{label}: direct and all-same-network route match exactly ({len(a["x_t"])} samples)')


if __name__ == '__main__':
  main()
