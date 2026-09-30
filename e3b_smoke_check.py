"""Check E3b identity/full smoke batches against E4 and make a contact sheet."""
import argparse
from pathlib import Path

import numpy as np
from PIL import Image

VARIANTS = ('x_mean', 'x_t', 'tweedie_xt')


def load(folder, checkpoint):
  filename = Path(folder) / f'ckpt_{checkpoint}' / 'samples_0.npz'
  with np.load(filename) as archive:
    return {key: np.asarray(archive[key]).copy() for key in archive.files}


def contact_sheet(images, filename, columns=8, count=64):
  chosen = images[:count]
  height, width = chosen.shape[1:3]
  rows = (len(chosen) + columns - 1) // columns
  canvas = Image.new('RGB', (columns * width, rows * height))
  for index, array in enumerate(chosen):
    canvas.paste(Image.fromarray(array),
                 ((index % columns) * width, (index // columns) * height))
  canvas.save(filename)


def main():
  ap = argparse.ArgumentParser()
  ap.add_argument('--baseline', required=True, help='E4 VP/extended eval folder')
  ap.add_argument('--none', required=True, help='E3b none smoke eval folder')
  ap.add_argument('--full', required=True, help='E3b full smoke eval folder')
  ap.add_argument('--checkpoint', default='260')
  ap.add_argument('--contact-sheet', required=True)
  args = ap.parse_args()
  baseline = load(args.baseline, args.checkpoint)
  identity = load(args.none, args.checkpoint)
  full = load(args.full, args.checkpoint)
  passed = True
  for variant in VARIANTS:
    key = f'samples_{variant}'
    a, b = baseline[key], identity[key]
    if a.shape != b.shape or a.dtype != b.dtype:
      raise ValueError(f'{key}: shape or dtype differs: {a.shape}/{a.dtype} vs {b.shape}/{b.dtype}')
    equal = np.array_equal(a, b)
    fraction = float(np.mean(a != b))
    difference = int(np.max(np.abs(a.astype(np.int16) - b.astype(np.int16))))
    print(f'{variant}: identical={equal}, changed_fraction={fraction:.6g}, max_abs={difference}')
    passed &= equal
  for label, sample, expected_fox in (
      ('none', identity, 0), ('full', full, 1001)):
    nfe = int(sample['nfe'])
    vp = int(sample['splice_vp_calls'])
    fox = int(sample['splice_fox_calls'])
    if nfe != 1001 or vp + fox != nfe or fox != expected_fox:
      raise ValueError(f'{label}: bad route counts: nfe={nfe}, vp={vp}, fox={fox}')
    print(f'{label}: nfe={nfe}, VP={vp}, FOX={fox}')
  contact_sheet(full['samples_tweedie_xt'], args.contact_sheet)
  print(f'contact sheet: {args.contact_sheet}')
  if not passed:
    raise SystemExit('Identity arrays differ; inspect before the full run.')


if __name__ == '__main__':
  main()
