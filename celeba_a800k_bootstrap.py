"""Conditional bootstrap intervals for DIAL 800k - A 800k and B 800k - A 800k on CelebA (CPU only).

Optional, started by hand after celeba_a800k_eval.py. The method and the code of
paper_diff_bootstrap.py, which is imported unchanged: each cell is resampled independently from
its own 50k pool_3_tweedie_xt features, checkpoints and reference statistics fixed; per solver
and seed, the mean over seeds, and the difference of contrasts between solvers.

One part, celeba_a800k: DIAL 800k, B 800k (the cells paper_diff_bootstrap.py reads) and A 800k
(celeba_a800k_50k_20261115, both lanes in one root). DIAL 800k and B 800k keep their indices in
paper_diff_bootstrap.CELLS, so their draws repeat those of its celeba part draw for draw; the A
800k cells take the next free indices, 36 to 39. The output is separate from
paper_diff_bootstrap_20261115, whose containers keep running on their own code.
"""
import json
import sys

import paper_diff_bootstrap as pdb

PART = 'celeba_a800k'
ROOT_KEY = 'a800k'
ROOTS = {'fox800': pdb.ROOTS['fox800'], 'gpu2q': pdb.ROOTS['gpu2q'], ROOT_KEY: 'celeba_a800k_50k_20261115'}
A800 = dict(dataset='CELEBA', label='A 800k', part=PART, root=ROOT_KEY, work='celeba_a',
            folder='eval_a800k_celeba_a_c160_m35_{method}_seed{seed}', ckpt=160, step=800001, checkpoint_id=None,
            result='celeba_a_c160_m35_{method}_seed{seed}',
            training='H200 to 200k, RTX 200k-400k, H200 400k-800k (celeba_a_800k_h200_20261113)',
            training_gpu=('H200', 'RTX'))
MODELS = {'celeba_dial800': dict(pdb.MODELS['celeba_dial800'], part=PART),
          'celeba_b800': dict(pdb.MODELS['celeba_b800'], part=PART),
          'celeba_a800': A800}
DIFFS = (('celeba_dial800', 'celeba_a800'), ('celeba_b800', 'celeba_a800'))
CELLS = [dict(name=pdb.cell_name(model, method, seed), model=model, method=method, seed=seed)
         for model in MODELS for seed in pdb.SEEDS for method in pdb.METHODS]
NEW = [cell['name'] for cell in CELLS if cell['name'] not in pdb.INDEX]  # The A 800k cells.
INDEX = {cell['name']: pdb.INDEX[cell['name']] if cell['name'] in pdb.INDEX else len(pdb.CELLS) + NEW.index(cell['name'])
         for cell in CELLS}

_audit_cell = pdb.audit_cell


def audit_cell(inputs, cell, reference_sha, out):
    """The audit of paper_diff_bootstrap; the GPU of an A 800k cell comes from the preflight of its lane."""
    record = _audit_cell(inputs, cell, reference_sha, out)
    if cell['model'] == 'celeba_a800':
        path = inputs / ROOT_KEY / 'logs' / 'preflight_s{}.json'.format(cell['seed'])
        record['sampling_gpu'] = (json.loads(path.read_text()).get('environment', {}).get('visible_cuda_devices')
                                  if path.exists() else None)
    return record


def install():
    """Points the functions of paper_diff_bootstrap at this part."""
    pdb.ROOTS, pdb.MODELS, pdb.DIFFS, pdb.PARTS = ROOTS, MODELS, DIFFS, (PART,)
    pdb.CELLS, pdb.INDEX, pdb.audit_cell = CELLS, INDEX, audit_cell


def check():
    pdb.check()  # The original tables, before install.
    assert len(CELLS) == 12 and len(set(INDEX.values())) == 12
    assert [INDEX[c['name']] for c in CELLS if c['model'] == 'celeba_a800'] == [36, 37, 38, 39]
    for model in ('celeba_dial800', 'celeba_b800'):
        for cell in CELLS:
            if cell['model'] == model:
                assert INDEX[cell['name']] == pdb.INDEX[cell['name']]
                assert pdb.MODELS[model]['part'] == 'celeba'
    install()
    assert [len(pdb.part_cells(p)) for p in pdb.PARTS] == [12]
    for left, right in DIFFS:
        assert MODELS[left]['dataset'] == MODELS[right]['dataset'] == 'CELEBA'
    print('A 800k bootstrap checks passed', flush=True)


def main():
    if '--check' in sys.argv[1:]:
        check()
        return
    install()
    pdb.main()


if __name__ == '__main__':
    main()
