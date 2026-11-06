"""Second training seed of the native VP/FOX pair: FOX, then VP, on one H200.

Both arms start from the configs stage 0 evaluates as native VP and FOX, so the
recipe is the May one. The seed, precision flags, environment and seeded data
pipeline differ. Compare the VP-FOX gap inside this pair; do not mix its absolute
FID with the May pair, which ran under other code and without these flags.

allow_tf32 and cudnn_benchmark follow A-G and the A/B seeds. The repository
default turns TF32 off, which would lose the 0.33 s/step this plan relies on.

Run through run_celeba_pair_seed43_h200.sh from a separate worktree. This file
is also the training config: main.py --config=celeba_pair_seed.py:fox.
"""
from datetime import datetime
import json
from pathlib import Path
import re
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parent
SEED = 43
TARGET_ITERS = 800000
SNAPSHOT_FREQ = 5000  # Same checkpoint indices as the native pair: 40 is 200k.
FINAL_CKPT = TARGET_ITERS // SNAPSHOT_FREQ
ARMS = ('fox', 'vp')
WORK = dict(fox='workdirs/celeba_fox_matern32_kappa1000_h200_seed43',
            vp='workdirs/celeba_vp_continuous_h200_seed43')
SMOKE = 'workdirs/_smoke_pair'
SMOKE_ITERS = 300  # cuDNN autotuning ends well before step 100.
SMOKE_RESUME_ITERS = 350
LOG_LINE = re.compile(r'(\d{4}-\d\d-\d\d \d\d:\d\d:\d\d,\d{3}) - step: (\d+), training_loss')


def get_config(arm):
    from celeba_stage0_eval import native_config
    if arm not in ARMS:
        raise ValueError('Unknown arm: ' + arm)
    config = native_config(arm)
    config.seed = SEED
    config.allow_tf32 = True
    config.cudnn_benchmark = True
    training = config.training
    training.batch_size = 128
    training.n_iters = TARGET_ITERS
    training.snapshot_freq = SNAPSHOT_FREQ
    training.snapshot_freq_for_preemption = 5000
    training.snapshot_sampling = False
    training.log_freq = 50
    training.eval_freq = 500
    return config


def flatten(value, prefix=''):
    if not isinstance(value, dict):
        return {prefix[:-1]: value}
    result = {}
    for key, item in value.items():
        result.update(flatten(item, prefix + key + '.'))
    return result


def differing_keys(left, right):
    left, right = flatten(left), flatten(right)
    return {key for key in set(left) | set(right)
            if key not in left or key not in right or left[key] != right[key]}


def check_pair(fox, vp):
    """Refuses a pair that the gap between its arms would not be fair for."""
    for name, manifest in (('fox', fox), ('vp', vp)):
        if manifest['config']['seed'] != SEED:
            raise ValueError(name + ': unexpected seed.')
        if 'H200' not in str(manifest['environment']['visible_cuda_devices']):
            raise ValueError(name + ': not trained on an H200.')
    for key in ('source_sha256', 'environment'):
        if fox[key] != vp[key]:
            raise ValueError('The arms differ in ' + key)
    extra = {key for key in differing_keys(fox['config'], vp['config'])
             if key != 'training.sde' and not key.startswith('model.fox_')}
    if extra:
        raise ValueError('The arms differ outside the SDE: {}'.format(sorted(extra)))


def read_manifest(work):
    return json.loads((ROOT / work / 'run_manifest.json').read_text())


def final_path(arm):
    return ROOT / WORK[arm] / 'checkpoints/checkpoint_{}.pth'.format(FINAL_CKPT)


def verify(arm):
    import torch
    manifest = read_manifest(WORK[arm])
    final = torch.load(str(final_path(arm)), map_location='cpu', weights_only=False)
    if (int(final['step']) != TARGET_ITERS + 1 or
            final.get('training_protocol_sha256') != manifest['training_protocol_sha256']):
        raise ValueError('The final checkpoint differs from this run: ' + arm)
    if arm == 'vp':
        check_pair(read_manifest(WORK['fox']), manifest)
    print('{} reached state.step {}: {}'.format(arm, int(final['step']), final_path(arm)), flush=True)


def run_main(arm, work, *overrides):
    subprocess.check_call([sys.executable, '-u', 'main.py', '--mode=train',
                           '--config=celeba_pair_seed.py:' + arm, '--workdir=' + work] + list(overrides),
                          cwd=str(ROOT))


def train(arm):
    """A rerun resumes; a finished arm is skipped, since resuming it replays all its batches."""
    if final_path(arm).exists():
        print('Already trained: ' + arm, flush=True)
    else:
        run_main(arm, WORK[arm])
    verify(arm)


def seconds_per_step(stdout, first=100, last=SMOKE_ITERS):
    times = {}
    for line in stdout.read_text().splitlines():
        match = LOG_LINE.search(line)
        if match:
            times.setdefault(int(match.group(2)),
                             datetime.strptime(match.group(1), '%Y-%m-%d %H:%M:%S,%f'))
    return (times[last] - times[first]).total_seconds() / (last - first)


def smoke():
    """Both arms for 300 updates, then a resume: the real image, GPU, data and config."""
    if (ROOT / SMOKE).exists():
        shutil.rmtree(str(ROOT / SMOKE))
    speed = {}
    for arm in ARMS:
        work = '{}/{}'.format(SMOKE, arm)
        run_main(arm, work, '--config.training.n_iters={}'.format(SMOKE_ITERS))
        speed[arm] = seconds_per_step(ROOT / work / 'stdout.txt')
        run_main(arm, work, '--config.training.n_iters={}'.format(SMOKE_RESUME_ITERS))
        if not (ROOT / work / 'checkpoints/checkpoint_final_step_{}.pth'.format(SMOKE_RESUME_ITERS + 1)).exists():
            raise ValueError('The resume did not reach its final state: ' + arm)
    check_pair(read_manifest(SMOKE + '/fox'), read_manifest(SMOKE + '/vp'))
    shutil.rmtree(str(ROOT / SMOKE))
    for arm in ARMS:
        print('{}: {:.3f} s/step, expected about 0.33; 800k takes {:.1f} h'.format(
            arm, speed[arm], speed[arm] * TARGET_ITERS / 3600), flush=True)


if __name__ == '__main__':
    if sys.argv[1:] == ['smoke']:
        smoke()
    elif len(sys.argv) == 3 and sys.argv[1] == 'train' and sys.argv[2] in ARMS:
        train(sys.argv[2])
    else:
        raise SystemExit('Usage: celeba_pair_seed.py smoke | train fox|vp')
