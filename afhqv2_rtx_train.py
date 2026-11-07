"""AFHQv2-64 on RTX PRO 6000: FOX, then A and VP, the CelebA recipe on a second dataset.

Each arm is its CelebA config with the data swapped: native FOX (Matern-3/2, kappa = 1000,
u = -5, constant drift, q(1) = 1) and native VP as stage 0 evaluates them, and A, the pure VP
with log-SNR labels. NCSN++, batch 128, TF32 as in the seed 43 pair, training seed 42.

The repository's AFHQv2 configs are not this recipe, so they are not used: their
apply_afhqv2_data sets centered=False, which is why data.centered is set back here. The August
pilot ran with [0, 1] inputs, so its FID does not compare with these runs. Centering touches
only the scaler pair: reference statistics come from raw PNGs, and samples pass the inverse
scaler before Inception. check_stats and check_data confirm both on the real files.

The file name of the FID statistics depends on the dataset path inside the container, so the
launcher mounts the data where the statistics were computed. Snapshots every 50k, resume
every 5k. The target is a parameter: n_iters is outside the training protocol, so FOX may
stop at 600k and go on later from the same worktree.

Run through run_afhqv2_rtx.sh from a separate worktree. This file is also the training
config: main.py --config=afhqv2_rtx_train.py:fox.
"""
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

import celeba_pair_seed as pair

ROOT = Path(__file__).resolve().parent
SEED = 42
SNAPSHOT_FREQ = 50000
RESUME_FREQ = 5000
ITERS = dict(fox=800000, a=200000, vp=200000)
ARMS = tuple(ITERS)
WORK = dict(fox='workdirs/afhqv2_fox_matern32_kappa1000_rtx_seed42',
            a='workdirs/afhqv2_A_vp_pvp_rtx_seed42',
            vp='workdirs/afhqv2_vp_rtx_seed42')
STATS = 'assets/stats/afhqv2_64_cd017afab009_stats.npz'
IMAGES = 15803
SMOKE = 'workdirs/_smoke_afhqv2'
SMOKE_ITERS = 300  # cuDNN autotuning ends well before step 100.
SMOKE_RESUME_ITERS = 350
EXPECTED_SPEED = 0.55
SLOW_SPEED = 0.60  # Above this FOX trains to FALLBACK_ITERS, to keep the evaluation date.
FALLBACK_ITERS = 600000


def base_config(arm):
    from celeba_stage0_eval import native_config
    from configs.factorial.celeba_a_vp_pvp import get_config as get_a
    return get_a() if arm == 'a' else native_config(arm)


def get_config(arm):
    from configs.default_afhqv2_configs import apply_afhqv2_data
    if arm not in ARMS:
        raise ValueError('Unknown arm: ' + arm)
    config = apply_afhqv2_data(base_config(arm))
    config.data.centered = True  # apply_afhqv2_data turns it off; CelebA trains on [-1, 1].
    config.seed = SEED
    config.allow_tf32 = True
    config.cudnn_benchmark = True
    training = config.training
    training.batch_size = 128
    training.n_iters = ITERS[arm]
    training.snapshot_freq = SNAPSHOT_FREQ
    training.snapshot_freq_for_preemption = RESUME_FREQ
    training.snapshot_sampling = False
    training.log_freq = 50
    training.eval_freq = 500
    return config


def read_manifest(work):
    return json.loads((ROOT / work / 'run_manifest.json').read_text())


def check_iters(iters):
    if iters <= 0 or iters % SNAPSHOT_FREQ:
        raise ValueError('The target must be a multiple of {}: {}'.format(SNAPSHOT_FREQ, iters))


def final_path(arm, iters):
    return ROOT / WORK[arm] / 'checkpoints/checkpoint_{}.pth'.format(iters // SNAPSHOT_FREQ)


def verify(arm, iters):
    import torch
    manifest = read_manifest(WORK[arm])
    final = torch.load(str(final_path(arm, iters)), map_location='cpu', weights_only=False)
    if (int(final['step']) != iters + 1 or
            final.get('training_protocol_sha256') != manifest['training_protocol_sha256']):
        raise ValueError('The final checkpoint differs from this run: ' + arm)
    config = manifest['config']
    if (config['seed'], config['data']['dataset'], config['data']['centered']) != (SEED, 'AFHQV2', True):
        raise ValueError('Unexpected run manifest: ' + arm)
    print('{} reached state.step {}: {}'.format(arm, int(final['step']), final_path(arm, iters)), flush=True)


def run_main(arm, work, iters):
    subprocess.check_call([sys.executable, '-u', 'main.py', '--mode=train', '--config=afhqv2_rtx_train.py:' + arm,
                           '--workdir=' + work, '--config.training.n_iters={}'.format(iters)], cwd=str(ROOT))


def train(arm, iters):
    """A rerun resumes; a finished arm is skipped, since resuming it replays all its batches."""
    check_iters(iters)
    if final_path(arm, iters).exists():
        print('Already trained: {} to {}'.format(arm, iters), flush=True)
    else:
        run_main(arm, WORK[arm], iters)
    verify(arm, iters)


def check_stats(config):
    """The file the evaluation will read, computed from these images."""
    import numpy as np
    import datasets
    import evaluation
    name = evaluation._get_stats_filename(config)
    if name != STATS:
        raise ValueError('Evaluation would use {} instead of {}: is the data mounted at the path '
                         'of the statistics?'.format(name, STATS))
    with np.load(str(ROOT / name)) as data:
        if str(data['dataset_identity']) != datasets.afhqv2_dataset_identity(config):
            raise ValueError('The statistics come from another dataset.')
        if int(data['num_examples']) != IMAGES or data['pool_3'].shape != (IMAGES, 2048):
            raise ValueError('The statistics have an unexpected size.')
    print('Statistics: {}, {} images'.format(name, IMAGES), flush=True)


def check_batch(images, config):
    """images in NHWC as the loader gives them: [0, 1] in, [-1, 1] to the network, [0, 1] back."""
    import numpy as np
    import datasets
    scaled = datasets.get_data_scaler(config)(images)
    back = datasets.get_data_inverse_scaler(config)(scaled)
    if images.min() < 0 or images.max() > 1 or images.max() < 0.5:
        raise ValueError('The loader does not give images in [0, 1].')
    if scaled.min() >= 0 or scaled.min() < -1 or scaled.max() > 1:
        raise ValueError('The network input is not in [-1, 1].')
    if np.abs(back - images).max() > 1e-6:
        raise ValueError('The inverse scaler does not return the images.')
    pixels = back * 255.
    if np.abs(pixels - np.round(pixels)).max() > 1e-3:
        raise ValueError('Images do not map back to 8-bit pixels.')
    print('Pixels: loader [{:.3f}; {:.3f}], network [{:.3f}; {:.3f}]'.format(
        images.min(), images.max(), scaled.min(), scaled.max()), flush=True)


def check_data(config):
    import datasets
    train_ds, _, _ = datasets.get_dataset(config, uniform_dequantization=config.data.uniform_dequantization)
    batch = next(iter(train_ds))['image'].numpy()
    check_batch(batch.reshape((-1,) + batch.shape[-3:]), config)


def report_speed(speed):
    for arm in ARMS:
        print('{}: {:.3f} s/step, expected about {}; {} updates take {:.1f} h'.format(
            arm, speed[arm], EXPECTED_SPEED, ITERS[arm], speed[arm] * ITERS[arm] / 3600), flush=True)
    if speed['fox'] > SLOW_SPEED:
        print('FOX is above {} s/step: train it to {}, queue fox GPU_ID {}'.format(
            SLOW_SPEED, FALLBACK_ITERS, FALLBACK_ITERS), flush=True)


def check_files():
    config = get_config('fox')
    check_stats(config)
    check_data(config)


def smoke():
    """Real data and statistics, then every arm for 300 updates and a resume."""
    # A separate process without the GPU: TF here would hold the memory the trainings need.
    subprocess.check_call([sys.executable, '-u', 'afhqv2_rtx_train.py', 'files'], cwd=str(ROOT),
                          env=dict(os.environ, CUDA_VISIBLE_DEVICES=''))
    if (ROOT / SMOKE).exists():
        shutil.rmtree(str(ROOT / SMOKE))
    speed = {}
    for arm in ARMS:
        work = '{}/{}'.format(SMOKE, arm)
        run_main(arm, work, SMOKE_ITERS)
        speed[arm] = pair.seconds_per_step(ROOT / work / 'stdout.txt', 100, SMOKE_ITERS)
        run_main(arm, work, SMOKE_RESUME_ITERS)
        if not (ROOT / work / 'checkpoints/checkpoint_final_step_{}.pth'.format(SMOKE_RESUME_ITERS + 1)).exists():
            raise ValueError('The resume did not reach its final state: ' + arm)
    shutil.rmtree(str(ROOT / SMOKE))
    report_speed(speed)


if __name__ == '__main__':
    if sys.argv[1:] == ['smoke']:
        smoke()
    elif sys.argv[1:] == ['files']:
        check_files()
    elif len(sys.argv) == 4 and sys.argv[1] == 'train' and sys.argv[2] in ARMS:
        train(sys.argv[2], int(sys.argv[3]))
    else:
        raise SystemExit('Usage: afhqv2_rtx_train.py smoke | files | train fox|a|vp ITERS')
