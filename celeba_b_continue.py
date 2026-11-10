"""Continue CelebA arm B, training seed 42, from 200k to 800k updates on the H200.

A copy of celeba_a_continue.py for B: the VP process with the FOX distribution of noise
levels, log-SNR labels, window [-10; 9.115]. The question is whether B leaves the ODE dip
the way native FOX does between 400k and 800k, so the curve over steps matters more than
the end point; celeba_b_curve_eval.py evaluates the snapshots while this trains.

Model, EMA, Adam and step come from checkpoint_40; RNG and the data position start anew,
as for A: the image is the RTX one with TF 2.15, which cannot replay the TF 2.4 batches.
The run is not a bitwise continuation, and a jump right after 200k must not be
interpreted. The fingerprint of checkpoint_40 is checked against the source manifest,
then replaced by this one. Snapshots stay every 5k, so 400k is checkpoint_80 and 800k is
checkpoint_160, the numbering of native FOX.

Run through run_celeba_b_continue_800k_h200.sh from a separate worktree. This file is
also the training config: main.py --config=celeba_b_continue.py.
"""
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent
SEED = 42
SNAPSHOT_FREQ = 5000
SOURCE_WORK = 'celeba_factorial_B_vp_pfox_50k'
WORK = 'workdirs/celeba_factorial_B_vp_pfox_800k_h200_seed42'
SOURCE = ROOT / 'workdirs' / SOURCE_WORK / 'checkpoints/checkpoint_40.pth'
SOURCE_MANIFEST = ROOT / 'workdirs/_training_inputs' / (SOURCE_WORK + '.json')
SOURCE_ID = '70f0b3d68a294f5282f860a7bb292825'  # The checkpoint stage 0 evaluated: B_c40_lo, 8.32.
SOURCE_CKPT = 40
SOURCE_STEP = 200001
TARGET_ITERS = 800000
FINAL_CKPT = 160


def read_source():
    return json.loads(SOURCE_MANIFEST.read_text())


def make_config(device):
    from ml_collections import ConfigDict
    from configs.factorial.celeba_b_vp_pfox import get_config as get_b
    base = get_b()
    config = ConfigDict(read_source()['config'])
    if (config.training.factorial_arm, config.seed, config.training.snapshot_freq) != ('B_vp_pfox', SEED, SNAPSHOT_FREQ):
        raise ValueError('Expected arm B, seed {} and snapshot_freq {} in the source manifest.'.format(SEED, SNAPSHOT_FREQ))
    config.eval = base.eval  # Manifests exclude evaluation settings.
    config.training.n_iters = TARGET_ITERS  # The other four fields are excluded too.
    config.training.log_freq = 50
    config.training.eval_freq = 1000
    config.training.snapshot_freq_for_preemption = 5000
    config.training.snapshot_sampling = False
    config.device = device
    return config


def get_config():
    return make_config('cuda:0')


def new_state(config, model_protocol):
    """A state without a training fingerprint accepts the H200 parent."""
    import losses
    from models import utils as mutils, ncsnpp  # noqa: F401, registers the model.
    from models.ema import ExponentialMovingAverage
    model = mutils.create_model(config)
    return dict(model=model, optimizer=losses.get_optimizer(config, model.parameters()),
                ema=ExponentialMovingAverage(model.parameters(), config.model.ema_rate),
                step=0, data_batches_consumed=0, eval_batches_consumed=0,
                model_protocol_sha256=model_protocol)


def validate_state(state, config):
    import torch
    parameters = [p for p in state['model'].parameters() if p.requires_grad]
    optimizer = state['optimizer']
    if len(optimizer.state) != len(parameters):
        raise ValueError('Missing Adam states for trainable parameters.')
    expected = dict(lr=config.optim.lr, betas=(config.optim.beta1, 0.999),
                    eps=config.optim.eps, weight_decay=config.optim.weight_decay)
    for group in optimizer.param_groups:
        for key, value in expected.items():
            if group[key] != value:
                raise ValueError('Source Adam {}={!r}, expected {!r}'.format(key, group[key], value))
    for parameter in parameters:
        moments = optimizer.state[parameter]
        if moments.get('step') is None or int(moments['step']) != int(state['step']):
            raise ValueError('Adam step differs from state.step.')
        for key in ('exp_avg', 'exp_avg_sq'):
            tensor = moments.get(key)
            if tensor is None or tensor.shape != parameter.shape or not bool(torch.isfinite(tensor).all()):
                raise ValueError('Invalid Adam moment: ' + key)
    if state['ema'].decay != config.model.ema_rate:
        raise ValueError('EMA decay changed.')
    for tensor in list(state['model'].state_dict().values()) + state['ema'].shadow_params:
        if not bool(torch.isfinite(tensor).all()):
            raise ValueError('Non-finite model or EMA tensor.')


def load_source(config):
    from artifact_utils import build_model_manifest
    from utils import restore_checkpoint
    source = read_source()
    model_protocol = source['model_manifest']['model_protocol_sha256']
    if build_model_manifest(config)['model_protocol_sha256'] != model_protocol:
        raise ValueError('The config does not rebuild the model of the source manifest.')
    state = new_state(config, model_protocol)
    restore_checkpoint(str(SOURCE), state, config.device, restore_rng=False)
    if state['checkpoint_id'] != SOURCE_ID:
        raise ValueError('Not the checkpoint stage 0 evaluated: ' + state['checkpoint_id'])
    if state['training_protocol_sha256'] != source['training_protocol_sha256']:
        raise ValueError('The checkpoint does not belong to the source manifest.')
    if int(state['step']) != SOURCE_STEP:
        raise ValueError('Expected state.step={}, got {}'.format(SOURCE_STEP, state['step']))
    validate_state(state, config)
    return state, source


def sha256(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def check():
    """Real weights on CPU, no writes: safe while both GPUs are busy."""
    config = make_config('cpu')
    state, _ = load_source(config)
    print('Check passed: step {}, checkpoint {}, {} batches consumed on H200.'.format(
        int(state['step']), state['checkpoint_id'], state['data_batches_consumed']), flush=True)


def import_parent():
    """Writes the new manifest, the lineage and checkpoint_40; safe to rerun."""
    import tensorflow as tf
    from artifact_utils import atomic_write_bytes, build_training_manifest, ensure_training_manifest
    from reproducibility import configure_reproducibility, seed_everything
    from utils import save_checkpoint
    config = get_config()
    configure_reproducibility(config, tensorflow_module=tf)
    manifest = build_training_manifest(config)
    workdir = ROOT / WORK
    imported = workdir / 'checkpoints/checkpoint_{}.pth'.format(SOURCE_CKPT)
    source = read_source()
    lineage = dict(
        schema_version=1, source_checkpoint='workdirs/{}/checkpoints/checkpoint_{}.pth'.format(SOURCE_WORK, SOURCE_CKPT),
        source_checkpoint_id=SOURCE_ID, source_checkpoint_sha256=sha256(SOURCE), source_state_step=SOURCE_STEP,
        source_training_protocol_sha256=source['training_protocol_sha256'],
        source_environment=source['environment'],
        restored_components=['model', 'optimizer', 'ema', 'step'],
        continuation_seed=config.seed, data_batches_consumed_at_import=0, eval_batches_consumed_at_import=0,
        rng_policy='New seed RNG stream in the RTX image on the H200; the data position is reset.',
        training_protocol_sha256=manifest['training_protocol_sha256'])
    ensure_training_manifest(str(workdir), manifest)
    path = workdir / 'continuation.json'
    if path.exists():
        if json.loads(path.read_text()) != lineage:
            raise ValueError('The workdir continues a different parent or protocol.')
    else:
        atomic_write_bytes(str(path), (json.dumps(lineage, sort_keys=True, indent=2) + '\n').encode(), overwrite=False)
    if imported.exists():
        print('Parent already imported: ' + str(imported), flush=True)
        return
    state, _ = load_source(config)
    state.update(training_protocol_sha256=manifest['training_protocol_sha256'],
                 data_batches_consumed=0, eval_batches_consumed=0)
    seed_everything(config.seed, tensorflow_module=tf)
    save_checkpoint(str(imported), state, overwrite=False)
    print('Imported step {} as {}'.format(SOURCE_STEP, imported), flush=True)


def verify():
    import torch
    manifest = json.loads((ROOT / WORK / 'run_manifest.json').read_text())
    path = ROOT / WORK / 'checkpoints/checkpoint_{}.pth'.format(FINAL_CKPT)
    final = torch.load(str(path), map_location='cpu', weights_only=False)
    if (int(final['step']) != TARGET_ITERS + 1 or
            final.get('training_protocol_sha256') != manifest['training_protocol_sha256']):
        raise ValueError('The final checkpoint differs from this continuation.')
    print('B reached state.step {}: {}'.format(int(final['step']), path), flush=True)


if __name__ == '__main__':
    modes = dict(check=check, verify=verify)
    modes['import'] = import_parent
    if len(sys.argv) != 2 or sys.argv[1] not in modes:
        raise SystemExit('Usage: celeba_b_continue.py check|import|verify')
    modes[sys.argv[1]]()
