"""Freeze six E3b splice evals; validate existing baseline metadata without generation."""
import argparse
import ast
import gc
import hashlib
import json
import os
from pathlib import Path
import shutil

VP = 'celeba_vp_continuous_50k'
FOX = 'celeba_fox_matern32_kappa1000_50k'
REFERENCE = 'assets/stats/celeba_64_0c2d89311192_stats.npz'
REFERENCE_SHA = '66f3e831815ccebf90fe477348fcd0de92e5d966b7ed1eef77dc80fab72b6327'
ROOT = Path('/workspace/score_sde_pytorch')
IDENTITY = Path('/opt/e3b/identity')
BASELINES = Path('/opt/e3b/baselines')
VARIANTS = ('x_mean', 'x_t', 'tweedie_xt')


def digest(path):
    result = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            result.update(block)
    return result.hexdigest()


def fingerprint(root):
    # This command also runs in a CPU-only snapshot container. Import no TF/Torch.
    tree = ast.parse((Path(root) / 'artifact_utils.py').read_text())
    function = next(node for node in tree.body
                    if isinstance(node, ast.FunctionDef) and node.name == '_source_fingerprint')
    namespace = dict(hashlib=hashlib, os=os)
    exec(compile(ast.Module(body=[function], type_ignores=[]), 'fingerprint', 'exec'), namespace)
    return namespace['_source_fingerprint'](str(root))


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2) + '\n')


def baseline_folder(seed):
    return f'eval_e4_endpoint_denoise_final50k_vp_extended_ckpt260_nfe1000_50000samples_batch512_seed{seed}'


def snapshot(source, output):
    source, output = Path(source), Path(output)
    before = fingerprint(source)
    runtime = output / 'runtime'
    runtime.mkdir()  # Never overwrite an existing frozen runtime.
    ignored = {'.git', '__pycache__', '.cache', '.tfds', '.torch_extensions',
               'figures', 'notes', 'tests', 'workdirs'}
    extensions = ('.py', '.sh', '.c', '.cc', '.cpp', '.cu', '.cuh', '.h')
    for current, directories, files in os.walk(str(source)):
        directories[:] = sorted(d for d in directories if d not in ignored)
        for name in files:
            if not name.endswith(extensions) and name not in ('Dockerfile', 'requirements.txt'):
                continue
            path = Path(current) / name
            target = runtime / path.relative_to(source)
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(str(path), str(target))
    if before != fingerprint(source) or before != fingerprint(runtime):
        raise RuntimeError('Source changed during snapshot, or frozen source is incomplete.')
    for directory in ('identity', 'logs', 'baselines', 'workdirs'):
        (output / directory).mkdir()
    for directory in ('logs', 'workdirs', 'celeba'):
        (runtime / directory).mkdir(exist_ok=True)
    (runtime / 'assets/stats').mkdir(parents=True, exist_ok=True)
    if digest(source / REFERENCE) != REFERENCE_SHA:
        raise RuntimeError('Unexpected CelebA reference.')
    shutil.copy2(str(source / REFERENCE), str(runtime / REFERENCE))
    inputs = {str(ROOT / REFERENCE): REFERENCE_SHA}
    for arm in (VP, FOX):
        relative = f'workdirs/{arm}/checkpoints/checkpoint_260.pth'
        inputs[str(ROOT / relative)] = digest(source / relative)
        (output / 'workdirs' / arm / 'checkpoints').mkdir(parents=True)
    for seed in (0, 1):
        origin = source / 'workdirs' / VP / baseline_folder(seed)
        target = output / 'baselines' / f'seed{seed}'
        target.mkdir(parents=True)
        for name in ('manifest.json', 'report_260.npz'):
            shutil.copy2(str(origin / name), str(target / name))
    files = {str(path.relative_to(runtime)): digest(path)
             for path in sorted(runtime.rglob('*')) if path.is_file()}
    baselines = {str(path.relative_to(output / 'baselines')): digest(path)
                 for path in sorted((output / 'baselines').rglob('*')) if path.is_file()}
    write_json(output / 'identity/snapshot.json', dict(
        source_sha256=before, runtime_files=files, baseline_files=baselines, inputs=inputs,
        seeds=[0, 1], routes=['head:-3.3', 'full', 'tail:-3.3'], num_samples=50000,
        batch_size=512, checkpoint=260, nfe=1001, endpoint=10.575749165617202,
        baseline_validation='metadata_only', baseline_generation=False))
    print(f'Frozen source: {before}; six 50k cells, sampling seeds 0/1.', flush=True)


def check():
    saved = json.loads((IDENTITY / 'snapshot.json').read_text())
    if fingerprint(ROOT) != saved['source_sha256']:
        raise RuntimeError('Frozen source fingerprint changed.')
    for relative, expected in saved['runtime_files'].items():
        if digest(ROOT / relative) != expected:
            raise RuntimeError(f'Frozen runtime file changed: {relative}')
    for relative, expected in saved['baseline_files'].items():
        if digest(BASELINES / relative) != expected:
            raise RuntimeError(f'Frozen baseline changed: {relative}')
    for path, expected in saved['inputs'].items():
        if digest(path) != expected:
            raise RuntimeError(f'Pinned checkpoint/reference changed: {path}')
    return saved


def preflight():
    import numpy as np
    import torch
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError('Expected exactly one visible CUDA GPU.')
    torch.cuda.init()
    from artifact_utils import _environment_manifest
    saved = check()
    environment = _environment_manifest()
    identities = {}
    for arm in (VP, FOX):
        path = ROOT / f'workdirs/{arm}/checkpoints/checkpoint_260.pth'
        state = torch.load(str(path), map_location='cpu', weights_only=False)
        if int(state['step']) != 1300002:
            raise RuntimeError(f'Expected final 1.3M checkpoint: {arm}')
        identities[arm] = state.get('checkpoint_id') or 'legacy-sha256-' + digest(path)
        del state
        gc.collect()
    for seed in (0, 1):
        folder = BASELINES / f'seed{seed}'
        manifest = json.loads((folder / 'manifest.json').read_text())
        config = manifest['config']
        expected = dict(batch_size=512, num_samples=50000, sampling_seed=seed,
                        sampling_num_scales=1000, endpoint_denoise_variants=True,
                        enable_sampling=True, enable_loss=False, enable_bpd=False)
        if any(config['eval'].get(k) != v for k, v in expected.items()):
            raise RuntimeError('Existing VP baseline uses a different eval protocol.')
        expected = dict(time_grid='uniform_logsnr', logsnr_min=-10.0,
                        logsnr_max=10.575749165617202, predictor='euler_maruyama',
                        corrector='none', noise_removal=True, probability_flow=False, method='pc')
        if any(config['sampling'].get(k) != v for k, v in expected.items()):
            raise RuntimeError('Existing VP baseline uses a different sampler.')
        if (config['training']['sde'] != 'vpsde' or config['seed'] != 42
                or config['data']['dataset'] != 'CELEBA' or config['data']['image_size'] != 64
                or not config['data']['centered'] or config['model'].get('noise_conditioning', 'time') != 'time'
                or not config['allow_tf32'] or not config['cudnn_benchmark']
                or config['deterministic'] or config['allow_fp16_reduced_precision_reduction']):
            raise RuntimeError('Unexpected VP baseline model/data/precision settings.')
        for key in ('python', 'numpy', 'tensorflow', 'torch', 'torch_cuda', 'cudnn', 'packages'):
            if environment[key] != manifest['environment'][key]:
                raise RuntimeError(f'Baseline/runtime environment differs: {key}')
        with np.load(str(folder / 'report_260.npz')) as report:
            if (str(report['checkpoint_id']) != identities[VP]
                    or int(report['checkpoint_step']) != 1300002 or int(report['num_samples']) != 50000
                    or int(report['nfe']) != 1001 or str(report['dataset_stats_id']) != REFERENCE_SHA):
                raise RuntimeError('Existing VP baseline checkpoint/reference/count mismatch.')
    write_json('logs/final50k_preflight.json', dict(status='passed', source_sha256=saved['source_sha256'],
                                                 checkpoint_ids=identities, environment=environment))
    print('Preflight passed: pinned source, checkpoints, reference, baseline protocols and environment.', flush=True)


def eval_folder(seed, route):
    suffix = route.replace(':', '')
    return ROOT / 'workdirs' / VP / f'eval_e3b_splice_final50k_{suffix}_ckpt260_nfe1000_50000samples_batch512_seed{seed}'


def report(seed, route):
    import numpy as np
    folder = eval_folder(seed, route)
    with np.load(str(folder / 'report_260.npz')) as result:
        if int(result['num_samples']) != 50000 or int(result['nfe']) != 1001:
            raise RuntimeError('Incomplete final splice report.')
        if str(result['dataset_stats_id']) != REFERENCE_SHA:
            raise RuntimeError('Final report uses a different reference.')
        vp, fox = int(result['splice_vp_calls']), int(result['splice_fox_calls'])
        if vp + fox != 1001 or (route == 'full' and fox != 1001):
            raise RuntimeError('Wrong final splice route counts.')
        if route.startswith(('head:', 'tail:')) and not (0 < vp < 1001 and 0 < fox < 1001):
            raise RuntimeError('Head/tail failed to switch networks.')
        if len(list((folder / 'ckpt_260').glob('statistics_*.npz'))) != 98:
            raise RuntimeError('Expected 98 saved statistics batches.')
        metrics = {f'{metric}_{variant}': float(result[f'{metric}_{variant}'])
                   for variant in VARIANTS for metric in ('IS', 'fid', 'kid')}
        if not all(np.isfinite(value) for value in metrics.values()):
            raise RuntimeError('Non-finite final metrics.')
        write_json(f'logs/seed{seed}_{route.replace(":", "")}_result.json',
                   dict(status='completed', seed=seed, route=route, **metrics))
    print(f'Validated final report: seed={seed}, route={route}, FID Tweedie={metrics["fid_tweedie_xt"]:.6f}', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=('snapshot', 'preflight', 'check', 'report'))
    parser.add_argument('arguments', nargs='*')
    args = parser.parse_args()
    if args.mode == 'snapshot':
        if len(args.arguments) != 2:
            parser.error('snapshot requires source and output')
        snapshot(*args.arguments)
    elif args.mode in ('preflight', 'check'):
        if args.arguments:
            parser.error('unexpected arguments')
        (preflight if args.mode == 'preflight' else check)()
    else:
        if len(args.arguments) != 2 or args.arguments[0] not in ('0', '1'):
            parser.error('expected sampling seed 0/1 and a route for report')
        if args.arguments[1] not in ('full', 'head:-3.3', 'tail:-3.3'):
            parser.error('unexpected route')
        report(int(args.arguments[0]), args.arguments[1])


if __name__ == '__main__':
    main()
