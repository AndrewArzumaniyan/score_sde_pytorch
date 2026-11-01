#!/usr/bin/env bash
# Docker-only audit/bootstrap of completed CelebA next15h + cached controls.
# Inline Python keeps the training source fingerprint unchanged.
set -euo pipefail

setting() { printenv "$1" || printf '%s\n' "$2"; }
mode=--help
if (( $# )); then mode="$1"; shift; fi
[[ $# == 0 ]] || { echo 'Expected one mode.' >&2; exit 2; }
if [[ "$mode" == --help ]]; then
  cat <<'HELP'
Usage: bash run_celeba_nfe_ab_cpu_bootstrap.sh --check|--pilot|--full
--check: CPU-only synthetic checks, no experiment inputs.
--pilot: audit eight completed trajectory cells, reproduce 14 FIDs, 5 repeats.
--full: continue the same audited pilot to 1000 repeats (not 1000 extra).
4 CPU / 4 GiB, nice 19, no GPU/network. NumPy/SciPy only.
Defaults: ../celeba_nfe_ab_15h_20261027_r2, ../celeba_denoiser_swap_10k,
          ../ab_replication_seed43; output ../celeba_nfe_ab_bootstrap_20261028.
Overrides CELEBA_NFE_BOOT_{NEXT,SWAP,AB,OUTPUT,IMAGE,CPUS,THREADS,MEMORY,NAME_PREFIX}.
Resume a stopped pilot/full container with docker start <existing-name>.
HELP
  exit
fi
if [[ "$mode" != --inside ]]; then
  case "$mode" in --check|--pilot|--full) ;; *) echo 'Use --help.' >&2; exit 2 ;; esac
  script="$(readlink -f "$0")"
  repo="$(dirname "$script")"
  parent="$(dirname "$repo")"
  root=/workspace/score_sde_pytorch
  image="$(setting CELEBA_NFE_BOOT_IMAGE score-sde-h200)"
  image_id="$(docker image inspect --format '{{.Id}}' "$image")"
  cpus="$(setting CELEBA_NFE_BOOT_CPUS 4)"
  threads="$(setting CELEBA_NFE_BOOT_THREADS 4)"
  memory="$(setting CELEBA_NFE_BOOT_MEMORY 4g)"
  [[ "$threads" =~ ^[1-9][0-9]*$ ]] || { echo 'Invalid thread count.' >&2; exit 2; }
  set -- run --init --network none --cpus "$cpus" --cpu-shares 128 \
    --memory "$memory" --memory-swap "$memory" --user "$(id -u):$(id -g)" \
    --env NVIDIA_VISIBLE_DEVICES=void --env CUDA_VISIBLE_DEVICES= \
    --env PYTHONDONTWRITEBYTECODE=1 --env BOOT_MODE="$mode" \
    --env OMP_NUM_THREADS="$threads" --env OPENBLAS_NUM_THREADS="$threads" \
    --env MKL_NUM_THREADS="$threads" --env NUMEXPR_NUM_THREADS="$threads" \
    --env TFDS_DATA_DIR="$(setting TFDS_DATA_DIR "$root/.tfds")" \
    --env TORCH_EXTENSIONS_DIR="$(setting TORCH_EXTENSIONS_DIR "$root/.torch_extensions")" \
    --env XDG_CACHE_HOME="$(setting XDG_CACHE_HOME "$root/.cache")" \
    --mount "type=bind,src=$repo,dst=$root,readonly" --workdir "$root" --entrypoint bash
  if [[ "$mode" == --check ]]; then
    docker "$@" --rm --env BOOT_IMAGE_ID="$image_id" \
      --mount "type=bind,src=$script,dst=/opt/bootstrap.sh,readonly" \
      "$image_id" /opt/bootstrap.sh --inside
    exit
  fi
  next="$(readlink -e "$(setting CELEBA_NFE_BOOT_NEXT "$parent/celeba_nfe_ab_15h_20261027_r2")")"
  swap="$(readlink -e "$(setting CELEBA_NFE_BOOT_SWAP "$parent/celeba_denoiser_swap_10k")")"
  ab="$(readlink -e "$(setting CELEBA_NFE_BOOT_AB "$parent/ab_replication_seed43")")"
  out="$(realpath -m "$(setting CELEBA_NFE_BOOT_OUTPUT "$parent/celeba_nfe_ab_bootstrap_20261028")")"
  for input in "$repo" "$next" "$swap" "$ab"; do
    [[ -d "$input" ]] || { echo "Missing input: $input" >&2; exit 1; }
    case "$out/" in "$input/"*) echo 'Output overlaps an input.' >&2; exit 2 ;; esac
    case "$input/" in "$out/"*) echo 'Output contains an input.' >&2; exit 2 ;; esac
  done
  [[ "$(cat "$next/results/queue.state")" == completed &&
     "$(cat "$swap/results/queue.state")" == completed ]] || {
    echo 'Both generation queues must be completed.' >&2; exit 1; }
  [[ -d "$ab/eval_workdirs" ]] || { echo 'Missing A/B eval_workdirs.' >&2; exit 1; }
  prefix="$(setting CELEBA_NFE_BOOT_NAME_PREFIX celeba-nfe-ab-bootstrap)"
  stage="$(printf '%s' "$mode" | cut -c3-)"
  name="$prefix-$stage"
  [[ "$name" =~ ^[a-zA-Z0-9][a-zA-Z0-9_.-]*$ ]] || { echo 'Invalid container name.' >&2; exit 2; }
  if docker inspect "$name" >/dev/null 2>&1; then
    echo "Container exists: $name; inspect it, then resume with docker start if appropriate." >&2
    exit 1
  fi
  if [[ "$mode" == --full && ! -s "$out/pilot.json" ]]; then
    echo 'Run the audited --pilot first.' >&2; exit 1
  fi
  mkdir -p "$out/runtime"
  if [[ -s "$out/runtime/image_id.txt" ]]; then
    image_id="$(cat "$out/runtime/image_id.txt")"
    docker image inspect "$image_id" >/dev/null
  else
    printf '%s\n' "$image_id" > "$out/runtime/image_id.txt"
  fi
  snapshot="$out/runtime/run_celeba_nfe_ab_cpu_bootstrap.sh"
  if [[ -e "$snapshot" ]]; then
    cmp -s "$script" "$snapshot" || { echo 'Script changed; use a new output.' >&2; exit 1; }
  else
    cp -- "$script" "$snapshot"
  fi
  docker "$@" -d --name "$name" --env BOOT_IMAGE_ID="$image_id" \
    --mount "type=bind,src=$next,dst=/input/next,readonly" \
    --mount "type=bind,src=$swap,dst=/input/swap,readonly" \
    --mount "type=bind,src=$ab/eval_workdirs,dst=/input/ab/eval_workdirs,readonly" \
    --mount "type=bind,src=$out,dst=/output" \
    --mount "type=bind,src=$snapshot,dst=/opt/bootstrap.sh,readonly" \
    "$image_id" /opt/bootstrap.sh --inside
  docker inspect "$name" > "$out/docker_inspect_$stage.json"
  printf 'Started %s; results %s\n' "$name" "$out"
  exit
fi

[[ -z "$CUDA_VISIBLE_DEVICES" && "$NVIDIA_VISIBLE_DEVICES" == void ]] || {
  echo 'This job must have no visible GPU.' >&2; exit 2; }
nice -n 19 python -u - <<'PY_NFE_BOOTSTRAP'
"""Fixed-reference FID bootstrap; no project imports, models or image generation."""
import fcntl
import gc
import hashlib
import json
import math
import os
from pathlib import Path
import resource
import time

import numpy as np
import scipy
from scipy import linalg

N, BATCH, DIM, ROUNDS = 10000, 512, 2048, 20
REFERENCE_SHA = '66f3e831815ccebf90fe477348fcd0de92e5d966b7ed1eef77dc80fab72b6327'
ROOT, OUT = Path('/input'), Path('/output')
EXTENDED, COMMON = 10.575749165617202, 9.115429865459795
INVENTORY = {}


def require(ok, message):
    if not ok:
        raise ValueError(message)


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1048576), b''):
            h.update(block)
    return h.hexdigest()


def observed(path):
    path = Path(path)
    value = digest(path)
    INVENTORY[str(path)] = value
    return value


def canonical_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                     ensure_ascii=True, allow_nan=False).encode()).hexdigest()


def array_hash(x):
    x = np.ascontiguousarray(x)
    h = hashlib.sha256()
    h.update(str(x.dtype).encode('ascii'))
    h.update(str(x.shape).encode('ascii'))
    h.update(x.tobytes())
    return h.hexdigest()


def scalar(z, key):
    require(key in z, f'Missing metadata: {key}')
    return np.asarray(z[key]).item()


def metadata(z, expected, path):
    for key, value in expected.items():
        require(key in z and np.array_equal(z[key], np.asarray(value)),
                f'Wrong/missing {key}: {path}')


def read_json(path):
    observed(path)
    return json.loads(Path(path).read_text())


def manifest(path):
    data = read_json(path)
    payload = dict(data)
    claimed = payload.pop('protocol_sha256')
    require(canonical_hash(payload) == claimed, f'Manifest hash: {path}')
    return data


def save(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + '\n')
    temporary.replace(path)


def derive_seed(round_id):
    payload = '\0'.join(('0', 'sampling-round', str(round_id))).encode()
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], 'big') % (2**31 - 1)


def check_endpoint(alpha, sigma, endpoint, batch, t):
    # VPSDE.marginal_prob returns the mean factor in NCHW and std as a vector.
    require(alpha.shape == (batch, 1, 1, 1) and sigma.shape == (batch,) and
            np.isfinite(alpha).all() and np.isfinite(sigma).all() and
            (alpha > 0).all() and (sigma > 0).all(), 'Invalid endpoint coefficients')
    require(np.isfinite(t) and 0 < t < 1, 'Invalid endpoint time')
    # All these trajectories use the VP forward process, beta_min=.1/max=20.
    q = .5 * (20. - .1) * t*t + .1*t
    value = -q - np.log(-np.expm1(-q))
    require(abs(value-endpoint) < 1e-3, 'Wrong endpoint log-SNR')
    # Original std uses float32 1-exp(-q), with cancellation near this endpoint.
    require(np.allclose(alpha, np.exp(-q/2), rtol=1e-6, atol=1e-7) and
            np.allclose(sigma, np.sqrt(-np.expm1(-q)), rtol=.002, atol=1e-7),
            'Endpoint coefficients inconsistent with saved time')


def moments(x, weights=None):
    count = len(x) if weights is None else int(weights.sum())
    require(count > 1, 'Too few samples')
    total = np.zeros(x.shape[1], dtype=np.float64)
    for start in range(0, len(x), 512):
        a = np.asarray(x[start:start + 512], dtype=np.float64)
        total += a.sum(0) if weights is None else weights[start:start + len(a)] @ a
    mean = total / count
    cov = np.zeros((x.shape[1], x.shape[1]), dtype=np.float64)
    for start in range(0, len(x), 512):
        a = np.asarray(x[start:start + 512], dtype=np.float64) - mean
        if weights is not None:
            a *= np.sqrt(weights[start:start + len(a)])[:, None]
        cov += a.T @ a
    return mean, cov / (count - 1)


def fid(x, real, weights=None):
    mean, cov = moments(x, weights)
    cross = real['sqrt'] @ cov @ real['sqrt']
    eig = linalg.eigvalsh((cross + cross.T) * .5, check_finite=False,
                         overwrite_a=True, driver='evr')
    require(eig.min() > -1e-5, 'FID matrix has materially negative eigenvalues')
    value = float(np.square(mean - real['mean']).sum() + np.trace(cov) +
                  float(real['trace']) - 2 * np.sqrt(np.maximum(eig, 0)).sum())
    require(np.isfinite(value), 'Non-finite FID')
    return value


def reference_moments(path):
    require(observed(path) == REFERENCE_SHA, 'Wrong real-reference SHA')
    with np.load(path, allow_pickle=False) as z:
        x = z['pool_3']
    require(x.ndim == 2 and x.shape[1] == DIM and len(x) > DIM and
            x.dtype == np.float32 and np.isfinite(x).all(), 'Invalid reference features')
    mean, cov = moments(x)
    eig, vectors = linalg.eigh(cov, check_finite=False)
    require(eig.min() > -1e-7, 'Invalid reference covariance')
    root = (vectors * np.sqrt(np.maximum(eig, 0))) @ vectors.T
    return dict(mean=mean, sqrt=root, trace=np.asarray(np.trace(cov)))


def specs():
    swap = ROOT / 'swap/results'
    nxt = ROOT / 'next/results'
    result = []
    for steps, base in ((1000, swap), (2000, nxt)):
        for route in ('vp', 'tail'):
            name = f'native_{route}_{steps}'
            folder = base / (f'{route}_seed0' if steps == 1000 else
                             f'native_{route}_nfe2000_seed0')
            result.append(dict(name=name, group='native', kind='swap', folder=str(folder),
                               runtime=str(ROOT / ('swap' if steps == 1000 else 'next') / 'runtime'),
                               route=route, steps=steps, ckpt=260, step=1300002,
                               variants=['tweedie_vp', 'tweedie_fox']))
    for arm, label in (('A', 'A_vp_pvp'), ('B', 'B_vp_pfox')):
        work = f'celeba_ab_replication_{label}_seed43_200k'
        folder = ROOT / f'ab/eval_workdirs/{work}/eval_ab_replication_seed43_{arm}_common_ckpt4_nfe1000_10000samples_batch512_seed0'
        result.append(dict(name=f'controlled_{arm}', group='controlled', kind='legacy',
                           folder=str(folder), arm=arm, route=arm, steps=1000, ckpt=4,
                           step=200001, variants=['tweedie_xt']))
    for route in ('head', 'tail'):
        result.append(dict(name=f'controlled_{route}', group='controlled', kind='swap',
                           folder=str(nxt / f'controlled_{route}_nfe1000_seed0'),
                           runtime=str(ROOT / 'next/runtime'), route=route, steps=1000,
                           ckpt=4, step=200001, variants=['tweedie_a', 'tweedie_b']))
    return result


def verify_runtime(root):
    table = root / 'runtime.sha256'
    observed(table)
    runtime = (root / 'runtime').resolve()
    for line in table.read_text().splitlines():
        claimed, relative = line.split('  ', 1)
        path = (runtime / relative).resolve()
        require(os.path.commonpath([str(runtime), str(path)]) == str(runtime), 'Unsafe runtime path')
        require(observed(path) == claimed, f'Frozen runtime changed: {path}')
    # Do not open checkpoint files: retain the hashes recorded at GPU evaluation.
    observed(root / 'inputs.sha256')


def validate_config(s, m):
    cfg, steps = m['config'], s['steps']
    require(cfg['seed'] == (42 if s['group'] == 'native' else 43), 'Wrong master model seed')
    ev, sp = cfg['eval'], cfg['sampling']
    expected = dict(num_samples=N, batch_size=BATCH, sampling_seed=0,
                    sampling_num_scales=steps, enable_sampling=True,
                    enable_loss=False, enable_bpd=False,
                    endpoint_denoise_variants=s['kind'] == 'legacy')
    require(all(ev.get(k) == v for k, v in expected.items()), f'Eval config: {s["name"]}')
    require(all(sp.get(k) == v for k, v in dict(
        method='pc', predictor='euler_maruyama', corrector='none',
        time_grid='uniform_logsnr', probability_flow=False,
        noise_removal=s['kind'] == 'legacy', logsnr_min=-10.0,
        logsnr_max=EXTENDED if s['group'] == 'native' else COMMON).items()),
        f'Sampling config: {s["name"]}')
    require(cfg['data']['dataset'] == 'CELEBA' and cfg['data']['image_size'] == 64 and
            cfg['data']['centered'] is True and cfg['data']['num_channels'] == 3,
            'Wrong image preprocessing')
    require(cfg.get('allow_tf32') is True and cfg.get('cudnn_benchmark') is True and
            cfg.get('deterministic') is False and
            cfg.get('allow_fp16_reduced_precision_reduction') is False, 'Wrong numeric protocol')
    if s['group'] == 'controlled':
        require(cfg['model']['noise_conditioning'] == 'logsnr' and
                cfg['training']['sde'] == 'vpsde', 'Wrong controlled conditioning/SDE')
        require(cfg['model']['logsnr_min'] == -10. and cfg['model']['logsnr_max'] == COMMON,
                'Wrong controlled model support')
        if s['kind'] == 'legacy':
            allocation = 'vp_time' if s['arm'] == 'A' else 'normalized_fox_time'
            require(cfg['training']['noise_distribution'] == allocation, 'Wrong controlled allocation')
    require(cfg['model']['beta_min'] == .1 and cfg['model']['beta_max'] == 20.,
            'Endpoint check requires the recorded VP beta schedule')
    if s['kind'] == 'swap':
        require(ev['denoiser_swap_route'] == s['route'] and
                ev['denoiser_swap_threshold'] == -3.3, 'Wrong swap route/threshold')
        e = m['experiment']
        require(e['paired_denoisers'] is True and e['trajectory_nfe'] == steps and
                e['nfe_per_output'] == steps + 1 and e['executed_nfe'] == steps + 2 and
                e['denoiser_calls'] == 2 and e['reference_sha256'] == REFERENCE_SHA and
                e['raw_state_coordinates'] == 'centered_NCHW_float32' and
                e['quantization'] == 'inverse_scaler; clip(image*255,0,255); uint8',
                'Wrong swap endpoint/output protocol')
        require(e.get('variants', s['variants']) == s['variants'], 'Wrong denoiser variants')
        require(e['launcher_sha256'] == observed(Path(s['runtime']) / 'run_celeba_denoiser_swap.sh'),
                'Wrong frozen evaluation launcher')


def control_record(s, summary):
    if s['kind'] == 'legacy':
        return summary['baseline_reports']['controlled'][s['arm']]
    if s['group'] == 'native' and s['steps'] == 1000:
        return summary['baseline_reports']['native'][s['route']]
    return None


def metric_record(s, summary):
    control = control_record(s, summary)
    if control is not None:
        return control['metrics']
    return summary['cells'][f'{s["group"]}_{s["route"]}_nfe{s["steps"]}_seed0']


def audit_cell(s, summary):
    folder = Path(s['folder'])
    mpath, rpath = folder / 'manifest.json', folder / f'report_{s["ckpt"]}.npz'
    m = manifest(mpath)
    validate_config(s, m)
    report_sha = observed(rpath)
    with np.load(rpath, allow_pickle=False) as z:
        require(all(np.isfinite(scalar(z, f'{metric}_{v}')) for v in s['variants']
                    for metric in ('IS', 'fid', 'kid')), f'Invalid report metric: {rpath}')
        identity = {k: scalar(z, k) for k in (
            'protocol_sha256', 'checkpoint_id', 'checkpoint_step', 'training_protocol_sha256')}
        metadata(z, dict(protocol_sha256=m['protocol_sha256'], checkpoint_step=s['step'],
                         num_samples=N, dataset_stats_id=REFERENCE_SHA), rpath)
        if 'sampling_seed' in z or s['kind'] == 'swap':
            metadata(z, dict(sampling_seed=0), rpath)
        targets = {v: float(scalar(z, f'fid_{v}')) for v in s['variants']}
        if s['kind'] == 'swap':
            e = m['experiment']
            primary, alternate = ('vp', 'fox') if s['group'] == 'native' else ('A', 'B')
            a, b = e['checkpoints'][primary], e['checkpoints'][alternate]
            require(a['checkpoint_step'] == b['checkpoint_step'] == s['step'],
                    'Wrong primary/alternate checkpoint steps')
            metadata(z, {k: a[k] for k in ('checkpoint_id', 'checkpoint_step',
                                         'training_protocol_sha256', 'checkpoint_sha256')}, rpath)
            identity.update(alt_checkpoint_id=b['checkpoint_id'], alt_checkpoint_step=b['checkpoint_step'],
                            alt_training_protocol_sha256=b['training_protocol_sha256'],
                            checkpoint_sha256=a['checkpoint_sha256'], grid_sha256=e['grid_sha256'],
                            route=s['route'], trajectory_nfe=s['steps'], denoiser_calls=2,
                            executed_nfe=s['steps'] + 2, nfe_per_output=s['steps'] + 1)
            if 'group' in z:
                require(scalar(z, 'group') == s['group'], 'Wrong group')
                identity['group'] = s['group']
            metadata(z, identity, rpath)
            claimed_statistics = scalar(z, 'statistics_sha256')
        else:
            metadata(z, dict(nfe=1001), rpath)
    baseline = control_record(s, summary)
    if baseline is not None:
        require(baseline['manifest_sha256'] == digest(mpath) and
                baseline['report_sha256'] == report_sha and
                baseline['protocol_sha256'] == m['protocol_sha256'], 'Cached baseline differs from GPU summary')
    expected_metrics = metric_record(s, summary)
    require(all(targets[v] == expected_metrics[f'fid_{v}'] for v in s['variants']),
            f'FID differs from completed summary: {s["name"]}')
    directory = folder / f'ckpt_{s["ckpt"]}'
    files = sorted(directory.glob('statistics_*.npz'), key=lambda p: int(p.stem.split('_')[-1]))
    require([p.name for p in files] == [f'statistics_{i}.npz' for i in range(ROUNDS)],
            f'Wrong statistics rounds: {directory}')
    caches = {}
    for v in s['variants']:
        name = s['name'] + '__' + v
        caches[v] = np.lib.format.open_memmap(OUT / 'cache' / f'{name}.npy', mode='w+',
                                             shape=(N, DIM), dtype=np.float32)
    hashes, state_hashes = [], []
    for i, path in enumerate(files):
        hashes.append(observed(path))
        expected = dict(identity, round_id=i, sampling_seed=derive_seed(i), num_samples=BATCH)
        if s['kind'] == 'swap':
            expected['sample_start'] = i * BATCH
        else:
            expected['nfe'] = 1001
        with np.load(path, allow_pickle=False) as z:
            metadata(z, expected, path)
            for v in s['variants']:
                for prefix, dim in (('pool_3', DIM), ('logits', 1008)):
                    a = z[f'{prefix}_{v}']
                    require(a.shape == (BATCH, dim) and a.dtype == np.float32 and
                            np.isfinite(a).all(), f'Invalid {prefix}_{v}: {path}')
                    if prefix == 'pool_3':
                        start, take = i * BATCH, min(BATCH, N - i * BATCH)
                        caches[v][start:start + take] = a[:take]
            if s['kind'] == 'swap':
                state_hash, samples_sha = scalar(z, 'state_sha256'), scalar(z, 'samples_sha256')
        if s['kind'] == 'swap':
            state_path, image_path = directory / f'state_{i}.npz', directory / f'samples_{i}.npz'
            observed(state_path)
            require(observed(image_path) == samples_sha, f'Images/statistics hash link: {image_path}')
            state_hashes.append(state_hash)
            with np.load(state_path, allow_pickle=False) as z:
                metadata(z, dict(expected, state_sha256=state_hash), state_path)
                x, history = z['x_endpoint'], z['route_history']
                require(x.shape == (BATCH, 3, 64, 64) and x.dtype == np.float32 and
                        np.isfinite(x).all() and array_hash(x) == state_hash, f'Invalid float state: {state_path}')
                e = m['experiment']
                fox, vp = e['route_fox_calls'], e['route_vp_calls']
                wanted = (np.ones(fox, dtype=bool).tolist() + np.zeros(vp, dtype=bool).tolist()
                          if s['route'] == 'head' else
                          np.zeros(vp, dtype=bool).tolist() + np.ones(fox, dtype=bool).tolist())
                require(fox + vp == s['steps'] and history.dtype == np.bool_ and
                        np.array_equal(history, np.asarray(wanted, dtype=bool)), 'Wrong saved route history')
                alpha, sigma = z['alpha'], z['sigma']
                endpoint = EXTENDED if s['group'] == 'native' else COMMON
                check_endpoint(alpha, sigma, endpoint, BATCH, float(scalar(z, 'endpoint_t')))
            with np.load(image_path, allow_pickle=False) as z:
                metadata(z, dict(expected, state_sha256=state_hash), image_path)
                for v in s['variants']:
                    a = z[f'samples_{v}']
                    require(a.shape == (BATCH, 64, 64, 3) and a.dtype == np.uint8, 'Wrong saved image shape/dtype')
            del x, history
    if s['kind'] == 'swap':
        require(hashlib.sha256(''.join(hashes).encode()).hexdigest() == claimed_statistics,
                'Report/statistics digest mismatch')
        require(len(set(state_hashes)) == ROUNDS, 'Duplicate endpoint states across rounds')
    for x in caches.values():
        x.flush()
    del caches
    return dict(spec=s, protocol_sha256=m['protocol_sha256'], source_sha256=m['source_sha256'],
                identity=identity, report_fids=targets, report_sha256=report_sha,
                model_checkpoints=m['experiment']['checkpoints'] if s['kind'] == 'swap' else None,
                pairing='same endpoint and row for denoisers only' if s['kind'] == 'swap' else 'single output',
                state_sha256=state_hashes)


def signature():
    return dict(script_sha256=digest('/opt/bootstrap.sh'), image_id=os.environ['BOOT_IMAGE_ID'],
                numpy=np.__version__, scipy=scipy.__version__,
                threads={k: os.environ.get(k) for k in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS')},
                count=N, seed=20261028, pilot_reps=5, full_reps=1000,
                resampling='independent trajectory cells including different N; paired denoisers within a cell',
                scope='fixed weights and empirical reference; generated-sample uncertainty only',
                input_roots=[str(ROOT / k) for k in ('next', 'swap', 'ab')])


def verify_audit(audit):
    require(audit['signature'] == signature(), 'Script/environment/settings changed; use new output')
    for path, claimed in audit['input_sha256'].items():
        require(digest(path) == claimed, f'Audited input changed: {path}')
    for path, claimed in audit['cache_sha256'].items():
        require(digest(OUT / path) == claimed, f'Audited cache changed: {path}')
    return audit


def prepare():
    target = OUT / 'audit.json'
    if target.exists():
        return verify_audit(json.loads(target.read_text()))
    require(not any(OUT.glob('draws_*.json')), 'Draws without audit')
    (OUT / 'cache').mkdir(exist_ok=True)
    for k in ('next', 'swap'):
        require((ROOT / k / 'results/queue.state').read_text().strip() == 'completed',
                f'Generation queue not completed: {k}')
        observed(ROOT / k / 'results/queue.state')
        verify_runtime(ROOT / k)
    summary = read_json(ROOT / 'next/results/summary.json')
    require(summary['status'] == 'completed' and summary['completed_cells'] == 4 and
            summary['num_samples'] == N and summary['sampling_seed'] == 0, 'Incomplete/wrong GPU summary')
    real = reference_moments(ROOT / 'next/runtime/assets/stats/celeba_64_0c2d89311192_stats.npz')
    np.savez(OUT / 'cache/reference.npz', **real)
    runs = []
    for s in specs():
        began = time.monotonic()
        run = audit_cell(s, summary)
        run['point_fids'] = {}
        for v, target_fid in run['report_fids'].items():
            name = s['name'] + '__' + v
            x = np.load(OUT / 'cache' / f'{name}.npy', mmap_mode='r')
            value = fid(x, real)
            require(abs(value - target_fid) < .001, f'FID fails reproduction: {name}: {value-target_fid}')
            run['point_fids'][v] = value
            print(f'AUDIT {name}: report={target_fid:.9f} recomputed={value:.9f}', flush=True)
            del x
        runs.append(run)
        print(f'AUDIT cell {s["name"]} passed, {time.monotonic()-began:.1f}s', flush=True)
        gc.collect()
    by_name = {r['spec']['name']:r for r in runs}
    native_checkpoints = by_name['native_vp_1000']['model_checkpoints']
    for r in runs:
        if r['spec']['group'] == 'native':
            require(r['model_checkpoints'] == native_checkpoints,
                    'Native weights/identities differ across trajectories or N')
        elif r['spec']['kind'] == 'swap':
            for arm in ('A', 'B'):
                baseline = by_name[f'controlled_{arm}']['identity']
                model = r['model_checkpoints'][arm]
                require(all(model[k] == baseline[k] for k in (
                    'checkpoint_id', 'checkpoint_step', 'training_protocol_sha256')),
                    'Controlled splice/baseline weights differ')
    audit = dict(status='passed', signature=signature(), reference_sha256=REFERENCE_SHA,
                 input_sha256=INVENTORY, runs=runs,
                 cache_sha256={str(p.relative_to(OUT)): digest(p) for p in sorted((OUT / 'cache').iterdir())},
                 limitations=['No model/checkpoint loading, image generation, or Inception recomputation',
                              'KID/IS values are checked for finiteness; confidence intervals are FID only',
                              'Independent-cell resampling is a working design, not guaranteed conservative'])
    verify_audit(audit)
    save(target, audit)
    print('AUDIT PASSED: 160 statistics shards; 14 FIDs reproduced.', flush=True)
    return audit


def interval(values, point):
    values = np.asarray(values)
    q = np.quantile(values, [.025, .5, .975])
    return dict(point=float(point), percentile_95=[float(q[0]), float(q[2])],
                bootstrap_median=float(q[1]), bootstrap_mean=float(values.mean()),
                bootstrap_bias=float(values.mean() - point),
                basic_95=[float(2*point-q[2]), float(2*point-q[0])])


def contrasts(group, p):
    if group == 'native':
        a, b, c, d = [p[k] for k in ('native_vp_1000', 'native_tail_1000',
                                     'native_vp_2000', 'native_tail_2000')]
        return dict(tail_minus_vp_1000=b[0]-a[0], tail_minus_vp_2000=d[0]-c[0],
                    contrast_change_2000_minus_1000=(d[0]-c[0])-(b[0]-a[0]),
                    vp_change_2000_minus_1000=c[0]-a[0], tail_change_2000_minus_1000=d[0]-b[0],
                    denoiser_on_vp_1000=a[1]-a[0], denoiser_on_tail_1000=b[1]-b[0],
                    denoiser_on_vp_2000=c[1]-c[0], denoiser_on_tail_2000=d[1]-d[0],
                    total_tail_1000=b[1]-a[0], total_tail_2000=d[1]-c[0])
    a, b, h, t = [p[k] for k in ('controlled_A', 'controlled_B', 'controlled_head', 'controlled_tail')]
    return dict(b_minus_a=b[0]-a[0], head_minus_a=h[0]-a[0], head_minus_b=h[0]-b[0],
                tail_minus_a=t[1]-a[0], tail_minus_b=t[1]-b[0],
                tail_with_a_denoiser_minus_a=t[0]-a[0], head_minus_tail_common_a=h[0]-t[0],
                denoiser_on_head=h[1]-h[0], denoiser_on_tail=t[1]-t[0])


def weights_for(group, rep, cell, count=N):
    rng = np.random.default_rng(np.random.SeedSequence([20261028, group, rep, cell]))
    return np.bincount(rng.integers(0, count, size=count), minlength=count)


def bootstrap(audit, group, requested, real):
    selected = [r for r in audit['runs'] if r['spec']['group'] == group]
    points = {r['spec']['name']: [r['point_fids'][v] for v in r['spec']['variants']] for r in selected}
    report_points = {r['spec']['name']: [r['report_fids'][v] for v in r['spec']['variants']] for r in selected}
    point_contrasts, report_contrasts = contrasts(group, points), contrasts(group, report_points)
    path = OUT / f'draws_{group}.json'
    fingerprint = canonical_hash(audit)
    state = dict(audit_fingerprint=fingerprint, group=group, completed_reps=0, draws=[], elapsed_seconds=0.)
    if path.exists():
        state = json.loads(path.read_text())
        require(state['audit_fingerprint'] == fingerprint and state['group'] == group, 'Resume fingerprint')
    done = state['completed_reps']
    require(done == len(state['draws']) and done <= requested, 'Invalid completed draw count')
    names = [r['spec']['name'] for r in selected]
    for row in state['draws']:
        require(set(row) == set(names) and all(len(row[r['spec']['name']]) == len(r['spec']['variants'])
                                              for r in selected), 'Wrong saved draw columns')
        require(all(np.isfinite(v).all() for v in row.values()), 'Non-finite saved draw')
    features = {r['spec']['name']: [np.load(OUT / 'cache' / (r['spec']['name'] + '__' + v + '.npy'),
                                           mmap_mode='r') for v in r['spec']['variants']] for r in selected}
    began = time.monotonic()
    previous = state['elapsed_seconds']
    for rep in range(done, requested):
        one = time.monotonic()
        row = {}
        for index, name in enumerate(names):
            counts = weights_for(0 if group == 'native' else 1, rep, index)
            row[name] = [fid(x, real, counts) for x in features[name]]
        state['draws'].append(row)
        state.update(completed_reps=rep+1, elapsed_seconds=previous+time.monotonic()-began)
        save(path, state)
        seconds = time.monotonic()-one
        print(f'BOOT {group} {rep+1}/{requested}: {seconds:.1f}s, '
              f'full-remaining estimate={seconds*(1000-rep-1)/3600:.2f}h; '
              f'peakRSS={resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024:.0f}MiB', flush=True)
    distributions = [contrasts(group, row) for row in state['draws']]
    result = dict(group=group, completed_reps=requested, pilot_only=requested < 1000,
                  status='completed', audit_fingerprint=fingerprint, signature=audit['signature'],
                  columns={r['spec']['name']:r['spec']['variants'] for r in selected},
                  point_fids=points, report_point_fids=report_points,
                  elapsed_seconds=state['elapsed_seconds'],
                  mean_seconds_per_rep=state['elapsed_seconds']/requested,
                  estimated_seconds_remaining_to_1000=state['elapsed_seconds']/requested*(1000-requested),
                  contrasts={k:dict(interval([d[k] for d in distributions], v),
                                    report_point=report_contrasts[k]) for k,v in point_contrasts.items()})
    save(OUT / f'summary_{group}.json', result)
    return result


def self_test():
    rng = np.random.default_rng(7)
    x = rng.normal(size=(64, 8)).astype(np.float32)
    counts = weights_for(0, 3, 0, 64)
    a, b = moments(x, counts)
    replicated = np.repeat(x, counts, axis=0).astype(np.float64)
    require(np.allclose(a, replicated.mean(0)) and np.allclose(b, np.cov(replicated, rowvar=False)),
            'Weighted moments differ from literal resampling')
    cov = np.cov(x.astype(np.float64), rowvar=False)
    eig, vec = linalg.eigh(cov)
    real = dict(mean=x.astype(np.float64).mean(0), sqrt=(vec*np.sqrt(eig))@vec.T, trace=np.trace(cov))
    require(abs(fid(x, real)) < 1e-9 and
            np.allclose(fid(x, real, counts), fid(replicated, real)), 'FID implementation')
    require(np.array_equal(counts, weights_for(0, 3, 0, 64)) and
            not np.array_equal(counts, weights_for(0, 3, 1, 64)) and
            not np.array_equal(counts, weights_for(0, 4, 0, 64)), 'Bootstrap RNG streams')
    t = .1
    q = .5*(20.-.1)*t*t+.1*t
    endpoint = -q-np.log(-np.expm1(-q))
    alpha = np.full((4, 1, 1, 1), np.exp(-q/2), dtype=np.float32)
    sigma = np.full(4, np.sqrt(-np.expm1(-q)), dtype=np.float32)
    check_endpoint(alpha, sigma, endpoint, 4, t)
    try:
        check_endpoint(alpha, sigma[:, None, None, None], endpoint, 4, t)
    except ValueError:
        pass
    else:
        raise AssertionError('Wrong VPSDE coefficient shapes accepted')
    metadata({'protocol_sha256': np.asarray('ok')}, {'protocol_sha256':'ok'}, 'legacy-without-seed')
    try:
        metadata({'sampling_seed':np.asarray(1)}, {'sampling_seed':0}, 'wrong-seed')
    except ValueError:
        pass
    else:
        raise AssertionError('Wrong seed accepted')
    fake = dict(baseline_reports=dict(
        native=dict(vp=dict(metrics='old-vp'), tail=dict(metrics='old-tail')),
        controlled=dict(A=dict(metrics='A'), B=dict(metrics='B'))), cells=dict(
            native_vp_nfe2000_seed0='new-vp', native_tail_nfe2000_seed0='new-tail',
            controlled_head_nfe1000_seed0='head', controlled_tail_nfe1000_seed0='tail'))
    expected = dict(native_vp_1000='old-vp', native_tail_1000='old-tail',
                    native_vp_2000='new-vp', native_tail_2000='new-tail',
                    controlled_A='A', controlled_B='B', controlled_head='head', controlled_tail='tail')
    require(all(metric_record(s, fake) == expected[s['name']] for s in specs()),
            'Wrong legacy/native/controlled metric source')
    require(np.allclose(interval([-1., 0., 1.], 0.)['basic_95'], [-.95, .95]), 'Interval transform')
    print('PASS: weighted moments/FID; independent RNG streams; endpoint shapes; legacy/controlled source mapping; intervals.', flush=True)


def main():
    mode = os.environ['BOOT_MODE']
    if mode == '--check':
        self_test()
        return
    require(mode in ('--pilot', '--full'), 'Wrong execution mode')
    lock = (OUT / '.bootstrap.lock').open('a')
    fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    audit = prepare()
    if mode == '--full':
        pilot = json.loads((OUT / 'pilot.json').read_text())
        require(pilot['status'] == 'passed' and pilot['audit_fingerprint'] == canonical_hash(audit) and
                pilot['completed_reps_per_group'] == 5, 'A passed 5-repeat pilot is required')
    requested = 5 if mode == '--pilot' else 1000
    with np.load(OUT / 'cache/reference.npz', allow_pickle=False) as z:
        real = {k:z[k] for k in z.files}
    result = {group:bootstrap(audit, group, requested, real) for group in ('native', 'controlled')}
    verify_audit(audit)
    if mode == '--pilot':
        save(OUT / 'pilot.json', dict(status='passed', audit_fingerprint=canonical_hash(audit),
             completed_reps_per_group=5,
             estimated_remaining_hours=sum(r['estimated_seconds_remaining_to_1000'] for r in result.values())/3600,
             summaries={k:f'summary_{k}.json' for k in result},
             note='Timing pilot only; do not interpret five-repeat confidence intervals'))
    else:
        save(OUT / 'completed.json', dict(status='completed', reps_per_group=1000,
             audit_fingerprint=canonical_hash(audit), summaries={k:f'summary_{k}.json' for k in result}))
    print(f'CelebA NFE/A-B CPU {mode} completed.', flush=True)


if __name__ == '__main__':
    main()
PY_NFE_BOOTSTRAP
