# Docker setup for H200 and RTX

This setup packages the current repository into a CUDA-enabled container intended for Hopper-class GPUs such as H200.

RTX PRO 6000 Blackwell uses the same build and run layout, with
`Dockerfile.rtx` and `docker/requirements.rtx.txt`. Both images use the root
`.dockerignore` and mount the repository at `/workspace/score_sde_pytorch`.
Datasets, workdirs, logs, and caches are excluded from the build context;
they remain available through the repository mount at runtime.

## What is included in H200

- Python 3.8
- TensorFlow 2.4 CPU-side stack used by this repository for datasets and evaluation utilities
- PyTorch 2.4.1 + CUDA 12.1
- TorchVision 0.19.1
- pinned auxiliary packages that were already validated in local and server runs

The container does not install `jax`, `tensorflow-io`, or `torchaudio`, because they are not required for the training path used in this fork.

## Build

```bash
./docker/build_h200.sh
```

or explicitly:

```bash
docker build -t score-sde-h200 -f Dockerfile .
```

## Run

```bash
./docker/run_h200.sh
```

This mounts the repository into the container and keeps TFDS cache, torch extensions, and XDG cache inside the project directory:

- `.tfds`
- `.torch_extensions`
- `.cache`

## Notes

- `TORCH_CUDA_ARCH_LIST=9.0` is set for Hopper/H200.
- The image uses a CUDA devel base so that optional custom ops can compile if the runtime allows it.
- The repository already contains safe fallbacks for custom ops, so training still works if those extensions are unavailable.
- TensorFlow is expected to run on the CPU side in this setup. PyTorch uses the GPU.

## RTX PRO 6000 Blackwell

Run from the repository root, just as for H200:

```bash
./docker/build_rtx.sh
./docker/run_rtx.sh
```

The default image is `score-sde-rtx:latest`. The run script selects the
allocated physical GPU 2 on tokugawa by its UUID; it appears as `cuda:0`
inside the container. To choose an image and GPU explicitly:

```bash
./docker/build_rtx.sh score-sde-rtx:latest
./docker/run_rtx.sh score-sde-rtx:latest GPU-25e966e6-f936-4c1d-370b-fbe8e47c0095
```

The RTX image uses CUDA 12.8.1, `TORCH_CUDA_ARCH_LIST=12.0`, PyTorch 2.7.1,
TorchVision 0.22.1, Python 3.10, and TensorFlow CPU 2.15.1. Python and the
TensorFlow stack also need updating to stay compatible with the newer
PyTorch environment. Its extension and XDG caches use `.torch_extensions_rtx`
and `.cache_rtx` to avoid mixing binaries from the two environments.

Both builds install the same pinned official EDM source and run the H200
validation scripts. RTX additionally checks dependency consistency, imports,
and endpoint-denoise sampling regressions. The complete RTX package list is
saved inside the image at `/opt/score-sde-rtx-environment/requirements.lock.txt`.
Ninja comes from the system package `ninja-build`; its version is checked
during the build. The duplicate Python package is omitted because pip 25.1.1
rejects the wheel tags of `ninja==1.11.1.1` in the server build.
Build checks run on CPU; GPU support must be checked on the server after
building. With `run_rtx.sh`, a basic check inside the container is:

```bash
python - <<'PY'
import torch
print('PyTorch:', torch.__version__, 'CUDA:', torch.version.cuda)
assert torch.cuda.is_available()
assert torch.cuda.device_count() == 1
print('GPU:', torch.cuda.get_device_name(0))
print('Compute capability:', torch.cuda.get_device_capability(0))
x = torch.randn(256, 256, device='cuda')
y = x @ x.T
torch.cuda.synchronize()
assert torch.isfinite(y).all().item()
print('CUDA check OK')
PY
```
