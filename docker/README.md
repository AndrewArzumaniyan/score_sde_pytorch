# Docker setup for H200

This setup packages the current repository into a CUDA-enabled container intended for Hopper-class GPUs such as H200.

## What is included

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
