FROM nvidia/cuda:12.1.1-cudnn8-devel-ubuntu20.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    TF_CPP_MIN_LOG_LEVEL=2 \
    CUDA_HOME=/usr/local/cuda \
    TORCH_CUDA_ARCH_LIST=9.0 \
    MAX_JOBS=8 \
    TFDS_DATA_DIR=/workspace/score_sde_pytorch/.tfds \
    TORCH_EXTENSIONS_DIR=/workspace/score_sde_pytorch/.torch_extensions \
    XDG_CACHE_HOME=/workspace/score_sde_pytorch/.cache

RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
    git \
    build-essential \
    ninja-build \
    python3.8 \
    python3.8-dev \
    python3-distutils \
    python3-pip \
    python3-setuptools \
    python3-wheel \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    && rm -rf /var/lib/apt/lists/*

RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.8 1 \
    && python -m pip install --upgrade pip setuptools wheel

# The repo is bind-mounted at container runtime and is typically host-owned
# (e.g. a non-root server user), while the container runs as root. Git's
# ownership check then refuses every command with "detected dubious
# ownership" until `git config --global --add safe.directory ...` is run
# inside that specific container -- and a `--global` fix does not survive
# `docker run --rm`, so it has to be redone by hand every session. Writing it
# to the system config here bakes it into the image instead, so it applies
# to every container regardless of the bind-mounted owner.
RUN git config --system --add safe.directory /workspace/score_sde_pytorch

WORKDIR /workspace/score_sde_pytorch

COPY docker/requirements.base.txt /tmp/requirements.base.txt

RUN python -m pip install -r /tmp/requirements.base.txt

RUN python -m pip install \
    --index-url https://download.pytorch.org/whl/cu121 \
    --extra-index-url https://pypi.org/simple \
    torch==2.4.1 torchvision==0.19.1 \
    && python -m pip install typing-extensions==4.12.2

COPY . /workspace/score_sde_pytorch

RUN mkdir -p "$TFDS_DATA_DIR" "$TORCH_EXTENSIONS_DIR" "$XDG_CACHE_HOME"

# The gitignored official source is fetched at its pinned revision so image
# contents and canonical validation do not depend on the caller's build tree.
RUN bash tools/install_official_edm.sh

RUN python -m py_compile main.py run_lib.py sde_lib.py edm_lib.py \
    artifact_utils.py reproducibility.py utils.py datasets.py evaluation.py \
    losses.py sampling.py models/ema.py models/edm.py models/edm_canonical.py \
    && python baseline_validation.py \
    && python canonical_edm_validation.py \
    && python reproducibility_validation.py \
    && python checkpoint_validation.py \
    && python artifact_validation.py \
    && if [ -f tests/test_fox_sde.py ]; then python -m unittest tests.test_fox_sde; fi

CMD ["/bin/bash"]
