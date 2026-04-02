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

WORKDIR /workspace/score_sde_pytorch

COPY docker/requirements.base.txt /tmp/requirements.base.txt

RUN python -m pip install -r /tmp/requirements.base.txt

RUN python -m pip install \
    --index-url https://download.pytorch.org/whl/cu121 \
    torch==2.4.1 torchvision==0.19.1 \
    && python -m pip install typing-extensions==4.12.2

COPY . /workspace/score_sde_pytorch

RUN mkdir -p "$TFDS_DATA_DIR" "$TORCH_EXTENSIONS_DIR" "$XDG_CACHE_HOME"

RUN python -m py_compile main.py run_lib.py sde_lib.py \
    && if [ -f tests/test_fox_sde.py ]; then python -m unittest tests.test_fox_sde; fi

CMD ["/bin/bash"]
