# Trellis2 generator
FROM pytorch/pytorch:2.8.0-cuda12.8-cudnn9-runtime
ENV DEBIAN_FRONTEND=noninteractive

LABEL name="trellis2-gen" maintainer="404gen"

# Install docker dependencies
# Installing python 3.11, enbaling channel
RUN apt update -y &&  \
    apt-get install software-properties-common -y &&  \
    apt update -y &&  \
    add-apt-repository ppa:deadsnakes/ppa -y &&  \
    apt update -y  &&\
    apt install -y wget gnupg

# Add NVIDIA repo for CUDA 12.8
RUN wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb && \
    dpkg -i cuda-keyring_1.1-1_all.deb && \
    rm cuda-keyring_1.1-1_all.deb

RUN apt-get update && apt-get install -y \
    build-essential \
    git \
    cmake \
    lld \
    zstd \
    ninja-build \
    python3.11  \
    python3-pip  \
    python3-wheel \
    python3-pybind11 \
    pybind11-dev \
    libjpeg-dev  \
    zlib1g-dev \
    cuda-toolkit-12-8 \
    cuda-nvcc-12-8 \
    cuda-libraries-dev-12-8 \
    cuda-nvtx-12-8 \
    librange-v3-dev \
    # libspdlog-dev \
    # libfmt-dev \
    && rm -rf /var/lib/apt/lists/*

# rebinding names for python, pip, gcc, g++
RUN update-alternatives --install /usr/bin/pip pip /usr/bin/pip3 120 &&  \
    update-alternatives --install /usr/bin/python python /usr/bin/python3 120 && \
    pip install "pybind11[global]==3.0.2"

# Set environment variables
ENV LD_LIBRARY_PATH=/usr/lib64:$LD_LIBRARY_PATH

# Set CUDA environment variables
ENV CUDA_HOME=/usr/local/cuda-12.8
ENV PATH=$CUDA_HOME/bin:$PATH
ENV LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
ENV TORCH_CUDA_ARCH_LIST="8.9;9.0;12.0"

WORKDIR /app

# ============================================================
# Phase 1: Heavy, rarely-changing installs (best cache layer)
# These depend on torch from the base image, NOT requirements.txt.
# Moving them before COPY requirements.txt ensures they are
# cached even when pip requirements change.
# ============================================================

# Install flash attention
RUN wget https://github.com/Dao-AILab/flash-attention/releases/download/v2.8.3/flash_attn-2.8.3+cu12torch2.8cxx11abiTRUE-cp311-cp311-linux_x86_64.whl && \
    pip install flash_attn-2.8.3+cu12torch2.8cxx11abiTRUE-cp311-cp311-linux_x86_64.whl && \
    rm flash_attn-2.8.3+cu12torch2.8cxx11abiTRUE-cp311-cp311-linux_x86_64.whl

# Install utils3d
RUN pip install git+https://github.com/EasternJournalist/utils3d.git@9a4eb15e4021b67b12c460c7057d642626897ec8

# Install Kaolin
RUN pip install kaolin==0.18.0 -f https://nvidia-kaolin.s3.us-east-2.amazonaws.com/torch-2.8.0_cu128.html

RUN git clone https://github.com/facebookresearch/DRTK.git /tmp/DRTK --recursive && \
    (cd /tmp/DRTK && git checkout c5a3e523435b4daa37572c0653d6686531c787b2) && \
    pip install /tmp/DRTK --no-build-isolation

# Install CuMesh
RUN git clone https://github.com/JeffreyXiang/CuMesh.git /tmp/CuMesh --recursive && \
    cd /tmp/CuMesh && git checkout 6f403664cc79e5a0f0c993d2d03976ace0e7f829 && \
    pip install /tmp/CuMesh --no-build-isolation && \
    rm -rf /tmp/CuMesh

# Install FlexGEMM
RUN git clone https://github.com/JeffreyXiang/FlexGEMM.git /tmp/FlexGEMM --recursive && \
    cd /tmp/FlexGEMM && git checkout de6411284d20a6d41362db27a59a6923aeceebfe && \
    pip install /tmp/FlexGEMM --no-build-isolation && \
    rm -rf /tmp/FlexGEMM

# Copy o-voxel and install with eigen
COPY o-voxel /tmp/o-voxel
RUN git clone --depth 1 --branch 3.4.0 https://gitlab.com/libeigen/eigen.git /tmp/o-voxel/third_party/eigen && \
    pip install /tmp/o-voxel --no-build-isolation && \
    rm -rf /tmp/o-voxel

RUN git clone https://github.com/PramaLLC/BEN2.git /tmp/BEN2 --recursive && \
    cd /tmp/BEN2 && git checkout 2c99a5da477b5523585bfa5c893888a6e818a8f6 && \
    pip install /tmp/BEN2 --no-build-isolation && \
    rm -rf /tmp/BEN2


COPY libuvula /tmp/libuvula
RUN git clone https://github.com/gabime/spdlog.git /tmp/libuvula/spdlog && \
    cd /tmp/libuvula/spdlog && \
    git fetch --tags && git checkout v1.17.0 && \
    mkdir build && cd build && \
    cmake .. \
    -DCMAKE_BUILD_TYPE=Release \
    -DSPDLOG_FMT_EXTERNAL=OFF \
    -DCMAKE_POSITION_INDEPENDENT_CODE=ON \
    -DSPDLOG_INSTALL=ON && \
    cmake --build . -j &&\
    cmake --install .


RUN pip install -r /tmp/libuvula/requirements.txt && \
    mkdir -p /tmp/libuvula/build && cd /tmp/libuvula/build && \
    cmake .. \
    -DCMAKE_PREFIX_PATH=$CONDA_PREFIX \
    -DCMAKE_INSTALL_PREFIX=$CONDA_PREFIX \
    -DCMAKE_BUILD_TYPE=Release \
    -DUVULA_BUILD_PYTHON_BINDINGS=ON \
    -DUVULA_BUILD_JS_BINDINGS=OFF \
    -DUVULA_BUILD_CLI=OFF \
    -DUVULA_ENABLE_EXTENSIVE_WARNINGS=OFF && \
    cmake --build . -- -j$(nproc)

RUN cd /tmp/libuvula && \
    pip install build && \
    python -m build -w -o /tmp/libuvula/wheels && \
    pip install /tmp/libuvula/wheels/pyuvula-0.0.0-cp311-cp311-linux_x86_64.whl && \
    rm -rf /tmp/libuvula

# ============================================================
# Phase 2: pip requirements (rebuilds only when requirements.txt changes)
# ============================================================
COPY requirements.txt /app/
RUN pip install -r requirements.txt

# ============================================================
# Phase 3: Application source (changes most often, rebuilds fastest)
# ============================================================
COPY . /app

EXPOSE 10006

# Set entrypoint to use conda environment
CMD ["python", "serve.py", "--port", "10006"]
