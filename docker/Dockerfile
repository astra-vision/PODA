# Use the official PyTorch image with CUDA 11.3
FROM nvidia/cuda:11.3.1-cudnn8-devel-ubuntu20.04

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV CUDA_HOME=/usr/local/cuda
ENV NVCC=/usr/local/cuda/bin/nvcc

# Install basic dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.10 \
    gcc-9 \
    g++-9 \
    git \
    wget \
    curl \
    ca-certificates \
    libopencv-dev \
    && rm -rf /var/lib/apt/lists/*

# Set gcc/g++ to use version 9
RUN update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-9 100 \
    && update-alternatives --install /usr/bin/g++ g++ /usr/bin/g++-9 100

# Install pip
RUN apt-get update && apt-get install -y --no-install-recommends \
    software-properties-common \
    && add-apt-repository ppa:deadsnakes/ppa \
    && apt-get update && apt-get install -y --no-install-recommends \
    python3.10 \
    python3.10-venv \
    python3.10-dev \
    python3.10-distutils \
    curl \
    && rm -rf /var/lib/apt/lists/*

RUN curl https://bootstrap.pypa.io/get-pip.py -o /tmp/get-pip.py \
    && python3.10 /tmp/get-pip.py \
    && rm /tmp/get-pip.py

# Install PyTorch with CUDA 11.3 and related dependencies
RUN pip uninstall -y numpy
RUN pip install numpy==1.26.1
RUN pip install torch==1.11.0+cu113 torchvision==0.12.0+cu113 -f https://download.pytorch.org/whl/cu113/torch_stable.html

# Install OpenCV
RUN pip install opencv-python

# Install MMCV and MMDetection
RUN pip install mmcv-full==1.6.0 -f https://download.openmmlab.com/mmcv/dist/cu113/torch1.11.0/index.html

# Set Python 3.10 as the default python
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.10 1

# Verify the installation
RUN python3 --version && \
    gcc --version && \
    nvcc --version && \
    python3 -c "import torch; print(torch.cuda.is_available())" && \
    python3 -c "import mmcv; print(mmcv.__version__)" && \
    python3 -c "import cv2; print(cv2.__version__)"
