# Docker image for MLauto code execution.
#
# This mirrors the autogluon-assistant approach:
#   - CUDA runtime base image
#   - Python 3.11 via Miniconda
#   - The generated bash scripts create per-iteration conda envs inside this container
#
# Build:  docker build -t mlauto-executor:latest .
# The container is used by shared/utils.py:execute_in_docker()

ARG BASE_IMAGE=nvidia/cuda:12.2.0-runtime-ubuntu22.04
FROM ${BASE_IMAGE}

ENV DEBIAN_FRONTEND=noninteractive
ENV RUNNING_IN_DOCKER=true

# System packages
RUN apt-get update && \
    apt-get install -y \
        software-properties-common build-essential \
        wget unzip curl git pciutils vim \
        ffmpeg libsm6 libxext6 && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Install Miniconda
RUN wget -q https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh \
        -O /tmp/miniconda.sh && \
    bash /tmp/miniconda.sh -b -p /opt/conda && \
    rm /tmp/miniconda.sh

ENV PATH="/opt/conda/bin:$PATH"

# Accept conda TOS and init
RUN conda init bash && \
    conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main || true && \
    conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r || true && \
    conda clean -a -y

# Working directory — bind mounts go here at runtime:
#   /workspace/data       ← input data (read-only)
#   /workspace/output     ← output dir (read-write)
#   /workspace/script.sh  ← bash script to execute
WORKDIR /workspace

CMD ["/bin/bash"]
