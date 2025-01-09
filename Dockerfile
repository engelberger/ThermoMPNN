FROM nvidia/cuda:12.1.0-cudnn8-devel-ubuntu22.04

# Avoid prompts from apt and set proper permissions
ENV DEBIAN_FRONTEND=noninteractive
RUN mkdir -p /tmp && chmod 1777 /tmp

# Install basic dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.10 \
    python3-pip \
    python3-venv \
    git \
    wget \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user
RUN useradd -ms /bin/bash vscode \
    && mkdir -p /home/vscode/.local/bin \
    && chown -R vscode:vscode /home/vscode

USER vscode
WORKDIR /home/vscode

# Create and activate virtual environment
ENV VIRTUAL_ENV=/home/vscode/venv
RUN python3 -m venv $VIRTUAL_ENV
ENV PATH="$VIRTUAL_ENV/bin:$PATH"

# Install PyTorch with CUDA support
RUN pip install --no-cache-dir torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Install other dependencies
RUN pip install --no-cache-dir \
    pytorch-lightning \
    joblib \
    omegaconf \
    pandas \
    numpy \
    tqdm \
    wandb \
    biopython

# Install MMseqs2
RUN cd /tmp \
    && wget https://mmseqs.com/latest/mmseqs-linux-avx2.tar.gz \
    && tar xvfz mmseqs-linux-avx2.tar.gz \
    && mv mmseqs/bin/mmseqs /home/vscode/.local/bin/ \
    && rm -rf mmseqs* 

ENV PATH="/home/vscode/.local/bin:$PATH"

# Set working directory to mounted code
WORKDIR /workspaces/ThermoMPNN