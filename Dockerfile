###############################
# → Builder stage
###############################
FROM runpod/pytorch:2.1.0-py3.10-cuda11.8.0-devel-ubuntu22.04 AS builder

# Make sure C++ extensions only compile for supported GPUs (sm_90 + PTX is safe)
# Adjust if using CUDA 12.x with updated architecture support.
ENV TORCH_CUDA_ARCH_LIST="8.6 8.6+PTX 9.0+PTX"

# This is optional — only if your repo is private.
# Use docker build --build-arg GITHUB_TOKEN=<your_PAT> .
# Then uncomment below git URL.
ARG GITHUB_TOKEN
ENV GITHUB_REPO="https://github.com/UpLiv-ai/trellis-runpod-deployment.git"

WORKDIR /workspace

RUN apt-get update -qq && \
    DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
      git build-essential cmake ninja-build curl python3-dev libgl1 pkg-config && \
    rm -rf /var/lib/apt/lists/*

# Create a clean venv and upgrade pip, setuptools, wheel
RUN python3 -m venv /venv && \
    /venv/bin/pip install --upgrade pip setuptools wheel

# Install base PyTorch + Xformers + Kaolin wheels via PyTorch index
RUN /venv/bin/pip install torch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2 xformers \
       --index-url https://download.pytorch.org/whl/cu118 && \
    /venv/bin/pip install --ignore-installed kaolin==0.17.0 \
       -f https://nvidia-kaolin.s3.us-east-2.amazonaws.com/torch-2.1.2_cu118.html

# Copy your app's requirements and install everything with no build isolation
COPY requirements.txt /workspace/
RUN TORCH_CUDA_ARCH_LIST="$TORCH_CUDA_ARCH_LIST" \
    /venv/bin/pip install --no-cache-dir --no-build-isolation -r requirements.txt

# Clone your deployment repo (handler.py + Trellis logic)
RUN git clone ${GITHUB_REPO} .

###############################
# → Runtime stage
###############################
FROM runpod/pytorch:2.1.0-py3.10-cuda11.8.0-devel-ubuntu22.04

ENV PYTHONUNBUFFERED=1 \
    MODE_TO_RUN=pod \
    WORKDIR=/workspace \
    PATH="/venv/bin:$PATH"

WORKDIR $WORKDIR

COPY --from=builder /venv /venv
COPY --from=builder /workspace /workspace

CMD ["bash", "./start.sh"]
