###############################
# → Builder stage (no GPU attached)
###############################
FROM runpod/pytorch:2.1.0-py3.10-cuda11.8.0-devel-ubuntu22.04 AS builder

# ⚠️ Must include at least one CUDA arch so torch.cpp_extension sees something
ENV TORCH_CUDA_ARCH_LIST="8.6+PTX"

# Build-arg for cache busting ⬆️
ARG CACHEBUST=1
RUN echo "Cache bust: $CACHEBUST"

WORKDIR /workspace

# System‑level tools + ninja
RUN apt-get update -qq \
 && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
      git build-essential curl cmake ninja-build libgl1 pkg-config python3-dev \
 && rm -rf /var/lib/apt/lists/*

# Create Python venv and upgrade tooling
RUN python3 -m venv /venv \
 && /venv/bin/pip install --upgrade pip setuptools wheel

# Pin numpy & scipy early for ABI compatibility
RUN /venv/bin/pip install --no-cache-dir \
      numpy==1.23.5 \
      scipy==1.9.3

# Install specific torch + xformers + kaolin
RUN /venv/bin/pip install torch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2 xformers \
       --index-url https://download.pytorch.org/whl/cu118 \
 && /venv/bin/pip install --ignore-installed kaolin==0.17.0 \
       -f https://nvidia-kaolin.s3.us-east-2.amazonaws.com/torch-2.1.2_cu118.html

# Copy and install Python requirements
COPY requirements.txt /workspace/
RUN /venv/bin/pip install --no-cache-dir --no-build-isolation -r requirements.txt

# 2) Now force-reinstall NumPy ≥1.24, SciPy 1.10.x, and OpenCV headless
RUN /venv/bin/pip install --no-cache-dir --no-deps --force-reinstall \
      numpy==1.24.3 \
      scipy==1.10.1 \
      opencv-python-headless==4.7.0.72


# Get your repository content (handler.py + trellis code)
COPY . /workspace

################################
# → Runtime stage (actual container)
################################
FROM runpod/pytorch:2.1.0-py3.10-cuda11.8.0-devel-ubuntu22.04

ENV PYTHONUNBUFFERED=1 \
    MODE_TO_RUN=serverless \
    WARP_DISABLE_CPU_TRACER=1 \
    PATH="/venv/bin:$PATH"

WORKDIR /workspace

# Copy venv and workspace from builder
COPY --from=builder /venv /venv
COPY --from=builder /workspace /workspace

# Ensure start.sh is executable and used as entrypoint ⬆️
RUN chmod +x /workspace/start.sh
ENTRYPOINT ["/workspace/start.sh"]
