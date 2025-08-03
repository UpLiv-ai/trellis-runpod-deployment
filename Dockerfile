###############################
# → Builder stage (incl. dependencies)
###############################
FROM runpod/pytorch:2.1.0-py3.10-cuda11.8.0-devel-ubuntu22.04 AS builder

ENV PYTHONUNBUFFERED=1 \
    MODE_TO_RUN=pod \
    WORKDIR=/workspace

WORKDIR $WORKDIR

# Install Git and build tools
RUN apt-get update -qq && \
    DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
      git build-essential curl python3-dev pkg-config cmake ninja-build libgl1 && \
    rm -rf /var/lib/apt/lists/*

# Create virtual environment and upgrade packaging
RUN python3 -m venv /venv && \
    /venv/bin/pip install --upgrade pip setuptools wheel

# Copy in your repository files (from GitHub build context)
COPY . .

# Install CUDA‑compatible PyTorch + kaolin, then other deps
RUN /venv/bin/pip install torch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2 xformers \
      --index-url https://download.pytorch.org/whl/cu118 && \
    /venv/bin/pip install --ignore-installed kaolin==0.17.0 \
      -f https://nvidia-kaolin.s3.us-east-2.amazonaws.com/torch-2.1.2_cu118.html

# Install rest of requirements
RUN /venv/bin/pip install --no-cache-dir --no-build-isolation -r requirements.txt

###############################
# → Runtime stage
###############################
FROM runpod/pytorch:2.1.0-py3.10-cuda11.8.0-devel-ubuntu22.04

ENV PYTHONUNBUFFERED=1 \
    MODE_TO_RUN=pod \
    WORKDIR=/workspace

WORKDIR $WORKDIR

COPY --from=builder /venv /venv
COPY --from=builder $WORKDIR $WORKDIR

ENV PATH="/venv/bin:$PATH"

CMD ["bash", "./start.sh"]
