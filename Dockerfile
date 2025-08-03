###############################
# → Selector stage (dual‑mode)  #
###############################
FROM runpod/pytorch:2.1.0-py3.10-cuda11.8.0-devel-ubuntu22.04 AS builder

ENV PYTHONUNBUFFERED=1 \
    MODE_TO_RUN=pod \
    WORKDIR=/workspace

WORKDIR $WORKDIR

# 1) Install Git and system build tools
RUN apt-get update -qq && \
    DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
      git build-essential curl python3-dev pkg-config cmake ninja-build libgl1 && \
    rm -rf /var/lib/apt/lists/*

# 2) Create and activate virtualenv; upgrade pip, setuptools, wheel
RUN python3 -m venv /venv && \
    /venv/bin/pip install --upgrade pip setuptools wheel

# 3) Clone your repo (includes handler.py, Trellis code, submodules)
RUN git clone https://github.com/UpLiv-ai/trellis-runpod-deployment.git $WORKDIR

# 4) Install PyTorch, xformers, kaolin using CUDA‑compatible wheels
RUN /venv/bin/pip install torch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2 xformers \
      --index-url https://download.pytorch.org/whl/cu118 && \
    /venv/bin/pip install --ignore-installed kaolin==0.17.0 \
      -f https://nvidia-kaolin.s3.us-east-2.amazonaws.com/torch-2.1.2_cu118.html

# 5) Install the rest of your dependencies—DISABLE build isolation
COPY requirements.txt .
RUN /venv/bin/pip install --no-cache-dir --no-build-isolation -r requirements.txt

################################
# → Runtime stage (final image) #
################################
FROM runpod/pytorch:2.1.0-py3.10-cuda11.8.0-devel-ubuntu22.04

ENV PYTHONUNBUFFERED=1 \
    MODE_TO_RUN=pod \
    WORKDIR=/workspace

WORKDIR $WORKDIR

# Copy virtualenv and app code from builder
COPY --from=builder /venv /venv
COPY --from=builder $WORKDIR $WORKDIR

ENV PATH="/venv/bin:$PATH"

CMD ["bash", "./start.sh"]
