#####################################
# → Builder stage (multi‑stage image) #
#####################################
FROM runpod/pytorch:2.1.0-py3.10-cuda11.8.0-devel-ubuntu22.04 AS builder

# Make logs unbuffered, define some defaults for Trellis-style Start.sh
ENV PYTHONUNBUFFERED=1 \
    MODE_TO_RUN=pod \
    WORKDIR=/workspace

WORKDIR ${WORKDIR}

# Required for pip installs that include CUDA-extensions; keeps image small
RUN apt-get update -qq && \
    DEBIAN_FRONTEND=noninteractive apt-get install -y \
      build-essential curl git libgl1 qtbase5-dev ninja-build ca-certificates && \
    rm -rf /var/lib/apt/lists/*

# Create isolated Python virtual env
RUN python3 -m venv /venv && \
    /venv/bin/pip install --upgrade pip setuptools wheel

# Copy entire GitHub repo context (Dockerfile, handler.py, start.sh, Trellis files, requirements.txt)
# This avoids trying to clone again inside the container
COPY . ${WORKDIR}

# Install fixed versions of torch + xformers from PyTorch index, then kaolin
RUN /venv/bin/pip install torch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2 \
       xformers --index-url https://download.pytorch.org/whl/cu118 && \
    /venv/bin/pip install --ignore-installed kaolin==0.17.0 \
       -f https://nvidia-kaolin.s3.us-east-2.amazonaws.com/torch-2.1.2_cu118.html

# Install all other Python dependencies from your repository
RUN /venv/bin/pip install --no-cache-dir -r requirements.txt

##################################
# → Runtime stage (minimal image) #
##################################
FROM runpod/pytorch:2.1.0-py3.10-cuda11.8.0-devel-ubuntu22.04

ENV PYTHONUNBUFFERED=1 \
    MODE_TO_RUN=pod \
    WORKDIR=/workspace \
    PATH="/venv/bin:${PATH}"

WORKDIR ${WORKDIR}

# Copies the full environment and application
COPY --from=builder /venv /venv
COPY --from=builder ${WORKDIR} ${WORKDIR}

# Run your trellis-runpod deployment entrypoint script
CMD ["bash", "./start.sh"]
