###############################
# → Selector stage (dual‑mode)  #
###############################
FROM runpod/pytorch:2.1.0-py3.10-cuda11.8.0-devel-ubuntu22.04 AS builder

# Force Python to stdout/stderr unbuffered for logs
ENV PYTHONUNBUFFERED=1 \
    MODE_TO_RUN=pod \
    WORKDIR=/workspace

WORKDIR $WORKDIR

# Install git and build tools (so pip can install Git‑based dependencies)
RUN apt-get update -qq && \
    DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
       git build-essential curl libgl1 && \
    rm -rf /var/lib/apt/lists/*

# Create isolated Python environment
RUN python3 -m venv /venv && \
    /venv/bin/pip install --upgrade pip

# Install specific versions before loading additional requirements
# Torch 2.1.2 + Xformers wheel via PyTorch index, then kaolin
RUN /venv/bin/pip install torch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2 xformers \
       --index-url https://download.pytorch.org/whl/cu118 && \
    /venv/bin/pip install --ignore-installed kaolin==0.17.0 \
       -f https://nvidia-kaolin.s3.us-east-2.amazonaws.com/torch-2.1.2_cu118.html

# Now install other dependencies
COPY requirements.txt .
RUN /venv/bin/pip install --no-cache-dir -r requirements.txt

# Clone your UpLiv‑ai trellis‑runpod‑deployment repo (including handler.py and Trellis code)
RUN git clone https://github.com/UpLiv-ai/trellis-runpod-deployment.git .

################################
# → Runtime stage (final image) #
################################
FROM runpod/pytorch:2.1.0-py3.10-cuda11.8.0-devel-ubuntu22.04

ENV PYTHONUNBUFFERED=1 \
    MODE_TO_RUN=pod \
    WORKDIR=/workspace

WORKDIR $WORKDIR

# Copy virtualenv and application files
COPY --from=builder /venv /venv
COPY --from=builder $WORKDIR $WORKDIR

ENV PATH="/venv/bin:$PATH"

# Entrypoint script chooses pod vs serverless
CMD ["bash", "./start.sh"]
