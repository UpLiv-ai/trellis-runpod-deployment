# Stage 1: Define the base image
# Use an official PyTorch image with Python 3.10 and CUDA 11.8, matching the original environment's core dependencies.
# The 'devel' tag includes the full CUDA toolkit, which is often necessary for compiling custom extensions.
FROM pytorch/pytorch:2.1.2-cuda11.8-cudnn8-devel

# Set environment variables for non-interactive installs and unbuffered Python output for better logging.
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
# Set environment variables for model performance, as seen in the original Dockerfile.
ENV ATTN_BACKEND=xformers
ENV SPCONV_ALGO=native

# Install system-level dependencies.
# - git is required for installing packages from GitHub.
# - libglm-dev was a dependency in the original Dockerfile for 3D graphics operations.
# Clean up apt cache to reduce image size.
RUN apt-get update && \
    apt-get install -y git libglm-dev && \
    rm -rf /var/lib/apt/lists/*

# Set the working directory inside the container.
WORKDIR /app

# Copy the consolidated requirements file into the working directory.
COPY requirements.txt.

# Install all Python dependencies from the single requirements.txt file.
# Using --no-cache-dir reduces the image size by not storing the pip cache.
RUN pip install --no-cache-dir -r requirements.txt

# Copy the entire build context (including handler.py and the trellis-stable-projectorz subdirectory)
# into the container's working directory. This brings in the model code, model weights, and handler script.
COPY..

# Add the cloned repository to the PYTHONPATH.
# This allows Python to find and import the 'trellis' package from the handler.py script.
ENV PYTHONPATH="${PYTHONPATH}:/app/trellis-stable-projectorz"

# Define the command to run when a worker container starts.
# This executes the handler script, which in turn starts the RunPod serverless listener.
# The '-u' flag ensures that Python output is sent straight to stdout/stderr without buffering.
CMD ["python", "-u", "handler.py"]