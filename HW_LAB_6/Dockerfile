# ---------------------------------------------------------------------------
# ClickTheLook — training container
#
# Build:
#   docker build -t clickthelook:latest .
#
# Run:
#   docker run --rm --gpus all \
#     -v /path/to/data:/data \
#     -e DATA_ROOT=/data \
#     clickthelook:latest
# ---------------------------------------------------------------------------

FROM python:3.11-slim

# System libraries required by OpenCV / Ultralytics
# libgl1-mesa-glx was renamed to libgl1 in Debian trixie (python:3.11-slim base)
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgl1 \
        libglib2.0-0 \
        libsm6 \
        libxext6 \
        libxrender-dev \
        git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python dependencies first so this layer is cached independently
# of source-code changes.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy only the files the training job needs.
COPY config.py       ./config.py
COPY src/            ./src/
COPY scripts/        ./scripts/

ENTRYPOINT ["python", "scripts/main.py"]
