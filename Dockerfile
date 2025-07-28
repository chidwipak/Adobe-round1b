# syntax=docker/dockerfile:1
FROM --platform=linux/amd64 python:3.11-slim AS builder

# Install build dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    wget \
    && rm -rf /var/lib/apt/lists/*

# Install PyTorch CPU-only (required for sentence-transformers)
RUN pip install --no-cache-dir torch==2.2.0 --index-url https://download.pytorch.org/whl/cpu

# Copy requirements and install all Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Final stage
FROM --platform=linux/amd64 python:3.11-slim

# Set environment variables for optimal performance and offline/CPU-only
ENV PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app \
    OMP_NUM_THREADS=1 \
    MKL_NUM_THREADS=1 \
    TRANSFORMERS_OFFLINE=1 \
    TOKENIZERS_PARALLELISM=false

# Create application directories
RUN mkdir -p /app/input /app/output /opt/models

# Copy Python environment from builder stage
COPY --from=builder /usr/local /usr/local

# Copy pre-downloaded models (if present in models directory)
COPY models/ /opt/models/

# Copy application source code
COPY src/ /app/src/

# Set working directory
WORKDIR /app

# Set executable permissions for CLI
RUN chmod +x /app/src/cli.py

# Default command for Round 1B (can be overridden)
CMD ["python", "/app/src/cli.py", "--input-dir", "/app/input", "--output-dir", "/app/output"] 