# Base image với CUDA support
FROM nvidia/cuda:12.1.0-cudnn8-runtime-ubuntu22.04

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3.10 \
    python3-pip \
    libsndfile1 \
    ffmpeg \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .

# Install PyTorch with CUDA 12.1 from official repo first
RUN pip3 install --no-cache-dir \
    torch==2.5.1 \
    torchvision==0.20.1 \
    torchaudio==2.5.1 \
    --index-url https://download.pytorch.org/whl/cu121

# Install other Python dependencies
RUN pip3 install --no-cache-dir -r requirements.txt

# Install additional dependencies for API
RUN pip3 install --no-cache-dir \
    fastapi==0.104.1 \
    uvicorn[standard]==0.24.0 \
    python-multipart==0.0.6

# Copy application code
COPY . .

# Create directories
RUN mkdir -p /app/pretrain_model /app/result /app/audio

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV DEVICE=cuda
ENV MODEL_PATH=/app/pretrain_model/EmoTalk.pth
ENV PORT=8000
ENV HOST=0.0.0.0

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Run FastAPI server
CMD ["python3", "fastapi_server.py"]
