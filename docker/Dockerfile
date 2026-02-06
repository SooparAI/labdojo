# ============================================================================
# Lab Dojo v8 - Serverless Inference Brain
# Runs vLLM with Qwen2.5-32B-Instruct-AWQ on Vast.ai Serverless
# ============================================================================

FROM nvidia/cuda:12.1.0-runtime-ubuntu22.04

# Environment
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV MODEL_NAME=Qwen/Qwen2.5-32B-Instruct-AWQ
ENV PORT=8000
ENV MAX_MODEL_LEN=32768
ENV GPU_MEMORY_UTILIZATION=0.95

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3.11 \
    python3-pip \
    git \
    curl \
    wget \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip
RUN pip3 install --no-cache-dir --upgrade pip

# Install vLLM and dependencies
RUN pip3 install --no-cache-dir \
    vllm==0.6.3 \
    fastapi==0.115.0 \
    uvicorn[standard]==0.30.6 \
    aiohttp==3.10.5 \
    pydantic==2.9.2

# Create app directory
WORKDIR /app

# Copy inference server
COPY inference_server.py /app/

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=120s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Start server
CMD ["python3", "inference_server.py"]
