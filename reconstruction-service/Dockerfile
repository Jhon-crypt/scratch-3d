# TripoSR needs PyTorch + CUDA; clone repo and install deps.
# Use -devel so nvcc/CUDA toolkit is present for building torchmcubes.
FROM pytorch/pytorch:2.1.0-cuda12.1-cudnn8-devel

ENV DEBIAN_FRONTEND=noninteractive
WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    git build-essential cmake ninja-build \
    && rm -rf /var/lib/apt/lists/*

# Clone TripoSR (single-image 3D reconstruction)
RUN git clone --depth 1 https://github.com/VAST-AI-Research/TripoSR.git /app/triposr

WORKDIR /app/triposr
RUN pip install --no-cache-dir -r requirements.txt

WORKDIR /app
COPY reconstruction-service /app/reconstruction-service
ENV PYTHONPATH=/app/triposr
WORKDIR /app/reconstruction-service

RUN pip install --no-cache-dir fastapi "uvicorn[standard]" pydantic python-multipart

EXPOSE 8000
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
