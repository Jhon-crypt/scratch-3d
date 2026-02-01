# View-synthesis: Zero123++ (image → consistent multi-view). Needs GPU.
FROM pytorch/pytorch:2.1.0-cuda12.1-cudnn8-runtime

ENV DEBIAN_FRONTEND=noninteractive
WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends git \
    && rm -rf /var/lib/apt/lists/*

# Clone Zero123++ for custom diffusers pipeline
RUN git clone --depth 1 https://github.com/SUDO-AI-3D/zero123plus.git /app/zero123plus

WORKDIR /app
COPY view-synthesis-service /app/view-synthesis-service
# Force numpy<2 so transformers 4.29 does not report "Numpy is not available" (NumPy 2.x incompatibility)
RUN pip install --no-cache-dir "numpy>=1.24,<2" && \
    pip install --no-cache-dir -r view-synthesis-service/requirements.txt

ENV ZERO123_CUSTOM_PIPELINE=/app/zero123plus/diffusers-support

WORKDIR /app/view-synthesis-service
EXPOSE 8000
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
