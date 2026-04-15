FROM nvidia/cuda:12.4.0-devel-ubuntu22.04

ARG DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y \
    python3.10 \
    python3-pip \
    python3.10-venv \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir \
    vllm>=0.4.0 \
    fastapi \
    uvicorn[standard] \
    httpx \
    pydantic

WORKDIR /app

COPY vllm_client.py /app/
COPY api_server.py /app/
COPY benchmark.py /app/

EXPOSE 8000

CMD ["python3", "-m", "uvicorn", "api_server:app", "--host", "0.0.0.0", "--port", "8000"]
