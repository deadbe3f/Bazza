ARG BASE_IMAGE=runpod-ml:latest
FROM ${BASE_IMAGE}

COPY requirements.txt /tmp/bazza/
RUN python -m pip install --no-cache-dir -r /tmp/bazza/requirements.txt
ENV PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
WORKDIR /workspace
