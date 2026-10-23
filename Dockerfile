ARG BASE_IMAGE=runpod-ml:latest
FROM ${BASE_IMAGE}

COPY requirements.txt requirements-live.txt /tmp/bazza/
RUN python -m pip install --no-cache-dir -r /tmp/bazza/requirements-live.txt
ENV PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
WORKDIR /workspace
