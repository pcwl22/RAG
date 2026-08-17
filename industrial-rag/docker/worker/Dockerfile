FROM python:3.11-slim

WORKDIR /app

ARG PYTORCH_INDEX_URL=https://download.pytorch.org/whl/cpu

ENV PIP_DEFAULT_TIMEOUT=1000 \
    PIP_RETRIES=10 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

RUN set -eu; \
    apt-get -o Acquire::Retries=5 -o Acquire::http::Timeout=120 update; \
    attempt=1; \
    until apt-get -o Acquire::Retries=5 -o Acquire::http::Timeout=120 \
        install -y --no-install-recommends \
        build-essential git libpoppler-cpp-dev tesseract-ocr \
        tesseract-ocr-chi-sim tesseract-ocr-chi-tra libreoffice; do \
        if [ "${attempt}" -ge 5 ]; then exit 1; fi; \
        attempt=$((attempt + 1)); \
        sleep 5; \
        apt-get -o Acquire::Retries=5 -o Acquire::http::Timeout=120 update; \
    done; \
    rm -rf /var/lib/apt/lists/*

COPY pyproject.toml README.md constraints-docker.txt ./
COPY app/ ./app/

RUN pip install --no-cache-dir --timeout 1000 --retries 10 \
        torch==2.4.0 --index-url ${PYTORCH_INDEX_URL}
RUN pip install --no-cache-dir --timeout 1000 --retries 10 \
        --constraint constraints-docker.txt .

# Runtime-only configuration changes must not invalidate the dependency layer.
COPY prompts/ ./prompts/
COPY config/ ./config/

RUN mkdir -p /app/data /app/models /app/logs

RUN useradd -m -u 1000 appuser && chown -R appuser:appuser /app
USER appuser

CMD ["celery", "-A", "app.workers.celery_app:celery_app", "worker", "--loglevel=info", "--pool=solo", "--concurrency=1"]
