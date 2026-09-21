# Immutable multi-platform index digest for the Python 3.12.14 slim runtime.
FROM python:3.12.14-slim@sha256:09f7da3bc104798d0afb40bc08d23ab2da20a76130cec1f2ef170848f5d85217

WORKDIR /app

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

COPY pyproject.toml README.md requirements-runtime.lock.txt requirements-torch-cpu.lock.txt ./

RUN pip install --no-cache-dir --timeout 1000 --retries 10 --no-deps \
        --require-hashes --index-url https://download.pytorch.org/whl/cpu \
        -r requirements-torch-cpu.lock.txt \
    && pip install --no-cache-dir --timeout 1000 --retries 10 \
        --require-hashes -r requirements-runtime.lock.txt \
    && python -c "import torch; assert torch.__version__ == '2.13.0+cpu'; assert torch.version.cuda is None"
ENV PYTHONPATH=/app
COPY app/ ./app/
RUN python -m pip check

# Runtime-only configuration changes must not invalidate the dependency layer.
COPY prompts/ ./prompts/
COPY config/ ./config/

RUN mkdir -p /app/data /app/models /app/logs

RUN groupadd --gid 10001 appgroup \
    && useradd --create-home --uid 10001 --gid 10001 --no-log-init appuser \
    && chown -R 10001:10001 /app
USER 10001:10001

CMD ["celery", "-A", "app.workers.celery_app:celery_app", "worker", "--loglevel=info", "--pool=solo", "--concurrency=1"]
