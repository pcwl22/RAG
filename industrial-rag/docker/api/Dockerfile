FROM pytorch/pytorch:2.4.0-cuda12.1-cudnn9-runtime

WORKDIR /app

ARG APT_MIRROR=""

ENV PIP_DEFAULT_TIMEOUT=1000 \
    PIP_RETRIES=10 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

RUN set -eu; \
    if [ -n "${APT_MIRROR}" ]; then \
        sed -i \
            -e "s|http://archive.ubuntu.com/ubuntu|${APT_MIRROR}|g" \
            -e "s|http://security.ubuntu.com/ubuntu|${APT_MIRROR}|g" \
            /etc/apt/sources.list; \
    fi; \
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
        --constraint constraints-docker.txt . gunicorn

# Runtime-only configuration changes must not invalidate the dependency layer.
COPY prompts/ ./prompts/
COPY config/ ./config/

RUN mkdir -p /app/data /app/models /app/logs

RUN useradd -m -u 1000 appuser && chown -R appuser:appuser /app
USER appuser

EXPOSE 8000

CMD ["gunicorn", "app.main:app", \
     "--workers", "1", \
     "--worker-class", "uvicorn.workers.UvicornWorker", \
     "--bind", "0.0.0.0:8000", \
     "--timeout", "120", \
     "--keep-alive", "5", \
     "--access-logfile", "-", \
     "--error-logfile", "-"]
