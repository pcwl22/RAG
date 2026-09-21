# Immutable multi-platform index digest for the official PyTorch 2.13/CUDA 12.6 runtime.
# GHCR remains reachable in environments where Docker Hub authentication is blocked.
FROM ghcr.io/pytorch/pytorch:2.13.0-cuda12.6-cudnn9-runtime@sha256:b2e6463673129d979402f11083769c9eaad9211487b6260f52525f55a0a82dac

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
        build-essential git libpoppler-cpp-dev python3-venv tesseract-ocr \
        tesseract-ocr-chi-sim tesseract-ocr-chi-tra libreoffice; do \
        if [ "${attempt}" -ge 5 ]; then exit 1; fi; \
        attempt=$((attempt + 1)); \
        sleep 5; \
        apt-get -o Acquire::Retries=5 -o Acquire::http::Timeout=120 update; \
    done; \
    rm -rf /var/lib/apt/lists/*

# GHCR's official runtime follows PEP 668. Keep the base Torch installation
# visible while installing application dependencies into an isolated prefix.
RUN if python -m pip show spin >/dev/null 2>&1; then \
        python -m pip uninstall --break-system-packages --yes spin; \
    fi
RUN python -m venv --system-site-packages /opt/venv
ENV VIRTUAL_ENV=/opt/venv \
    PATH="/opt/venv/bin:${PATH}"

COPY pyproject.toml README.md requirements-runtime.lock.txt ./

RUN python -c "import torch; assert torch.__version__.split('+', 1)[0] == '2.13.0'; assert str(torch.version.cuda).startswith('12.6')" \
    && pip install --no-cache-dir --timeout 1000 --retries 10 \
        --require-hashes -r requirements-runtime.lock.txt
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

EXPOSE 8000

CMD ["gunicorn", "app.main:app", \
     "--workers", "1", \
     "--worker-class", "uvicorn.workers.UvicornWorker", \
     "--bind", "0.0.0.0:8000", \
     "--timeout", "120", \
     "--keep-alive", "5", \
     "--access-logfile", "-", \
     "--error-logfile", "-"]
