# Immutable multi-platform index digest for the official PyTorch 2.13/CUDA 12.6 runtime.
# GHCR remains reachable in environments where Docker Hub authentication is blocked.
FROM ghcr.io/pytorch/pytorch:2.14.0-cuda12.6-cudnn9-runtime@sha256:ac7effd0cd05c10c7ae8458b7c864fea6eaccfed31123db648bb080085635901

WORKDIR /app

ARG APT_MIRROR=""
ARG SECURITY_REFRESH=2026-09-21

ENV PIP_DEFAULT_TIMEOUT=1000 \
    PIP_RETRIES=10 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

COPY pyproject.toml README.md requirements-runtime.lock.txt ./

# The application uses wheel-based PyMuPDF and native Python parsers. OCR and
# office-conversion executables are not invoked by the runtime, so shipping the
# former 750+ MB toolchain only expanded the attack surface. python3-venv is a
# build-only helper and is removed in the same layer that creates the venv.
# hadolint ignore=DL3008
RUN set -eu; \
    test -n "${SECURITY_REFRESH}"; \
    if [ -n "${APT_MIRROR}" ]; then \
        sed -i \
            -e "s|http://archive.ubuntu.com/ubuntu|${APT_MIRROR}|g" \
            -e "s|http://security.ubuntu.com/ubuntu|${APT_MIRROR}|g" \
            /etc/apt/sources.list; \
    fi; \
    apt-get -o Acquire::Retries=5 -o Acquire::http::Timeout=120 update; \
    apt-get -o Acquire::Retries=5 -o Acquire::http::Timeout=120 \
        install -y --no-install-recommends --only-upgrade linux-libc-dev; \
    apt-get -o Acquire::Retries=5 -o Acquire::http::Timeout=120 \
        install -y --no-install-recommends python3-venv; \
    python -m pip uninstall --break-system-packages --yes \
        pillow spin torchaudio torchvision; \
    python -m pip check; \
    python -m venv --system-site-packages /opt/venv; \
    /opt/venv/bin/python -c "import torch; assert torch.__version__.split('+', 1)[0] == '2.13.0'; assert str(torch.version.cuda).startswith('12.6')"; \
    /opt/venv/bin/pip install --no-cache-dir --timeout 1000 --retries 10 \
        --require-hashes -r requirements-runtime.lock.txt; \
    /opt/venv/bin/python -m pip check; \
    apt-get purge -y --auto-remove python3-venv; \
    rm -rf /var/lib/apt/lists/*
ENV VIRTUAL_ENV=/opt/venv \
    PATH="/opt/venv/bin:${PATH}"
ENV PYTHONPATH=/app
COPY app/ ./app/

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
