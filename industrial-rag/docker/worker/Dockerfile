# Immutable multi-platform index digest for the Python 3.12.14 slim runtime.
FROM python:3.12.14-slim@sha256:09f7da3bc104798d0afb40bc08d23ab2da20a76130cec1f2ef170848f5d85217

WORKDIR /app

ENV PIP_DEFAULT_TIMEOUT=1000 \
    PIP_RETRIES=10 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

COPY pyproject.toml README.md requirements-runtime.lock.txt requirements-torch-cpu.lock.txt ./

RUN python -m venv /opt/venv \
    && /opt/venv/bin/pip install --no-cache-dir --timeout 1000 --retries 10 --no-deps \
        --require-hashes --index-url https://download.pytorch.org/whl/cpu \
        -r requirements-torch-cpu.lock.txt \
    && /opt/venv/bin/pip install --no-cache-dir --timeout 1000 --retries 10 \
        --require-hashes -r requirements-runtime.lock.txt \
    && /opt/venv/bin/python -c "import torch; assert torch.__version__ == '2.13.0+cpu'; assert torch.version.cuda is None" \
    && /opt/venv/bin/python -m pip check
ENV VIRTUAL_ENV=/opt/venv \
    PATH="/opt/venv/bin:${PATH}" \
    PYTHONPATH=/app
COPY app/ ./app/

# Runtime-only configuration changes must not invalidate the dependency layer.
COPY prompts/ ./prompts/
COPY config/ ./config/

RUN mkdir -p /app/data /app/models /app/logs

RUN groupadd --gid 10001 appgroup \
    && useradd --create-home --uid 10001 --gid 10001 --no-log-init appuser \
    && chown -R 10001:10001 /app
USER 10001:10001

CMD ["celery", "-A", "app.workers.celery_app:celery_app", "worker", "--loglevel=info", "--pool=solo", "--concurrency=1"]
