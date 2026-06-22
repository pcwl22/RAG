FROM python:3.11-slim

WORKDIR /app

# 安装系统依赖
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    libpoppler-cpp-dev \
    tesseract-ocr \
    tesseract-ocr-chi-sim \
    tesseract-ocr-chi-tra \
    libreoffice \
    && rm -rf /var/lib/apt/lists/*

# 安装Python依赖
COPY pyproject.toml .
RUN pip install --no-cache-dir -e .

# 复制源代码
COPY src/ ./src/
COPY configs/ ./configs/

# 创建数据目录
RUN mkdir -p /app/data /app/models

# 非root用户运行
RUN useradd -m -u 1000 appuser && chown -R appuser:appuser /app
USER appuser

CMD ["celery", "-A", "src.workers.celery_app", "worker", "--loglevel=info", "--concurrency=4"]
