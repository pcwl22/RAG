#!/bin/bash

echo "========================================"
echo "RAG 知识库系统 - 启动脚本"
echo "========================================"
echo

# 检查虚拟环境
if [[ -z "$VIRTUAL_ENV" ]]; then
    echo "[错误] 请先激活虚拟环境！"
    echo
    echo "激活命令:"
    echo "  source venv/bin/activate"
    echo
    exit 1
fi

echo "[1/4] 检查 Docker 服务..."
if ! docker ps > /dev/null 2>&1; then
    echo "[错误] Docker 未运行，请先启动 Docker"
    exit 1
fi
echo "✓ Docker 运行正常"

echo
echo "[2/4] 启动 PostgreSQL 和 Redis..."
if ! docker-compose up -d; then
    echo "[错误] Docker Compose 启动失败"
    exit 1
fi
echo "✓ 数据库服务已启动"

echo
echo "[3/4] 等待数据库就绪..."
sleep 5
echo "✓ 数据库就绪"

echo
echo "[4/4] 启动 FastAPI 服务..."
echo
echo "========================================"
echo "API 服务运行中..."
echo "========================================"
echo "API 文档: http://localhost:8000/docs"
echo "健康检查: http://localhost:8000/health"
echo
echo "按 Ctrl+C 停止服务"
echo "========================================"
echo

python -m uvicorn src.api.main:app --reload --host 0.0.0.0 --port 8000
