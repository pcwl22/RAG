"""
文档摄入API路由

任务队列根据配置 `queue.provider` 选择：
- "memory"（笔记本默认）：使用 FastAPI BackgroundTasks + 内存状态表。
- "rabbitmq" / "celery"：延迟导入 Celery 任务。
"""
import json
import uuid
from pathlib import Path

from fastapi import APIRouter, BackgroundTasks, File, Form, HTTPException, UploadFile
from pydantic import BaseModel

from src.core.config import get_settings
from src.utils.logger import get_logger

logger = get_logger(__name__)
router = APIRouter()
config = get_settings()


# ---------------------------------------------------------------------------
# 内存任务状态表（provider == "memory" 时使用）
# ---------------------------------------------------------------------------
# 结构: {task_id: {"status": str, "progress": int, "total_chunks": int, "error": str | None}}
_task_registry: dict[str, dict] = {}


def _get_upload_dir() -> Path:
    """获取上传目录（跨平台，可配置，默认相对路径）。"""
    upload_dir = (
        config.get("document_processing", {}).get("upload_dir") or "./data/uploads"
    )
    path = Path(upload_dir)
    path.mkdir(parents=True, exist_ok=True)
    return path


def _queue_provider() -> str:
    return config.get("queue", {}).get("provider", "memory").lower()


def _process_document_memory(
    task_id: str, file_path: str, filename: str, partition: str, metadata: dict
) -> None:
    """内存模式下的后台文档处理。"""
    import asyncio

    _task_registry[task_id] = {
        "status": "processing",
        "progress": 0,
        "total_chunks": 0,
        "error": None,
    }
    try:
        # 调用处理流水线
        from src.ingest.pipeline import process_document

        # 在新的事件循环中运行异步任务
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        result = loop.run_until_complete(
            process_document(file_path, filename, partition, metadata)
        )
        loop.close()

        if result["status"] == "completed":
            _task_registry[task_id] = {
                "status": "completed",
                "progress": 100,
                "total_chunks": result["total_chunks"],
                "error": None,
            }
            logger.info(f"Document processed: {task_id}, {result['total_chunks']} chunks")
        else:
            _task_registry[task_id] = {
                "status": "failed",
                "progress": 0,
                "total_chunks": 0,
                "error": result.get("error", "Unknown error"),
            }
    except Exception as e:
        logger.error(f"Document processing failed for task {task_id}: {e}")
        _task_registry[task_id] = {
            "status": "failed",
            "progress": 0,
            "total_chunks": 0,
            "error": str(e),
        }


class DocumentIngestResponse(BaseModel):
    """文档摄入响应"""

    task_id: str
    filename: str
    status: str
    message: str


class DocumentStatus(BaseModel):
    """文档状态"""

    task_id: str
    status: str
    progress: int
    total_chunks: int
    error: str | None = None


@router.post("/documents/ingest", response_model=DocumentIngestResponse)
async def ingest_document(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    partition: str = Form(default="text"),
    metadata: str = Form(default="{}"),
) -> DocumentIngestResponse:
    """
    摄入文档

    Args:
        background_tasks: FastAPI 后台任务（memory 模式使用）
        file: 上传的文件
        partition: 分区名称 (text/table/image/mixed)
        metadata: 额外元数据 (JSON字符串)

    Returns:
        摄入任务信息
    """
    # 验证文件类型
    file_ext = file.filename.split(".")[-1].lower() if file.filename else ""
    supported_formats = config["document_processing"]["supported_formats"]

    if file_ext not in supported_formats:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file format. Supported: {', '.join(supported_formats)}",
        )

    # 验证文件大小
    max_size = config["document_processing"]["max_file_size"]
    content = await file.read()
    if len(content) > max_size:
        raise HTTPException(
            status_code=400,
            detail=f"File too large. Max size: {max_size / 1024 / 1024:.0f}MB",
        )

    # 生成任务ID
    task_id = str(uuid.uuid4())

    # 保存文件（跨平台路径）
    upload_dir = _get_upload_dir()
    file_path = upload_dir / f"{task_id}_{file.filename}"
    with open(file_path, "wb") as f:
        f.write(content)

    # 解析元数据
    try:
        metadata_dict = json.loads(metadata)
    except json.JSONDecodeError:
        metadata_dict = {}

    # 根据队列提供方分发任务
    provider = _queue_provider()
    if provider in ("rabbitmq", "celery"):
        # 延迟导入，避免 memory 模式下强依赖 Celery
        from src.workers.tasks import process_document_task

        task = process_document_task.delay(
            file_path=str(file_path),
            filename=file.filename or "unknown",
            partition=partition,
            metadata=metadata_dict,
        )
        task_id = task.id
    else:
        # 内存模式：FastAPI 后台任务
        _task_registry[task_id] = {
            "status": "pending",
            "progress": 0,
            "total_chunks": 0,
            "error": None,
        }
        background_tasks.add_task(
            _process_document_memory,
            task_id=task_id,
            file_path=str(file_path),
            filename=file.filename or "unknown",
            partition=partition,
            metadata=metadata_dict,
        )

    logger.info(f"Document ingest task submitted: {task_id}", extra={"task_id": task_id})

    return DocumentIngestResponse(
        task_id=task_id,
        filename=file.filename or "unknown",
        status="processing",
        message="Document is being processed",
    )


@router.get("/documents/status/{task_id}", response_model=DocumentStatus)
async def get_document_status(task_id: str) -> DocumentStatus:
    """
    获取文档处理状态

    Args:
        task_id: 任务ID

    Returns:
        任务状态
    """
    provider = _queue_provider()

    if provider in ("rabbitmq", "celery"):
        from celery.result import AsyncResult

        result = AsyncResult(task_id)

        status_map = {
            "PENDING": "pending",
            "STARTED": "processing",
            "SUCCESS": "completed",
            "FAILURE": "failed",
            "RETRY": "retrying",
        }
        status = status_map.get(result.state, "unknown")

        response_data = {
            "task_id": task_id,
            "status": status,
            "progress": 0,
            "total_chunks": 0,
            "error": None,
        }
        if result.state == "SUCCESS":
            info = result.info or {}
            response_data["progress"] = 100
            response_data["total_chunks"] = info.get("total_chunks", 0)
        elif result.state == "FAILURE":
            response_data["error"] = str(result.info)

        return DocumentStatus(**response_data)

    # 内存模式
    state = _task_registry.get(task_id)
    if state is None:
        raise HTTPException(status_code=404, detail=f"Task {task_id} not found")

    return DocumentStatus(task_id=task_id, **state)


@router.delete("/documents/{document_id}")
async def delete_document(document_id: str) -> dict:
    """
    删除文档

    Args:
        document_id: 文档ID

    Returns:
        删除结果
    """
    from src.storage.storage_adapter import delete_document

    try:
        await delete_document(document_id)
        return {"status": "success", "message": f"Document {document_id} deleted"}
    except Exception as e:
        logger.error(f"Failed to delete document {document_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/documents")
async def list_documents(skip: int = 0, limit: int = 100) -> dict:
    """
    列出所有文档

    Args:
        skip: 跳过数量
        limit: 返回数量

    Returns:
        文档列表
    """
    from src.storage.storage_adapter import list_documents as list_docs

    try:
        documents = await list_docs(skip=skip, limit=limit)
        return {"total": len(documents), "documents": documents}
    except Exception as e:
        logger.error(f"Failed to list documents: {e}")
        raise HTTPException(status_code=500, detail=str(e))
