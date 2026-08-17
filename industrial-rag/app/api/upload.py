"""
文档摄入API路由

任务队列根据配置 `queue.provider` 选择：
- "memory"（笔记本默认）：使用 FastAPI BackgroundTasks + 内存状态表。
- "rabbitmq" / "celery"：延迟导入 Celery 任务。
"""
import asyncio
import json
import os
import threading
import uuid
from collections import OrderedDict
from collections.abc import Coroutine
from pathlib import Path
from typing import Annotated, Any

from fastapi import APIRouter, BackgroundTasks, File, Form, HTTPException, Query, UploadFile
from pydantic import BaseModel

from app.auth import current_tenant_id, normalize_tenant_id
from app.parser.document_parser import SUPPORTED_EXTENSIONS
from app.utils.config import get_settings, resolve_queue_provider
from app.utils.logger import get_logger
from app.utils.upload_files import quarantine_failed_upload
from app.utils.upload_validation import validate_uploaded_document
from app.vectorstore import storage_adapter

logger = get_logger(__name__)
router = APIRouter()
config = get_settings()


# ---------------------------------------------------------------------------
# 内存任务状态表（provider == "memory" 时使用）
# ---------------------------------------------------------------------------
# 结构: {task_id: {"status": str, "progress": int, "total_chunks": int, "error": str | None}}
# OrderedDict because eviction must be least-recently-used. A plain dict keeps
# its original insertion order when a key is reassigned, so a task that is still
# being updated would be evicted ahead of untouched older entries.
_task_registry: OrderedDict[tuple[str, str], dict] = OrderedDict()
_MAX_IN_MEMORY_TASKS = 1000
_task_registry_lock = threading.RLock()
INGESTION_ERROR_CODE = "DOCUMENT_INGESTION_FAILED"
INGESTION_ERROR_MESSAGE = "Document ingestion failed"


def _public_task_state(state: dict) -> dict:
    """Remove internal exception details from a task state returned by the API."""
    public_state = dict(state)
    if public_state.get("status") == "failed":
        public_state["error"] = INGESTION_ERROR_MESSAGE
        public_state["error_code"] = public_state.get("error_code") or INGESTION_ERROR_CODE
    return public_state


def _remember_task(task_id: str, state: dict, tenant_id: str | None = None) -> None:
    """Keep the local fallback bounded; distributed deployments should use Celery."""
    key = (normalize_tenant_id(tenant_id or current_tenant_id()), task_id)
    with _task_registry_lock:
        _task_registry[key] = state
        # Mark this task as most recently used so an in-flight task is not
        # evicted while stale completed entries survive.
        _task_registry.move_to_end(key)
        while len(_task_registry) > _MAX_IN_MEMORY_TASKS:
            _task_registry.popitem(last=False)


async def _persist_task_state(
    task_id: str,
    state: dict,
    tenant_id: str | None = None,
) -> None:
    resolved_tenant = normalize_tenant_id(tenant_id or current_tenant_id())
    _remember_task(task_id, state, resolved_tenant)
    try:
        from app.utils.cache import set_task_state

        await set_task_state(task_id, state, tenant_id=resolved_tenant)
    except Exception:
        logger.debug("Task state persistence unavailable", exc_info=True)


async def _load_task_state(task_id: str, tenant_id: str | None = None) -> dict | None:
    resolved_tenant = normalize_tenant_id(tenant_id or current_tenant_id())
    key = (resolved_tenant, task_id)
    with _task_registry_lock:
        state = _task_registry.get(key)
        if state is not None:
            # Polling a task counts as a use: a client watching an in-flight task
            # must not lose it to eviction while untouched entries survive.
            _task_registry.move_to_end(key)
    if state is not None:
        return state
    try:
        from app.utils.cache import get_task_state

        state = await get_task_state(task_id, tenant_id=resolved_tenant)
        if state:
            _remember_task(task_id, state, resolved_tenant)
        return state
    except Exception:
        return None


def _get_upload_dir() -> Path:
    """获取上传目录（跨平台，可配置，默认相对路径）。"""
    upload_dir = (
        config.get("document_processing", {}).get("upload_dir") or "./data/uploads"
    )
    path = Path(upload_dir)
    path.mkdir(parents=True, exist_ok=True)
    return path


def _quarantine_failed_upload(file_path: str, task_id: str) -> None:
    """Retain failed memory-mode inputs for an operator or retry job."""
    try:
        processing = config.get("document_processing", {})
        destination = quarantine_failed_upload(
            file_path,
            retention_seconds=int(processing.get("failed_upload_retention_seconds", 604800)),
            max_files=int(processing.get("failed_upload_max_files", 100)),
        )
        if destination is not None:
            logger.warning("Failed upload retained for retry: %s", destination)
    except OSError:
        logger.warning("Could not retain failed upload: %s", file_path, exc_info=True)


def _queue_provider() -> str:
    return resolve_queue_provider(config)


def _run_on_app_loop(
    app_loop: asyncio.AbstractEventLoop | None,
    coroutine: Coroutine[Any, Any, Any],
    operation: str,
) -> bool:
    """Run Redis-bound work on the FastAPI loop, never on the worker loop."""
    if app_loop is None or app_loop.is_closed():
        coroutine.close()
        logger.warning("Skipped %s because the application loop is unavailable", operation)
        return False
    try:
        future = asyncio.run_coroutine_threadsafe(coroutine, app_loop)
        future.result(timeout=10)
        return True
    except Exception:
        logger.warning("Failed to %s on the application loop", operation, exc_info=True)
        return False


def _publish_task_state(
    task_id: str,
    state: dict,
    app_loop: asyncio.AbstractEventLoop | None,
    tenant_id: str,
) -> None:
    """Publish immediately in memory and persist through the owning async loop."""
    _remember_task(task_id, state, tenant_id)
    if app_loop is not None:
        _run_on_app_loop(
            app_loop,
            _persist_task_state(task_id, state, tenant_id),
            f"persist task state for {task_id}",
        )


def _process_document_memory(
    task_id: str,
    file_path: str,
    filename: str,
    partition: str,
    metadata: dict,
    tenant_id: str | None = None,
    app_loop: asyncio.AbstractEventLoop | None = None,
) -> None:
    """内存模式下的后台文档处理。"""
    import asyncio

    tenant_id = normalize_tenant_id(tenant_id or current_tenant_id())
    completed = False
    state = {
        "status": "processing",
        "progress": 0,
        "total_chunks": 0,
        "error": None,
        "error_code": None,
    }
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    _publish_task_state(task_id, state, app_loop, tenant_id)
    try:
        # 调用处理流水线
        from app.service.ingest_service import process_document

        # 在新的事件循环中运行异步任务
        result = loop.run_until_complete(
            process_document(
                file_path,
                filename,
                partition,
                metadata,
                invalidate_cache=False,
                tenant_id=tenant_id,
            )
        )
        if result["status"] == "completed":
            completed = True
            if app_loop is not None:
                from app.utils.cache import invalidate_semantic_cache

                _run_on_app_loop(
                    app_loop,
                    invalidate_semantic_cache(tenant_id),
                    "invalidate semantic cache after ingestion",
                )
            _publish_task_state(task_id, {
                "status": "completed",
                "progress": 100,
                "total_chunks": result["total_chunks"],
                "error": None,
            }, app_loop, tenant_id)
            logger.info(f"Document processed: {task_id}, {result['total_chunks']} chunks")
        else:
            logger.error(
                "Document processing returned failure for task %s: %s",
                task_id,
                result.get("error", "unknown error"),
            )
            _publish_task_state(task_id, {
                "status": "failed",
                "progress": 0,
                "total_chunks": 0,
                "error": INGESTION_ERROR_MESSAGE,
                "error_code": INGESTION_ERROR_CODE,
            }, app_loop, tenant_id)
    except Exception as e:
        logger.error(f"Document processing failed for task {task_id}: {e}")
        _publish_task_state(task_id, {
            "status": "failed",
            "progress": 0,
            "total_chunks": 0,
            "error": INGESTION_ERROR_MESSAGE,
            "error_code": INGESTION_ERROR_CODE,
        }, app_loop, tenant_id)
    finally:
        if completed:
            try:
                os.unlink(file_path)
            except FileNotFoundError:
                pass
        else:
            _quarantine_failed_upload(file_path, task_id)
        loop.close()


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
    error_code: str | None = None


@router.post("/documents/ingest", response_model=DocumentIngestResponse)
async def ingest_document(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    partition: str = Form(default="text", max_length=100),
    metadata: str = Form(default="{}", max_length=65536),
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
    safe_filename = Path(file.filename or "unknown").name
    file_ext = Path(safe_filename).suffix.removeprefix(".").lower()
    configured_formats = {
        str(item).lower()
        for item in config["document_processing"].get("supported_formats", [])
    }
    supported_formats = sorted(configured_formats & SUPPORTED_EXTENSIONS)

    if file_ext not in supported_formats:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file format. Supported: {', '.join(supported_formats)}",
        )

    # Stream to disk so a large upload does not consume process memory.
    max_size = config["document_processing"]["max_file_size"]
    task_id = str(uuid.uuid4())
    upload_dir = _get_upload_dir()
    file_path = upload_dir / f"{task_id}_{safe_filename}"
    bytes_written = 0
    try:
        with open(file_path, "wb") as output:
            while chunk := await file.read(1024 * 1024):
                bytes_written += len(chunk)
                if bytes_written > max_size:
                    raise HTTPException(
                        status_code=400,
                        detail=f"File too large. Max size: {max_size / 1024 / 1024:.0f}MB",
                    )
                output.write(chunk)
    except Exception:
        try:
            os.unlink(file_path)
        except FileNotFoundError:
            pass
        raise

    try:
        validate_uploaded_document(file_path, file_ext, config["document_processing"])
    except ValueError as exc:
        try:
            os.unlink(file_path)
        except FileNotFoundError:
            pass
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    # 解析元数据
    try:
        metadata_dict = json.loads(metadata)
    except json.JSONDecodeError as exc:
        try:
            os.unlink(file_path)
        except FileNotFoundError:
            pass
        raise HTTPException(status_code=400, detail="metadata must be valid JSON") from exc
    if not isinstance(metadata_dict, dict):
        try:
            os.unlink(file_path)
        except FileNotFoundError:
            pass
        raise HTTPException(status_code=400, detail="metadata must be a JSON object")

    # 根据队列提供方分发任务
    tenant_id = current_tenant_id()
    provider = _queue_provider()
    if provider in ("rabbitmq", "celery"):
        # 延迟导入，避免 memory 模式下强依赖 Celery
        try:
            from app.workers.tasks import process_document_task

            task = process_document_task.delay(
                file_path=str(file_path),
                filename=safe_filename,
                partition=partition,
                metadata=metadata_dict,
                tenant_id=tenant_id,
            )
            task_id = task.id
            await _persist_task_state(
                task_id,
                {
                    "status": "pending",
                    "progress": 0,
                    "total_chunks": 0,
                    "error": None,
                    "error_code": None,
                },
                tenant_id,
            )
        except Exception as exc:
            try:
                os.unlink(file_path)
            except FileNotFoundError:
                pass
            raise HTTPException(status_code=503, detail="Document queue unavailable") from exc
    else:
        # 内存模式：FastAPI 后台任务
        initial_state = {
            "status": "pending",
            "progress": 0,
            "total_chunks": 0,
            "error": None,
            "error_code": None,
        }
        await _persist_task_state(task_id, initial_state, tenant_id)
        app_loop = asyncio.get_running_loop()
        background_tasks.add_task(
            _process_document_memory,
            task_id=task_id,
            file_path=str(file_path),
            filename=safe_filename,
            partition=partition,
            metadata=metadata_dict,
            tenant_id=tenant_id,
            app_loop=app_loop,
        )

    logger.info(f"Document ingest task submitted: {task_id}", extra={"task_id": task_id})

    return DocumentIngestResponse(
        task_id=task_id,
        filename=safe_filename,
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
    tenant_id = current_tenant_id()
    provider = _queue_provider()

    if provider in ("rabbitmq", "celery"):
        from app.workers.celery_app import celery_app

        if celery_app is None:
            raise HTTPException(status_code=503, detail="Document queue unavailable")
        result = celery_app.AsyncResult(task_id)
        known_state = await _load_task_state(task_id, tenant_id)
        if known_state is None:
            raise HTTPException(status_code=404, detail=f"Task {task_id} not found")

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
            "error_code": None,
        }
        if result.state == "PENDING" and known_state:
            response_data.update(known_state)
        if result.state == "SUCCESS":
            info = result.info or {}
            if isinstance(info, dict) and info.get("status") == "failed":
                response_data["status"] = "failed"
                response_data["error"] = INGESTION_ERROR_MESSAGE
                response_data["error_code"] = INGESTION_ERROR_CODE
            else:
                response_data["progress"] = 100
                response_data["total_chunks"] = info.get("total_chunks", 0) if isinstance(info, dict) else 0
        elif result.state == "FAILURE":
            logger.error("Celery document task %s failed: %s", task_id, result.info)
            response_data["error"] = INGESTION_ERROR_MESSAGE
            response_data["error_code"] = INGESTION_ERROR_CODE

        return DocumentStatus(**_public_task_state(response_data))

    # 内存模式
    state = await _load_task_state(task_id, tenant_id)
    if state is None:
        raise HTTPException(status_code=404, detail=f"Task {task_id} not found")

    return DocumentStatus(task_id=task_id, **_public_task_state(state))


@router.delete("/documents/{document_id}")
async def delete_document(document_id: str) -> dict:
    """
    删除文档

    Args:
        document_id: 文档ID

    Returns:
        删除结果
    """
    from app.vectorstore.storage_adapter import delete_document

    try:
        deleted = await delete_document(document_id)
        if not deleted:
            raise HTTPException(status_code=404, detail="Document not found")
        return {"status": "success", "message": f"Document {document_id} deleted"}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to delete document {document_id}: {e}")
        raise HTTPException(status_code=500, detail="Document deletion failed") from e


@router.get("/documents")
async def list_documents(
    skip: Annotated[int, Query(ge=0)] = 0,
    limit: Annotated[int, Query(ge=1, le=500)] = 100,
) -> dict:
    """
    列出所有文档

    Args:
        skip: 跳过数量
        limit: 返回数量

    Returns:
        文档列表
    """
    try:
        documents, total = await asyncio.gather(
            storage_adapter.list_documents(skip=skip, limit=limit),
            storage_adapter.count_documents(),
        )
        return {"total": total, "documents": documents}
    except Exception as e:
        logger.error(f"Failed to list documents: {e}")
        raise HTTPException(status_code=500, detail="Document listing failed") from e


@router.get("/documents/{document_id}/chunks")
async def list_document_chunks(
    document_id: str,
    skip: Annotated[int, Query(ge=0)] = 0,
    limit: Annotated[int, Query(ge=1, le=5000)] = 2000,
) -> dict:
    """
    列出某个文档真实入库切片。

    Args:
        document_id: 文档ID
        skip: 跳过数量
        limit: 返回数量

    Returns:
        切片列表，content 为数据库实际存储并参与检索的文本
    """
    from app.vectorstore.storage_adapter import list_document_chunks as list_chunks

    try:
        bounded_limit = max(1, min(limit, 5000))
        data = await list_chunks(document_id=document_id, skip=skip, limit=bounded_limit)
        return {
            "document_id": document_id,
            "total": data.get("total", 0),
            "chunks": data.get("chunks", []),
        }
    except Exception as e:
        logger.error(f"Failed to list document chunks for {document_id}: {e}")
        raise HTTPException(status_code=500, detail="Document chunk listing failed") from e
