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
import time
import uuid
from collections import OrderedDict
from collections.abc import Coroutine
from pathlib import Path
from typing import Annotated, Any

from fastapi import APIRouter, BackgroundTasks, File, Form, HTTPException, Query, UploadFile
from pydantic import BaseModel

from app.auth import current_tenant_id, normalize_tenant_id
from app.parser.document_parser import SUPPORTED_EXTENSIONS
from app.storage.object_store import (
    build_upload_manifest_key,
    build_upload_object_key,
    get_object_store,
    sanitize_upload_filename,
    upload_filename_extension,
    upload_filename_identity,
)
from app.utils.config import get_settings, resolve_object_storage_config, resolve_queue_provider
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
    *,
    require_durable: bool = False,
) -> bool:
    resolved_tenant = normalize_tenant_id(tenant_id or current_tenant_id())
    _remember_task(task_id, state, resolved_tenant)
    try:
        from app.utils.cache import set_task_state

        persisted = await set_task_state(task_id, state, tenant_id=resolved_tenant)
        if not persisted and require_durable:
            logger.warning(
                "Task state was retained only in process memory",
                extra={"task_id": task_id, "tenant_id": resolved_tenant},
            )
        return persisted
    except Exception:
        log = logger.warning if require_durable else logger.debug
        log("Task state persistence unavailable", exc_info=True)
        return False


async def _load_task_state(
    task_id: str,
    tenant_id: str | None = None,
    *,
    require_backend: bool = False,
) -> dict | None:
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

        state = await get_task_state(
            task_id,
            tenant_id=resolved_tenant,
            require_backend=require_backend,
        )
        if state:
            _remember_task(task_id, state, resolved_tenant)
        return state
    except Exception:
        if require_backend:
            raise
        return None


def _get_upload_dir() -> Path:
    """获取上传目录（跨平台，可配置，默认相对路径）。"""
    upload_dir = (
        config.get("document_processing", {}).get("upload_dir") or "./data/uploads"
    )
    path = Path(upload_dir)
    path.mkdir(parents=True, exist_ok=True)
    return path


def _unlink_if_exists(file_path: str | Path) -> None:
    try:
        os.unlink(file_path)
    except FileNotFoundError:
        pass


async def _remove_upload_file(file_path: str | Path) -> None:
    """Keep potentially slow filesystem cleanup off the request event loop."""
    await asyncio.to_thread(_unlink_if_exists, file_path)


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
    return str(resolve_queue_provider(config))


def _object_storage_for_queue(provider: str) -> Any | None:
    """Return the configured store; Celery never falls back to local paths."""
    if provider != "celery":
        return None
    settings = resolve_object_storage_config(config)
    if not settings["enabled"]:
        raise HTTPException(
            status_code=503,
            detail="Object storage is required for Celery document ingestion",
        )
    return get_object_store(config)


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

    resolved_tenant = str(normalize_tenant_id(tenant_id or current_tenant_id()))
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
    _publish_task_state(task_id, state, app_loop, resolved_tenant)
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
                tenant_id=resolved_tenant,
            )
        )
        if result["status"] == "completed":
            completed = True
            if app_loop is not None:
                from app.utils.cache import invalidate_semantic_cache

                _run_on_app_loop(
                    app_loop,
                    invalidate_semantic_cache(resolved_tenant),
                    "invalidate semantic cache after ingestion",
                )
            _publish_task_state(task_id, {
                "status": "completed",
                "progress": 100,
                "total_chunks": result["total_chunks"],
                "document_id": result.get("document_id"),
                "error": None,
            }, app_loop, resolved_tenant)
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
            }, app_loop, resolved_tenant)
    except Exception as e:
        logger.error(f"Document processing failed for task {task_id}: {e}")
        _publish_task_state(task_id, {
            "status": "failed",
            "progress": 0,
            "total_chunks": 0,
            "error": INGESTION_ERROR_MESSAGE,
            "error_code": INGESTION_ERROR_CODE,
        }, app_loop, resolved_tenant)
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
    document_id: str | None = None
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
    display_filename = sanitize_upload_filename(file.filename)
    file_ext = upload_filename_extension(file.filename)
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
    upload_dir = await asyncio.to_thread(_get_upload_dir)
    # The client filename is display metadata only.  The local storage identity
    # is generated by the server so Unicode normalization and truncation can
    # never make two uploads overwrite one another.
    file_path = upload_dir / f"{task_id}.{file_ext}"
    bytes_written = 0
    try:
        output = await asyncio.to_thread(file_path.open, "wb")
        try:
            while chunk := await file.read(1024 * 1024):
                bytes_written += len(chunk)
                if bytes_written > max_size:
                    raise HTTPException(
                        status_code=400,
                        detail=f"File too large. Max size: {max_size / 1024 / 1024:.0f}MB",
                    )
                # Local and network-backed upload directories may block. Keep
                # the request event loop available while the chunk is flushed.
                await asyncio.to_thread(output.write, chunk)
        finally:
            await asyncio.to_thread(output.close)
    except Exception:
        await _remove_upload_file(file_path)
        raise

    try:
        await asyncio.to_thread(
            validate_uploaded_document,
            file_path,
            file_ext,
            config["document_processing"],
        )
    except ValueError as exc:
        await _remove_upload_file(file_path)
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    # 解析元数据
    try:
        metadata_dict = json.loads(metadata)
    except json.JSONDecodeError as exc:
        await _remove_upload_file(file_path)
        raise HTTPException(status_code=400, detail="metadata must be valid JSON") from exc
    if not isinstance(metadata_dict, dict):
        await _remove_upload_file(file_path)
        raise HTTPException(status_code=400, detail="metadata must be a JSON object")
    # When no domain source id was supplied, derive replacement identity from
    # the complete normalized basename before the bounded display name is
    # truncated.  Explicit source ids remain supported for managed connectors.
    metadata_dict.setdefault(
        "source_id",
        f"upload-name-sha256:{upload_filename_identity(file.filename, partition)}",
    )

    # 根据队列提供方分发任务
    tenant_id = current_tenant_id()
    provider = _queue_provider()
    if provider == "celery":
        # 延迟导入，避免 memory 模式下强依赖 Celery
        object_key = build_upload_object_key(tenant_id, task_id, display_filename)
        manifest_key = build_upload_manifest_key(object_key)
        object_store = None
        dispatch_attempted = False
        uploaded_object_keys: list[str] = []
        manifest_payload = {
            "version": 1,
            "task_id": task_id,
            "object_key": object_key,
            "tenant_id": tenant_id,
            "filename": display_filename,
            "partition": partition,
            "metadata": metadata_dict,
            "created_at_unix": int(time.time()),
            "dispatch_attempts": 0,
            "last_dispatched_at_unix": 0,
        }
        try:
            object_store = _object_storage_for_queue(provider)
            if object_store is None:
                raise HTTPException(
                    status_code=503,
                    detail="Object storage is required for Celery document ingestion",
                )
            # Persist the outbox intent first. A process crash can then be
            # diagnosed/reconciled without leaving an undiscoverable payload.
            uploaded_object_keys.append(manifest_key)
            await asyncio.to_thread(
                object_store.put_json,
                manifest_key,
                manifest_payload,
            )
            uploaded_object_keys.append(object_key)
            await asyncio.to_thread(object_store.upload_file, file_path, object_key)
            await _remove_upload_file(file_path)

            # A distributed worker must never be accepted before the task can
            # be discovered from another API replica.  Use the upload UUID as
            # the Celery id so the durable receipt exists before enqueueing.
            initial_state_persisted = await _persist_task_state(
                task_id,
                {
                    "status": "pending",
                    "progress": 0,
                    "total_chunks": 0,
                    "error": None,
                    "error_code": None,
                },
                tenant_id,
                require_durable=True,
            )
            if not initial_state_persisted:
                raise HTTPException(status_code=503, detail="Task state store unavailable")

            from app.service.upload_reconciler import reconciliation_settings
            from app.workers.celery_app import task_default_queue
            from app.workers.tasks import process_document_task

            # Once broker dispatch starts, a timeout is ambiguous: the broker
            # may have accepted the message even if the API never received its
            # confirmation.  Retain the envelope for worker/reconciliation.
            dispatch_attempted = True
            producer_timeout = int(
                reconciliation_settings(config)["producer_timeout_seconds"]
            )
            await asyncio.wait_for(
                asyncio.to_thread(
                    process_document_task.apply_async,
                    kwargs={"object_key": object_key, "tenant_id": tenant_id},
                    task_id=task_id,
                    # Bind the producer explicitly to the same validated queue
                    # contract as the worker. The protected release canary sets
                    # a unique per-Pod queue so stable workers cannot consume it.
                    queue=task_default_queue(),
                ),
                timeout=producer_timeout,
            )
            # This marker is an optimization only. If it fails after the
            # broker accepted the message, the durable reconciler safely
            # redispatches the same id after its cooldown.
            manifest_payload["dispatch_attempts"] = 1
            manifest_payload["last_dispatched_at_unix"] = int(time.time())
            try:
                await asyncio.to_thread(
                    object_store.put_json,
                    manifest_key,
                    manifest_payload,
                )
            except Exception:
                logger.warning(
                    "Upload dispatch marker could not be persisted",
                    extra={"task_id": task_id},
                    exc_info=True,
                )
        except Exception as exc:
            if object_store is not None and not dispatch_attempted:
                for uploaded_key in uploaded_object_keys:
                    try:
                        await asyncio.to_thread(object_store.delete_object, uploaded_key)
                    except Exception:
                        logger.warning("Uploaded object cleanup failed", exc_info=True)
            await _remove_upload_file(file_path)
            raise HTTPException(status_code=503, detail="Document queue unavailable") from exc
    elif provider == "rabbitmq":
        # Kept as a compatibility alias for older settings; it still resolves
        # to Celery and therefore must use object storage as well.
        raise HTTPException(status_code=503, detail="RabbitMQ provider is no longer supported")
    else:
        # 内存模式：FastAPI 后台任务
        initial_state = {
            "status": "pending",
            "progress": 0,
            "total_chunks": 0,
            "document_id": None,
            "error": None,
            "error_code": None,
        }
        await _persist_task_state(task_id, initial_state, tenant_id)
        app_loop = asyncio.get_running_loop()
        background_tasks.add_task(
            _process_document_memory,
            task_id=task_id,
            file_path=str(file_path),
            filename=display_filename,
            partition=partition,
            metadata=metadata_dict,
            tenant_id=tenant_id,
            app_loop=app_loop,
        )

    logger.info(f"Document ingest task submitted: {task_id}", extra={"task_id": task_id})

    return DocumentIngestResponse(
        task_id=task_id,
        filename=display_filename,
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
        try:
            known_state = await _load_task_state(
                task_id,
                tenant_id,
                require_backend=True,
            )
        except Exception as exc:
            from app.utils.cache import TaskStateBackendUnavailable

            if isinstance(exc, TaskStateBackendUnavailable):
                raise HTTPException(
                    status_code=503,
                    detail="Task status temporarily unavailable",
                    headers={"Retry-After": "2"},
                ) from exc
            raise
        # The tenant-scoped durable receipt is the authorization binding for a
        # Celery id.  Never expose a result-backend state for a guessed id that
        # belongs to another tenant, even when Celery reports it as completed.
        if known_state is None:
            raise HTTPException(status_code=404, detail=f"Task {task_id} not found")
        try:
            result = celery_app.AsyncResult(task_id)
        except Exception as exc:
            logger.warning("Celery result backend is unavailable", exc_info=True)
            raise HTTPException(
                status_code=503,
                detail="Task status temporarily unavailable",
                headers={"Retry-After": "2"},
            ) from exc
        try:
            result_state = str(result.state)
            result_info = result.info if result_state != "PENDING" else None
        except Exception as exc:
            logger.warning("Celery result backend is unavailable", exc_info=True)
            raise HTTPException(
                status_code=503,
                detail="Task status temporarily unavailable",
                headers={"Retry-After": "2"},
            ) from exc
        status_map = {
            "PENDING": "pending",
            "STARTED": "processing",
            "PROGRESS": "processing",
            "SUCCESS": "completed",
            "FAILURE": "failed",
            "RETRY": "retrying",
        }
        status = status_map.get(result_state, "unknown")

        response_data = {
            "task_id": task_id,
            "status": status,
            "progress": 0,
            "total_chunks": 0,
            "error": None,
            "error_code": None,
        }
        if result_state == "PENDING" and known_state:
            response_data.update(known_state)
        elif result_state in {"STARTED", "PROGRESS"}:
            info = result_info or {}
            if isinstance(info, dict):
                response_data["progress"] = int(info.get("progress", 0) or 0)
                response_data["total_chunks"] = int(info.get("total_chunks", 0) or 0)
        elif result_state == "SUCCESS":
            info = result_info or {}
            if isinstance(info, dict) and info.get("status") == "failed":
                response_data["status"] = "failed"
                response_data["error"] = INGESTION_ERROR_MESSAGE
                response_data["error_code"] = INGESTION_ERROR_CODE
            else:
                response_data["progress"] = 100
                response_data["total_chunks"] = info.get("total_chunks", 0) if isinstance(info, dict) else 0
                document_id = info.get("document_id") if isinstance(info, dict) else None
                if isinstance(document_id, str) and document_id:
                    response_data["document_id"] = document_id
        elif result_state == "FAILURE":
            logger.error("Celery document task %s failed: %s", task_id, result_info)
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
