"""Document ingestion workflow service."""
import hashlib
import time
import uuid
from typing import Any

from app.embedding.embedder import encode_texts, get_embedding_runtime_info
from app.parser.chunk import chunk_text
from app.parser.document_parser import parse_document
from app.parser.legal_parser import build_legal_article_chunks
from app.parser.parent_child_chunking import chunk_text_parent_child
from app.utils.config import get_settings
from app.utils.inference import run_inference
from app.utils.logger import get_logger

logger = get_logger(__name__)


def _doc_processing_config() -> dict:
    return get_settings().get("document_processing", {})


def _chunking_config() -> dict:
    return _doc_processing_config().get("chunking", {})


def _storage_backend():
    from app.vectorstore.storage_adapter import replace_document

    return replace_document


def _source_key(filename: str, partition: str, metadata: dict[str, Any]) -> str:
    """Build a stable source identity for document-level replacement."""
    explicit = str(metadata.get("source_id") or "").strip()
    material = explicit or f"{partition.strip().casefold()}\0{filename.strip().casefold()}"
    return hashlib.sha256(material.encode("utf-8")).hexdigest()


def _dedupe_chunk_id(base_id: str, seen: set[str]) -> str:
    chunk_id = base_id
    suffix = 2
    while chunk_id in seen:
        chunk_id = f"{base_id}_{suffix}"
        suffix += 1
    seen.add(chunk_id)
    return chunk_id


async def process_document(
    file_path: str,
    filename: str,
    partition: str = "general",
    metadata: dict | None = None,
    *,
    invalidate_cache: bool = True,
) -> dict[str, Any]:
    """Parse, chunk, embed and store a document."""
    logger.info("Processing document: %s", filename)
    pipeline_start = time.perf_counter()

    try:
        text = parse_document(file_path)
        if not text or len(text.strip()) < 50:
            raise ValueError("Parsed text is too short or empty")

        cfg = _chunking_config()
        strategy = cfg.get("strategy", "recursive")
        child_chunks: list[dict[str, Any]] | None = None
        chunk_metadata_overrides: list[dict[str, Any]] | None = None
        chunk_start = time.perf_counter()

        legal_chunks = build_legal_article_chunks(text, filename)
        if legal_chunks:
            chunks = [item["content"] for item in legal_chunks]
            chunk_metadata_overrides = [item["metadata"] for item in legal_chunks]
            effective_strategy = "legal_article"
        elif strategy == "parent_child":
            chunk_result = chunk_text_parent_child(text)
            child_chunks = chunk_result["child_chunks"]
            chunks = [child["child_content"] for child in child_chunks]
            effective_strategy = strategy
        else:
            chunks = chunk_text(text)
            effective_strategy = strategy

        chunk_elapsed = time.perf_counter() - chunk_start

        if not chunks:
            raise ValueError("No chunks generated")

        document_id = str(uuid.uuid4())
        base_metadata = dict(metadata or {})
        source_key = _source_key(filename, partition, base_metadata)
        batch_size = int(_doc_processing_config().get("batch_size", 16))
        embeddings: list[list[float]] = []
        embedding_runtime = get_embedding_runtime_info()
        embedding_start = time.perf_counter()

        logger.info(
            "Embedding %s chunks",
            len(chunks),
            extra={
                "document_id": document_id,
                "document_filename": filename,
                "chunk_strategy": effective_strategy,
                "chunk_count": len(chunks),
                "chunk_duration_seconds": round(chunk_elapsed, 4),
                "embedding_device": embedding_runtime["resolved_device"],
                "embedding_gpu_name": embedding_runtime["gpu_name"],
                "cuda_available": embedding_runtime["cuda_available"],
            },
        )
        for start in range(0, len(chunks), batch_size):
            batch = chunks[start : start + batch_size]
            embeddings.extend(await run_inference(encode_texts, batch, batch_size=len(batch)))

        embedding_elapsed = time.perf_counter() - embedding_start

        chunk_ids: list[str] = []
        chunk_metadatas: list[dict[str, Any]] = []
        seen_chunk_ids: set[str] = set()

        for index, chunk in enumerate(chunks):
            chunk_meta: dict[str, Any] = {
                **base_metadata,
                "document_id": document_id,
                "filename": filename,
                "chunk_index": index,
                "total_chunks": len(chunks),
                "chunk_length": len(chunk),
                "chunk_strategy": effective_strategy,
                "partition": partition,
                "source_key": source_key,
            }

            if chunk_metadata_overrides:
                chunk_meta.update(chunk_metadata_overrides[index])

            # Database row IDs are version-specific. Stable legal identities
            # remain in metadata.semantic_chunk_id for controlled exact recall.
            chunk_id_base = f"{document_id}_chunk_{index}"
            chunk_id = _dedupe_chunk_id(chunk_id_base, seen_chunk_ids)
            chunk_ids.append(chunk_id)
            chunk_meta["chunk_id"] = chunk_id

            if child_chunks:
                child = child_chunks[index]
                chunk_meta.update(
                    {
                        "parent_id": child["parent_id"],
                        "parent_content": child["parent_content"],
                        "child_id": child["child_id"],
                        "child_index": child["child_index"],
                        "total_children": child["total_children"],
                    }
                )

            chunk_metadatas.append(chunk_meta)

        replace_document = _storage_backend()
        await replace_document(
            source_key=source_key,
            filename=filename,
            ids=chunk_ids,
            embeddings=embeddings,
            documents=chunks,
            metadatas=chunk_metadatas,
            partition=partition,
        )
        if invalidate_cache:
            try:
                from app.utils.cache import invalidate_semantic_cache

                await invalidate_semantic_cache()
            except Exception:
                logger.debug("Semantic cache invalidation unavailable", exc_info=True)

        total_elapsed = time.perf_counter() - pipeline_start
        logger.info(
            "Document processed successfully: %s",
            document_id,
            extra={
                "document_id": document_id,
                "document_filename": filename,
                "chunk_strategy": effective_strategy,
                "chunk_count": len(chunks),
                "chunk_duration_seconds": round(chunk_elapsed, 4),
                "embedding_duration_seconds": round(embedding_elapsed, 4),
                "pipeline_duration_seconds": round(total_elapsed, 4),
                "embedding_device": embedding_runtime["resolved_device"],
                "embedding_gpu_name": embedding_runtime["gpu_name"],
            },
        )
        return {
            "document_id": document_id,
            "source_key": source_key,
            "total_chunks": len(chunks),
            "status": "completed",
        }

    except Exception as exc:
        logger.error("Document processing failed: %s", exc, exc_info=True)
        return {"document_id": None, "total_chunks": 0, "status": "failed", "error": str(exc)}


__all__ = ["parse_document", "process_document"]
