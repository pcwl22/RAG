"""Storage adapter that routes to PostgreSQL or ChromaDB based on config."""
from typing import Any

from app.utils.config import get_settings
from app.utils.logger import get_logger

logger = get_logger(__name__)

_backend: str | None = None


def _get_backend() -> str:
    """Determine which storage backend to use based on config."""
    global _backend
    if _backend is not None:
        return _backend

    config = get_settings()
    use_postgres = config.get("rag", {}).get("retrieval", {}).get("use_postgres", False)
    _backend = "postgres" if use_postgres else "chroma"
    logger.info(f"Using storage backend: {_backend}")
    return _backend


async def init_vector_store() -> None:
    """Initialize the configured storage backend."""
    backend = _get_backend()
    if backend == "postgres":
        from app.vectorstore.postgres_store import init_postgres_store
        await init_postgres_store()
    else:
        from app.vectorstore.chroma_store import init_vector_store as init_chroma
        await init_chroma()


async def close_vector_store() -> None:
    """Close the configured storage backend."""
    backend = _get_backend()
    if backend == "postgres":
        from app.vectorstore.postgres_store import close_postgres_store
        await close_postgres_store()
    else:
        from app.vectorstore.chroma_store import close_vector_store as close_chroma
        await close_chroma()


async def add_documents(
    ids: list[str],
    embeddings: list[list[float]],
    documents: list[str],
    metadatas: list[dict] | None = None,
    partition: str = "general",
) -> int:
    """Add documents to the configured storage backend."""
    backend = _get_backend()
    if backend == "postgres":
        from app.vectorstore.postgres_store import add_documents as add_postgres
        return await add_postgres(ids, embeddings, documents, metadatas, partition)
    else:
        from app.vectorstore.chroma_store import add_documents as add_chroma
        return await add_chroma(ids, embeddings, documents, metadatas, partition)


async def search(
    query_embedding: list[float],
    top_k: int = 5,
    partition: str | None = None,
    where: dict | None = None,
) -> list[dict]:
    """Search using the configured storage backend.

    For PostgreSQL with hybrid search enabled, this will use vector search only.
    Use hybrid_search() for combined vector + keyword retrieval.
    """
    backend = _get_backend()
    if backend == "postgres":
        from app.vectorstore.postgres_store import vector_search
        return await vector_search(query_embedding, top_k, partition)
    else:
        from app.vectorstore.chroma_store import search as search_chroma
        return await search_chroma(query_embedding, top_k, partition, where)


async def hybrid_search(
    query: str,
    query_embedding: list[float],
    top_k: int = 10,
    partition: str | None = None,
) -> list[dict]:
    """Hybrid search (vector + keyword) with RRF fusion.

    Only available for PostgreSQL backend.
    Falls back to vector-only search for ChromaDB.
    """
    backend = _get_backend()
    if backend == "postgres":
        from app.vectorstore.postgres_store import hybrid_search as hybrid_pg
        config = get_settings().get("rag", {}).get("retrieval", {})
        return await hybrid_pg(
            query=query,
            query_embedding=query_embedding,
            top_k=top_k,
            partition=partition,
            enable_rrf=True,
            enable_dynamic_topk=config.get("enable_dynamic_topk", True),
            rrf_k=config.get("rrf_k", 60),
            threshold_ratio=config.get("dynamic_topk_threshold", 0.5),
        )
    else:
        logger.warning("Hybrid search not available for ChromaDB, falling back to vector search")
        return await search(query_embedding, top_k, partition)


async def delete_document(document_id: str) -> None:
    """Delete a document from the configured storage backend."""
    backend = _get_backend()
    if backend == "postgres":
        from app.vectorstore.postgres_store import delete_document as delete_postgres
        await delete_postgres(document_id)
    else:
        from app.vectorstore.chroma_store import delete_document as delete_chroma
        await delete_chroma(document_id)


async def list_documents(skip: int = 0, limit: int = 100) -> list[dict]:
    """List documents from the configured storage backend."""
    backend = _get_backend()
    if backend == "postgres":
        from app.vectorstore.postgres_store import list_documents as list_postgres
        return await list_postgres(skip, limit)
    else:
        from app.vectorstore.chroma_store import list_documents as list_chroma
        return await list_chroma(skip, limit)
