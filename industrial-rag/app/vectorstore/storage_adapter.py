"""Storage adapter for the PostgreSQL/pgvector backend."""
from app.utils.config import get_settings
from app.utils.logger import get_logger

logger = get_logger(__name__)


async def init_vector_store() -> None:
    """Initialize the PostgreSQL storage backend."""
    from app.vectorstore.postgres_store import init_postgres_store

    logger.info("Using storage backend: postgres")
    await init_postgres_store()


async def close_vector_store() -> None:
    """Close the PostgreSQL storage backend."""
    from app.vectorstore.postgres_store import close_postgres_store

    await close_postgres_store()


async def add_documents(
    ids: list[str],
    embeddings: list[list[float]],
    documents: list[str],
    metadatas: list[dict] | None = None,
    partition: str = "general",
) -> int:
    """Add documents to PostgreSQL."""
    from app.vectorstore.postgres_store import add_documents as add_postgres

    return await add_postgres(ids, embeddings, documents, metadatas, partition)


async def search(
    query_embedding: list[float],
    top_k: int = 5,
    partition: str | None = None,
    where: dict | None = None,
) -> list[dict]:
    """Run vector search in PostgreSQL.

    The ``where`` argument is accepted for API compatibility with older callers,
    but PostgreSQL filtering is currently handled through explicit parameters.
    """
    from app.vectorstore.postgres_store import vector_search

    return await vector_search(query_embedding, top_k, partition)


async def get_documents_by_ids(ids: list[str], partition: str | None = None) -> list[dict]:
    """Fetch documents by exact PostgreSQL primary keys."""
    from app.vectorstore.postgres_store import get_documents_by_ids as get_postgres_documents

    return await get_postgres_documents(ids, partition)


async def get_documents_by_citations(
    article_numbers: list[str], law_names: list[str], partition: str | None = None
) -> list[dict]:
    """Fetch explicitly cited provisions from PostgreSQL."""
    from app.vectorstore.postgres_store import (
        get_documents_by_citations as get_postgres_citations,
    )

    return await get_postgres_citations(article_numbers, law_names, partition)


async def hybrid_search(
    query: str,
    query_embedding: list[float],
    top_k: int = 10,
    partition: str | None = None,
) -> list[dict]:
    """Run PostgreSQL hybrid search (vector + keyword) with RRF fusion."""
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


async def delete_document(document_id: str) -> None:
    """Delete a document from PostgreSQL."""
    from app.vectorstore.postgres_store import delete_document as delete_postgres

    await delete_postgres(document_id)


async def list_documents(skip: int = 0, limit: int = 100) -> list[dict]:
    """List documents from PostgreSQL."""
    from app.vectorstore.postgres_store import list_documents as list_postgres

    return await list_postgres(skip, limit)


async def list_document_chunks(document_id: str, skip: int = 0, limit: int = 2000) -> dict:
    """List stored chunks for a document from PostgreSQL."""
    from app.vectorstore.postgres_store import list_document_chunks as list_chunks_postgres

    return await list_chunks_postgres(document_id, skip, limit)
