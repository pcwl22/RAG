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


async def check_vector_store_health() -> bool:
    """Return whether PostgreSQL currently accepts a lightweight query."""
    from app.vectorstore.postgres_store import check_postgres_health

    return await check_postgres_health()


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


async def replace_document(
    source_key: str,
    filename: str,
    ids: list[str],
    embeddings: list[list[float]],
    documents: list[str],
    metadatas: list[dict],
    partition: str = "general",
) -> int:
    """Atomically replace every chunk belonging to one logical source."""
    from app.vectorstore.postgres_store import replace_document as replace_postgres

    return await replace_postgres(
        source_key,
        filename,
        ids,
        embeddings,
        documents,
        metadatas,
        partition,
    )


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
    """Fetch documents by primary key or stable semantic chunk identity."""
    from app.vectorstore.postgres_store import get_documents_by_ids as get_postgres_documents

    return await get_postgres_documents(ids, partition)


async def get_documents_by_citations(
    citation_pairs: list[tuple[str, str]], partition: str | None = None
) -> list[dict]:
    """Fetch explicitly cited provisions from PostgreSQL."""
    from app.vectorstore.postgres_store import (
        get_documents_by_citations as get_postgres_citations,
    )

    return await get_postgres_citations(citation_pairs, partition)


async def hybrid_search(
    query: str,
    query_embedding: list[float],
    top_k: int = 10,
    partition: str | None = None,
    enable_rrf: bool | None = None,
    enable_dynamic_topk: bool | None = None,
) -> list[dict]:
    """Run PostgreSQL hybrid search (vector + keyword) with RRF fusion."""
    from app.vectorstore.postgres_store import hybrid_search as hybrid_pg

    config = get_settings().get("rag", {}).get("retrieval", {})
    return await hybrid_pg(
        query=query,
        query_embedding=query_embedding,
        top_k=top_k,
        partition=partition,
        enable_rrf=(
            bool(config.get("enable_rrf", True))
            if enable_rrf is None
            else enable_rrf
        ),
        enable_dynamic_topk=(
            bool(config.get("enable_dynamic_topk", True))
            if enable_dynamic_topk is None
            else enable_dynamic_topk
        ),
        rrf_k=config.get("rrf_k", 60),
        threshold_ratio=config.get("dynamic_topk_threshold", 0.5),
    )


async def delete_document(document_id: str) -> bool:
    """Delete a document from PostgreSQL."""
    from app.vectorstore.postgres_store import delete_document as delete_postgres

    return await delete_postgres(document_id)


async def list_documents(skip: int = 0, limit: int = 100) -> list[dict]:
    """List documents from PostgreSQL."""
    from app.vectorstore.postgres_store import list_documents as list_postgres

    return await list_postgres(skip, limit)


async def count_documents() -> int:
    """Return the total number of grouped source documents."""
    from app.vectorstore.postgres_store import count_documents as count_postgres

    return await count_postgres()


async def list_document_chunks(document_id: str, skip: int = 0, limit: int = 2000) -> dict:
    """List stored chunks for a document from PostgreSQL."""
    from app.vectorstore.postgres_store import list_document_chunks as list_chunks_postgres

    return await list_chunks_postgres(document_id, skip, limit)
