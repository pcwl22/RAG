"""
向量存储模块 - Chroma 实现

Chroma 是嵌入式向量数据库，本地零外部服务（persistent 模式），
也支持连接独立的 Chroma Server（http 模式）。

文档类型分区（text/table/image/...）通过 metadata 的 `partition` 字段实现，
检索时用 `where` 过滤——Chroma 没有 Milvus 那样的原生 partition。

对外提供的主要接口：
- init_vector_store() / close_vector_store()   生命周期
- add_documents(...)                            写入
- search(...)                                   向量检索
- delete_document(document_id)                  按文档删除
- list_documents(skip, limit)                   列出文档
"""
from typing import Any

from app.utils.config import get_settings
from app.utils.logger import get_logger

logger = get_logger(__name__)

# 全局客户端与集合
_client: Any = None
_collection: Any = None


def _chroma_config() -> dict:
    return get_settings().get("chroma", {})


async def init_vector_store() -> None:
    """初始化 Chroma 客户端并获取/创建集合。"""
    global _client, _collection

    if _collection is not None:
        return

    import chromadb

    cfg = _chroma_config()
    mode = cfg.get("mode", "persistent").lower()

    if mode == "http":
        host = cfg.get("host", "localhost")
        port = cfg.get("port", 8000)
        logger.info(f"Connecting to Chroma server at {host}:{port}")
        _client = chromadb.HttpClient(host=host, port=port)
    else:
        persist_dir = cfg.get("persist_directory", "./data/chroma")
        # 确保目录存在（跨平台）
        from pathlib import Path

        Path(persist_dir).mkdir(parents=True, exist_ok=True)
        logger.info(f"Opening persistent Chroma at {persist_dir}")
        _client = chromadb.PersistentClient(path=persist_dir)

    coll_cfg = cfg.get("collection", {})
    name = coll_cfg.get("name", "rag_documents")
    distance = coll_cfg.get("distance", "cosine")

    _collection = _client.get_or_create_collection(
        name=name,
        metadata={"hnsw:space": distance},
    )
    logger.info(
        f"Chroma collection ready: {name} "
        f"(distance={distance}, count={_collection.count()})"
    )


async def close_vector_store() -> None:
    """关闭/释放 Chroma 客户端。

    PersistentClient 会自动持久化，无需显式 flush；这里只清理引用。
    """
    global _client, _collection
    logger.info("Closing Chroma client")
    _collection = None
    _client = None


def _get_collection() -> Any:
    if _collection is None:
        raise RuntimeError(
            "向量库未初始化，请先在应用启动时调用 init_vector_store()"
        )
    return _collection


async def add_documents(
    ids: list[str],
    embeddings: list[list[float]],
    documents: list[str],
    metadatas: list[dict] | None = None,
    partition: str = "text",
) -> int:
    """写入一批文档块。

    Args:
        ids: 每个块的唯一 ID
        embeddings: 对应的向量
        documents: 原文内容
        metadatas: 每个块的元数据（会自动注入 partition 字段）
        partition: 文档类型分区（写入 metadata.partition）

    Returns:
        写入的块数量
    """
    coll = _get_collection()

    if metadatas is None:
        metadatas = [{} for _ in ids]
    # 注入分区标记（Chroma 元数据值需为标量）
    for md in metadatas:
        md.setdefault("partition", partition)

    coll.add(
        ids=ids,
        embeddings=embeddings,
        documents=documents,
        metadatas=metadatas,
    )
    logger.info(f"Added {len(ids)} chunks to partition '{partition}'")
    return len(ids)


async def search(
    query_embedding: list[float],
    top_k: int = 5,
    partition: str | None = None,
    where: dict | None = None,
) -> list[dict]:
    """向量检索。

    Args:
        query_embedding: 查询向量
        top_k: 返回数量
        partition: 限定分区（None 表示全部）
        where: 额外的 metadata 过滤条件

    Returns:
        结果列表，每项含 id / content / score / metadata / chunk_index
    """
    coll = _get_collection()

    # 组装过滤条件
    filters: dict = {}
    if partition:
        filters["partition"] = partition
    if where:
        filters.update(where)

    query_kwargs: dict = {
        "query_embeddings": [query_embedding],
        "n_results": top_k,
    }
    if filters:
        query_kwargs["where"] = filters

    res = coll.query(**query_kwargs)

    # Chroma 返回的是每个 query 一组结果，取第 0 组
    out: list[dict] = []
    ids = res.get("ids", [[]])[0]
    docs = res.get("documents", [[]])[0]
    metas = res.get("metadatas", [[]])[0]
    dists = res.get("distances", [[]])[0]

    for i, _id in enumerate(ids):
        distance = dists[i] if i < len(dists) else 0.0
        # cosine 距离 -> 相似度分数（distance 越小越相似）
        score = 1.0 - distance
        meta = metas[i] if i < len(metas) else {}
        out.append(
            {
                "id": _id,
                "content": docs[i] if i < len(docs) else "",
                "score": score,
                "metadata": meta,
                "chunk_index": meta.get("chunk_index", 0),
            }
        )
    return out


async def delete_document(document_id: str) -> None:
    """删除某个文档的所有块（按 metadata.document_id 匹配）。"""
    coll = _get_collection()
    coll.delete(where={"document_id": document_id})
    logger.info(f"Deleted document {document_id}")


async def list_documents(skip: int = 0, limit: int = 100) -> list[dict]:
    """列出已入库的文档（按 document_id 去重聚合）。

    Chroma 没有原生的文档级视图，这里扫描 metadata 聚合出文档清单。
    """
    coll = _get_collection()

    got = coll.get(include=["metadatas"])
    metas = got.get("metadatas", []) or []

    # 按 document_id 聚合
    docs: dict[str, dict] = {}
    for md in metas:
        doc_id = md.get("document_id")
        if not doc_id:
            continue
        if doc_id not in docs:
            docs[doc_id] = {
                "document_id": doc_id,
                "filename": md.get("filename", "unknown"),
                "partition": md.get("partition", "text"),
                "chunk_count": 0,
            }
        docs[doc_id]["chunk_count"] += 1

    items = list(docs.values())
    return items[skip : skip + limit]
