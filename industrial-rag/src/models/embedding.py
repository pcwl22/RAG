"""
嵌入模型管理

使用 sentence-transformers 加载 BGE-M3 模型。
由于 FlagEmbedding 在 Windows + Python 3.12 环境存在兼容性问题（Segmentation Fault），
改用 sentence-transformers，并默认 CPU 模式以保证稳定性。
"""
from typing import List

from src.core.config import get_settings
from src.utils.logger import get_logger

logger = get_logger(__name__)

# 全局嵌入模型实例
_embedding_model = None


def load_embedding_model():
    """
    加载嵌入模型（sentence-transformers，CPU 模式）

    Returns:
        嵌入模型实例
    """
    global _embedding_model

    if _embedding_model is not None:
        return _embedding_model

    from sentence_transformers import SentenceTransformer

    config = get_settings()
    embed_config = config["embedding"]
    model_path = embed_config["model_path"]

    # 设备选择（GPU 已验证可用）
    device = embed_config.get("device", "cuda")

    logger.info(f"Loading embedding model: {model_path} (device={device})")

    try:
        _embedding_model = SentenceTransformer(model_path, device=device)
        logger.info("Embedding model loaded successfully")
        return _embedding_model
    except Exception as e:
        logger.error(f"Failed to load embedding model: {e}")
        raise


def get_embedding_model():
    """
    获取嵌入模型实例

    Returns:
        嵌入模型
    """
    if _embedding_model is None:
        return load_embedding_model()
    return _embedding_model


def encode_texts(texts: List[str], batch_size: int = 32) -> List[List[float]]:
    """
    编码文本为向量

    Args:
        texts: 文本列表
        batch_size: 批处理大小

    Returns:
        向量列表（每个向量为 list[float]）
    """
    model = get_embedding_model()
    config = get_settings()
    embed_config = config["embedding"]

    try:
        # sentence-transformers 的 encode 接口
        embeddings = model.encode(
            texts,
            batch_size=batch_size,
            normalize_embeddings=embed_config.get("normalize_embeddings", True),
            show_progress_bar=False,
        )

        # embeddings 是 numpy 数组
        return embeddings.tolist()

    except Exception as e:
        logger.error(f"Failed to encode texts: {e}")
        raise


def encode_query(query: str) -> List[float]:
    """
    编码单个查询

    Args:
        query: 查询文本

    Returns:
        查询向量
    """
    embeddings = encode_texts([query], batch_size=1)
    return embeddings[0]
