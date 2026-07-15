"""
Embedding model helpers.

This module loads the BGE-M3 embedding model with sentence-transformers.
FlagEmbedding had compatibility issues in the local Windows + Python 3.12
environment, so device selection is controlled by config and can use GPU.
"""
from __future__ import annotations

from app.utils.config import get_settings
from app.utils.logger import get_logger

logger = get_logger(__name__)

_embedding_model = None


def resolve_torch_device(configured_device: str | None) -> str:
    """Return a usable torch device, falling back to CPU when CUDA is unavailable."""
    device = str(configured_device or "cpu")
    if not device.startswith("cuda"):
        return device

    try:
        import torch

        if torch.cuda.is_available():
            return device
    except Exception:
        pass

    logger.warning("Configured device %s is unavailable; falling back to cpu", device)
    return "cpu"


def get_embedding_runtime_info() -> dict[str, str | bool | None]:
    """Return runtime device information for the embedding model."""
    config = get_settings()
    device = config.get("embedding", {}).get("device", "cuda")
    resolved_config_device = resolve_torch_device(device)
    info: dict[str, str | bool | None] = {
        "configured_device": device,
        "resolved_device": resolved_config_device,
        "gpu_name": None,
        "cuda_available": None,
    }

    try:
        import torch

        cuda_available = torch.cuda.is_available()
        info["cuda_available"] = cuda_available

        if _embedding_model is not None:
            resolved_device = getattr(_embedding_model, "device", None)
            if resolved_device is not None:
                info["resolved_device"] = str(resolved_device)

        if cuda_available and str(device).startswith("cuda"):
            gpu_index = 0
            if ":" in str(device):
                try:
                    gpu_index = int(str(device).split(":", 1)[1])
                except ValueError:
                    gpu_index = 0
            info["gpu_name"] = torch.cuda.get_device_name(gpu_index)
            if info["resolved_device"] is None:
                info["resolved_device"] = f"cuda:{gpu_index}"
    except Exception:
        pass

    return info


def load_embedding_model():
    """
    Load the embedding model with sentence-transformers.

    Returns:
        Embedding model instance.
    """
    global _embedding_model

    if _embedding_model is not None:
        return _embedding_model

    from sentence_transformers import SentenceTransformer

    config = get_settings()
    embed_config = config["embedding"]
    model_path = embed_config["model_path"]

    # Device selection is driven by embedding.device in config, with a local
    # CPU fallback for environments where torch was installed without CUDA.
    device = resolve_torch_device(embed_config.get("device", "cuda"))

    logger.info(f"Loading embedding model: {model_path} (device={device})")

    try:
        _embedding_model = SentenceTransformer(model_path, device=device)
        runtime_info = get_embedding_runtime_info()
        logger.info(
            "Embedding model loaded successfully",
            extra={
                "configured_device": runtime_info["configured_device"],
                "resolved_device": runtime_info["resolved_device"],
                "cuda_available": runtime_info["cuda_available"],
                "gpu_name": runtime_info["gpu_name"],
            },
        )
        return _embedding_model
    except Exception as e:
        logger.error(f"Failed to load embedding model: {e}")
        raise


def get_embedding_model():
    """
    Return the cached embedding model instance.

    Returns:
        Embedding model instance.
    """
    if _embedding_model is None:
        return load_embedding_model()
    return _embedding_model


def encode_texts(texts: list[str], batch_size: int = 32) -> list[list[float]]:
    """
    Encode a list of texts into embedding vectors.

    Args:
        texts: Input text list.
        batch_size: Batch size for encoding.

    Returns:
        List of embedding vectors.
    """
    model = get_embedding_model()
    config = get_settings()
    embed_config = config["embedding"]

    try:
        embeddings = model.encode(
            texts,
            batch_size=batch_size,
            normalize_embeddings=embed_config.get("normalize_embeddings", True),
            show_progress_bar=False,
        )
        return embeddings.tolist()

    except Exception as e:
        logger.error(f"Failed to encode texts: {e}")
        raise


def encode_query(query: str) -> list[float]:
    """
    Encode a single query.

    Args:
        query: Query text.

    Returns:
        Query embedding vector.
    """
    embeddings = encode_texts([query], batch_size=1)
    return embeddings[0]
