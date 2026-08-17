"""
Embedding model helpers.

This module loads the BGE-M3 embedding model with sentence-transformers.
FlagEmbedding had compatibility issues in the local Windows + Python 3.12
environment, so device selection is controlled by config and can use GPU.
"""
from __future__ import annotations

import os
import time
from typing import Any

from app.utils.config import get_settings
from app.utils.logger import get_logger
from app.utils.metrics import EMBEDDING_DURATION, EMBEDDING_REQUESTS

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
    device = os.getenv("EMBEDDING_DEVICE") or config.get("embedding", {}).get("device", "cuda")
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


def load_embedding_model() -> Any:
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
    device = resolve_torch_device(
        os.getenv("EMBEDDING_DEVICE") or embed_config.get("device", "cuda")
    )

    logger.info(f"Loading embedding model: {model_path} (device={device})")

    try:
        _embedding_model = SentenceTransformer(model_path, device=device)
        # embedding.max_length participates in the corpus fingerprint recorded by
        # the vector store, so it has to actually govern truncation. Without this
        # the setting is inert: changing it would force a corpus rebuild through
        # the fingerprint while leaving the produced vectors identical.
        configured_max_length = embed_config.get("max_length")
        if configured_max_length is not None:
            max_length = int(configured_max_length)
            if max_length < 1:
                raise ValueError("embedding.max_length must be at least 1")
            model_limit = int(getattr(_embedding_model, "max_seq_length", max_length))
            if max_length > model_limit:
                logger.warning(
                    "Configured embedding.max_length %s exceeds the model limit %s; "
                    "using the model limit",
                    max_length,
                    model_limit,
                )
            else:
                _embedding_model.max_seq_length = max_length
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


def get_embedding_model() -> Any:
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

    started = time.perf_counter()
    outcome = "success"
    try:
        embeddings = model.encode(
            texts,
            batch_size=batch_size,
            normalize_embeddings=embed_config.get("normalize_embeddings", True),
            show_progress_bar=False,
        )
        vectors: list[list[float]] = embeddings.tolist()
        return vectors

    except Exception as e:
        outcome = "error"
        logger.error(f"Failed to encode texts: {e}")
        raise
    finally:
        EMBEDDING_REQUESTS.labels(outcome=outcome).inc()
        EMBEDDING_DURATION.observe(time.perf_counter() - started)


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
