"""Text chunking helpers."""
import re
from typing import Any

from app.parser.legal_parser import LegalDocumentParser, LegalSection
from app.parser.parent_child_chunker import ParentChildChunker
from app.parser.parent_child_chunking import chunk_text_parent_child
from app.utils.config import get_config_section
from app.utils.logger import get_logger

logger = get_logger(__name__)

DEFAULT_SEPARATORS = [
    "\n# ",
    "\n## ",
    "\n### ",
    "\n\n",
    "\n",
    "。",
    "；",
    "，",
    ". ",
    "; ",
    ", ",
    " ",
    "",
]


def _doc_processing_config() -> dict[str, Any]:
    return get_config_section("document_processing")


def _chunking_config() -> dict[str, Any]:
    return get_config_section("document_processing", "chunking")


def _normalize_text(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t]+\n", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _sliding_window(text: str, chunk_size: int, chunk_overlap: int) -> list[str]:
    step = max(1, chunk_size - chunk_overlap)
    return [
        text[i : i + chunk_size].strip()
        for i in range(0, len(text), step)
        if text[i : i + chunk_size].strip()
    ]


def _split_text(text: str, separators: list[str], chunk_size: int, chunk_overlap: int) -> list[str]:
    if len(text) <= chunk_size:
        return [text.strip()] if text.strip() else []
    if not separators:
        return _sliding_window(text, chunk_size, chunk_overlap)

    separator = separators[0]
    if separator == "":
        return _sliding_window(text, chunk_size, chunk_overlap)
    if separator not in text:
        return _split_text(text, separators[1:], chunk_size, chunk_overlap)

    pieces: list[str] = []
    for index, split in enumerate(text.split(separator)):
        if not split.strip():
            continue
        piece = split if index == 0 else f"{separator}{split}"
        if len(piece) > chunk_size:
            pieces.extend(_split_text(piece, separators[1:], chunk_size, chunk_overlap))
        else:
            pieces.append(piece.strip())
    return pieces


def _merge_splits(splits: list[str], chunk_size: int, chunk_overlap: int) -> list[str]:
    chunks: list[str] = []
    current: list[str] = []
    current_len = 0

    for split in splits:
        split = split.strip()
        if not split:
            continue
        sep_len = 1 if current else 0
        if current and current_len + sep_len + len(split) > chunk_size:
            chunk = "\n".join(current).strip()
            if chunk:
                chunks.append(chunk)

            overlap_parts: list[str] = []
            overlap_len = 0
            for part in reversed(current):
                part_len = len(part) + (1 if overlap_parts else 0)
                if overlap_len + part_len > chunk_overlap:
                    break
                overlap_parts.insert(0, part)
                overlap_len += part_len
            current = overlap_parts
            current_len = sum(len(part) for part in current) + max(0, len(current) - 1)

        current.append(split)
        current_len += len(split) + sep_len

    if current:
        chunk = "\n".join(current).strip()
        if chunk:
            chunks.append(chunk)
    return chunks


def chunk_text_recursive(text: str, chunk_size: int, chunk_overlap: int) -> list[str]:
    """Chunk text recursively using configured separators."""
    if chunk_size <= 0:
        raise ValueError("chunk_size must be greater than 0")
    if chunk_overlap >= chunk_size:
        chunk_overlap = max(0, chunk_size // 5)

    cfg = _chunking_config()
    separators = cfg.get("separators") or DEFAULT_SEPARATORS
    if "" not in separators:
        separators = [*separators, ""]

    raw_chunks = _split_text(_normalize_text(text), separators, chunk_size, chunk_overlap)
    chunks = _merge_splits(raw_chunks, chunk_size, chunk_overlap)
    min_chunk_size = int(cfg.get("min_chunk_size", 80))
    if len(chunks) > 1:
        chunks = [chunk for chunk in chunks if len(chunk) >= min_chunk_size]
    return chunks


def chunk_text(text: str) -> list[str]:
    """Chunk text using the configured strategy."""
    cfg = _chunking_config()
    strategy = cfg.get("strategy", "recursive")
    chunk_size = int(cfg.get("chunk_size", 800))
    chunk_overlap = int(cfg.get("chunk_overlap", 120))
    text = _normalize_text(text)

    if strategy == "parent_child":
        result = chunk_text_parent_child(text)
        return [child["child_content"] for child in result["child_chunks"]]
    if strategy == "recursive":
        return chunk_text_recursive(text, chunk_size, chunk_overlap)
    if strategy == "fixed":
        return _sliding_window(text, chunk_size, chunk_overlap)
    if strategy == "semantic":
        # Silently falling back produced a corpus whose chunk_strategy metadata
        # and chunking fingerprint claimed "semantic" while the text was split
        # recursively. Fail instead of misreporting how the corpus was built.
        raise ValueError(
            "Semantic chunking is not implemented. Set "
            "document_processing.chunking.strategy to one of: "
            "recursive, parent_child, fixed."
        )
    raise ValueError(f"Unknown chunking strategy: {strategy}")


__all__ = [
    "chunk_text",
    "chunk_text_recursive",
    "chunk_text_parent_child",
    "ParentChildChunker",
    "LegalDocumentParser",
    "LegalSection",
]
