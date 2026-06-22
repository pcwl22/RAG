"""Document ingestion pipeline."""
import re
import uuid
from pathlib import Path
from typing import Any

from src.core.config import get_settings
from src.models.embedding import encode_texts
from src.utils.logger import get_logger

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


def _doc_processing_config() -> dict:
    return get_settings().get("document_processing", {})


def _chunking_config() -> dict:
    return _doc_processing_config().get("chunking", {})


def parse_pdf(file_path: str) -> str:
    """Parse PDF with MinerU first, then fall back to PyMuPDF."""
    logger.info("Parsing PDF: %s", file_path)
    cfg = _doc_processing_config().get("pdf", {})

    try:
        from magic_pdf.pipe.UNIPipe import UNIPipe
        from magic_pdf.rw.DiskReaderWriter import DiskReaderWriter

        with open(file_path, "rb") as f:
            pdf_bytes = f.read()

        reader_writer = DiskReaderWriter(str(Path(file_path).parent))
        pipe = UNIPipe(pdf_bytes, {"_pdf_type": ""}, reader_writer)
        parse_method = cfg.get("parse_method", "auto")
        pipe.pipe_classify()
        if parse_method in {"ocr", "txt"} or pipe.classify_result:
            pipe.pipe_analyze()
            pipe.pipe_parse()

        return pipe.pipe_mk_markdown(
            "pipe.md",
            drop_mode="none",
            md_make_mode=cfg.get("output_format", "md"),
        )
    except Exception as exc:
        logger.warning("MinerU parsing failed, falling back to PyMuPDF: %s", exc)
        try:
            import fitz

            doc = fitz.open(file_path)
            return "\n".join(page.get_text() for page in doc)
        except Exception as fallback_exc:
            raise RuntimeError(f"Failed to parse PDF: {file_path}") from fallback_exc


def parse_docx(file_path: str) -> str:
    from docx import Document

    doc = Document(file_path)
    return "\n".join(para.text for para in doc.paragraphs if para.text.strip())


def parse_excel(file_path: str) -> str:
    import pandas as pd

    sheets = pd.read_excel(file_path, sheet_name=None)
    parts: list[str] = []
    for sheet_name, df in sheets.items():
        parts.append(f"## {sheet_name}\n")
        parts.append(df.to_markdown(index=False))
    return "\n\n".join(parts)


def parse_text(file_path: str) -> str:
    with open(file_path, "r", encoding="utf-8") as f:
        return f.read()


def parse_document(file_path: str) -> str:
    ext = Path(file_path).suffix.lower()
    if ext == ".pdf":
        return parse_pdf(file_path)
    if ext in {".docx", ".doc"}:
        return parse_docx(file_path)
    if ext in {".xlsx", ".xls"}:
        return parse_excel(file_path)
    if ext in {".txt", ".md"}:
        return parse_text(file_path)
    raise ValueError(f"Unsupported file format: {ext}")


def _normalize_text(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t]+\n", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _sliding_window(text: str, chunk_size: int, chunk_overlap: int) -> list[str]:
    step = max(1, chunk_size - chunk_overlap)
    return [text[i : i + chunk_size].strip() for i in range(0, len(text), step) if text[i : i + chunk_size].strip()]


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
    cfg = _chunking_config()
    strategy = cfg.get("strategy", "recursive")
    chunk_size = int(cfg.get("chunk_size", 800))
    chunk_overlap = int(cfg.get("chunk_overlap", 120))
    text = _normalize_text(text)

    if strategy == "parent_child":
        from src.ingest.parent_child_chunking import chunk_text_parent_child

        result = chunk_text_parent_child(text)
        return [child["child_content"] for child in result["child_chunks"]]
    if strategy == "recursive":
        return chunk_text_recursive(text, chunk_size, chunk_overlap)
    if strategy == "fixed":
        return _sliding_window(text, chunk_size, chunk_overlap)
    if strategy == "semantic":
        logger.warning("Semantic chunking is not implemented; using recursive chunking")
        return chunk_text_recursive(text, chunk_size, chunk_overlap)
    raise ValueError(f"Unknown chunking strategy: {strategy}")


def _storage_backend():
    from src.storage.storage_adapter import add_documents
    return add_documents


async def process_document(
    file_path: str,
    filename: str,
    partition: str = "general",
    metadata: dict | None = None,
) -> dict[str, Any]:
    """Parse, chunk, embed and store a document."""
    logger.info("Processing document: %s", filename)

    try:
        text = parse_document(file_path)
        if not text or len(text.strip()) < 50:
            raise ValueError("Parsed text is too short or empty")

        cfg = _chunking_config()
        strategy = cfg.get("strategy", "recursive")
        child_chunks: list[dict[str, Any]] | None = None

        if strategy == "parent_child":
            from src.ingest.parent_child_chunking import chunk_text_parent_child

            chunk_result = chunk_text_parent_child(text)
            child_chunks = chunk_result["child_chunks"]
            chunks = [child["child_content"] for child in child_chunks]
        else:
            chunks = chunk_text(text)

        if not chunks:
            raise ValueError("No chunks generated")

        document_id = str(uuid.uuid4())
        batch_size = int(_doc_processing_config().get("batch_size", 16))
        embeddings: list[list[float]] = []

        logger.info("Embedding %s chunks", len(chunks))
        for start in range(0, len(chunks), batch_size):
            batch = chunks[start : start + batch_size]
            embeddings.extend(encode_texts(batch, batch_size=len(batch)))

        base_metadata = metadata or {}
        chunk_ids: list[str] = []
        chunk_metadatas: list[dict[str, Any]] = []

        for index, chunk in enumerate(chunks):
            chunk_id = f"{document_id}_chunk_{index}"
            chunk_ids.append(chunk_id)

            chunk_meta: dict[str, Any] = {
                **base_metadata,
                "document_id": document_id,
                "filename": filename,
                "chunk_index": index,
                "total_chunks": len(chunks),
                "chunk_length": len(chunk),
                "partition": partition,
            }

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

        add_documents = _storage_backend()
        await add_documents(
            ids=chunk_ids,
            embeddings=embeddings,
            documents=chunks,
            metadatas=chunk_metadatas,
            partition=partition,
        )

        logger.info("Document processed successfully: %s", document_id)
        return {"document_id": document_id, "total_chunks": len(chunks), "status": "completed"}

    except Exception as exc:
        logger.error("Document processing failed: %s", exc, exc_info=True)
        return {"document_id": None, "total_chunks": 0, "status": "failed", "error": str(exc)}
