"""PDF parsing helpers."""
from pathlib import Path
from typing import Any

from app.utils.config import get_config_section
from app.utils.logger import get_logger

logger = get_logger(__name__)


def _doc_processing_config() -> dict[str, Any]:
    return get_config_section("document_processing")


def parse_pdf(file_path: str) -> str:
    """Parse PDF with the configured engine and a controlled fallback."""
    logger.info("Parsing PDF: %s", file_path)
    cfg = _doc_processing_config().get("pdf", {})
    engine = str(cfg.get("engine", "auto")).strip().lower()

    if engine == "pymupdf":
        return _parse_with_pymupdf(file_path)

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

        return str(
            pipe.pipe_mk_markdown(
                "pipe.md",
                drop_mode="none",
                md_make_mode=cfg.get("output_format", "md"),
            )
        )
    except Exception as exc:
        logger.warning("MinerU parsing failed, falling back to PyMuPDF: %s", exc)
        return _parse_with_pymupdf(file_path)


def _parse_with_pymupdf(file_path: str) -> str:
    try:
        import fitz

        with fitz.open(file_path) as doc:
            return "\n".join(page.get_text() for page in doc)
    except Exception as exc:
        raise RuntimeError(f"Failed to parse PDF: {file_path}") from exc


__all__ = ["parse_pdf"]
