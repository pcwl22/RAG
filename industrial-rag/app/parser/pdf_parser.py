"""PDF parsing helpers."""
from pathlib import Path

from app.utils.config import get_settings
from app.utils.logger import get_logger

logger = get_logger(__name__)


def _doc_processing_config() -> dict:
    return get_settings().get("document_processing", {})


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


__all__ = ["parse_pdf"]
