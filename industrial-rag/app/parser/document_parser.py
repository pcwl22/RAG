"""Document parser dispatch."""
from pathlib import Path

from app.parser.docx_parser import parse_docx
from app.parser.pdf_parser import parse_pdf

SUPPORTED_EXTENSIONS = frozenset({"pdf", "docx", "xlsx", "txt", "md"})


def parse_excel(file_path: str) -> str:
    """Parse Excel workbooks into markdown tables."""
    import pandas as pd

    sheets = pd.read_excel(file_path, sheet_name=None)
    parts: list[str] = []
    for sheet_name, df in sheets.items():
        parts.append(f"## {sheet_name}\n")
        try:
            parts.append(df.to_markdown(index=False))
        except ImportError:
            columns = [str(column) for column in df.columns]
            rows = [[str(value) for value in row] for row in df.itertuples(index=False, name=None)]
            parts.append(_markdown_table(columns, rows))
    return "\n\n".join(parts)


def _markdown_table(columns: list[str], rows: list[list[str]]) -> str:
    header = "| " + " | ".join(columns) + " |"
    separator = "| " + " | ".join("---" for _ in columns) + " |"
    body = ["| " + " | ".join(row) + " |" for row in rows]
    return "\n".join([header, separator, *body])


def parse_text(file_path: str) -> str:
    """Read a UTF-8 text or markdown file."""
    with open(file_path, encoding="utf-8") as f:
        return f.read()


def parse_document(file_path: str) -> str:
    """Parse supported document formats into text."""
    ext = Path(file_path).suffix.lower()
    if ext == ".pdf":
        return parse_pdf(file_path)
    if ext == ".docx":
        return parse_docx(file_path)
    if ext == ".xlsx":
        return parse_excel(file_path)
    if ext in {".txt", ".md"}:
        return parse_text(file_path)
    raise ValueError(f"Unsupported file format: {ext}")


__all__ = [
    "SUPPORTED_EXTENSIONS",
    "parse_document",
    "parse_pdf",
    "parse_docx",
    "parse_excel",
    "parse_text",
]
