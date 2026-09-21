"""Word document parsing helpers."""
from collections.abc import Iterator
from typing import Any


def _markdown_table(table: object) -> str:
    """Serialize a Word table without losing cells during text extraction."""
    rows = [
        [
            cell.text.replace("|", r"\|").replace("\n", " ").strip()
            for cell in row.cells
        ]
        for row in table.rows  # type: ignore[attr-defined]
    ]
    if not rows:
        return ""
    width = max(len(row) for row in rows)
    normalized = [row + [""] * (width - len(row)) for row in rows]
    header = "| " + " | ".join(normalized[0]) + " |"
    separator = "| " + " | ".join("---" for _ in range(width)) + " |"
    body = ["| " + " | ".join(row) + " |" for row in normalized[1:]]
    return "\n".join([header, separator, *body])


def _iter_block_items(document: object) -> Iterator[Any]:
    """Yield paragraphs and tables in their original document order."""
    from docx.document import Document as DocumentType
    from docx.oxml.table import CT_Tbl
    from docx.oxml.text.paragraph import CT_P
    from docx.table import Table
    from docx.text.paragraph import Paragraph

    if not isinstance(document, DocumentType):
        raise TypeError("document must be a python-docx Document")
    for child in document.element.body.iterchildren():
        if isinstance(child, CT_P):
            yield Paragraph(child, document)
        elif isinstance(child, CT_Tbl):
            yield Table(child, document)


def parse_docx(file_path: str) -> str:
    """Parse Word paragraphs and tables into ordered plain/markdown text."""
    from docx import Document
    from docx.table import Table
    from docx.text.paragraph import Paragraph

    doc = Document(file_path)
    parts: list[str] = []
    for block in _iter_block_items(doc):
        if isinstance(block, Paragraph):
            text = block.text.strip()
        elif isinstance(block, Table):
            text = _markdown_table(block)
        else:  # pragma: no cover - guarded by _iter_block_items
            continue
        if text:
            parts.append(text)
    return "\n\n".join(parts)


__all__ = ["parse_docx"]
