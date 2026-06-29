"""Word document parsing helpers."""


def parse_docx(file_path: str) -> str:
    """Parse Word documents into plain text."""
    from docx import Document

    doc = Document(file_path)
    return "\n".join(para.text for para in doc.paragraphs if para.text.strip())


__all__ = ["parse_docx"]
