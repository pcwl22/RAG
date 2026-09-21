"""Validate uploaded document containers before invoking complex parsers."""

from pathlib import Path
from typing import Any
from zipfile import BadZipFile, ZipFile


def _validate_office_archive(path: Path, extension: str, config: dict[str, Any]) -> None:
    max_entries = int(config.get("max_archive_entries", 5000))
    max_uncompressed = int(config.get("max_archive_uncompressed_size", 512 * 1024 * 1024))
    max_ratio = float(config.get("max_archive_compression_ratio", 200))
    required = "word/document.xml" if extension == "docx" else "xl/workbook.xml"
    try:
        with ZipFile(path) as archive:
            entries = archive.infolist()
            if len(entries) > max_entries:
                raise ValueError("Office archive contains too many entries")
            if required not in archive.namelist():
                raise ValueError(f"Invalid {extension.upper()} container")
            total = 0
            for entry in entries:
                total += int(entry.file_size)
                if total > max_uncompressed:
                    raise ValueError("Office archive expands beyond the configured limit")
                if entry.file_size and entry.file_size / max(1, entry.compress_size) > max_ratio:
                    raise ValueError("Office archive compression ratio is suspicious")
    except BadZipFile as exc:
        raise ValueError(f"Invalid {extension.upper()} ZIP container") from exc


def validate_uploaded_document(
    file_path: str | Path,
    extension: str,
    config: dict[str, Any],
) -> None:
    """Reject mismatched signatures, binary text, and unsafe Office archives."""
    path = Path(file_path)
    if path.stat().st_size <= 0:
        raise ValueError("Uploaded file is empty")
    with path.open("rb") as stream:
        header = stream.read(64 * 1024)

    extension = extension.lower().removeprefix(".")
    if extension == "pdf":
        if not header.lstrip().startswith(b"%PDF-"):
            raise ValueError("File extension does not match PDF content")
        return
    if extension in {"docx", "xlsx"}:
        if not header.startswith(b"PK"):
            raise ValueError(f"File extension does not match {extension.upper()} content")
        _validate_office_archive(path, extension, config)
        return
    if extension in {"txt", "md"}:
        if b"\x00" in header:
            raise ValueError("Text upload contains binary NUL bytes")
        try:
            header.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise ValueError("Text upload must be valid UTF-8") from exc


__all__ = ["validate_uploaded_document"]
