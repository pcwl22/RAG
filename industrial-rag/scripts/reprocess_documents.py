"""Reprocess uploaded documents for one tenant through the canonical ingest path."""

from __future__ import annotations

import argparse
import asyncio
import json
import re
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.auth import normalize_tenant_id  # noqa: E402
from app.service.ingest_service import process_document  # noqa: E402
from app.utils.config import get_settings  # noqa: E402
from app.utils.logger import get_logger  # noqa: E402
from app.vectorstore.storage_adapter import (  # noqa: E402
    close_vector_store,
    init_vector_store,
)

logger = get_logger(__name__)
_UUID_PREFIX = re.compile(
    r"^[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}_"
)


def _configured_upload_dir() -> Path:
    configured = get_settings().get("document_processing", {}).get("upload_dir", "./data/uploads")
    path = Path(str(configured)).expanduser()
    return path if path.is_absolute() else PROJECT_ROOT / path


def _original_filename(path: Path) -> str:
    return _UUID_PREFIX.sub("", path.name, count=1) or path.name


async def reprocess_documents(
    tenant_id: str,
    *,
    upload_dir: str | Path | None = None,
    pattern: str = "*.pdf",
) -> dict[str, int]:
    """Reprocess matching files for one tenant with idempotent source replacement."""
    resolved_tenant = normalize_tenant_id(tenant_id)
    source_dir = Path(upload_dir).expanduser() if upload_dir is not None else _configured_upload_dir()
    if not source_dir.is_absolute():
        source_dir = PROJECT_ROOT / source_dir
    files = sorted(path for path in source_dir.glob(pattern) if path.is_file())
    logger.info("Found %s files for tenant %s in %s", len(files), resolved_tenant, source_dir)

    await init_vector_store()
    processed = 0
    failed = 0
    try:
        for source in files:
            filename = _original_filename(source)
            try:
                result = await process_document(
                    str(source),
                    filename,
                    partition="general",
                    metadata={"source": "reprocess-script"},
                    tenant_id=resolved_tenant,
                )
                if result.get("status") != "completed":
                    raise RuntimeError(result.get("error") or "document processing failed")
                processed += 1
                logger.info("Reprocessed %s (%s chunks)", filename, result.get("total_chunks", 0))
            except Exception as exc:
                failed += 1
                logger.error("Failed to reprocess %s: %s", source, exc, exc_info=True)
    finally:
        await close_vector_store()

    return {"processed": processed, "failed": failed, "total": len(files)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tenant-id", required=True, help="Tenant UUID that owns the source files")
    parser.add_argument("--upload-dir", help="Override the configured upload directory")
    parser.add_argument("--pattern", default="*.pdf", help="Glob pattern, default: *.pdf")
    args = parser.parse_args()
    result = asyncio.run(
        reprocess_documents(args.tenant_id, upload_dir=args.upload_dir, pattern=args.pattern)
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if result["failed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
