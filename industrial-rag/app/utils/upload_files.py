"""Retention helpers for failed upload artifacts."""
import time
from pathlib import Path


def quarantine_failed_upload(
    file_path: str,
    *,
    retention_seconds: int = 604800,
    max_files: int = 100,
) -> Path | None:
    source = Path(file_path)
    if not source.exists():
        return None

    failed_dir = source.parent / "failed"
    failed_dir.mkdir(parents=True, exist_ok=True)
    cutoff = time.time() - max(0, retention_seconds)
    retained = sorted(
        (path for path in failed_dir.iterdir() if path.is_file()),
        key=lambda path: path.stat().st_mtime,
    )
    for path in retained:
        if path.stat().st_mtime < cutoff:
            path.unlink(missing_ok=True)
    retained = [path for path in retained if path.exists()]
    overflow = max(0, len(retained) - max(0, max_files - 1))
    for path in retained[:overflow]:
        path.unlink(missing_ok=True)

    destination = failed_dir / source.name
    source.replace(destination)
    return destination
