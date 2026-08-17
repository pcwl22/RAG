"""Create an atomic custom-format PostgreSQL backup with pg_dump."""

from __future__ import annotations

import argparse
import json
import subprocess
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    # The packaged path is the canonical one for type checking; the runtime
    # fallback below only exists so `python scripts/backup_postgres.py` works.
    from scripts.postgres_cli import PostgresTarget
else:
    try:
        from scripts.postgres_cli import PostgresTarget
    except ModuleNotFoundError:  # Direct execution adds only scripts/ to sys.path.
        from postgres_cli import PostgresTarget

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BACKUP_DIR = PROJECT_ROOT / "data" / "backups" / "postgres"


def default_output(database: str) -> Path:
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    return DEFAULT_BACKUP_DIR / f"{database}-{timestamp}.dump"


def backup_database(
    target: PostgresTarget,
    output: Path,
    *,
    executable: str = "pg_dump",
    overwrite: bool = False,
    runner: Callable[..., subprocess.CompletedProcess] = subprocess.run,
) -> Path:
    destination = output.expanduser().resolve()
    if destination.exists() and not overwrite:
        raise FileExistsError(f"Backup already exists: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_name(f".{destination.name}.partial")
    if partial.exists():
        partial.unlink()

    command = [
        executable,
        *target.connection_args(),
        "--format=custom",
        "--compress=6",
        "--no-owner",
        "--no-privileges",
        "--file",
        str(partial),
    ]
    try:
        runner(command, env=target.subprocess_env(), check=True)
        if not partial.is_file() or partial.stat().st_size == 0:
            raise RuntimeError("pg_dump completed without producing a non-empty backup")
        partial.replace(destination)
    except Exception:
        if partial.exists():
            partial.unlink()
        raise
    return destination


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--host")
    parser.add_argument("--port")
    parser.add_argument("--database")
    parser.add_argument("--user")
    parser.add_argument("--pg-dump", default="pg_dump")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    target = PostgresTarget.from_env(
        host=args.host,
        port=args.port,
        database=args.database,
        user=args.user,
    )
    output = backup_database(
        target,
        args.output or default_output(target.database),
        executable=args.pg_dump,
        overwrite=args.overwrite,
    )
    print(json.dumps({"status": "completed", "database": target.database, "output": str(output)}))


if __name__ == "__main__":
    main()
