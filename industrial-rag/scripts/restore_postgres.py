"""Validate and restore a custom-format PostgreSQL backup with pg_restore."""

from __future__ import annotations

import argparse
import json
import subprocess
from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    # The packaged path is the canonical one for type checking; the runtime
    # fallback below only exists so `python scripts/restore_postgres.py` works.
    from scripts.postgres_cli import PostgresTarget
else:
    try:
        from scripts.postgres_cli import PostgresTarget
    except ModuleNotFoundError:  # Direct execution adds only scripts/ to sys.path.
        from postgres_cli import PostgresTarget


def restore_database(
    target: PostgresTarget,
    source: Path,
    *,
    confirmed_database: str,
    executable: str = "pg_restore",
    clean: bool = False,
    runner: Callable[..., subprocess.CompletedProcess] = subprocess.run,
) -> Path:
    if confirmed_database != target.database:
        raise ValueError(
            "Restore confirmation does not match the target database; "
            f"pass --confirm-database {target.database}"
        )
    archive = source.expanduser().resolve()
    if not archive.is_file():
        raise FileNotFoundError(f"Backup archive not found: {archive}")

    environment = target.subprocess_env()
    runner([executable, "--list", str(archive)], env=environment, check=True)
    command = [
        executable,
        *target.connection_args(),
        "--exit-on-error",
        "--single-transaction",
        "--no-owner",
        "--no-privileges",
    ]
    if clean:
        command.extend(["--clean", "--if-exists"])
    command.append(str(archive))
    runner(command, env=environment, check=True)
    return archive


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--confirm-database", required=True)
    parser.add_argument("--host")
    parser.add_argument("--port")
    parser.add_argument("--database")
    parser.add_argument("--user")
    parser.add_argument("--pg-restore", default="pg_restore")
    parser.add_argument(
        "--clean",
        action="store_true",
        help="Drop matching objects before restore; use only for an approved recovery operation",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    target = PostgresTarget.from_env(
        host=args.host,
        port=args.port,
        database=args.database,
        user=args.user,
    )
    archive = restore_database(
        target,
        args.input,
        confirmed_database=args.confirm_database,
        executable=args.pg_restore,
        clean=args.clean,
    )
    print(json.dumps({"status": "completed", "database": target.database, "input": str(archive)}))


if __name__ == "__main__":
    main()
