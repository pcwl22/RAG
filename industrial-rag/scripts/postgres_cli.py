"""Shared PostgreSQL command-line configuration for maintenance scripts."""

from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import dataclass


@dataclass(frozen=True)
class PostgresTarget:
    host: str
    port: str
    database: str
    user: str
    password: str | None = None

    @classmethod
    def from_env(
        cls,
        environ: Mapping[str, str] | None = None,
        *,
        host: str | None = None,
        port: str | None = None,
        database: str | None = None,
        user: str | None = None,
    ) -> PostgresTarget:
        env = environ if environ is not None else os.environ
        return cls(
            host=host or env.get("PGHOST") or env.get("POSTGRES_HOST") or "127.0.0.1",
            port=str(port or env.get("PGPORT") or env.get("POSTGRES_PORT") or "5432"),
            database=database or env.get("PGDATABASE") or env.get("POSTGRES_DB") or "rag_db",
            user=user or env.get("PGUSER") or env.get("POSTGRES_USER") or "postgres",
            password=env.get("PGPASSWORD") or env.get("POSTGRES_PASSWORD"),
        )

    def connection_args(self) -> list[str]:
        return [
            "--host",
            self.host,
            "--port",
            self.port,
            "--username",
            self.user,
            "--dbname",
            self.database,
        ]

    def subprocess_env(self, environ: Mapping[str, str] | None = None) -> dict[str, str]:
        result = dict(environ if environ is not None else os.environ)
        if self.password:
            result["PGPASSWORD"] = self.password
        return result
