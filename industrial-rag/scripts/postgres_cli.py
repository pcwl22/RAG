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
    sslmode: str | None = None
    sslrootcert: str | None = None

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
        secure_required = str(env.get("RAG_SECURE_MODE", "")).strip().lower() in {
            "1",
            "true",
            "yes",
            "on",
        } or str(env.get("RAG_ENV", "")).strip().lower() == "base"

        def selected(
            explicit: str | None,
            standard_name: str,
            canonical_name: str,
            default: str,
        ) -> str:
            if explicit:
                return explicit
            if secure_required:
                return env.get(canonical_name) or env.get(standard_name) or default
            return env.get(standard_name) or env.get(canonical_name) or default

        target = cls(
            host=selected(host, "PGHOST", "POSTGRES_HOST", "127.0.0.1"),
            port=selected(port, "PGPORT", "POSTGRES_PORT", "5432"),
            database=selected(database, "PGDATABASE", "POSTGRES_DB", "rag_db"),
            user=selected(user, "PGUSER", "POSTGRES_USER", "postgres"),
            password=(
                env.get("POSTGRES_PASSWORD") or env.get("PGPASSWORD")
                if secure_required
                else env.get("PGPASSWORD") or env.get("POSTGRES_PASSWORD")
            ),
            # Release validation owns the canonical POSTGRES_* values. Do not
            # allow stale ambient libpq variables to downgrade maintenance TLS.
            sslmode=env.get("POSTGRES_SSLMODE") or env.get("PGSSLMODE"),
            sslrootcert=env.get("POSTGRES_SSLROOTCERT") or env.get("PGSSLROOTCERT"),
        )
        if secure_required:
            if target.sslmode != "verify-full":
                raise ValueError(
                    "PostgreSQL maintenance requires sslmode=verify-full in "
                    "secure/production mode"
                )
            if not str(target.sslrootcert or "").strip():
                raise ValueError(
                    "PostgreSQL maintenance requires a root certificate in "
                    "secure/production mode"
                )
        return target

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
        if self.sslmode:
            result["PGSSLMODE"] = self.sslmode
        if self.sslrootcert:
            result["PGSSLROOTCERT"] = self.sslrootcert
        return result
