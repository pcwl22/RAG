import asyncio
import hashlib
from pathlib import Path

import pytest

from app.vectorstore import postgres_store
from scripts.backup_postgres import backup_database
from scripts.postgres_cli import PostgresTarget
from scripts.restore_postgres import restore_database


class _ConcurrentIndexCursor:
    def __init__(self, connection):
        self.connection = connection
        self._row = None

    def execute(self, statement, params=None):
        normalized = " ".join(str(statement).split())
        self.connection.events.append(
            ("execute", normalized, self.connection.autocommit, params)
        )
        self._row = None
        if normalized.startswith("SELECT checksum FROM rag_schema_migrations"):
            if self.connection.migration_checksum is not None:
                self._row = (self.connection.migration_checksum,)
            return
        if normalized.startswith("SELECT index_row.indisvalid"):
            state = self.connection.index_states.get(params[0])
            self._row = None if state is None else (state,)
            return
        if normalized.startswith("DROP INDEX CONCURRENTLY"):
            index_name = normalized.rsplit("public.", 1)[1]
            self.connection.index_states.pop(index_name, None)
            return
        if normalized.startswith("CREATE INDEX CONCURRENTLY"):
            index_name = normalized.split("IF NOT EXISTS ", 1)[1].split()[0]
            if index_name == self.connection.fail_index:
                raise RuntimeError("simulated concurrent index failure")
            self.connection.index_states[index_name] = True
            return
        if normalized.startswith("INSERT INTO rag_schema_migrations"):
            self.connection.migration_checksum = params[1]

    def fetchone(self):
        return self._row

    def close(self):
        self.connection.events.append(("close", self.connection.autocommit))


class _ConcurrentIndexConnection:
    def __init__(self, *, index_states=None, fail_index=None):
        self.autocommit = False
        self.index_states = dict(index_states or {})
        self.fail_index = fail_index
        self.migration_checksum = None
        self.events = []

    def cursor(self):
        return _ConcurrentIndexCursor(self)

    def commit(self):
        self.events.append(("commit", self.autocommit))

    def rollback(self):
        self.events.append(("rollback", self.autocommit))


class _StagedMigrationCursor:
    def __init__(self, connection):
        self.connection = connection
        self._rows = []
        self._row = None

    def execute(self, statement, params=None):
        normalized = " ".join(str(statement).split())
        self.connection.events.append(
            ("execute", normalized, self.connection.autocommit, params)
        )
        self._rows = []
        self._row = None
        if normalized.startswith("SELECT checksum FROM rag_schema_migrations"):
            checksum = self.connection.migrations.get(params[0])
            self._row = None if checksum is None else (checksum,)
            return
        if normalized.startswith("WITH batch AS"):
            if self.connection.rows_locked:
                updated = 0
            else:
                updated = min(int(params[0]), self.connection.null_rows)
                self.connection.null_rows -= updated
            self._rows = [(1,)] * updated
            return
        if normalized.startswith(
            "SELECT COUNT(*) FROM public.documents WHERE tenant_id IS NULL"
        ):
            self._row = (self.connection.null_rows,)
            return
        if normalized.startswith("SELECT constraint_row.conname, array_agg"):
            self._row = (
                None
                if self.connection.primary_key is None
                else (self.connection.primary_key[0], list(self.connection.primary_key[1]))
            )
            return
        if normalized.startswith(
            "SELECT index_row.indisvalid AND index_row.indisready, index_row.indisunique"
        ):
            state = self.connection.composite_index
            self._row = (
                None if state is None else (state[0], state[1], list(state[2]))
            )
            return
        if normalized.startswith("DROP INDEX CONCURRENTLY"):
            self.connection.composite_index = None
            return
        if normalized.startswith("CREATE UNIQUE INDEX CONCURRENTLY"):
            if self.connection.fail_index:
                self.connection.composite_index = (
                    False,
                    True,
                    ("tenant_id", "id"),
                )
                raise RuntimeError("simulated unique index failure")
            self.connection.composite_index = (True, True, ("tenant_id", "id"))
            return
        if normalized.startswith("ALTER TABLE public.documents"):
            if self.connection.fail_cutover:
                raise RuntimeError("simulated lock timeout")
            self.connection.primary_key = ("documents_pkey", ("tenant_id", "id"))
            self.connection.composite_index = None
            return
        if normalized.startswith("INSERT INTO rag_schema_migrations"):
            self.connection.migrations[params[0]] = params[1]

    def fetchone(self):
        return self._row

    def fetchall(self):
        return self._rows

    def close(self):
        self.connection.events.append(("close", self.connection.autocommit))


class _StagedMigrationConnection:
    def __init__(
        self,
        *,
        null_rows=0,
        rows_locked=False,
        primary_key=("documents_pkey", ("id",)),
        composite_index=None,
        fail_index=False,
        fail_cutover=False,
    ):
        self.autocommit = False
        self.null_rows = null_rows
        self.rows_locked = rows_locked
        self.primary_key = primary_key
        self.composite_index = composite_index
        self.fail_index = fail_index
        self.fail_cutover = fail_cutover
        self.migrations = {}
        self.events = []

    def cursor(self):
        return _StagedMigrationCursor(self)

    def commit(self):
        self.events.append(("commit", self.autocommit))

    def rollback(self):
        self.events.append(("rollback", self.autocommit))


def test_migration_can_use_a_separate_bounded_statement_timeout(monkeypatch):
    monkeypatch.setattr(
        postgres_store,
        "get_config_section",
        lambda *_path: {"command_timeout": 60, "lock_timeout": 5},
    )
    monkeypatch.setenv("POSTGRES_COMMAND_TIMEOUT_SECONDS", "840")
    monkeypatch.setenv("POSTGRES_LOCK_TIMEOUT_SECONDS", "30")

    config = postgres_store._postgres_config()

    assert config["command_timeout"] == 840
    assert config["lock_timeout"] == 30


def test_tenant_backfill_commits_bounded_skip_locked_batches(monkeypatch):
    migration = postgres_store.STAGED_TENANT_MIGRATIONS[1]
    connection = _StagedMigrationConnection(null_rows=5)
    monkeypatch.setenv("POSTGRES_MIGRATION_BATCH_SIZE", "2")

    postgres_store._backfill_tenant_ids(connection, migration)

    updates = [
        event
        for event in connection.events
        if event[0] == "execute" and event[1].startswith("WITH batch AS")
    ]
    assert len(updates) == 4
    assert all("FOR UPDATE SKIP LOCKED" in event[1] for event in updates)
    assert connection.null_rows == 0
    assert connection.migrations[migration[0]] == migration[1]
    assert sum(event[0] == "commit" for event in connection.events) >= 4

    update_count = len(updates)
    postgres_store._backfill_tenant_ids(connection, migration)
    assert (
        sum(
            event[0] == "execute" and event[1].startswith("WITH batch AS")
            for event in connection.events
        )
        == update_count
    )


def test_tenant_backfill_does_not_mark_locked_rows_and_resumes(monkeypatch):
    migration = postgres_store.STAGED_TENANT_MIGRATIONS[1]
    connection = _StagedMigrationConnection(null_rows=3, rows_locked=True)
    monkeypatch.setenv("POSTGRES_MIGRATION_BATCH_SIZE", "2")

    with pytest.raises(RuntimeError, match="rows locked"):
        postgres_store._backfill_tenant_ids(connection, migration)
    assert migration[0] not in connection.migrations
    assert connection.null_rows == 3

    connection.rows_locked = False
    postgres_store._backfill_tenant_ids(connection, migration)
    assert connection.null_rows == 0
    assert connection.migrations[migration[0]] == migration[1]


def test_online_tenant_ddl_uses_a_bounded_lock_wait(monkeypatch):
    expand = postgres_store.STAGED_TENANT_MIGRATIONS[0]
    constraints = postgres_store.STAGED_TENANT_MIGRATIONS[2]
    connection = _StagedMigrationConnection()
    monkeypatch.setenv("POSTGRES_MIGRATION_LOCK_TIMEOUT_SECONDS", "2")
    monkeypatch.setattr(
        postgres_store,
        "_tenant_constraints",
        lambda _cur: (
            ("c", True, "CHECK (tenant_id IS NOT NULL)"),
            ("f", True, "FOREIGN KEY (tenant_id) REFERENCES rag_tenants(id)"),
        ),
    )
    monkeypatch.setattr(postgres_store, "_tenant_column_not_null", lambda _cur: True)

    postgres_store._expand_tenant_schema(connection, expand)
    postgres_store._validate_tenant_constraints(connection, constraints)

    lock_timeout_events = [
        event
        for event in connection.events
        if event[0] == "execute" and event[1].startswith("SELECT set_config('lock_timeout'")
    ]
    assert len(lock_timeout_events) == 2
    assert all(event[3] == ("2s",) for event in lock_timeout_events)
    assert connection.migrations[expand[0]] == expand[1]
    assert connection.migrations[constraints[0]] == constraints[1]


def test_composite_index_and_cutover_retry_after_bounded_lock_failure(monkeypatch):
    legacy_migration = postgres_store.LEGACY_SCHEMA_MIGRATIONS[1]
    index_migration = postgres_store.STAGED_TENANT_MIGRATIONS[3]
    cutover_migration = postgres_store.STAGED_TENANT_MIGRATIONS[4]
    connection = _StagedMigrationConnection(fail_cutover=True)
    monkeypatch.setenv("POSTGRES_ALLOW_LEGACY_PK_CUTOVER", "true")
    monkeypatch.setenv("POSTGRES_PK_CUTOVER_LOCK_TIMEOUT_SECONDS", "1")

    postgres_store._prepare_composite_key_index(connection, index_migration)
    assert connection.composite_index == (True, True, ("tenant_id", "id"))
    assert connection.migrations[index_migration[0]] == index_migration[1]

    with pytest.raises(RuntimeError, match="simulated lock timeout"):
        postgres_store._cutover_composite_primary_key(
            connection, cutover_migration, legacy_migration
        )
    assert connection.primary_key == ("documents_pkey", ("id",))
    assert cutover_migration[0] not in connection.migrations
    assert legacy_migration[0] not in connection.migrations

    connection.fail_cutover = False
    postgres_store._cutover_composite_primary_key(
        connection, cutover_migration, legacy_migration
    )
    assert connection.primary_key == ("documents_pkey", ("tenant_id", "id"))
    assert connection.migrations[cutover_migration[0]] == cutover_migration[1]
    assert connection.migrations[legacy_migration[0]] == legacy_migration[1]
    assert any(
        event[0] == "execute"
        and event[1].startswith("SELECT set_config('lock_timeout'")
        and event[3] == ("1s",)
        for event in connection.events
    )


def test_legacy_primary_key_cutover_requires_explicit_maintenance_gate(
    monkeypatch,
):
    legacy_migration = postgres_store.LEGACY_SCHEMA_MIGRATIONS[1]
    cutover_migration = postgres_store.STAGED_TENANT_MIGRATIONS[4]
    connection = _StagedMigrationConnection(
        composite_index=(True, True, ("tenant_id", "id"))
    )
    monkeypatch.delenv("POSTGRES_ALLOW_LEGACY_PK_CUTOVER", raising=False)

    with pytest.raises(RuntimeError, match="traffic"):
        postgres_store._cutover_composite_primary_key(
            connection, cutover_migration, legacy_migration
        )
    assert connection.primary_key == ("documents_pkey", ("id",))
    assert connection.migrations == {}


def test_concurrent_initializers_wait_for_schema_readiness(monkeypatch):
    class FakePool:
        def __init__(self):
            self.closed = False

        def closeall(self):
            self.closed = True

    fake_pool = FakePool()
    schema_started = asyncio.Event()
    release_schema = asyncio.Event()
    schema_calls = 0
    pool_calls = 0

    def create_pool(*args, **kwargs):
        nonlocal pool_calls
        pool_calls += 1
        return fake_pool

    async def execute_schema(*args, **kwargs):
        nonlocal schema_calls
        schema_calls += 1
        schema_started.set()
        await release_schema.wait()

    monkeypatch.setattr(postgres_store, "_pool", None)
    monkeypatch.setattr(postgres_store, "_executor", None)
    monkeypatch.setattr(postgres_store, "_schema_ready", False)
    monkeypatch.setattr(postgres_store, "_init_lock", asyncio.Lock())
    monkeypatch.setattr(postgres_store, "_postgres_config", lambda: {"max_pool_size": 2})
    monkeypatch.setattr(postgres_store, "_auto_migrate_enabled", lambda: True)
    monkeypatch.setattr(
        postgres_store.psycopg2.pool,
        "ThreadedConnectionPool",
        create_pool,
    )
    monkeypatch.setattr(postgres_store, "_execute_sync", execute_schema)

    async def run():
        first = asyncio.create_task(postgres_store.init_postgres_store())
        await schema_started.wait()
        second = asyncio.create_task(postgres_store.init_postgres_store())
        await asyncio.sleep(0)

        assert second.done() is False
        assert postgres_store._schema_ready is False
        with pytest.raises(RuntimeError, match="schema is not ready"):
            postgres_store._connection()
        assert await postgres_store.check_postgres_health() is False

        release_schema.set()
        await asyncio.gather(first, second)

        assert postgres_store._schema_ready is True
        assert schema_calls == 1
        assert pool_calls == 1
        await postgres_store.close_postgres_store()

    asyncio.run(run())
    assert fake_pool.closed is True


def test_postgres_pool_receives_verified_tls_settings(monkeypatch):
    captured = {}

    class FakePool:
        def closeall(self):
            captured["closed"] = True

    def create_pool(**kwargs):
        captured["pool"] = kwargs
        return FakePool()

    async def complete_schema(*_args, **_kwargs):
        return None

    monkeypatch.setattr(postgres_store, "_pool", None)
    monkeypatch.setattr(postgres_store, "_executor", None)
    monkeypatch.setattr(postgres_store, "_schema_ready", False)
    monkeypatch.setattr(postgres_store, "_init_lock", asyncio.Lock())
    monkeypatch.setattr(
        postgres_store,
        "_postgres_config",
        lambda: {
            "host": "postgres.internal",
            "port": 5432,
            "database": "rag",
            "user": "rag_runtime",
            "password": "secret",
            "min_pool_size": 2,
            "max_pool_size": 7,
            "sslmode": "verify-full",
            "sslrootcert": "/etc/ssl/certs/internal-ca.pem",
        },
    )
    monkeypatch.setattr(postgres_store, "_auto_migrate_enabled", lambda: True)
    monkeypatch.setattr(
        postgres_store.psycopg2.pool,
        "ThreadedConnectionPool",
        create_pool,
    )
    monkeypatch.setattr(postgres_store, "_execute_sync", complete_schema)

    async def run():
        await postgres_store.init_postgres_store()
        await postgres_store.close_postgres_store()

    asyncio.run(run())

    assert captured["pool"]["sslmode"] == "verify-full"
    assert captured["pool"]["sslrootcert"] == "/etc/ssl/certs/internal-ca.pem"
    assert captured["pool"]["minconn"] == 2
    assert captured["pool"]["maxconn"] == 7
    assert captured["closed"] is True


def test_postgres_advisory_lease_is_released_before_pool_return(monkeypatch):
    events = []

    class Cursor:
        def execute(self, statement, params):
            events.append((statement, params))

        def fetchone(self):
            return (True,)

        def close(self):
            events.append(("cursor:close", ()))

    class Connection:
        def cursor(self):
            return Cursor()

        def rollback(self):
            events.append(("rollback", ()))

    class Pool:
        def __init__(self):
            self.connection = Connection()

        def getconn(self):
            return self.connection

        def putconn(self, connection, close=False):
            assert connection is self.connection
            events.append(("pool:return", (close,)))

    monkeypatch.setattr(postgres_store, "_pool", Pool())

    with postgres_store.postgres_advisory_lease("document-ingest:task-1"):
        events.append(("work", ()))

    assert "pg_try_advisory_lock" in events[0][0]
    assert events[1] == ("work", ())
    assert "pg_advisory_unlock" in events[2][0]
    assert events[-2:] == [("rollback", ()), ("pool:return", (False,))]


def test_concurrent_index_migration_repairs_invalid_index_and_records_last():
    first_index = postgres_store.CONCURRENT_INDEX_DEFINITIONS[0][0]
    second_index = postgres_store.CONCURRENT_INDEX_DEFINITIONS[1][0]
    connection = _ConcurrentIndexConnection(
        index_states={first_index: False, second_index: True}
    )

    postgres_store._run_search_index_migration_concurrently(
        connection,
        "search-index-migration",
        "sha256:expected",
    )

    executed = [event for event in connection.events if event[0] == "execute"]
    ddl = [event for event in executed if " INDEX CONCURRENTLY " in event[1]]
    record_position = next(
        index
        for index, event in enumerate(connection.events)
        if event[0] == "execute"
        and event[1].startswith("INSERT INTO rag_schema_migrations")
    )
    last_ddl_position = max(
        index
        for index, event in enumerate(connection.events)
        if event[0] == "execute" and " INDEX CONCURRENTLY " in event[1]
    )

    assert all(event[2] is True for event in ddl)
    assert any(
        event[1]
        == f"DROP INDEX CONCURRENTLY IF EXISTS public.{first_index}"
        for event in ddl
    )
    assert not any(
        event[1].startswith("CREATE INDEX CONCURRENTLY IF NOT EXISTS ")
        and event[1].split("IF NOT EXISTS ", 1)[1].split()[0] == second_index
        for event in ddl
    )
    assert last_ddl_position < record_position
    assert connection.migration_checksum == "sha256:expected"
    assert connection.autocommit is False
    assert any("pg_advisory_lock" in event[1] for event in executed)
    assert any("pg_advisory_unlock" in event[1] for event in executed)


def test_concurrent_index_migration_failure_is_safely_rerunnable():
    first_index = postgres_store.CONCURRENT_INDEX_DEFINITIONS[0][0]
    failed_index = postgres_store.CONCURRENT_INDEX_DEFINITIONS[1][0]
    connection = _ConcurrentIndexConnection(
        index_states={first_index: False},
        fail_index=failed_index,
    )

    with pytest.raises(RuntimeError, match="simulated concurrent index failure"):
        postgres_store._run_search_index_migration_concurrently(
            connection,
            "search-index-migration",
            "sha256:expected",
        )

    assert connection.migration_checksum is None
    assert connection.index_states[first_index] is True
    assert connection.autocommit is False
    assert any(
        event[0] == "execute" and "pg_advisory_unlock" in event[1]
        for event in connection.events
    )

    connection.fail_index = None
    postgres_store._run_search_index_migration_concurrently(
        connection,
        "search-index-migration",
        "sha256:expected",
    )
    first_index_creates = [
        event
        for event in connection.events
        if event[0] == "execute"
        and event[1].startswith(
            f"CREATE INDEX CONCURRENTLY IF NOT EXISTS {first_index}"
        )
    ]

    assert len(first_index_creates) == 1
    assert connection.migration_checksum == "sha256:expected"

    ddl_count = sum(
        event[0] == "execute" and " INDEX CONCURRENTLY " in event[1]
        for event in connection.events
    )
    postgres_store._run_search_index_migration_concurrently(
        connection,
        "search-index-migration",
        "sha256:expected",
    )
    assert (
        sum(
            event[0] == "execute" and " INDEX CONCURRENTLY " in event[1]
            for event in connection.events
        )
        == ddl_count
    )


def test_postgres_target_prefers_standard_pg_environment():
    target = PostgresTarget.from_env(
        {
            "PGHOST": "database.internal",
            "PGPORT": "6432",
            "PGDATABASE": "production_rag",
            "PGUSER": "backup_user",
            "PGPASSWORD": "secret",
            "PGSSLMODE": "verify-full",
            "PGSSLROOTCERT": "/etc/ssl/certs/ca-certificates.crt",
            "POSTGRES_HOST": "ignored",
        }
    )

    assert target.host == "database.internal"
    assert target.port == "6432"
    assert target.database == "production_rag"
    assert target.user == "backup_user"
    assert target.sslmode == "verify-full"
    assert target.sslrootcert == "/etc/ssl/certs/ca-certificates.crt"


def test_postgres_target_maps_release_tls_values_to_libpq_environment():
    target = PostgresTarget.from_env(
        {
            "POSTGRES_SSLMODE": "verify-full",
            "POSTGRES_SSLROOTCERT": "/run/secrets/postgres-ca.crt",
        }
    )

    environment = target.subprocess_env({})
    assert environment["PGSSLMODE"] == "verify-full"
    assert environment["PGSSLROOTCERT"] == "/run/secrets/postgres-ca.crt"


def test_postgres_target_canonical_tls_cannot_be_downgraded_by_ambient_pg_values():
    target = PostgresTarget.from_env(
        {
            "RAG_ENV": "base",
            "POSTGRES_HOST": "production-db.internal",
            "POSTGRES_DB": "production_rag",
            "POSTGRES_PASSWORD": "release-password",
            "POSTGRES_SSLMODE": "verify-full",
            "POSTGRES_SSLROOTCERT": "/run/secrets/postgres-ca.crt",
            "PGHOST": "wrong-db.internal",
            "PGDATABASE": "wrong_database",
            "PGPASSWORD": "stale-password",
            "PGSSLMODE": "disable",
            "PGSSLROOTCERT": "/tmp/untrusted-ca.crt",
        }
    )

    assert target.host == "production-db.internal"
    assert target.database == "production_rag"
    assert target.password == "release-password"
    assert target.sslmode == "verify-full"
    assert target.sslrootcert == "/run/secrets/postgres-ca.crt"
    environment = target.subprocess_env({"PGSSLMODE": "disable"})
    assert environment["PGSSLMODE"] == "verify-full"
    assert environment["PGSSLROOTCERT"] == "/run/secrets/postgres-ca.crt"


def test_postgres_target_fails_closed_on_weak_production_tls():
    with pytest.raises(ValueError, match="verify-full"):
        PostgresTarget.from_env(
            {
                "RAG_ENV": "base",
                "POSTGRES_SSLMODE": "disable",
                "POSTGRES_SSLROOTCERT": "/run/secrets/postgres-ca.crt",
            }
        )


def test_backup_is_atomic_and_password_is_not_in_command(tmp_path):
    calls = []
    target = PostgresTarget("localhost", "5432", "rag_db", "backup", "secret")
    output = tmp_path / "rag.dump"

    def fake_run(command, **kwargs):
        calls.append((command, kwargs))
        partial = Path(command[command.index("--file") + 1])
        partial.write_bytes(b"postgres archive")

    result = backup_database(target, output, runner=fake_run)

    assert result == output.resolve()
    assert output.read_bytes() == b"postgres archive"
    assert "secret" not in calls[0][0]
    assert calls[0][1]["env"]["PGPASSWORD"] == "secret"
    assert calls[0][1]["check"] is True


def test_restore_requires_exact_database_confirmation(tmp_path):
    archive = tmp_path / "rag.dump"
    archive.write_bytes(b"postgres archive")
    target = PostgresTarget("localhost", "5432", "rag_db", "postgres")

    with pytest.raises(ValueError, match="confirmation does not match"):
        restore_database(
            target,
            archive,
            confirmed_database="another_database",
            runner=lambda *_args, **_kwargs: pytest.fail("restore must not run"),
        )


def test_restore_validates_archive_before_clean_restore(tmp_path):
    archive = tmp_path / "rag.dump"
    archive.write_bytes(b"postgres archive")
    target = PostgresTarget("localhost", "5432", "rag_db", "postgres", "secret")
    calls = []

    def fake_run(command, **kwargs):
        calls.append((command, kwargs))

    restore_database(
        target,
        archive,
        confirmed_database="rag_db",
        clean=True,
        runner=fake_run,
    )

    assert calls[0][0] == ["pg_restore", "--list", str(archive.resolve())]
    assert "--clean" in calls[1][0]
    assert "--if-exists" in calls[1][0]
    assert "--single-transaction" in calls[1][0]
    assert all(call[1]["check"] is True for call in calls)


def test_sql_migration_checksum_is_derived_from_file_content():
    migration_id, checksum, sql_text = postgres_store.CONTENT_SCHEMA_MIGRATIONS[0]

    assert migration_id == "20260805_01_schema_v3_contract"
    assert checksum == "sha256:" + hashlib.sha256(sql_text.encode("utf-8")).hexdigest()
    assert "documents_tenant_isolation" in sql_text
