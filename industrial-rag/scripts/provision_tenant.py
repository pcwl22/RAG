"""Create or rename a tenant in the shared-schema PostgreSQL store."""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.auth import normalize_tenant_id  # noqa: E402
from app.vectorstore import postgres_store  # noqa: E402


async def provision_tenant(tenant_id: str, name: str) -> dict[str, str]:
    resolved_id = normalize_tenant_id(tenant_id)
    resolved_name = name.strip()
    if not resolved_name:
        raise ValueError("Tenant name must not be empty")

    await postgres_store.init_postgres_store()
    conn = postgres_store._connection()
    cur = conn.cursor()
    try:
        cur.execute(
            """
            INSERT INTO rag_tenants (id, name)
            VALUES (%s::uuid, %s)
            ON CONFLICT (id) DO UPDATE SET name = EXCLUDED.name
            """,
            (resolved_id, resolved_name),
        )
        conn.commit()
        return {"tenant_id": resolved_id, "name": resolved_name, "status": "ready"}
    except Exception:
        conn.rollback()
        raise
    finally:
        cur.close()
        postgres_store._return_connection(conn)
        await postgres_store.close_postgres_store()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tenant-id", required=True, help="Tenant UUID used in OIDC tokens")
    parser.add_argument("--name", required=True, help="Human-readable tenant name")
    args = parser.parse_args()
    result = asyncio.run(provision_tenant(args.tenant_id, args.name))
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
