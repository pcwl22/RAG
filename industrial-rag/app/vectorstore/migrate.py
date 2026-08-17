"""Run PostgreSQL schema migration and runtime-role provisioning once."""

import asyncio
import os

from app.vectorstore.postgres_store import close_postgres_store, init_postgres_store


async def migrate() -> None:
    os.environ["POSTGRES_AUTO_MIGRATE"] = "true"
    try:
        await init_postgres_store()
    finally:
        await close_postgres_store()


if __name__ == "__main__":
    asyncio.run(migrate())
