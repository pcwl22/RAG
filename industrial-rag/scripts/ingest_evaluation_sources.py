"""Ingest and verify supplemental legal sources used by the evaluation suite."""
from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
WORKSPACE_ROOT = PROJECT_ROOT.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.parser.document_parser import parse_document  # noqa: E402
from app.parser.legal_parser import build_legal_article_chunks  # noqa: E402
from app.service.ingest_service import process_document  # noqa: E402
from app.vectorstore.storage_adapter import (  # noqa: E402
    close_vector_store,
    init_vector_store,
    list_documents,
)

EXPECTED_IDS = {
    "劳动合同法实施条例_劳动合同的订立_5条",
    "劳动合同法实施条例_劳动合同的订立_6条",
}


def find_source() -> Path:
    matches = sorted((WORKSPACE_ROOT / "date").glob("*实施条例*官方摘录.txt"))
    if len(matches) != 1:
        raise RuntimeError(f"Expected one official regulation excerpt, found {len(matches)}")
    return matches[0]


def verify_parser(source: Path) -> list[str]:
    chunks = build_legal_article_chunks(parse_document(str(source)), source.name)
    raw_ids = [(item.get("metadata") or {}).get("semantic_chunk_id") for item in chunks]
    ids = [str(chunk_id) for chunk_id in raw_ids if chunk_id is not None]
    missing = sorted(EXPECTED_IDS - set(ids))
    if missing:
        raise RuntimeError(f"Parser did not produce required article IDs: {missing}; got: {ids}")
    return ids


async def run(force: bool) -> dict:
    source = find_source()
    parsed_ids = verify_parser(source)
    await init_vector_store()
    try:
        existing = await list_documents(0, 1000)
        already_ingested = any(item.get("filename") == source.name for item in existing)
        if already_ingested and not force:
            result = {"status": "already_ingested", "filename": source.name}
        else:
            result = await process_document(
                str(source),
                source.name,
                metadata={
                    "official_source_url": (
                        "https://www.gov.cn/zwgk/2008-09/19/content_1099470.htm"
                    ),
                    "source_authority": "中华人民共和国国务院",
                    "source_scope": "第二章第四条至第七条官方摘录",
                },
            )
            if result.get("status") != "completed":
                raise RuntimeError(f"Ingestion failed: {result}")
        return {**result, "filename": source.name, "verified_article_ids": parsed_ids}
    finally:
        await close_vector_store()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    print(json.dumps(asyncio.run(run(args.force)), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
