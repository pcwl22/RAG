import asyncio

from scripts import reprocess_documents as reprocess


def test_original_filename_only_strips_uuid_upload_prefix():
    prefixed = "123e4567-e89b-12d3-a456-426614174000_contract.pdf"
    assert reprocess._original_filename(reprocess.Path(prefixed)) == "contract.pdf"
    assert reprocess._original_filename(reprocess.Path("legal_contract.pdf")) == "legal_contract.pdf"


def test_reprocess_uses_explicit_tenant_and_canonical_ingest(monkeypatch, tmp_path):
    tenant_id = "00000000-0000-0000-0000-00000000000a"
    (tmp_path / "123e4567-e89b-12d3-a456-426614174000_first.pdf").write_bytes(b"pdf")
    (tmp_path / "second.pdf").write_bytes(b"pdf")
    calls = []

    async def init():
        calls.append("init")

    async def close():
        calls.append("close")

    async def process(file_path, filename, partition, metadata, *, tenant_id):
        calls.append((filename, partition, metadata, tenant_id))
        return {"status": "completed", "total_chunks": 1}

    monkeypatch.setattr(reprocess, "init_vector_store", init)
    monkeypatch.setattr(reprocess, "close_vector_store", close)
    monkeypatch.setattr(reprocess, "process_document", process)

    result = asyncio.run(reprocess.reprocess_documents(tenant_id, upload_dir=tmp_path))

    assert result == {"processed": 2, "failed": 0, "total": 2}
    assert calls[0] == "init"
    assert calls[-1] == "close"
    assert [call[0] for call in calls[1:-1]] == ["first.pdf", "second.pdf"]
    assert all(call[3] == tenant_id for call in calls[1:-1])
