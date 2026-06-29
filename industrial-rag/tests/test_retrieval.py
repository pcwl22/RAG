"""Retrieval helper tests."""
import asyncio

from app.retrieval import dense, hybrid, reranker
from app.vectorstore.postgres_store import _extract_chinese_keywords


def test_candidate_count_bounds():
    config = {
        "rerank_candidate_multiplier": 4,
        "rerank_min_candidates": 20,
        "rerank_max_candidates": 80,
    }

    assert dense._candidate_count(5, True, config) == 20
    assert dense._candidate_count(30, True, config) == 80
    assert hybrid._candidate_count(5, False, config) == 5


def test_dense_deduplicates_by_document_and_chunk():
    docs = [
        {"id": "a", "score": 0.9, "metadata": {"document_id": "doc1", "chunk_index": 0}},
        {"id": "a_dup", "score": 0.8, "metadata": {"document_id": "doc1", "chunk_index": 0}},
        {"id": "b", "score": 0.7, "metadata": {"document_id": "doc1", "chunk_index": 1}},
        {"id": "c", "score": 0.6, "metadata": {"document_id": "doc2", "chunk_index": 0}},
    ]

    deduped = dense._deduplicate_results(docs, max_per_document=1)

    assert [doc["id"] for doc in deduped] == ["a", "c"]


def test_reranker_falls_back_when_model_unavailable(monkeypatch):
    docs = [
        {"id": "a", "score": 0.9, "content": "first"},
        {"id": "b", "score": 0.8, "content": "second"},
    ]
    monkeypatch.setattr(reranker, "load_reranker", lambda: None)

    assert reranker.rerank_documents("query", docs, top_n=1) == [docs[0]]


def test_legal_keyword_extraction_uses_query_terms_without_article_mapping():
    keywords = _extract_chinese_keywords("多次贩卖含依托咪酯的上头电子烟，如何定罪？")

    assert "第三百四十七条" not in keywords
    assert "第三百五十七条" not in keywords
    assert any("依托咪酯" in keyword for keyword in keywords)
    assert any("电子烟" in keyword for keyword in keywords)


def test_retrievers_default_threshold_matches_config_default(monkeypatch):
    docs = [
        {"id": "low", "score": 0.04, "metadata": {"document_id": "doc1", "chunk_index": 0}},
        {"id": "kept", "score": 0.05, "metadata": {"document_id": "doc1", "chunk_index": 1}},
    ]

    async def fake_search(**kwargs):
        return docs

    async def fake_hybrid_search(**kwargs):
        return docs

    monkeypatch.setattr(dense, "_retrieval_config", lambda: {})
    monkeypatch.setattr(hybrid, "_retrieval_config", lambda: {})
    monkeypatch.setattr(dense, "encode_query", lambda query: [0.0])
    monkeypatch.setattr(hybrid, "encode_query", lambda query: [0.0])
    monkeypatch.setattr(dense.storage_adapter, "search", fake_search)
    monkeypatch.setattr(hybrid.storage_adapter, "hybrid_search", fake_hybrid_search)

    async def run():
        dense_results = await dense.RetrievalEngine().retrieve("query", enable_rerank=False)
        hybrid_results = await hybrid.HybridRetrievalEngine().retrieve("query", enable_rerank=False)

        assert [doc["id"] for doc in dense_results] == ["kept"]
        assert [doc["id"] for doc in hybrid_results] == ["kept"]

    asyncio.run(run())
