"""Retrieval helper tests."""
import asyncio

import pytest

from app.retrieval import dense, hybrid, reranker
from app.retrieval.domain_signal_map import is_known_out_of_scope, load_out_of_scope_signal_groups
from app.retrieval.legal_concept_map import match_legal_concept_articles
from app.retrieval.result_merge import merge_retrieval_results, result_score
from app.vectorstore import postgres_store
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


def test_result_score_uses_first_valid_rank_score():
    assert (
        result_score(
            {
                "score": "0.62",
                "rrf_score": 9.0,
                "metadata": {"rerank_prob": "not-a-number"},
            }
        )
        == 0.62
    )
    assert result_score({"metadata": {"rerank_prob": 0.0}, "score": 0.9}) == 0.0


def test_merge_prioritizes_controlled_article_ids():
    results = merge_retrieval_results(
        [
            [
                {"id": "unmapped-high", "score": 0.99, "metadata": {}},
                {
                    "id": "mapped-low",
                    "score": 0.2,
                    "metadata": {"semantic_chunk_id": "劳动合同法_劳动合同的订立_7条"},
                },
            ]
        ],
        top_k=1,
        queries=["mapped query"],
        priority_ids=["劳动合同法_劳动合同的订立_7条"],
    )

    assert [doc["id"] for doc in results] == ["mapped-low"]
    assert results[0]["mapped_article_priority"] == 1


def test_legal_keyword_extraction_uses_query_terms_without_article_mapping():
    keywords = _extract_chinese_keywords("多次贩卖含依托咪酯的上头电子烟，如何定罪？")

    assert "第三百四十七条" not in keywords
    assert "第三百五十七条" not in keywords
    assert any("依托咪酯" in keyword for keyword in keywords)
    assert any("电子烟" in keyword for keyword in keywords)


def test_legal_keyword_extraction_preserves_semantic_chunk_ids():
    keywords = _extract_chinese_keywords(
        "未缴社保 经济补偿 劳动合同法_劳动合同的解除和终止_46条"
    )

    assert "劳动合同法_劳动合同的解除和终止_46条" in keywords


def test_controlled_mappings_cover_comparison_and_coerced_accomplice():
    comparison = match_legal_concept_articles("盗窃罪与职务侵占罪的核心区别是什么？")
    comparison_ids = {
        article["semantic_chunk_id"]
        for mapping in comparison
        for article in mapping["articles"]
    }
    coerced = match_legal_concept_articles("受胁迫参加犯罪的胁从犯依法应如何处罚？")
    coerced_ids = {
        article["semantic_chunk_id"]
        for mapping in coerced
        for article in mapping["articles"]
    }

    assert "刑法_分则_侵犯财产罪_盗窃罪_264条" in comparison_ids
    assert "刑法_分则_侵犯财产罪_职务侵占罪_271条" in comparison_ids
    assert "刑法_总则_犯罪_共同犯罪_28条" in coerced_ids


def test_explicit_legal_citations_extracts_multiple_articles_and_longest_law_name():
    pairs = hybrid._explicit_legal_citations(
        "对比《中华人民共和国劳动合同法实施条例》第五条与第六条。"
    )
    assert pairs == [
        ("中华人民共和国劳动合同法实施条例", "第五条"),
        ("中华人民共和国劳动合同法实施条例", "第六条"),
    ]


def test_explicit_legal_citations_preserve_cross_law_pairing():
    assert hybrid._explicit_legal_citations(
        "比较《中华人民共和国民法典》第一条与《中华人民共和国刑法》第二条"
    ) == [
        ("中华人民共和国民法典", "第一条"),
        ("中华人民共和国刑法", "第二条"),
    ]


def test_explicit_legal_citations_normalize_common_law_aliases():
    assert hybrid._explicit_legal_citations(
        "比较民法典第一条、第二条与刑法第三条"
    ) == [
        ("中华人民共和国民法典", "第一条"),
        ("中华人民共和国民法典", "第二条"),
        ("中华人民共和国刑法", "第三条"),
    ]
    assert hybrid._explicit_legal_citations("劳动合同法实施条例第五条") == [
        ("中华人民共和国劳动合同法实施条例", "第五条")
    ]


def test_internal_recall_query_cannot_trigger_exact_citation_lookup(monkeypatch):
    exact_calls = []

    async def fake_hybrid_search(**kwargs):
        return [{"id": "fuzzy", "score": 0.9, "metadata": {}}]

    async def fake_exact_lookup(citation_pairs, partition=None):
        exact_calls.append((citation_pairs, partition))
        return [{"id": "exact", "score": 1.0, "metadata": {}}]

    monkeypatch.setattr(hybrid, "encode_query", lambda query: [0.0])
    monkeypatch.setattr(hybrid.storage_adapter, "hybrid_search", fake_hybrid_search)
    monkeypatch.setattr(
        hybrid.storage_adapter, "get_documents_by_citations", fake_exact_lookup
    )

    async def run():
        results = await hybrid.HybridRetrievalEngine().retrieve(
            "中华人民共和国刑法 第二十八条 胁从犯",
            enable_rerank=False,
            similarity_threshold=0.0,
            enable_exact_citations=False,
        )
        assert [doc["id"] for doc in results] == ["fuzzy"]

    asyncio.run(run())
    assert exact_calls == []


def test_out_of_scope_signals_distinguish_procedure_from_criminal_provision():
    assert len(load_out_of_scope_signal_groups()) >= 2
    assert is_known_out_of_scope("增值税专用发票抵扣期限如何规定") is True
    assert is_known_out_of_scope("出口退税备案材料有哪些") is True
    assert is_known_out_of_scope("无线电频率许可如何申请") is True
    assert is_known_out_of_scope("虚开增值税专用发票如何定罪") is False
    assert is_known_out_of_scope("无线电设备损坏应如何承担侵权责任") is False


def test_keyword_search_applies_partition_to_all_or_terms(monkeypatch):
    captured = {}

    class FakeCursor:
        def execute(self, sql, params):
            captured["sql"] = " ".join(sql.split())
            captured["params"] = params

        def fetchall(self):
            return []

        def close(self):
            pass

    class FakeConnection:
        def rollback(self):
            captured["rolled_back"] = True

        def cursor(self, *args, **kwargs):
            return FakeCursor()

    class FakePool:
        def putconn(self, conn, close=False):
            captured["returned_connection"] = conn
            captured["connection_discarded"] = close

    monkeypatch.setattr(postgres_store, "_connection", lambda: FakeConnection())
    monkeypatch.setattr(postgres_store, "_pool", FakePool())
    monkeypatch.setattr(postgres_store, "_extract_chinese_keywords", lambda query: ["社保", "补偿"])

    assert postgres_store._keyword_search_sync("社保补偿", top_k=5, partition="labor") == []
    assert "WHERE (" in captured["sql"]
    assert ") AND partition = %s" in captured["sql"]
    assert captured["params"][-2:] == ["labor", 5]
    assert captured["rolled_back"] is True
    assert captured["connection_discarded"] is False


def test_failed_rollback_discards_connection(monkeypatch):
    captured = {}

    class BrokenConnection:
        def rollback(self):
            raise RuntimeError("connection lost")

    class FakePool:
        def putconn(self, conn, close=False):
            captured["connection"] = conn
            captured["discarded"] = close

    connection = BrokenConnection()
    monkeypatch.setattr(postgres_store, "_pool", FakePool())

    postgres_store._return_connection(connection)

    assert captured == {"connection": connection, "discarded": True}


def test_postgres_initialization_failure_disposes_runtime(monkeypatch):
    class FakePool:
        def __init__(self):
            self.closed = False

        def closeall(self):
            self.closed = True

    fake_pool = FakePool()
    monkeypatch.setattr(postgres_store, "_pool", None)
    monkeypatch.setattr(postgres_store, "_executor", None)
    monkeypatch.setattr(
        postgres_store.psycopg2.pool,
        "ThreadedConnectionPool",
        lambda *args, **kwargs: fake_pool,
    )

    async def fail_schema(*args, **kwargs):
        raise PermissionError("schema denied")

    monkeypatch.setattr(postgres_store, "_execute_sync", fail_schema)

    async def run():
        with pytest.raises(PermissionError, match="schema denied"):
            await postgres_store.init_postgres_store()

    asyncio.run(run())
    assert fake_pool.closed is True
    assert postgres_store._pool is None
    assert postgres_store._executor is None


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
